from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from .schema_modal import Chat, User_Session
from app.services.llm import GeminiLLMClient
from app.graph.system_prompt import CHAT_SUMMARY_SYSTEM_PROMPT, CHAT_TITLE_SYSTEM_PROMPT


def get_or_create_session(db: Session, session_id: str, ip_address: str) -> User_Session:
    session = (
        db.query(User_Session).filter(User_Session.session_id == session_id).first()
    )

    if session:
        return session

    session = User_Session(session_id=session_id, ip_address=ip_address)

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def get_chunk_by_chunk_id(db: Session, chunk_id: str):
    return db.query(Chat).filter(Chat.chunk_id == chunk_id).first()


def get_active_chunk(db: Session, conversation_id: str):
    """Returns the conversation's currently-appendable chunk - the one
    chunk row per conversation with `is_active=True` (see `save_message`
    below). This is also always the conversation's most recently created
    chunk, so it doubles as "the chunk to show when opening a
    conversation"."""
    return (
        db.query(Chat)
        .filter(Chat.conversation_id == conversation_id, Chat.is_active == True)
        .first()
    )


def list_recent_chats(db: Session, session_id: str, days: int = 7):
    """Returns one row per conversation (not per chunk) from the last
    `days` days, newest-active-first, for the sidebar's history list.
    Untitled conversations are excluded - a conversation is titled as soon
    as its first exchange is saved (see `save_message`), so a missing
    title means that exchange never completed and there's nothing worth
    showing for it."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    chunks = (
        db.query(Chat)
        .filter(
            Chat.session_id == session_id,
            Chat.created_at >= cutoff,
            Chat.chat_title.isnot(None),
            Chat.chat_title != "",
        )
        .order_by(Chat.created_at.desc())
        .all()
    )

    # `chat_title` is copied onto every chunk of a conversation (see
    # `save_message`), so any one chunk represents its conversation. Since
    # `chunks` is already newest-first, the first chunk seen for a given
    # conversation_id is that conversation's most recent activity.
    seen_conversations = set()
    conversations = []
    for chunk in chunks:
        if chunk.conversation_id in seen_conversations:
            continue
        seen_conversations.add(chunk.conversation_id)
        conversations.append(chunk)

    return conversations


CHUNK_SIZE = 20


def _summarize_chat(
    llm_client: GeminiLLMClient,
    messages: list[dict],
    previous_summary: str | None,
) -> str:
    """Summarizes a chunk's user/assistant messages into a single summary,
    folding in `previous_summary` (the summary carried over from the chunk
    before this one, if any) so context isn't lost across chunks."""
    conversation = "\n".join(f"{m['role']}: {m['message']}" for m in messages)

    user_query = (
        f"Previous summary:\n{previous_summary}\n\nNew messages:\n{conversation}"
        if previous_summary
        else f"Conversation:\n{conversation}"
    )

    return llm_client.generate_response(
        system_prompt=CHAT_SUMMARY_SYSTEM_PROMPT,
        user_query=user_query,
    )


def _generate_chat_title(
    llm_client: GeminiLLMClient,
    question: str,
    answer: str,
) -> str:
    """Generates a short title from a conversation's first exchange, for
    the sidebar's history list. The system prompt already asks for 2-5
    words, but that's enforced here too in case the model doesn't comply."""
    title = llm_client.generate_response(
        system_prompt=CHAT_TITLE_SYSTEM_PROMPT,
        user_query=f"User: {question}\nAssistant: {answer}",
    )
    return " ".join(title.strip().strip('"').split()[:5])


def save_message(
    db: Session,
    session_id: str,
    conversation_id: str,
    message: dict,
    llm_client: GeminiLLMClient,
    ip_address: str,
):
    # One conversation_id can span several chunk rows (see CHUNK_SIZE
    # below) - always look up the currently-active one rather than a
    # caller-supplied chunk id, so a chunk split partway through an
    # exchange (see `store_exchange`) is picked up transparently.
    active_chunk = get_active_chunk(db, conversation_id)

    if active_chunk is None:
        get_or_create_session(db, session_id, ip_address)
        chunk = Chat(
            session_id=session_id, conversation_id=conversation_id, messages=[message]
        )

        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        return chunk

    # Active chunk has space

    if len(active_chunk.messages) < CHUNK_SIZE:
        active_chunk.messages.append(message)

        db.commit()
        db.refresh(active_chunk)

        # Title the conversation once its first exchange (user + assistant)
        # is in - this only fires once per conversation, right after its
        # first chunk is created.
        if active_chunk.chat_title is None and len(active_chunk.messages) == 2:
            active_chunk.chat_title = _generate_chat_title(
                llm_client=llm_client,
                question=active_chunk.messages[0]["message"],
                answer=active_chunk.messages[1]["message"],
            )
            db.commit()
            db.refresh(active_chunk)

        return active_chunk
    else:
        chat_summary = _summarize_chat(
            llm_client=llm_client,
            messages=active_chunk.messages,
            previous_summary=active_chunk.previous_chunk_summary,
        )

        active_chunk.is_active = False

        new_chunk = Chat(
            session_id=session_id,
            conversation_id=conversation_id,
            messages=[message],
            previous_chunk_id=active_chunk.chunk_id,
            previous_chunk_summary=chat_summary,
            chat_title=active_chunk.chat_title,
            is_active=True,
        )

        db.add(new_chunk)
        db.commit()
        db.refresh(new_chunk)

        return new_chunk


def store_exchange(
    db: Session,
    session_id: str,
    conversation_id: str,
    user_query: str,
    llm_response: str,
    llm_client: GeminiLLMClient,
    ip_address: str,
) -> Chat:
    """Persists one user/assistant exchange as two messages via
    `save_message`. Returns the chunk row the assistant's message ended up
    in - `conversation_id` is stable and never differs from the one passed
    in, but the returned chunk's `chat_title` reflects one just generated
    for a brand-new conversation."""
    chunk = None
    for role, content in (("user", user_query), ("assistant", llm_response)):
        chunk = save_message(
            db=db,
            session_id=session_id,
            conversation_id=conversation_id,
            message={
                "role": role,
                "message": content,
                "created_at": datetime.utcnow().isoformat(),
            },
            llm_client=llm_client,
            ip_address=ip_address,
        )

    return chunk
