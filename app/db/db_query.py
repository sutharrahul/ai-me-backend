from sqlalchemy.orm import Session
from .schema_modal import Chat
from app.services.llm import GeminiLLMClient
from app.graph.system_prompt import CHAT_SUMMARY_SYSTEM_PROMPT


def get_active_chat(db: Session, session_id: str):
    active_chat = (
        db.query(Chat)
        .filter(Chat.session_id == session_id, Chat.is_active == True)
        .first()
    )
    return active_chat


MAX_MESSAGE = 20


def _summarize_chat(
    llm_client: GeminiLLMClient,
    messages: list[dict],
    previous_summary: str | None,
) -> str:
    """Summarizes a chat's user/assistant messages into a single summary,
    folding in `previous_summary` (the summary carried over from the chat
    before this one, if any) so context isn't lost across rotations."""
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


def save_message(db: Session, session_id: str, message: dict, llm_client: GeminiLLMClient):
    # get active chat
    active_chat = get_active_chat(db, session_id)

    if active_chat is None:
        chat = Chat(session_id=session_id, messages=[message])

        db.add(chat)
        db.commit()
        db.refresh(chat)

        return chat

    # Active chat has space

    if len(active_chat.messages) < MAX_MESSAGE:
        active_chat.messages.append(message)


        db.commit()
        db.refresh(active_chat)

        return active_chat
    else:

        chat_summary = _summarize_chat(
            llm_client=llm_client,
            messages=active_chat.messages,
            previous_summary=active_chat.previous_chat_summary,
        )

        active_chat.is_active = False

        new_chat = Chat(
            session_id=session_id,
            messages=[message],
            previous_chat_id=active_chat.chat_id,
            previous_chat_summary=chat_summary,
            is_active=True,
        )

        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)

        return new_chat
