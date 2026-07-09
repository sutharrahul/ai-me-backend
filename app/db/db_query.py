from sqlalchemy.orm import Session
from .schema_modal import Chat, User_Session


def get_or_create_session(db: Session, session_id: str, ip_address: str):
    session = (
        db.query(User_Session).filter(User_Session.session_id == session_id).first()
    )

    if session:
        return session

    session = User_Session(
        session_id=session_id,
        ip_address=ip_address,
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def get_active_chat(db: Session, session_id: str):
    active_chat = (
        db.query(Chat)
        .filter(Chat.session_id == session_id, Chat.is_active == True)
        .first()
    )
    return active_chat


MAX_MESSAGE = 20


def save_message(db: Session, session_id: str, message: dict):
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

        # extract all chat from DB, give to LLM for chat_summary
        all_chat_messages, previous_summary = (active_chat.messages, active_chat.previous_chat_summary)
        # I give all_chat_message list to LLM for chat sumamry
        chat_summary = None

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
