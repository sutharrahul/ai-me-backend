from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import (
    Column,
    Integer,
    String,
    JSON,
    DateTime,
    ForeignKey,
    Boolean,
)
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableList


import uuid



class Base(DeclarativeBase):
    pass


class Chat(Base):
    """One storage chunk. A conversation is one or more chunk rows sharing
    the same `conversation_id` - see `save_message`/`CHUNK_SIZE` in
    `app/db/db_query.py` for how a conversation splits into chunks once it
    grows past CHUNK_SIZE messages, and `is_active` for which chunk is
    currently being appended to."""

    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("user_session.session_id"), nullable=False)
    # Stable for the whole conversation - the id the frontend/API calls
    # "chat_id" (see `ChatSummary`/`ChatDetailResponse` in
    # `app/models/schemas.py`). Shared by every chunk of one conversation;
    # unlike `chunk_id` below, this never changes.
    conversation_id = Column(String, nullable=False, index=True)
    # Identifies this specific chunk row. Internal only - never sent to the
    # frontend as a chat identity, only used to link chunks together via
    # `previous_chunk_id` and to fetch a specific older chunk for
    # pagination (`GET /chat/chunk/{chunk_id}`).
    chunk_id = Column(
        String,
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )
    messages = Column(MutableList.as_mutable(JSON), nullable=False)
    # Generated once, from the conversation's first exchange, then copied
    # onto every later chunk of the same conversation (see `save_message`)
    # so `list_recent_chats` can read it off any one chunk.
    chat_title = Column(String, nullable=True)
    previous_chunk_id = Column(String, nullable=True)
    previous_chunk_summary = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)

    session = relationship("User_Session", back_populates="chats")


class User_Session(Base):
    __tablename__ = "user_session"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False, index=True, unique=True)
    ip_address = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    chats = relationship("Chat", back_populates="session")
