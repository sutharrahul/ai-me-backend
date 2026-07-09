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
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("user_session.session_id"), nullable=False)
    chat_id = Column(
        String,
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )
    messages = Column(MutableList.as_mutable(JSON), nullable=False)
    previous_chat_id = Column(String, nullable=True)
    previous_chat_summary = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)

    session = relationship("Session", back_populates="chats")


class User_Session(Base):
    __tablename__ = "user_session"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False, index=True, unique=True)
    ip_address = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    chats = relationship("Chat", back_populates="session")
