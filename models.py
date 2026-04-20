import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="employee")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    upload_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    chunk_count = Column(Integer, default=0)
    active = Column(Boolean, default=True)
    status = Column(String, default="processing")


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
    chunk_index = Column(Integer)

    def get_embedding(self):
        return json.loads(self.embedding)

    def set_embedding(self, vec):
        self.embedding = json.dumps(vec)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    title = Column(String, default="New Chat")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TopQuestion(Base):
    __tablename__ = "top_questions"
    id = Column(Integer, primary_key=True)
    question_text = Column(String, nullable=False, unique=True)
    count = Column(Integer, default=1)
    last_asked = Column(DateTime, default=lambda: datetime.now(timezone.utc))
