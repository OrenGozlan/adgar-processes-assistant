import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

db_path = "/data/adgar.db" if os.path.isdir("/data") else "./adgar.db"
engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from models import User, Document, Chunk, ChatSession, Message, TopQuestion  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate(engine)


def _migrate(eng):
    from sqlalchemy import text, inspect
    insp = inspect(eng)
    if "documents" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("documents")]
        if "status" not in cols:
            with eng.begin() as conn:
                conn.execute(text("ALTER TABLE documents ADD COLUMN status VARCHAR DEFAULT 'ready'"))
