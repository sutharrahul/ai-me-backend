from app.config import get_settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

settings = get_settings()

DB_URI = settings.db_uri.get_secret_value()

engine = create_engine(DB_URI, echo=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
