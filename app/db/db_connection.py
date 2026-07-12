from app.config import get_settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

settings = get_settings()

DB_URI = settings.db_uri.get_secret_value()

engine = create_engine(
    DB_URI,
    # Verbose SQL logging is a dev aid; in production it floods the platform logs.
    echo=settings.environment != "production",
    # Managed Postgres drops idle connections. Without this, the first request
    # after an idle period fails on a dead pooled connection.
    pool_pre_ping=True,
)

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
