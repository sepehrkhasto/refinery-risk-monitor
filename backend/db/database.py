"""
SQLAlchemy database configuration supporting both SQLite and PostgreSQL.
Includes connection pooling, connection health checks, and automatic reconnect handling.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from backend.config import settings

connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args["check_same_thread"] = False
    pool_class = None
else:
    pool_class = QueuePool

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    poolclass=pool_class,
    pool_size=10 if pool_class else 0,
    max_overflow=20 if pool_class else 0,
    pool_pre_ping=True,
    echo=settings.DB_ECHO,
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()