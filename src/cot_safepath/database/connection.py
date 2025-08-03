"""
Database connection management for CoT SafePath Filter.
"""

import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from alembic import command
from alembic.config import Config

from ..exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and configuration."""
    
    def __init__(self, database_url: str = None, **kwargs):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql://safepath:password@localhost:5432/safepath_dev"
        )
        
        # Database configuration
        self.pool_size = int(os.getenv("DATABASE_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "30"))
        self.pool_timeout = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
        self.echo = os.getenv("DATABASE_ECHO", "false").lower() == "true"
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            echo=self.echo,
            **kwargs
        )
        
        # Configure session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Setup connection event listeners
        self._setup_event_listeners()
        
        logger.info(f"Database manager initialized with URL: {self._mask_url(self.database_url)}")
    
    def _setup_event_listeners(self):
        """Setup database event listeners for monitoring and optimization."""
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance (if using SQLite)."""
            if "sqlite" in self.database_url:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def log_queries(conn, cursor, statement, parameters, context, executemany):
            """Log SQL queries for debugging (only in debug mode)."""
            if self.echo:
                logger.debug(f"SQL Query: {statement}")
                if parameters:
                    logger.debug(f"Parameters: {parameters}")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        try:
            return self.SessionLocal()
        except Exception as e:
            logger.error(f"Failed to create database session: {e}")
            raise ConfigurationError(f"Database session creation failed: {e}")
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables."""
        try:
            from .models import Base
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise ConfigurationError(f"Table creation failed: {e}")
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        try:
            from .models import Base
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise ConfigurationError(f"Table drop failed: {e}")
    
    def run_migrations(self, alembic_cfg_path: str = "alembic.ini"):
        """Run Alembic database migrations."""
        try:
            alembic_cfg = Config(alembic_cfg_path)
            command.upgrade(alembic_cfg, "head")
            logger.info("Database migrations completed successfully")
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            raise ConfigurationError(f"Migration failed: {e}")
    
    def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection check successful")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get database connection information."""
        return {
            "url": self._mask_url(self.database_url),
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "echo": self.echo,
        }
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in database URL."""
        if "@" in url:
            parts = url.split("@")
            if "://" in parts[0]:
                protocol_user = parts[0].split("://")
                if ":" in protocol_user[1]:
                    user_pass = protocol_user[1].split(":")
                    masked = f"{protocol_user[0]}://{user_pass[0]}:***@{parts[1]}"
                    return masked
        return url
    
    def close(self):
        """Close database connections."""
        try:
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def init_database(database_url: str = None, **kwargs) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(database_url, **kwargs)
    return _db_manager


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = init_database()
    return _db_manager


def get_database_session() -> Session:
    """Get a new database session from the global manager."""
    return get_database_manager().get_session()


@contextmanager
def database_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db_manager = get_database_manager()
    with db_manager.session_scope() as session:
        yield session