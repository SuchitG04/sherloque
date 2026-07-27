from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pgvector.asyncpg import register_vector
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Database
    db_async_driver: str | None = None
    db_user: str | None = None
    db_password: str | None = None
    db_host: str | None = None
    db_port: str | None = None
    db_name: str | None = None
    # db_supports_schema: bool = False
    db_direct_async_str: str | None = None

    @property
    def engine_async_str(self) -> str:
        if self.db_direct_async_str:
            return self.db_direct_async_str
        if self.db_host:
            return f"postgresql+{self.db_async_driver}://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        raise ValueError("Database connection information is incomplete.")

    # 3rd party
    fireworks_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_embed_model: str = "openai/text-embedding-3-small"
    openrouter_rerank_model: str = "cohere/rerank-v3.5"


class Settings(DefaultSettings):
    pass


@lru_cache()
def get_settings() -> Settings:
    env_file = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_file)
    return Settings()


@lru_cache()
def get_async_engine() -> AsyncEngine:
    settings = get_settings()
    engine = create_async_engine(
        settings.engine_async_str,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

    # Register the pgvector type on every new asyncpg connection so vector
    # columns bind/return as Python lists (no manual ::vector string casts).
    # run_async hands the raw asyncpg connection to register_vector.
    @event.listens_for(engine.sync_engine, "connect")
    def _register_pgvector(dbapi_connection, _connection_record):
        dbapi_connection.run_async(register_vector)

    return engine


@lru_cache()
def get_async_lifespan():
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine = get_async_engine()
        async with engine.begin() as conn:
            await conn.run_sync()
        yield


__all__ = [
    "Settings",
    "get_settings",
    "get_async_engine",
    "get_async_lifespan",
]
