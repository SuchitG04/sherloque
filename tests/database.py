import unittest
from uuid import uuid4

from pgvector.asyncpg import register_vector
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from sherloque.config import get_settings


class DatabaseTestCase(unittest.IsolatedAsyncioTestCase):
    engine: AsyncEngine

    async def asyncSetUp(self) -> None:
        try:
            database_url = get_settings().engine_async_str
        except ValueError:
            self.skipTest("test database is not configured")

        self.schema = f"sherloque_test_{uuid4().hex}"
        self.engine = create_async_engine(
            database_url,
            connect_args={
                "server_settings": {"search_path": f"{self.schema},public"}
            },
        )

        @event.listens_for(self.engine.sync_engine, "connect")
        def register_vector_type(dbapi_connection, _connection_record):
            dbapi_connection.run_async(register_vector)

        async with self.engine.begin() as connection:
            await connection.execute(text(f"CREATE SCHEMA {self.schema}"))
            await connection.execute(
                text(
                    """
                    CREATE TABLE document (
                        _id SERIAL PRIMARY KEY,
                        -- Production documents may have only a URL or body;
                        -- keep the fixture's nullability aligned with init_db.
                        title TEXT,
                        full_text TEXT NOT NULL,
                        len INTEGER NOT NULL,
                        embedding VECTOR(768)
                    )
                    """
                )
            )
            await connection.execute(
                text(
                    """
                    CREATE TABLE token (
                        _id SERIAL PRIMARY KEY,
                        token TEXT NOT NULL UNIQUE,
                        doc_freq INTEGER NOT NULL
                    )
                    """
                )
            )
            await connection.execute(
                text(
                    """
                    CREATE TABLE term_doc_stats (
                        doc_id INTEGER NOT NULL REFERENCES document(_id),
                        token_id INTEGER NOT NULL REFERENCES token(_id),
                        tf INTEGER NOT NULL,
                        PRIMARY KEY (doc_id, token_id)
                    )
                    """
                )
            )

    async def asyncTearDown(self) -> None:
        if not hasattr(self, "engine"):
            return
        try:
            async with self.engine.begin() as connection:
                await connection.execute(
                    text(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")
                )
        finally:
            await self.engine.dispose()
