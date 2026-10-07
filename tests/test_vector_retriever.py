import math
import unittest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import text

from sherloque.retrieve.vector import (
    EMBED_DIM,
    QWEN3_QUERY_TASK,
    VectorRetriever,
    VectorRetrieverConfig,
)
from tests.database import DatabaseTestCase


class VectorRetrieverUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_a_normalized_qwen_query_embedding(self) -> None:
        connection = AsyncMock()
        connection.execute.return_value = []
        engine = MagicMock()
        engine.connect.return_value.__aenter__.return_value = connection
        model = MagicMock()
        model.embed = AsyncMock(
            return_value=[[1.0] + [0.0] * (EMBED_DIM - 1)]
        )
        retriever = VectorRetriever(
            engine,
            VectorRetrieverConfig(),
            model,
        )

        await retriever.retrieve(query="banana bread")

        model.embed.assert_awaited_once_with(
            documents=[f"Instruct: {QWEN3_QUERY_TASK}\nQuery: banana bread"],
            dimensions=EMBED_DIM,
            normalize=True,
        )


class VectorRetrieverIntegrationTests(DatabaseTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        root_half = math.sqrt(0.5)
        self.query_vector = [1.0] + [0.0] * (EMBED_DIM - 1)
        documents = [
            ("Closest", self.query_vector),
            ("Unrelated", [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)),
            ("Related", [root_half, root_half] + [0.0] * (EMBED_DIM - 2)),
        ]
        async with self.engine.begin() as connection:
            for title, embedding in documents:
                await connection.execute(
                    text(
                        """
                        INSERT INTO document (title, full_text, len, embedding)
                        VALUES (:title, :title, 1, :embedding)
                        """
                    ),
                    {"title": title, "embedding": embedding},
                )

    async def test_orders_documents_by_cosine_similarity(self) -> None:
        model = MagicMock()
        model.embed = AsyncMock(return_value=[self.query_vector])
        retriever = VectorRetriever(
            self.engine,
            VectorRetrieverConfig(),
            model,
        )

        results = await retriever.retrieve(query="anything", top_k=2)

        self.assertEqual(
            [result.doc_title for result in results],
            ["Closest", "Related"],
        )
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertGreater(results[0].score, results[1].score)

    async def test_returns_empty_title_for_an_untitled_document(self) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO document (title, full_text, len, embedding)
                    VALUES (NULL, 'Untitled', 1, :embedding)
                    """
                ),
                {"embedding": self.query_vector},
            )
        model = MagicMock()
        model.embed = AsyncMock(return_value=[self.query_vector])
        retriever = VectorRetriever(self.engine, VectorRetrieverConfig(), model)

        results = await retriever.retrieve(query="anything", top_k=4)

        self.assertIn("", [result.doc_title for result in results])
