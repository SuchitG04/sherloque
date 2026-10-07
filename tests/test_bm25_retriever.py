from sqlalchemy import text

from sherloque.retrieve.bm25 import BM25Retriever, BM25RetrieverConfig
from tests.database import DatabaseTestCase


class BM25RetrieverTests(DatabaseTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        async with self.engine.begin() as connection:
            documents = [
                ("Python", "Python asynchronous database tutorial", 4),
                ("Banana bread", "Banana bread recipe with walnuts", 5),
                ("Vector search", "Vector search with PostgreSQL", 4),
            ]
            document_ids = {}
            for title, full_text, length in documents:
                result = await connection.execute(
                    text(
                        """
                        INSERT INTO document (title, full_text, len)
                        VALUES (:title, :full_text, :len)
                        RETURNING _id
                        """
                    ),
                    {"title": title, "full_text": full_text, "len": length},
                )
                document_ids[title] = result.scalar_one()

            terms = {
                "banana": ("Banana bread", 1),
                "recipe": ("Banana bread", 1),
                "database": ("Python", 1),
                "search": ("Vector search", 1),
            }
            for token, (title, frequency) in terms.items():
                result = await connection.execute(
                    text(
                        """
                        INSERT INTO token (token, doc_freq)
                        VALUES (:token, 1)
                        RETURNING _id
                        """
                    ),
                    {"token": token},
                )
                await connection.execute(
                    text(
                        """
                        INSERT INTO term_doc_stats (doc_id, token_id, tf)
                        VALUES (:doc_id, :token_id, :tf)
                        """
                    ),
                    {
                        "doc_id": document_ids[title],
                        "token_id": result.scalar_one(),
                        "tf": frequency,
                    },
                )

        self.retriever = BM25Retriever(self.engine, BM25RetrieverConfig())

    async def test_matching_document_ranks_first(self) -> None:
        results = await self.retriever.retrieve(query="banana recipes")

        self.assertEqual(results[0].doc_title, "Banana bread")
        self.assertGreater(results[0].score, 0)

    async def test_top_k_and_unknown_terms(self) -> None:
        results = await self.retriever.retrieve(
            query="database recipe search",
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertEqual(await self.retriever.retrieve(query="astronomy"), [])

    async def test_returns_empty_title_for_an_untitled_document(self) -> None:
        async with self.engine.begin() as connection:
            document_id = (
                await connection.execute(
                    text(
                        """
                        INSERT INTO document (title, full_text, len)
                        VALUES (NULL, 'Untitled astronomy note', 3)
                        RETURNING _id
                        """
                    )
                )
            ).scalar_one()
            token_id = (
                await connection.execute(
                    text(
                        """
                        INSERT INTO token (token, doc_freq)
                        VALUES ('astronomy', 1)
                        RETURNING _id
                        """
                    )
                )
            ).scalar_one()
            await connection.execute(
                text(
                    """
                    INSERT INTO term_doc_stats (doc_id, token_id, tf)
                    VALUES (:doc_id, :token_id, 1)
                    """
                ),
                {"doc_id": document_id, "token_id": token_id},
            )

        results = await self.retriever.retrieve(query="astronomy")

        self.assertEqual(results[0].doc_title, "")
