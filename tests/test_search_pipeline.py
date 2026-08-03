from sqlalchemy import text

from sherloque.query_engine import QueryEngine
from sherloque.rank import CrossEncoderReRanker
from sherloque.retrieve import (
    BM25Retriever,
    BM25RetrieverConfig,
    VectorRetriever,
    VectorRetrieverConfig,
)
from sherloque.retrieve.vector import EMBED_DIM, QWEN3_QUERY_TASK
from tests.database import DatabaseTestCase


class DeterministicSearchModel:
    """Deterministic substitute for the external embedding/reranking service."""

    def __init__(self, query_embedding: list[float]) -> None:
        self.query_embedding = query_embedding
        self.embedding_inputs: list[str] = []
        self.rerank_query: str | None = None
        self.rerank_documents: list[str] = []

    async def embed(self, *, documents: list[str], **kwargs) -> list[list[float]]:
        self.embedding_inputs = documents
        return [self.query_embedding for _ in documents]

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        **kwargs,
    ) -> list[tuple[int, float]]:
        self.rerank_query = query
        self.rerank_documents = documents
        scores = [
            (index, 1.0 if "asyncio" in document.lower() else 0.1)
            for index, document in enumerate(documents)
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)


class SearchPipelineTests(DatabaseTestCase):
    async def test_real_retrieval_fusion_and_reranking_pipeline(self) -> None:
        query_embedding = [1.0] + [0.0] * (EMBED_DIM - 1)
        related_embedding = [0.8, 0.6] + [0.0] * (EMBED_DIM - 2)
        unrelated_embedding = [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)

        async with self.engine.begin() as connection:
            rows = [
                (
                    "Lexical distractor",
                    "Python database tutorial",
                    3,
                    query_embedding,
                ),
                (
                    "Semantic answer",
                    "An asyncio guide for Python concurrency",
                    6,
                    related_embedding,
                ),
                (
                    "Unrelated",
                    "Banana bread recipe",
                    3,
                    unrelated_embedding,
                ),
            ]
            document_ids: dict[str, int] = {}
            for title, full_text, length, embedding in rows:
                result = await connection.execute(
                    text(
                        """
                        INSERT INTO document (title, full_text, len, embedding)
                        VALUES (:title, :full_text, :len, :embedding)
                        RETURNING _id
                        """
                    ),
                    {
                        "title": title,
                        "full_text": full_text,
                        "len": length,
                        "embedding": embedding,
                    },
                )
                document_ids[title] = result.scalar_one()

            python_token_id = (
                await connection.execute(
                    text(
                        """
                        INSERT INTO token (token, doc_freq)
                        VALUES ('python', 2)
                        RETURNING _id
                        """
                    )
                )
            ).scalar_one()
            for title in ("Lexical distractor", "Semantic answer"):
                await connection.execute(
                    text(
                        """
                        INSERT INTO term_doc_stats (doc_id, token_id, tf)
                        VALUES (:doc_id, :token_id, 1)
                        """
                    ),
                    {
                        "doc_id": document_ids[title],
                        "token_id": python_token_id,
                    },
                )

        model = DeterministicSearchModel(query_embedding)
        engine = QueryEngine(
            retrievers=[
                BM25Retriever(
                    self.engine,
                    BM25RetrieverConfig(top_k=3),
                ),
                VectorRetriever(
                    self.engine,
                    VectorRetrieverConfig(top_k=3),
                    model,
                ),
            ],
            rankers=[CrossEncoderReRanker(self.engine, model)],
        )

        results = await engine.search(query="python")

        self.assertEqual(
            model.embedding_inputs,
            [f"Instruct: {QWEN3_QUERY_TASK}\nQuery: python"],
        )
        self.assertEqual(model.rerank_query, "python")
        self.assertEqual(
            set(model.rerank_documents),
            {
                "Python database tutorial",
                "An asyncio guide for Python concurrency",
                "Banana bread recipe",
            },
        )
        self.assertEqual(results[0].doc_title, "Semantic answer")
        self.assertEqual(results[0].doc_id, document_ids["Semantic answer"])
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual(
            {result.doc_title for result in results},
            {"Lexical distractor", "Semantic answer", "Unrelated"},
        )
