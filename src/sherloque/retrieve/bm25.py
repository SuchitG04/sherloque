from dataclasses import dataclass

from nltk import word_tokenize, WordNetLemmatizer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .base import RetrieverResult, BaseRetriever

@dataclass
class BM25RetrieverConfig:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 10


class BM25Retriever(BaseRetriever):
    BM25_SQL = text(
        # @formatter:off
        """
        WITH q_tokens AS (
            SELECT t._id AS token_id, t.doc_freq
            FROM token t
            WHERE t.token = ANY(:tokens)
        ), stats AS (
            SELECT COUNT(*)::float AS total_docs, AVG(len) ::float AS avgdl
            FROM document
        )
        SELECT d.title AS doc_title, d._id as doc_id, SUM(
            LN(
                (stats.total_docs - q.doc_freq + 0.5) / (q.doc_freq + 0.5) + 1
            ) * (tds.tf * (:k1 + 1)) / (
                tds.tf + :k1 * (1 - :b + :b * d.len / stats.avgdl)
           )
        ) ::float AS score
        FROM term_doc_stats tds
        JOIN q_tokens q ON tds.token_id = q.token_id
        JOIN document d ON tds.doc_id = d._id
        CROSS JOIN stats
        GROUP BY d._id
        ORDER BY score DESC
        LIMIT :top_k
        """
        # @formatter:on
    )

    def __init__(self, engine: AsyncEngine, config: BM25RetrieverConfig):
        self.engine = engine
        self.config = config
        self.lemmatizer = WordNetLemmatizer()

    def _preprocess_query(self, query: str) -> list[str]:
        """Lowercase, tokenize, drop non-alphanumeric tokens, and lemmatize."""
        return [
            self.lemmatizer.lemmatize(tok)
            for tok in word_tokenize(query.lower())
            if any(ch.isalnum() for ch in tok)
        ]

    async def retrieve(
        self,
        *,
        query: str,
        k1: float | None = None,
        b: float | None = None,
        top_k: int | None = None
    ) -> list[RetrieverResult]:
        k1 = k1 or self.config.k1
        b = b or self.config.b
        top_k = top_k or self.config.top_k

        tokens = self._preprocess_query(query)
        async with self.engine.connect() as conn:
            cur = await conn.execute(
                self.BM25_SQL,
                {"tokens": tokens, "k1": k1, "b": b, "top_k": top_k},
            )
            rows = [record for record in cur]
        results = [RetrieverResult(doc_id=row.doc_id, doc_title=row.doc_title or "", score=row.score) for row in rows]
        return results


__all__ = [
    "BM25RetrieverConfig",
    "BM25Retriever",
]
