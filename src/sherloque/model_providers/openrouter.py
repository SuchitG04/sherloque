from typing import Any

import httpx

from .base import BaseModelProvider


class OpenRouterModelProvider(BaseModelProvider):
    """Native OpenRouter embeddings and reranking API client."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        embed_model: str = "openai/text-embedding-3-small",
        rerank_model: str = "cohere/rerank-v3.5",
        base_url: str = BASE_URL,
        timeout: float | httpx.Timeout = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        owns_http_client = http_client is None

        if http_client is None:
            headers = {"Content-Type": "application/json"}
            if api_key is not None:
                headers["Authorization"] = f"Bearer {api_key}"
            http_client = httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                base_url=base_url.rstrip("/"),
            )

        super().__init__(
            http_client=http_client,
            owns_http_client=owns_http_client,
        )

    async def embed(
        self,
        *,
        documents: list[str],
        dimensions: int | None = None,
        normalize: bool = True,
        encoding_format: str = "float",
        **kwargs: Any,
    ) -> list[list[float]]:
        if not documents:
            return []
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        if encoding_format != "float":
            raise ValueError(
                "Sherloque's numeric embedding interface requires float encoding"
            )

        payload: dict[str, Any] = dict(kwargs)
        payload.update(
            {
                "model": self.embed_model,
                "input": documents,
                "encoding_format": encoding_format,
            }
        )
        if dimensions is not None:
            payload["dimensions"] = dimensions

        response = await self._post("/embeddings", payload)
        try:
            data = response.json()["data"]
            indexed_vectors = {
                int(item["index"]): [float(value) for value in item["embedding"]]
                for item in data
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid OpenRouter embeddings response") from exc

        expected_indexes = set(range(len(documents)))
        if len(data) != len(documents) or set(indexed_vectors) != expected_indexes:
            raise ValueError(
                "OpenRouter embeddings response indexes do not match inputs"
            )

        vectors = [indexed_vectors[index] for index in range(len(documents))]
        vector_lengths = {len(vector) for vector in vectors}
        if len(vector_lengths) != 1:
            raise ValueError(
                "OpenRouter embeddings response has inconsistent dimensions"
            )
        if dimensions is not None and vector_lengths != {dimensions}:
            raise ValueError(
                "OpenRouter embeddings response does not match requested dimensions"
            )
        if normalize:
            vectors = [self._l2_normalize(vector) for vector in vectors]
        return vectors

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        if not documents or top_n == 0:
            return []
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be non-negative")

        payload: dict[str, Any] = dict(kwargs)
        payload.update(
            {
                "model": self.rerank_model,
                "query": query,
                "documents": documents,
            }
        )
        if top_n is not None:
            payload["top_n"] = top_n

        response = await self._post("/rerank", payload)
        try:
            raw_results = response.json()["results"]
            results = [
                (int(item["index"]), float(item["relevance_score"]))
                for item in raw_results
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid OpenRouter rerank response") from exc

        if any(index < 0 or index >= len(documents) for index, _ in results):
            raise ValueError("OpenRouter rerank response contains an invalid index")
        if len({index for index, _ in results}) != len(results):
            raise ValueError("OpenRouter rerank response contains duplicate indexes")

        results.sort(key=lambda item: item[1], reverse=True)
        return results


__all__ = ["OpenRouterModelProvider"]
