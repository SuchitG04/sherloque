import json
import unittest

import httpx

from sherloque.model_providers import OpenRouterModelProvider
from sherloque.config import Settings


class OpenRouterModelProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"index": 1, "embedding": [0.0, 2.0]},
                            {"index": 0, "embedding": [3.0, 4.0]},
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 2, "relevance_score": 0.4},
                        {"index": 0, "relevance_score": 0.9},
                    ]
                },
            )

        self.client = httpx.AsyncClient(
            base_url="https://openrouter.test/api/v1",
            transport=httpx.MockTransport(handler),
        )
        self.provider = OpenRouterModelProvider(
            embed_model="test/embedder",
            rerank_model="test/reranker",
            http_client=self.client,
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_builds_native_request_and_maps_original_indexes(self) -> None:
        results = await self.provider.rerank(
            query="capital of France",
            documents=["Paris", "Berlin", "Madrid"],
            top_n=2,
        )

        request = self.requests[0]
        self.assertEqual(request.url.path, "/api/v1/rerank")
        self.assertEqual(
            json.loads(request.content),
            {
                "model": "test/reranker",
                "query": "capital of France",
                "documents": ["Paris", "Berlin", "Madrid"],
                "top_n": 2,
            },
        )
        self.assertEqual(results, [(0, 0.9), (2, 0.4)])

    async def test_embed_builds_request_restores_order_and_normalizes(self) -> None:
        vectors = await self.provider.embed(
            documents=["first", "second"],
            dimensions=2,
            user="test-user",
        )

        request = self.requests[0]
        self.assertEqual(request.url.path, "/api/v1/embeddings")
        self.assertEqual(
            json.loads(request.content),
            {
                "model": "test/embedder",
                "input": ["first", "second"],
                "encoding_format": "float",
                "dimensions": 2,
                "user": "test-user",
            },
        )
        self.assertEqual(vectors[0], [0.6, 0.8])
        self.assertEqual(vectors[1], [0.0, 1.0])

    async def test_embed_can_return_raw_vectors(self) -> None:
        vectors = await self.provider.embed(
            documents=["first", "second"],
            normalize=False,
        )

        self.assertEqual(vectors, [[3.0, 4.0], [0.0, 2.0]])

    async def test_empty_embed_skips_request(self) -> None:
        self.assertEqual(await self.provider.embed(documents=[]), [])
        self.assertEqual(self.requests, [])

    async def test_empty_documents_and_zero_top_n_skip_request(self) -> None:
        self.assertEqual(
            await self.provider.rerank(query="query", documents=[]),
            [],
        )
        self.assertEqual(
            await self.provider.rerank(
                query="query",
                documents=["doc"],
                top_n=0,
            ),
            [],
        )
        self.assertEqual(self.requests, [])

    async def test_invalid_arguments_and_response_indexes_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_n"):
            await self.provider.rerank(
                query="query",
                documents=["doc"],
                top_n=-1,
            )

        with self.assertRaisesRegex(ValueError, "invalid index"):
            await self.provider.rerank(
                query="query",
                documents=["only one"],
            )

    async def test_http_errors_propagate(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        client = httpx.AsyncClient(
            base_url="https://openrouter.test/api/v1",
            transport=httpx.MockTransport(handler),
        )
        provider = OpenRouterModelProvider(http_client=client)
        try:
            with self.assertRaises(httpx.HTTPStatusError):
                await provider.rerank(query="query", documents=["doc"])
        finally:
            await client.aclose()

    async def test_model_configuration(self) -> None:
        self.assertEqual(self.provider.embed_model, "test/embedder")
        self.assertEqual(self.provider.rerank_model, "test/reranker")

    async def test_invalid_embedding_arguments_fail_before_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions"):
            await self.provider.embed(documents=["doc"], dimensions=0)
        with self.assertRaisesRegex(ValueError, "float encoding"):
            await self.provider.embed(
                documents=["doc"],
                encoding_format="base64",
            )
        self.assertEqual(self.requests, [])

    async def test_invalid_embedding_response_indexes_raise(self) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 4, "embedding": [1.0]}]},
            )

        client = httpx.AsyncClient(
            base_url="https://openrouter.test/api/v1",
            transport=httpx.MockTransport(handler),
        )
        provider = OpenRouterModelProvider(http_client=client)
        try:
            with self.assertRaisesRegex(ValueError, "indexes"):
                await provider.embed(documents=["doc"])
        finally:
            await client.aclose()

    async def test_settings_expose_secret_and_models_without_hardcoding_key(
        self,
    ) -> None:
        settings = Settings(
            openrouter_api_key="secret",
            openrouter_embed_model="vendor/custom-embedder",
            openrouter_rerank_model="vendor/custom-reranker",
        )

        self.assertEqual(settings.openrouter_api_key, "secret")
        self.assertEqual(
            settings.openrouter_embed_model,
            "vendor/custom-embedder",
        )
        self.assertEqual(
            settings.openrouter_rerank_model,
            "vendor/custom-reranker",
        )
