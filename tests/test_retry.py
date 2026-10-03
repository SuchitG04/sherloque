import unittest
from unittest.mock import AsyncMock, patch

import httpx

from sherloque.model_providers._retry import with_retry


def http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/embeddings")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_http_errors(self) -> None:
        operation = AsyncMock(
            side_effect=[http_error(429), http_error(520), "success"]
        )
        wrapped = with_retry(max_retries=3, delay=0)(operation)

        with patch(
            "sherloque.model_providers._retry.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            result = await wrapped()

        self.assertEqual(result, "success")
        self.assertEqual(operation.await_count, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_does_not_retry_client_errors(self) -> None:
        operation = AsyncMock(side_effect=http_error(400))
        wrapped = with_retry(max_retries=3, delay=0)(operation)

        with patch(
            "sherloque.model_providers._retry.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            with self.assertRaises(httpx.HTTPStatusError):
                await wrapped()

        operation.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_stops_after_max_retries_on_transport_errors(self) -> None:
        request = httpx.Request("POST", "https://example.test/embeddings")
        operation = AsyncMock(side_effect=httpx.ConnectError("offline", request=request))
        wrapped = with_retry(max_retries=3, delay=0)(operation)

        with patch(
            "sherloque.model_providers._retry.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            with self.assertRaises(httpx.ConnectError):
                await wrapped()

        self.assertEqual(operation.await_count, 3)
        self.assertEqual(sleep.await_count, 2)
