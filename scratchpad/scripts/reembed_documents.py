"""Replace document embeddings through a configured model provider.

This is intentionally destructive: every selected document.embedding value is
overwritten. Progress is committed and checkpointed after each batch so an
interrupted run can resume without rebilling completed documents.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx
from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sherloque.config import get_async_engine, get_settings  # noqa: E402
from sherloque.model_providers import OpenRouterModelProvider  # noqa: E402

LOG = logging.getLogger("reembed_documents")

SELECT_BATCH = text(
    """
    SELECT _id, full_text
    FROM document
    WHERE _id > :after_id
    ORDER BY _id
    LIMIT :limit
    """
)
UPDATE_EMBEDDING = text(
    "UPDATE document SET embedding = :embedding WHERE _id = :doc_id"
)


def load_checkpoint(
    path: Path,
    model: str,
    dimensions: int,
    corpus_count: int,
    corpus_max_id: int,
) -> dict:
    if not path.exists():
        return {
            "model": model,
            "dimensions": dimensions,
            "corpus_count": corpus_count,
            "corpus_max_id": corpus_max_id,
            "last_id": 0,
            "updated": 0,
        }
    state = json.loads(path.read_text())
    expected = {
        "model": model,
        "dimensions": dimensions,
        "corpus_count": corpus_count,
        "corpus_max_id": corpus_max_id,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint does not match this model/corpus")
    if state.get("updated", -1) > corpus_count or state.get(
        "last_id", -1
    ) > corpus_max_id:
        raise ValueError("checkpoint progress exceeds the current corpus")
    return state


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True))
    temporary.replace(path)


async def embed_with_backoff(
    provider: OpenRouterModelProvider,
    texts: list[str],
    dimensions: int,
) -> list[list[float]]:
    delay = 2.0
    for attempt in range(8):
        try:
            return await provider.embed(
                documents=texts,
                dimensions=dimensions,
                normalize=True,
            )
        except ValueError:
            if len(texts) == 1:
                if attempt == 7:
                    raise
                LOG.warning("provider rejected one document; backing off %.1fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            midpoint = len(texts) // 2
            LOG.warning(
                "provider rejected batch of %d documents; splitting it",
                len(texts),
            )
            first = await embed_with_backoff(
                provider,
                texts[:midpoint],
                dimensions,
            )
            second = await embed_with_backoff(
                provider,
                texts[midpoint:],
                dimensions,
            )
            return first + second
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {429, 500, 502, 503, 529}:
                raise
            if attempt == 7:
                raise
            LOG.warning(
                "provider status %d; backing off %.1fs",
                exc.response.status_code,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


async def async_main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = get_settings()
    if not args.yes_replace_all:
        raise ValueError(
            "refusing destructive replacement without --yes-replace-all"
        )
    if args.dimensions <= 0 or args.batch_size <= 0 or args.request_delay < 0:
        raise ValueError(
            "dimensions/batch-size must be positive and delay non-negative"
        )
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured")

    checkpoint = Path(args.checkpoint)
    engine = get_async_engine()
    provider = OpenRouterModelProvider(
        api_key=settings.openrouter_api_key,
        embed_model=args.model,
        rerank_model=settings.openrouter_rerank_model,
    )
    try:
        async with engine.connect() as connection:
            corpus = (
                await connection.execute(
                    text(
                        "SELECT COUNT(*) AS docs, "
                        "COALESCE(MAX(_id), 0) AS max_id FROM document"
                    )
                )
            ).one()
        total = corpus.docs
        state = load_checkpoint(
            checkpoint,
            args.model,
            args.dimensions,
            total,
            corpus.max_id,
        )
        LOG.info(
            "re-embedding %d documents with %s at %d dimensions; resuming after "
            "id=%d (%d already updated)",
            total,
            args.model,
            args.dimensions,
            state["last_id"],
            state["updated"],
        )

        while True:
            async with engine.connect() as connection:
                rows = list(
                    await connection.execute(
                        SELECT_BATCH,
                        {
                            "after_id": state["last_id"],
                            "limit": args.batch_size,
                        },
                    )
                )
            if not rows:
                break

            vectors = await embed_with_backoff(
                provider,
                [row.full_text or "" for row in rows],
                args.dimensions,
            )
            async with engine.begin() as connection:
                for row, embedding in zip(rows, vectors):
                    await connection.execute(
                        UPDATE_EMBEDDING,
                        {"doc_id": row._id, "embedding": embedding},
                    )

            state["last_id"] = rows[-1]._id
            state["updated"] += len(rows)
            save_checkpoint(checkpoint, state)
            LOG.info("updated %d/%d documents", state["updated"], total)
            if args.request_delay:
                await asyncio.sleep(args.request_delay)

        async with engine.connect() as connection:
            validation = (
                await connection.execute(
                    text(
                        """
                        SELECT COUNT(*) AS docs,
                               COUNT(*) FILTER (WHERE embedding IS NULL) AS nulls,
                               MIN(vector_dims(embedding)) AS min_dim,
                               MAX(vector_dims(embedding)) AS max_dim
                        FROM document
                        """
                    )
                )
            ).one()
        print(
            f"documents={validation.docs} updated={state['updated']} "
            f"null_embeddings={validation.nulls} "
            f"min_dim={validation.min_dim} max_dim={validation.max_dim}"
        )
    finally:
        await provider.aclose()
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="qwen/qwen3-embedding-4b",
    )
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint",
        default="/tmp/sherloque-openrouter-reembed.json",
    )
    parser.add_argument(
        "--yes-replace-all",
        action="store_true",
        help="confirm destructive replacement of every document embedding",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
