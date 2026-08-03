# Deferred tests

## External integrations

- Run embeddings and similarity reranking against a local Ollama model.
- Run Fireworks embeddings and cross-encoder reranking with an opt-in API key.
- Run embeddings against the real OpenAI API with an opt-in API key.

## End-to-end coverage

- Crawl and index a small local site, then verify that its pages become searchable.

## Search quality

- Evaluate BM25, vector retrieval, fusion, and reranking on labeled queries using recall, MRR, and NDCG.

## Reliability and performance

- Provision PostgreSQL with pgvector in CI instead of relying on a preconfigured database.
- Test concurrent searches, large candidate sets, model batching, and latency budgets.
