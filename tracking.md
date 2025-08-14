### Work Log

- [Init] Created this tracking file to log progress.

- [Embeddings]
  - Added auto-detection of embedding dimension in `context_manager/llm/embeddings.py` via `get_sentence_embedding_dimension()`.
  - Exposed `get_dimension()` API for consumers.

- [LongTermMemory]
  - Auto-align FAISS index dimension with embedding provider; validate embedding dimension on add.
  - Normalize embeddings and queries to support cosine similarity via L2 distance proxy.
  - Ensure float32 when adding/searching FAISS.
  - Added persistence: `save(dir)`/`load(dir)` writing `index.faiss` and `entries.json`.
  - Added weighted ranking: configurable `similarity_weight`, `recency_weight`, `importance_weight`, and `recency_half_life_seconds` for exponential decay.

- [Orchestrator]
  - Auto-align LTM dimension to embedding model; warn when config dimension mismatches model.
  - Budget-aware assembly that prunes LTM hits first, then recent turns, before hard truncation.
  - Exposed `save_memory()` / `load_memory()` passthroughs.
  - Added session/task scoping (`session_id`, `task_id`) with isolation flags and filtering of LTM results.
  - Added lightweight hybrid retrieval: keyword-based filtering of LTM hits before budget pruning.

- [Summarizer]
  - Added `summarize_texts()` utility with graceful fallback.

- [LLM Adapters]
  - Added proper async support using `AsyncOpenAI`/`AsyncAnthropic` when available; otherwise offload sync calls to thread executors.

- [Dev Setup]
  - Added `requirements-dev.txt` (pytest, coverage, linters, type-checker).

- [Tests]
  - Added `tests/test_persistence_and_budget.py` covering:
    - Embedding/index dimension alignment and add.
    - Cosine similarity proxy ordering monotonicity.
    - Budget-aware pruning under tight budgets.
    - LTM save/load roundtrip.
  - Added `tests/test_summarizer_and_debug.py` covering:
    - Summarizer fallback when LLM adapter raises errors.
    - Summary metadata fields.
    - `debug_context_building` counts and budget compliance.
    - TokenCounter tokenizer-failure fallback path.
  - Added `tests/test_session_isolation.py` covering:
    - Session-scoped memories are not retrieved across sessions unless cross-session is allowed.
  - Added `tests/test_hybrid_and_summary.py` covering:
    - Keyword-based filtering prioritizes matching LTM.
    - `summarize_texts()` fallback behavior.
  - Added `tests/test_weighted_ranking.py` covering:
    - Importance boosts ranking when weighted.
    - Recency boosts ranking with exponential half-life.

- [Docs]
  - Updated `docs/api_reference.md` with new constructor params, persistence APIs, scoping helpers, summarizer method, and weighted ranking config.
  - Updated `docs/core_concepts.md` to reflect scoping and hybrid retrieval.

Next up:
- Consider adding a simple importance heuristics helper and examples in docs.
