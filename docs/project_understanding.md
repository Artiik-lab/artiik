### Project Understanding and Context Engineering Synthesis

This document summarizes the project based on `README.md` and synthesizes it with "Context engineering for agents" principles.

### What this project is

- **Goal**: A drop-in context/memory layer for AI agents that abstracts away context window limits.
- **Core features**: Short-term memory (STM), long-term memory (LTM) with FAISS, hierarchical summarization, token budget management, LLM adapters (OpenAI/Anthropic), and debug utilities.
- **Typical flow**:
  1. `observe(user_input, response)` stores turns in STM and summarizes/offsloads to LTM on overflow.
  2. `build_context(user_input)` assembles recent STM turns + relevant LTM hits + current input, then truncates to token budget.
  3. `query_memory(query)` performs semantic search over LTM.

### Mapping to context engineering pillars

- **Write context**:
  - STM stores raw turns; summarizer writes chunk summaries into LTM.
  - Missing: explicit scratchpads/task notes; persistence for LTM across sessions; instructional memory.

- **Select context**:
  - LTM similarity search via FAISS; filter by score; include `recent_k` STM turns.
  - Missing: hybrid retrieval (keyword + dense), recency/importance re-ranking, tool-output selection policies, per-task scoping.

- **Compress context**:
  - Chunk-level summarization on STM overflow; hierarchical summarizer available.
  - Missing: budget-aware pre-call summarization; periodic hierarchical merges; summary refresh/aging.

- **Isolate context**:
  - Current design is single shared memory per `ContextManager` instance.
  - Missing: multi-session isolation (user/session IDs), task-scoped memory, and sandboxed tool traces.

### Current implementation status (from code)

- `core`
  - `ContextManager`: orchestrates token counting, STM, LTM, summarizer, and context assembly.
  - `Config`: Pydantic models for LLM/memory/vector store; debug flags.
- `memory`
  - `ShortTermMemory`: token-aware deque with eviction; chunking for summarization.
  - `LongTermMemory`: FAISS HNSW index + in-memory entries; add/search/delete/rebuild.
  - `HierarchicalSummarizer`: LLM-based chunk and higher-level summaries.
- `llm`
  - Adapters for OpenAI/Anthropic with sync/async methods; embedding provider via sentence-transformers.
- `utils`
  - `TokenCounter` with tiktoken, truncation helpers.
- `examples`
  - `SimpleAgent` with mock tools and ContextManager integration.
- `tests`
  - Coverage for STM basics, LTM add/search, ContextManager init/observe/build_context/query_memory, TokenCounter.

### Notable gaps and issues

- **Embedding dimension mismatch**: `VectorStoreConfig.dimension` defaults to 768, while `EmbeddingProvider` default model is 384. `ContextManager` initializes `LongTermMemory` with `config.vector_store.dimension`, which can mismatch actual embedding size and break FAISS operations.
- **FAISS metric semantics**: Using `IndexHNSWFlat` defaults to L2. For cosine, embeddings should be L2-normalized and searched with inner product, or index created with the appropriate metric. Current similarity post-processing `1 - d/max(d)` is a heuristic.
- **Async adapters**: `OpenAIAdapter.generate` and `AnthropicAdapter.generate` use `await` on sync clients. Proper async variants require `AsyncOpenAI` / `AsyncAnthropic` or running sync calls in executors.
- **Summarization trigger**: Only fires on STM overflow. No pre-call budget-aware compression or periodic hierarchical merges.
- **Persistence**: LTM is in-memory only. No disk persistence (e.g., FAISS write/read) or external vector store backend despite a `provider` field in config.
- **Session/task isolation**: No `session_id`/`conversation_id`/`task_id` scoping or per-scope retrieval policies.
- **Selection quality**: No re-ranking (e.g., cross-encoder), no importance/recency weighting, no deduplication beyond score threshold.
- **Docs vs code**: Docs refer to `requirements-dev.txt` and broader test suite structure not present in repo; advanced async/batch examples are illustrative but not implemented in package.

### Recommended additions (high-impact first)

1. Fix embedding dimension handling
   - Default `vector_store.dimension` to the embedding model dimension or auto-detect from `EmbeddingProvider`.
   - Validate dimension on startup; raise clear error if mismatched.
2. Correct adapter async behavior
   - Use `AsyncOpenAI`/`AsyncAnthropic` for async paths; keep sync methods purely sync.
3. Improve retrieval quality
   - L2-normalize embeddings and switch to inner-product search for cosine; or configure FAISS metric accordingly.
   - Add optional hybrid retrieval (BM25 + dense) and simple recency/importance boosts.
4. Add persistence layer
   - Save/load FAISS index + entries (texts/metadata) to disk; or provide pluggable backends (e.g., SQLite + FAISS, Qdrant/Pinecone).
5. Budget-aware compression
   - Before LLM calls, estimate tokens and summarize/prune to target utilization; allow configurable policies.
6. Context isolation
   - Introduce `session_id` and `task_id` scoping; maintain separate namespaces or filtered views over LTM.
7. Monitoring and config validation
   - Expose warnings for misconfiguration (dimensions, budgets); add lightweight metrics on context build.
8. Developer ergonomics
   - Add `requirements-dev.txt`; expand tests to avoid heavy network/model downloads via mocks.

### Test plan to add

- **Unit tests**
  - Dimension validation: error when embedding size != index dimension; auto-align path.
  - FAISS cosine vs L2: verify normalized embeddings produce monotonic similarity ordering.
  - Summarization fallback: summarizer returns fallback on adapter errors; metadata computed correctly.
  - Budget-aware truncation: `build_context` enforces `prompt_token_budget` under various sizes.
  - LTM delete/rebuild: delete updates index; subsequent search excludes deleted entry.
  - Debug API: `debug_context_building` returns consistent counts and token stats.
  - TokenCounter fallback: simulate tokenizer failure and assert char/4 estimation path used.

- **Integration tests** (with mocks)
  - Overflow path: add turns to exceed STM; ensure `_summarize_and_offload` is invoked and summary stored in LTM.
  - Retrieval quality: add diverse memories; assert thresholding and ranking place relevant items first.
  - Session isolation (once added): memories from session A are not retrieved for session B without cross-session retrieval enabled.

- **Performance/regression**
  - Context build latency under N turns and M LTM entries (no external calls).
  - Memory usage sanity for default configuration.

### Suggested near-term roadmap

1. Align embedding dimension and FAISS metric; add config validation and tests.
2. Add disk persistence (save/load) for LTM entries and FAISS index.
3. Implement budget-aware summarization before LLM invocation; add tests.
4. Correct async adapter implementations; add adapter unit tests with mocks.
5. Introduce basic session scoping in `ContextManager` and retrieval; add tests.
6. Add `requirements-dev.txt` and expand test suite as above.



