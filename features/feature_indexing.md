# 📄 PRD – File, Document & Codebase Ingestion

## Problem

The library currently handles conversational memory (STM + LTM) very well, but it doesn’t directly ingest *external knowledge sources* like files, docs, or codebases. For building AI coding agents, research assistants, or domain-specific copilots, it’s essential to index and retrieve from such resources alongside conversational context.

## Goals

* Allow developers to index files, documents, and codebases into LTM.
* Make retrieval seamlessly blend conversational memory + document/code memory.
* Ensure scalability (chunking, metadata scoping, persistence).
* Keep the API consistent with the existing memory API.

------------
* Instead of inventing a new “document memory,” we should **reuse your existing LTM (`LongTermMemory`)** because it already supports arbitrary `MemoryEntry` objects.
* The new feature is really about adding a **file/doc ingestion pipeline** on top, which:

  * loads and chunks documents/code,
  * feeds them into `add_memory` with proper metadata,
  * and then retrieval just works as-is (because LTM → FAISS → scoring → hybrid filtering → budget assembly).
* Scoping (`session_id`, `task_id`) also works naturally for documents, so a codebase could be one “session” or “task.”
* STM stays conversation-focused, but documents bypass STM and go directly into LTM.
----------

## Functional Requirements

1. **Ingestion API**

   * `ContextManager.ingest_file(path, **metadata)`
   * `ContextManager.ingest_directory(path, file_types, recursive=True, **metadata)`
   * `ContextManager.ingest_text(text, source_id, **metadata)`
     Each ingestion method:
   * loads content,
   * splits into chunks,
   * calls `add_memory(chunk, metadata)` for each.

2. **Chunking Layer**

   * Default: split by tokens (using the same tokenizer as STM budget counting).
   * Configurable `chunk_size` and `chunk_overlap`.
   * Special strategies:

     * Markdown/Docs → paragraph-aware splitting.
     * Code → function/class-level splitting with optional fallback to lines.

3. **Metadata Enrichment**

   * Store useful metadata in each `MemoryEntry`:

     * `source_type`: file/doc/code
     * `source_id`: path, filename, or repo hash
     * `line_numbers` or `chunk_index`
     * Inherits `session_id`/`task_id` if provided.

4. **Retrieval Integration**

   * Retrieval pipeline unchanged (LTM → FAISS search → scoring → hybrid keyword filter → budget).
   * Context assembly will now include **both conversation + external documents**.

5. **Persistence**

   * `save_memory` / `load_memory` continues to persist FAISS + JSON.
   * Document/code chunks behave like any other LTM entries.

6. **Debugging & Monitoring**

   * `debug_context_building(input)` should show document/code hits distinctly (e.g., `[DOC]`, `[CODE]` prefix in debug logs).
   * `get_stats()` should report number of entries per `source_type`.

## Non-Goals

* Full code semantic parsing (AST-level).
* Automatic sync with changing repos (MVP is manual re-index).

## Example Usage

```python
# Create context manager for coding agent
cm = ContextManager(config, session_id="user_456", task_id="coding_project")

# Ingest repo files
cm.ingest_directory("./my_repo", file_types=[".py", ".md"], recursive=True, importance=0.8)

# Ask coding agent a question
user_input = "How is the database connection handled in this project?"
context = cm.build_context(user_input)

assistant_response = llm.generate(context)
cm.observe(user_input, assistant_response)
```

---

This keeps it aligned with your current pipeline (STM ↔ LTM) and extends it with an ingestion layer without breaking abstraction.

