# Demos

Practical scripts that showcase how to use ContextManager in real scenarios.

## Prerequisites

- Python 3.9+
- Install deps:

```bash
pip install -r requirements.txt
```

- First run will download a SentenceTransformer model (~90MB).
- Some demos can use OpenAI if you provide an API key:

```bash
export OPENAI_API_KEY="your-key"
```

## Run any demo

```bash
python3 Demos/demo_basic_chat.py
```

## Included demos

- demo_basic_chat.py
  - Minimal Build → Call → Observe loop with fallback responses (no API key required)
- demo_openai_chat.py
  - Same flow but uses OpenAI for real responses (requires `OPENAI_API_KEY`)
- demo_session_and_task_scoping.py
  - Shows session/task isolation vs. allowed cross-session/task retrieval
- demo_persistence.py
  - Saves LTM to disk and loads it in a new manager instance
- demo_weighted_ranking.py
  - Demonstrates ranking by importance and recency via `MemoryConfig` weights
- demo_debugging_and_hybrid.py
  - Shows debug info for context building and hybrid (keyword + vector) retrieval

## Tips

- If you hit FAISS or NumPy warnings on macOS, consider:
  - `brew install libomp`
  - Using a stable NumPy version (e.g., 1.26.x)
