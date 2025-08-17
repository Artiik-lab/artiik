# Codebase Assistant (Real LLM)

A real-world demo that indexes a local repository (code + docs) into ContextManager’s Long‑Term Memory and answers questions about it with OpenAI.

## Why this demo?
- Showcases ingestion (files/dirs) → retrieval → budget-aware context building.
- Demonstrates session/task scoping, importance-weighted ranking, and persistence.

## Requirements
- Python 3.9+
- Install project dependencies:

```bash
pip install -r requirements.txt
```

- OpenAI key:

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Usage

```bash
python3 Demos/use_cases/codebase_assistant/agent.py \
  --repo-path ./ \
  --question "How does long-term memory search compute ranking?" \
  --persist-dir data/codebase_assistant \
  --file-types .py,.md \
  --model gpt-4
```

Notes:
- First run ingests and persists memory. Subsequent runs load from `--persist-dir`.
- Use `--file-types` to filter (e.g., `.py,.md,.txt`), and `--no-recursive` to disable recursion.
- You can adjust ranking by providing `--ltm-importance` during ingestion.

## What it does
1. Loads existing memory if present; otherwise ingests files and persists FAISS + entries.
2. Builds a budget-aware context combining recent interaction + relevant chunks from the repo.
3. Calls OpenAI (via adapter) for a grounded answer.
4. Observes the interaction to keep conversation state for follow-ups.

## Example Questions
- "Where is user authentication handled?"
- "How does the LTM convert FAISS distances to cosine similarity proxy?"
- "Show me how summarization triggers when STM is full."
- "What files implement session/task scoping?"
