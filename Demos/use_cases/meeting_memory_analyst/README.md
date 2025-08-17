# Meeting Memory Analyst (Context-Only)

This demo ingests meeting transcripts/notes into ContextManager’s Long‑Term Memory and builds a token‑optimized context for downstream LLMs. It does NOT call any LLM itself; you own the model call. The goal is to show how to use the library to assemble high‑signal context for your agent.

## Why this demo?
- Realistic ingestion of meeting notes (e.g., `.md`, `.txt`, `.vtt`, `.srt`).
- Scoping (session/task) and persistence between runs.
- Budget‑aware prompt assembly for your chosen LLM.

## Requirements
- Python 3.9+
- Install project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 Demos/use_cases/meeting_memory_analyst/agent.py \
  --notes-path ./meetings \
  --question "What decisions did we make about Q4 roadmap?" \
  --persist-dir data/meeting_memory \
  --file-types .md,.txt,.vtt,.srt \
  --session-id team_alpha --task-id q4_planning \
  --budget 6000 --ltm-k 10 --recent-k 5
```

Output includes:
- A single context string (printed to stdout) that you can pass to your LLM.
- Optional debug metrics (counts and token usage).

## Integrating with your LLM
This demo does not use any internal adapters. After you get the context string, call your LLM of choice:

Example (OpenAI SDK shown only for illustration; not used by the demo):
```python
# pseudo-code
messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": context},
]
response = client.chat.completions.create(model="gpt-4o", messages=messages)
assistant = response.choices[0].message.content

# Optionally record the interaction afterwards
cm.observe(user_input, assistant)
```

## Tips
- Re-run with new `--question` to get fresh budget‑aware context.
- After your LLM returns, consider calling `cm.observe(user_input, assistant_response)` to grow conversation memory.
- Use `--persist-dir` to avoid re‑ingestion on subsequent runs.
