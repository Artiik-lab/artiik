#!/usr/bin/env python3
"""
Codebase Assistant demo using ContextManager with real OpenAI calls.

Features:
- Ingest code/docs from a local repository into LTM (token-chunked)
- Persist and reload FAISS index and entries
- Build budget-aware context (recent + relevant LTM)
- Answer user question via OpenAI and observe the interaction for follow-ups
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from context_manager.core import ContextManager, Config
from context_manager.llm.adapters import create_llm_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Codebase Assistant (ContextManager + OpenAI)")
    p.add_argument("--repo-path", type=str, required=True, help="Path to repository root to ingest")
    p.add_argument("--persist-dir", type=str, default="data/codebase_assistant", help="Directory to save/load LTM")
    p.add_argument("--file-types", type=str, default=".py,.md", help="Comma-separated list of file extensions to include")
    p.add_argument("--no-recursive", action="store_true", help="Disable recursive ingestion")
    p.add_argument("--ltm-importance", type=float, default=0.8, help="Default importance metadata for ingested chunks")
    p.add_argument("--model", type=str, default="gpt-4", help="OpenAI model to use")
    p.add_argument("--question", type=str, required=True, help="Question to ask about the repo")
    p.add_argument("--session-id", type=str, default="codebase_session", help="Session id for scoping")
    p.add_argument("--task-id", type=str, default="codebase_task", help="Task id for scoping")
    return p.parse_args()


def ensure_openai_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this demo.")
    return api_key


def maybe_ingest(cm: ContextManager, repo_path: str, file_types: list[str], recursive: bool, importance: float, persist_dir: str) -> None:
    # If persisted data exists, load it; otherwise ingest and save
    idx = Path(persist_dir) / "index.faiss"
    ents = Path(persist_dir) / "entries.json"
    if idx.exists() and ents.exists():
        print(f"Loading existing memory from {persist_dir}...")
        cm.load_memory(persist_dir)
        return

    print(f"Ingesting repository at {repo_path}...")
    total = cm.ingest_directory(repo_path, file_types=file_types, recursive=recursive, importance=importance)
    print(f"Ingested {total} chunks. Persisting memory to {persist_dir}...")
    cm.save_memory(persist_dir)


def main() -> None:
    args = parse_args()
    api_key = ensure_openai_key()

    cfg = Config()
    cfg.llm.api_key = api_key
    cfg.llm.model = args.model
    cfg.memory.ltm_hits_k = 12
    cfg.memory.prompt_token_budget = 6000

    cm = ContextManager(cfg, session_id=args.session_id, task_id=args.task_id)
    llm = create_llm_adapter("openai", api_key=api_key, model=args.model)

    file_types = [e.strip() for e in args.file_types.split(",") if e.strip()]
    maybe_ingest(cm, args.repo_path, file_types, (not args.no_recursive), args.ltm_importance, args.persist_dir)

    # Ask the question using memory-enhanced context
    context = cm.build_context(args.question)
    prompt = (
        "You are a helpful coding assistant. Use the provided context snippets from the codebase to answer the question. "
        "Cite file names or functions when relevant. If unsure, say what further files to inspect.\n\n"
        f"User: {context}\n\nAssistant:"
    )
    print("\n🔎 Asking:", args.question)
    response = llm.generate_sync(prompt, max_tokens=800, temperature=0.4)
    cm.observe(args.question, response)

    print("\n✅ Answer:\n")
    print(response)

    # Show brief debug
    dbg = cm.debug_context_building(args.question)
    print("\n📊 Debug:")
    print(f"  recent_turns: {dbg['recent_turns_count']}")
    print(f"  ltm_hits: {dbg['ltm_results_count']}")
    print(f"  final_tokens: {dbg['final_context_tokens']}/{dbg['context_budget']}")


if __name__ == "__main__":
    main()


