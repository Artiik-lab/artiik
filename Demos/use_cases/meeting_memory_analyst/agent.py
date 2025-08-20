#!/usr/bin/env python3
"""
Meeting Memory Analyst demo that focuses on context assembly only.

It ingests meeting notes (files in a directory), persists LTM, and builds a
token-budgeted context you can pass to your own LLM. This script does NOT call
any LLM; adapters are used internally by the library for summarization only.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from context_manager.core import ContextManager, Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meeting Memory Analyst (Context Only)")
    p.add_argument("--notes-path", type=str, required=True, help="Path to meeting notes directory")
    p.add_argument("--persist-dir", type=str, default="data/meeting_memory", help="Directory to save/load LTM")
    p.add_argument("--file-types", type=str, default=".md,.txt,.vtt,.srt", help="Comma-separated extensions")
    p.add_argument("--no-recursive", action="store_true", help="Disable recursive ingestion")
    p.add_argument("--ltm-importance", type=float, default=0.7, help="Default importance for ingested chunks")
    p.add_argument("--session-id", type=str, default="team_meetings", help="Session id")
    p.add_argument("--task-id", type=str, default="planning_cycle", help="Task id")
    p.add_argument("--budget", type=int, default=6000, help="Prompt token budget")
    p.add_argument("--ltm-k", type=int, default=10, help="Number of LTM hits to retrieve")
    p.add_argument("--recent-k", type=int, default=5, help="Number of recent turns to keep")
    p.add_argument("--question", type=str, required=True, help="Question to ask about the meetings")
    # Optional: call OpenAI Responses API (not required; default is context-only)
    p.add_argument("--call-openai", action="store_true", help="Call OpenAI Responses API with the assembled context")
    p.add_argument("--openai-model", type=str, default="gpt-4o", help="OpenAI model for Responses API (e.g., gpt-4o)")
    p.add_argument(
        "--system",
        type=str,
        default=(
            "You are a helpful meeting memory analyst. Use the provided context snippets to answer the user question. "
            "Be concise and point to concrete decisions, dates, owners, and action items. If uncertain, state what is missing."
        ),
        help="System instruction prepended to the context for the model.",
    )
    return p.parse_args()


def maybe_ingest(cm: ContextManager, notes_path: str, file_types: list[str], recursive: bool, importance: float, persist_dir: str) -> None:
    idx = Path(persist_dir) / "index.faiss"
    ents = Path(persist_dir) / "entries.json"
    if idx.exists() and ents.exists():
        print(f"Loading existing memory from {persist_dir}...")
        cm.load_memory(persist_dir)
        return

    print(f"Ingesting meeting notes at {notes_path}...")
    total = cm.ingest_directory(notes_path, file_types=file_types, recursive=recursive, importance=importance)
    print(f"Ingested {total} chunks. Persisting memory to {persist_dir}...")
    cm.save_memory(persist_dir)


def main() -> None:
    args = parse_args()

    cfg = Config()
    cfg.memory.prompt_token_budget = args.budget
    cfg.memory.ltm_hits_k = args.ltm_k
    cfg.memory.recent_k = args.recent_k

    cm = ContextManager(cfg, session_id=args.session_id, task_id=args.task_id)

    file_types = [e.strip() for e in args.file_types.split(",") if e.strip()]
    maybe_ingest(cm, args.notes_path, file_types, (not args.no_recursive), args.ltm_importance, args.persist_dir)

    # Build context only (no LLM call here)
    context = cm.build_context(args.question)

    print("\n🔎 Question:")
    print(args.question)
    print("\n🧠 Assembled Context (pass this to your LLM):\n")
    print(context)

    # Optionally call OpenAI Responses API (no chat.completions)
    if args.call_openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n⚠️  Skipping OpenAI call: OPENAI_API_KEY is not set.")
        else:
            try:
                from openai import OpenAI  # imported here to avoid hard dependency when not used

                client = OpenAI(api_key=api_key)
                prompt = f"System: {args.system}\n\nUser: {context}\n\nAssistant:"
                resp = client.responses.create(model=args.openai_model, input=prompt)

                # Best-effort extraction across SDK versions
                answer = None
                try:
                    answer = getattr(resp, "output_text", None)
                except Exception:
                    answer = None
                if not answer:
                    try:
                        # Newer SDKs: resp.output[0].content[0].text.value
                        output = getattr(resp, "output", None)
                        if output and len(output) > 0:
                            content = getattr(output[0], "content", None)
                            if content and len(content) > 0:
                                text_obj = getattr(content[0], "text", None)
                                if text_obj is not None:
                                    answer = getattr(text_obj, "value", None) or str(text_obj)
                    except Exception:
                        pass
                if not answer:
                    answer = str(resp)

                print("\n🤖 Assistant (OpenAI Responses):\n")
                print(answer)

                # Optionally record turn
                cm.observe(args.question, answer)
            except Exception as e:
                print(f"\n❌ OpenAI call failed: {e}")

    dbg = cm.debug_context_building(args.question)
    print("\n📊 Debug:")
    print(f"  recent_turns: {dbg['recent_turns_count']}")
    print(f"  ltm_hits: {dbg['ltm_results_count']}")
    print(f"  final_tokens: {dbg['final_context_tokens']}/{dbg['context_budget']}")



if __name__ == "__main__":
    main()


