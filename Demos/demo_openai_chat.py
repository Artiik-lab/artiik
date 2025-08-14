#!/usr/bin/env python3
"""
Demo that uses ContextManager with OpenAI for real responses.
Requires OPENAI_API_KEY to be set in the environment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager.core import ContextManager, Config
from context_manager.llm.adapters import create_llm_adapter


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY to run this demo with real LLM responses.")
        return

    cfg = Config()
    cfg.llm.api_key = api_key

    cm = ContextManager(cfg)
    llm = create_llm_adapter("openai", api_key=api_key, model=cfg.llm.model)

    conversation = [
        "Hello! I'm planning a team offsite in Lisbon.",
        "We have 20 people. Suggest activities and budget.",
        "What did we decide about dates?",
    ]

    print("\n🤖 Demo: OpenAI Chat\n" + "=" * 30)
    for i, user in enumerate(conversation, 1):
        context = cm.build_context(user)
        prompt = f"User: {context}\n\nAssistant:"
        response = llm.generate_sync(prompt, max_tokens=300, temperature=0.5)
        cm.observe(user, response)
        print(f"\nUser {i}: {user}")
        print(f"Assistant: {response[:500]}")

    dbg = cm.debug_context_building("Summarize our Lisbon plan")
    print("\n🔎 Debug:")
    print(f"  recent_turns: {dbg['recent_turns_count']}")
    print(f"  ltm_hits: {dbg['ltm_results_count']}")
    print(f"  final_tokens: {dbg['final_context_tokens']}/{dbg['context_budget']}")


if __name__ == "__main__":
    main()


