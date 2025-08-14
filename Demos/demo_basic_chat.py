#!/usr/bin/env python3
"""
Minimal demo that uses ContextManager without calling an external LLM.
It simulates an assistant response and shows how memory evolves.
"""

import os
import sys

# Ensure local import works when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager.core import ContextManager, Config


def fake_llm_reply(prompt: str) -> str:
    # Simulate a reply without external API calls
    return f"[Simulated reply] {prompt[:120]}..."


def main():
    cfg = Config()
    cm = ContextManager(cfg)

    conversation = [
        "Hi! I'm researching a trip to Japan.",
        "I want to visit Tokyo and Kyoto. Any must-see spots?",
        "Remind me later about cherry blossom season.",
        "What's my current plan so far?",
    ]

    print("\n🧠 Demo: Basic Chat (no external LLM)\n" + "=" * 40)
    for i, user in enumerate(conversation, 1):
        context = cm.build_context(user)
        reply = fake_llm_reply(context)
        cm.observe(user, reply)
        print(f"\nUser {i}: {user}")
        print(f"Assistant: {reply[:200]}")

    stats = cm.get_stats()
    print("\n📊 Memory Stats:")
    print(f"  STM turns: {stats['short_term_memory']['num_turns']}")
    print(f"  LTM entries: {stats['long_term_memory']['num_entries']}")


if __name__ == "__main__":
    main()


