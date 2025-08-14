#!/usr/bin/env python3
"""
Show debug info and hybrid retrieval behavior.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager.core import ContextManager, Config


def main():
    cfg = Config()
    cm = ContextManager(cfg)

    # Seed some memories; one matches the keyword strongly
    cm.add_memory("We discussed Python packaging best practices", {"topic": "dev"})
    cm.add_memory("Completely unrelated gardening tips", {"topic": "garden"})

    # Add a couple recent turns
    cm.observe("Tell me about Python wheels", "Sure, wheels are built artifacts...")
    cm.observe("How do I publish to PyPI?", "Use build + twine...")

    query = "How to build Python wheels?"
    context = cm.build_context(query)
    dbg = cm.debug_context_building(query)

    print("\n🧪 Debug & Hybrid Retrieval Demo")
    print("Context preview:\n", context[:400], "...", sep="")
    print("\nDebug:")
    print(f"  recent_turns: {dbg['recent_turns_count']}")
    print(f"  ltm_hits: {dbg['ltm_results_count']}")
    print(f"  final_tokens: {dbg['final_context_tokens']}/{dbg['context_budget']}")


if __name__ == "__main__":
    main()


