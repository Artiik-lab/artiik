#!/usr/bin/env python3
"""
Demonstrate saving and loading long-term memory (FAISS + entries) from disk.
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager.core import ContextManager, Config


def main():
    cfg = Config()
    cm = ContextManager(cfg)

    cm.add_memory("User likes coffee", {"importance": 0.5})
    cm.add_memory("Project deadline is next Friday", {"importance": 0.9})

    tmpdir = tempfile.mkdtemp(prefix="cm_demo_")
    try:
        cm.save_memory(tmpdir)
        print(f"Saved memory to {tmpdir}")

        cm2 = ContextManager(cfg)
        cm2.load_memory(tmpdir)
        print("Loaded memory into a new manager instance")

        print("Query 'deadline':")
        print(cm2.query_memory("deadline", k=5))
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()


