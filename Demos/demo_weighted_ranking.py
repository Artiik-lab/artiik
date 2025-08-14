#!/usr/bin/env python3
"""
Demonstrate ranking configuration: similarity, recency, and importance weights.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager.core import ContextManager, Config


def main():
    cfg = Config()
    # Emphasize importance over similarity/recency
    cfg.memory.similarity_weight = 0.1
    cfg.memory.recency_weight = 0.0
    cfg.memory.importance_weight = 1.0

    cm = ContextManager(cfg)
    cm.add_memory("low importance note", {"importance": 0.2})
    cm.add_memory("high importance note", {"importance": 0.9})

    print("\n🏷️ Importance-weighted ranking (query='note'):")
    print(cm.query_memory("note", k=2))

    # Shift to recency emphasis
    cfg.memory.similarity_weight = 0.0
    cfg.memory.importance_weight = 0.0
    cfg.memory.recency_weight = 1.0
    cfg.memory.recency_half_life_seconds = 3600

    cm2 = ContextManager(cfg)
    now = time.time()
    cm2.add_memory("older item")
    cm2.long_term_memory.entries[-1].timestamp = now - 7200  # 2 hours
    cm2.add_memory("newer item")
    cm2.long_term_memory.entries[-1].timestamp = now - 60  # 1 minute

    print("\n⏱️ Recency-weighted ranking (query='item'):")
    print(cm2.query_memory("item", k=2))


if __name__ == "__main__":
    main()


