#!/usr/bin/env python3
"""
Demonstrate session and task scoping for memory isolation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager.core import ContextManager, Config


def main():
    cfg = Config()

    # Two sessions
    cm_a = ContextManager(cfg, session_id="sessA")
    cm_b = ContextManager(cfg, session_id="sessB")

    cm_a.add_memory("Alice prefers Python", {"importance": 0.8})
    cm_b.add_memory("Bob prefers JavaScript", {"importance": 0.8})

    print("\n🔐 Session isolation")
    print("A queries 'prefers'")
    print(cm_a.query_memory("prefers", k=5))
    print("B queries 'prefers'")
    print(cm_b.query_memory("prefers", k=5))

    # Allow cross-session on B
    cm_b.set_session("sessB", allow_cross_session=True)
    print("\n🔓 B allows cross-session, queries 'prefers'")
    print(cm_b.query_memory("prefers", k=5))

    # Task scoping
    cm_x = ContextManager(cfg, task_id="taskX")
    cm_y = ContextManager(cfg, task_id="taskY")
    cm_x.add_memory("Task X: deploy staging")
    cm_y.add_memory("Task Y: prepare launch email")

    print("\n🧩 Task isolation")
    print("X queries 'Task'")
    print(cm_x.query_memory("Task", k=5))
    print("Y queries 'Task'")
    print(cm_y.query_memory("Task", k=5))

    cm_y.set_task("taskY", allow_cross_task=True)
    print("\n🧩 Y allows cross-task, queries 'Task'")
    print(cm_y.query_memory("Task", k=5))


if __name__ == "__main__":
    main()


