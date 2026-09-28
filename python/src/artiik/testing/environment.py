"""Deterministic tools for scripted sessions."""

from __future__ import annotations

import hashlib
import json

from artiik.messages import JSONValue
from artiik.testing.tokens import CHARS_PER_TOKEN


class Environment:
    """Runs tool calls: every call returns text of a fixed size, derived from the call.

    The same call always returns the same output, so sessions are reproducible.
    ``calls`` records every call, in order.
    """

    def __init__(self, *, output_tokens: int = 200) -> None:
        self.output_tokens = output_tokens
        self.calls: list[tuple[str, JSONValue]] = []

    def run(self, name: str, arguments: JSONValue) -> str:
        """Run a tool and return its output."""
        self.calls.append((name, arguments))
        payload = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        header = f"{name} output {digest[:12]}\n"
        line = f"log {digest[:8]} ok\n"
        size = max(self.output_tokens * CHARS_PER_TOKEN - len(header), 0)
        return header + (line * (size // len(line) + 1))[:size]
