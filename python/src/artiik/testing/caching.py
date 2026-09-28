"""Prompt-cache simulations, so fake usage numbers behave like real ones.

- Anthropic caches the prefix up to each cache breakpoint, and a later request
  reads the longest cached prefix found within a lookback window of blocks
  before one of its breakpoints.
- OpenAI caches automatically: a request reads the longest prefix seen before,
  counted in 128-token increments once it reaches 1,024 tokens.

Both ignore time-to-live, and the Anthropic one ignores the minimum cacheable
prompt length. That's enough for tests that run in seconds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    """One cacheable piece of a prompt: a tool definition, a system block or a content block."""

    key: str
    """A canonical serialization of the piece, without its ``cache_control``."""

    tokens: int
    breakpoint: bool = False


@dataclass(frozen=True)
class AnthropicUsage:
    """How a request's input tokens split between the cache and fresh input."""

    cache_read: int
    cache_write: int
    uncached: int


class AnthropicCache:
    """Simulates Anthropic prompt caching."""

    def __init__(self, *, lookback: int = 20) -> None:
        self.lookback = lookback
        self._written: set[str] = set()

    def account(self, units: Sequence[Unit]) -> AnthropicUsage:
        """Split a request's tokens, and write the new prefixes to the cache."""
        breakpoints = [index for index, unit in enumerate(units) if unit.breakpoint]
        if not breakpoints:
            return AnthropicUsage(
                cache_read=0, cache_write=0, uncached=sum(unit.tokens for unit in units)
            )
        hashes, cumulative = _prefixes(units)
        total = cumulative[-1]
        read_at = -1
        for point in breakpoints:
            for position in range(point, max(point - self.lookback, 0) - 1, -1):
                if hashes[position] in self._written:
                    read_at = max(read_at, position)
                    break
        for point in breakpoints:
            if point > read_at:
                self._written.add(hashes[point])
        last = breakpoints[-1]
        cache_read = cumulative[read_at] if read_at >= 0 else 0
        cache_write = cumulative[last] - cache_read if last > read_at else 0
        return AnthropicUsage(
            cache_read=cache_read, cache_write=cache_write, uncached=total - cumulative[last]
        )


class OpenAICache:
    """Simulates OpenAI automatic prompt caching."""

    minimum_tokens = 1_024
    increment = 128

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def account(self, units: Sequence[Unit]) -> int:
        """Return the cached tokens for a request, and remember its prefixes."""
        hashes, cumulative = _prefixes(units)
        matched = 0
        for index in range(len(units) - 1, -1, -1):
            if hashes[index] in self._seen:
                matched = cumulative[index]
                break
        self._seen.update(hashes)
        if matched < self.minimum_tokens:
            return 0
        return matched // self.increment * self.increment


def _prefixes(units: Sequence[Unit]) -> tuple[list[str], list[int]]:
    digest = hashlib.sha256()
    hashes: list[str] = []
    cumulative: list[int] = []
    total = 0
    for unit in units:
        digest.update(unit.key.encode())
        digest.update(b"\x00")
        hashes.append(digest.copy().hexdigest())
        total += unit.tokens
        cumulative.append(total)
    return hashes, cumulative
