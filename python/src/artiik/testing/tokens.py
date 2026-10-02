"""A deterministic stand-in tokenizer: about four characters per token.

The fake clients use it for usage numbers and context-window checks, and tests
use it to compute exact budgets. It isn't meant to match any real tokenizer. A
:class:`Tokenizer` with other settings stands in for a different model family,
for example to check that artiik's estimator calibrates itself.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass

from artiik.messages import (
    Block,
    Compaction,
    Document,
    Image,
    JSONValue,
    Message,
    Opaque,
    RedactedThinking,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)

CHARS_PER_TOKEN = 4
MEDIA_TOKENS = 1_000
"""What an image or a document costs, whatever its size."""

BLOCK_OVERHEAD = 1
MESSAGE_OVERHEAD = 3


@dataclass(frozen=True)
class Tokenizer:
    """Counts tokens from characters, plus fixed costs per block, message and media item."""

    chars_per_token: float = CHARS_PER_TOKEN
    block_overhead: int = BLOCK_OVERHEAD
    message_overhead: int = MESSAGE_OVERHEAD
    media_tokens: int = MEDIA_TOKENS

    def count_text(self, text: str) -> int:
        """Tokens in a piece of text."""
        return math.ceil(len(text) / self.chars_per_token)

    def count_json(self, value: JSONValue) -> int:
        """Tokens in a JSON value, counted on its compact serialization."""
        return self.count_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def count_block(self, block: Block) -> int:
        """Tokens in one content block, including its overhead."""
        return self._content(block) + self.block_overhead

    def count_message(self, message: Message) -> int:
        """Tokens in one message, including its overhead."""
        return self.message_overhead + sum(self.count_block(block) for block in message.blocks)

    def count_messages(self, messages: Iterable[Message]) -> int:
        """Tokens in a sequence of messages."""
        return sum(self.count_message(message) for message in messages)

    def _content(self, block: Block) -> int:
        match block:
            case Text():
                return self.count_text(block.text)
            case Image() | Document():
                return self.media_tokens
            case ToolUse():
                arguments = (
                    self.count_text(block.raw_arguments)
                    if block.raw_arguments is not None
                    else self.count_json(block.input)
                )
                return self.count_text(block.name) + arguments
            case ToolResult():
                if isinstance(block.content, str):
                    return self.count_text(block.content)
                if block.content is None:
                    return 0
                return sum(self.count_block(item) for item in block.content)
            case Thinking():
                return self.count_text(block.thinking)
            case RedactedThinking():
                return self.count_text(block.data)
            case Compaction():
                summary = block.summary
                return (
                    self.count_text(summary) if summary is not None else self.count_json(block.data)
                )
            case Opaque():
                return self.count_json(block.data)


DEFAULT = Tokenizer()
"""The tokenizer the fakes use unless they're given another."""


def count_text(text: str) -> int:
    """Tokens in a piece of text, with the default tokenizer."""
    return DEFAULT.count_text(text)


def count_json(value: JSONValue) -> int:
    """Tokens in a JSON value, with the default tokenizer."""
    return DEFAULT.count_json(value)


def count_block(block: Block) -> int:
    """Tokens in one content block, with the default tokenizer."""
    return DEFAULT.count_block(block)


def count_message(message: Message) -> int:
    """Tokens in one message, with the default tokenizer."""
    return DEFAULT.count_message(message)


def count_messages(messages: Iterable[Message]) -> int:
    """Tokens in a sequence of messages, with the default tokenizer."""
    return DEFAULT.count_messages(messages)
