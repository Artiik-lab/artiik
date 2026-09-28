"""A deterministic stand-in tokenizer: about four characters per token.

The fake clients use it for usage numbers and context-window checks, and tests
use it to compute exact budgets. It isn't meant to match any real tokenizer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

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


def count_text(text: str) -> int:
    """Tokens in a piece of text."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def count_json(value: JSONValue) -> int:
    """Tokens in a JSON value, counted on its compact serialization."""
    return count_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def count_block(block: Block) -> int:
    """Tokens in one content block, including its overhead."""
    return _block_content(block) + BLOCK_OVERHEAD


def count_message(message: Message) -> int:
    """Tokens in one message, including its overhead."""
    return MESSAGE_OVERHEAD + sum(count_block(block) for block in message.blocks)


def count_messages(messages: Iterable[Message]) -> int:
    """Tokens in a sequence of messages."""
    return sum(count_message(message) for message in messages)


def _block_content(block: Block) -> int:
    match block:
        case Text():
            return count_text(block.text)
        case Image() | Document():
            return MEDIA_TOKENS
        case ToolUse():
            arguments = (
                count_text(block.raw_arguments)
                if block.raw_arguments is not None
                else count_json(block.input)
            )
            return count_text(block.name) + arguments
        case ToolResult():
            if isinstance(block.content, str):
                return count_text(block.content)
            if block.content is None:
                return 0
            return sum(count_block(item) for item in block.content)
        case Thinking():
            return count_text(block.thinking)
        case RedactedThinking():
            return count_text(block.data)
        case Compaction():
            summary = block.summary
            return count_text(summary) if summary is not None else count_json(block.data)
        case Opaque():
            return count_json(block.data)
