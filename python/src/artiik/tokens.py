"""Token accounting: estimates that calibrate themselves, and exact counts when you opt in.

artiik needs a request's size before sending it. The provider's count of the
previous request, its ``usage``, is the ground truth, so a
:class:`~artiik.context.Context` only estimates what was added since. An
estimate starts from the number of characters, at about four per token, and
a ratio per API. The :class:`Estimator` then adjusts that ratio per model from
the usage the provider reports, because tokenizers differ between model
families.

Exact counts are opt-in: :class:`AnthropicTokenCounter` calls Anthropic's
token counting endpoint, and :class:`TiktokenCounter` counts OpenAI requests
with the ``tiktoken`` extra.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, cast

from artiik.errors import FormatError
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.formats._json import plain, to_object
from artiik.messages import (
    Block,
    Compaction,
    Document,
    Format,
    Image,
    JSONValue,
    Message,
    Opaque,
    RedactedThinking,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
    last_compaction,
)

CHARS_PER_TOKEN = 4.0
"""The unit of a raw estimate: an estimate counts characters in fours."""

MESSAGE_TOKENS = 3
"""Fixed tokens per message: its role and delimiters."""

BLOCK_TOKENS = 1
"""Fixed tokens per content block."""

MEDIA_TOKENS: Mapping[Format, int] = MappingProxyType(
    {
        Format.ANTHROPIC_MESSAGES: 1_600,
        Format.OPENAI_RESPONSES: 1_000,
        Format.OPENAI_CHAT: 1_000,
    }
)
"""A flat cost per image or document, until the provider's usage shows the real one."""


@dataclass(frozen=True, slots=True)
class Tally:
    """What an estimate is made from: characters of text, fixed tokens and media blocks."""

    chars: int = 0
    fixed: int = 0
    media: int = 0

    def __add__(self, other: Tally) -> Tally:
        return Tally(
            chars=self.chars + other.chars,
            fixed=self.fixed + other.fixed,
            media=self.media + other.media,
        )

    def __sub__(self, other: Tally) -> Tally:
        return Tally(
            chars=self.chars - other.chars,
            fixed=self.fixed - other.fixed,
            media=self.media - other.media,
        )


def tally_block(block: Block) -> Tally:
    """The raw material of one content block's estimate."""
    match block:
        case Text():
            return Tally(chars=len(block.text), fixed=BLOCK_TOKENS)
        case Image() | Document():
            return Tally(fixed=BLOCK_TOKENS, media=1)
        case ToolUse():
            arguments = (
                block.raw_arguments if block.raw_arguments is not None else _dumps(block.input)
            )
            return Tally(chars=len(block.name) + len(arguments), fixed=BLOCK_TOKENS)
        case ToolResult():
            if isinstance(block.content, str):
                content = Tally(chars=len(block.content))
            elif block.content is None:
                content = Tally()
            else:
                content = tally_blocks(block.content)
            return content + Tally(fixed=BLOCK_TOKENS)
        case Thinking():
            return Tally(chars=len(block.thinking), fixed=BLOCK_TOKENS)
        case RedactedThinking():
            return Tally(chars=len(block.data), fixed=BLOCK_TOKENS)
        case Compaction():
            summary = block.summary
            chars = len(summary) if summary is not None else len(_dumps(block.data))
            return Tally(chars=chars, fixed=BLOCK_TOKENS)
        case Opaque():
            return Tally(chars=len(_dumps(block.data)), fixed=BLOCK_TOKENS)


def tally_blocks(blocks: Iterable[Block]) -> Tally:
    """The raw material of several blocks' estimate."""
    return sum((tally_block(block) for block in blocks), Tally())


def tally_message(message: Message) -> Tally:
    """The raw material of one message's estimate, framing included."""
    return tally_blocks(message.blocks) + Tally(fixed=MESSAGE_TOKENS)


def tally_json(value: JSONValue) -> Tally:
    """The raw material of a JSON value's estimate, such as a tool definition."""
    return Tally(chars=len(_dumps(value)))


def tally_request(api: Format, request: Mapping[str, Any]) -> Tally:
    """The raw material of a whole request's estimate: tools, system prompt and conversation.

    For the Responses API, only the input from the latest compaction item on
    counts, as that's all the model reads.
    """
    data = to_object(plain(dict(request)), "request")
    tally = Tally()
    tools = data.get("tools")
    if isinstance(tools, list):
        tally += sum((tally_json(tool) for tool in tools), Tally())
    messages: list[Message]
    match api:
        case Format.ANTHROPIC_MESSAGES:
            if "system" in data:
                tally += tally_blocks(anthropic_messages.parse_system(data["system"]).blocks)
            raw = data.get("messages")
            messages = anthropic_messages.parse_messages(raw if isinstance(raw, list) else [])
        case Format.OPENAI_RESPONSES:
            instructions = data.get("instructions")
            if isinstance(instructions, str):
                tally += tally_message(Message.from_text("system", instructions))
            raw = data.get("input")
            items = [{"role": "user", "content": raw}] if isinstance(raw, str) else raw
            messages = openai_responses.parse_items(items if isinstance(items, list) else [])
            messages = messages[last_compaction(messages) :]
        case Format.OPENAI_CHAT:
            raw = data.get("messages")
            messages = openai_chat.parse_messages(raw if isinstance(raw, list) else [])
    return tally + sum((tally_message(message) for message in messages), Tally())


class Estimator:
    """Estimates prompt sizes, calibrated per model from the usage providers report.

    An estimate is ``ratio * chars / 4``, rounded up, plus the fixed tokens and
    a flat cost per image or document. ``ratio`` starts from a guess per API.
    After each call, :meth:`observe` compares what was sent with the tokens the
    provider counted, and the ratio follows, recent calls weighing most. Each
    context makes its own estimator; pass one estimator to several contexts to
    share the calibration.
    """

    START: ClassVar[Mapping[Format, float]] = MappingProxyType(
        {
            Format.ANTHROPIC_MESSAGES: 1.25,
            Format.OPENAI_RESPONSES: 1.1,
            Format.OPENAI_CHAT: 1.1,
        }
    )
    """Starting ratios. They lean high: a first estimate should overshoot, not undershoot."""

    MIN_RATIO: ClassVar[float] = 0.25
    MAX_RATIO: ClassVar[float] = 4.0

    def __init__(self, *, decay: float = 0.7) -> None:
        if not 0 <= decay < 1:
            raise ValueError(f"decay must be at least 0 and below 1, got {decay}")
        self.decay = decay
        self._sums: dict[tuple[Format, str], tuple[float, float]] = {}

    def ratio(self, api: Format, model: str) -> float:
        """The current tokens-per-four-characters ratio for a model."""
        sums = self._sums.get((api, model))
        if sums is None or sums[1] <= 0:
            return self.START[api]
        return min(max(sums[0] / sums[1], self.MIN_RATIO), self.MAX_RATIO)

    def estimate(self, tally: Tally, *, api: Format, model: str) -> int:
        """Estimate the tokens of a tally."""
        text = tally.chars / CHARS_PER_TOKEN * self.ratio(api, model)
        return math.ceil(text) + tally.fixed + tally.media * MEDIA_TOKENS[api]

    def observe(self, tally: Tally, actual: int, *, api: Format, model: str) -> None:
        """Learn from a request made of ``tally`` that the provider counted as ``actual`` tokens."""
        text = tally.chars / CHARS_PER_TOKEN
        measured = actual - tally.fixed - tally.media * MEDIA_TOKENS[api]
        if text <= 0 or measured <= 0:
            return
        weight, units = self._sums.get((api, model), (0.0, 0.0))
        self._sums[(api, model)] = (weight * self.decay + measured, units * self.decay + text)


class TokenCounter(Protocol):
    """Counts a request's input tokens exactly, or as exactly as the provider allows."""

    def count(self, api: Format, request: Mapping[str, Any]) -> int:
        """Count the input tokens of a request in the given format."""
        ...


class AnthropicTokenCounter:
    """Exact counts from Anthropic's token counting endpoint, ``messages.count_tokens``.

    ``client`` is an ``anthropic.Anthropic`` client. Each count is an API call,
    so a context only counts when its estimate comes close to the budget.
    Reference: https://docs.anthropic.com/en/api/messages-count-tokens
    """

    PARAMETERS: ClassVar[tuple[str, ...]] = (
        "model",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "thinking",
        "mcp_servers",
    )
    BETA_PARAMETERS: ClassVar[tuple[str, ...]] = ("betas", "context_management")

    def __init__(self, client: Any) -> None:
        self._client = client

    def count(self, api: Format, request: Mapping[str, Any]) -> int:
        """Count a Messages API request, through the beta endpoint when it has ``betas``."""
        if api is not Format.ANTHROPIC_MESSAGES:
            raise ValueError(
                f"AnthropicTokenCounter counts {Format.ANTHROPIC_MESSAGES.value} requests"
            )
        params = {key: request[key] for key in self.PARAMETERS if key in request}
        if "betas" in request:
            params.update({key: request[key] for key in self.BETA_PARAMETERS if key in request})
            endpoint = self._client.beta.messages.count_tokens
        else:
            endpoint = self._client.messages.count_tokens
        result = to_object(plain(endpoint(**params)), "count_tokens")
        tokens = result.get("input_tokens")
        if not isinstance(tokens, int):
            raise FormatError("count_tokens.input_tokens: missing")
        return tokens


class Encoding(Protocol):
    """The part of a ``tiktoken.Encoding`` that :class:`TiktokenCounter` uses."""

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special tokens as plain text."""
        ...


class TiktokenCounter:
    """Counts OpenAI requests with tiktoken: exactly for text, by estimate for the rest.

    Needs the ``tiktoken`` extra: ``pip install 'artiik[tiktoken]'``. The
    framing follows OpenAI's guidance for chat models: three tokens per
    message, one more for a name, and three to prime the reply. Tool
    definitions are counted on their JSON, and images, files and opaque items
    use the estimator's flat costs, so those parts are approximate.
    """

    MESSAGE_TOKENS: ClassVar[int] = 3
    REPLY_TOKENS: ClassVar[int] = 3

    def __init__(self, encoding: str | Encoding = "o200k_base") -> None:
        if isinstance(encoding, str):
            try:
                tiktoken = importlib.import_module("tiktoken")
            except ImportError as error:
                raise ImportError(
                    "TiktokenCounter needs tiktoken: pip install 'artiik[tiktoken]'"
                ) from error
            encoding = cast(Encoding, tiktoken.get_encoding(encoding))
        self._encoding = encoding

    def count(self, api: Format, request: Mapping[str, Any]) -> int:
        """Count a Responses or Chat Completions request."""
        data = to_object(plain(dict(request)), "request")
        messages: list[Message]
        match api:
            case Format.OPENAI_RESPONSES:
                raw = data.get("input", [])
                items = [{"role": "user", "content": raw}] if isinstance(raw, str) else raw
                messages = openai_responses.parse_items(items if isinstance(items, list) else [])
                messages = messages[last_compaction(messages) :]
                instructions = data.get("instructions")
                if isinstance(instructions, str):
                    messages.insert(0, Message.from_text("system", instructions))
            case Format.OPENAI_CHAT:
                raw = data.get("messages", [])
                messages = openai_chat.parse_messages(raw if isinstance(raw, list) else [])
            case Format.ANTHROPIC_MESSAGES:
                raise ValueError(
                    "TiktokenCounter counts OpenAI requests; use AnthropicTokenCounter"
                )
        total = self.REPLY_TOKENS + sum(self._message(api, message) for message in messages)
        tools = data.get("tools")
        if isinstance(tools, list) and tools:
            total += self._text(_dumps(tools))
        return total

    def _message(self, api: Format, message: Message) -> int:
        tokens = self.MESSAGE_TOKENS + self._text(message.role)
        name = message.extra.get("name")
        if isinstance(name, str):
            tokens += 1 + self._text(name)
        return tokens + sum(self._block(api, block) for block in message.blocks)

    def _block(self, api: Format, block: Block) -> int:
        match block:
            case Text():
                return self._text(block.text)
            case Image() | Document():
                return MEDIA_TOKENS[api]
            case ToolUse():
                arguments = (
                    block.raw_arguments if block.raw_arguments is not None else _dumps(block.input)
                )
                return self._text(block.name) + self._text(arguments)
            case ToolResult():
                if isinstance(block.content, str):
                    return self._text(block.content)
                if block.content is None:
                    return 0
                return sum(self._block(api, item) for item in block.content)
            case Thinking():
                return self._text(block.thinking)
            case RedactedThinking():
                return self._text(block.data)
            case Compaction() | Opaque():
                return self._text(_dumps(block.data))

    def _text(self, text: str) -> int:
        return len(self._encoding.encode_ordinary(text)) if text else 0


def _dumps(value: JSONValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
