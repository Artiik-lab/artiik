"""Calls recorded by the fake clients, and helpers that read them in the neutral model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.messages import Format, JSONObject, JSONValue, Message, last_compaction
from artiik.testing.errors import FakeAPIError
from artiik.testing.tokens import DEFAULT, Tokenizer


@dataclass(frozen=True)
class RecordedCall:
    """One call to a fake client: the request as sent, and the response or the error."""

    index: int
    endpoint: str
    format: Format
    request: JSONObject
    response: JSONObject | None = None
    error: FakeAPIError | None = None
    prompt_tokens: int | None = None
    """The prompt's size as the fake counted it, when it got that far."""

    @property
    def ok(self) -> bool:
        """Whether the call succeeded."""
        return self.error is None

    @property
    def is_compaction(self) -> bool:
        """Whether the call asked for a compaction instead of a reply."""
        return "compaction" in self.request or self.endpoint == "responses.compact"

    def conversation(self) -> list[Message]:
        """The request's messages (or Responses input items) in the neutral model."""
        return list(self._conversation)

    @cached_property
    def _conversation(self) -> tuple[Message, ...]:
        # Parsed once: several invariants read the same calls.
        return tuple(parse_conversation(self.format, self.request))

    def system(self) -> list[Message]:
        """The request's system prompt, if it's sent outside the messages."""
        return parse_system(self.format, self.request)

    def tokens(self) -> int:
        """The size of the prompt the model reads: the fake's count, or the default tokenizer's."""
        if self.prompt_tokens is not None:
            return self.prompt_tokens
        return request_tokens(self.format, self.request)


def entries(fmt: Format, request: JSONObject) -> list[JSONValue]:
    """The conversation entries of a request: its messages, or its Responses input items."""
    if fmt is Format.OPENAI_RESPONSES:
        raw = request.get("input", [])
        if isinstance(raw, str):
            return [{"role": "user", "content": raw}]
    else:
        raw = request.get("messages", [])
    return list(raw) if isinstance(raw, list) else []


def parse_conversation(fmt: Format, request: JSONObject) -> list[Message]:
    """Parse a request's conversation entries."""
    items = entries(fmt, request)
    match fmt:
        case Format.ANTHROPIC_MESSAGES:
            return anthropic_messages.parse_messages(items)
        case Format.OPENAI_RESPONSES:
            return openai_responses.parse_items(items)
        case Format.OPENAI_CHAT:
            return openai_chat.parse_messages(items)


def parse_system(fmt: Format, request: JSONObject) -> list[Message]:
    """Parse a request's system prompt when it's a separate parameter."""
    if fmt is Format.ANTHROPIC_MESSAGES and "system" in request:
        return [anthropic_messages.parse_system(request["system"])]
    instructions = request.get("instructions")
    if fmt is Format.OPENAI_RESPONSES and isinstance(instructions, str):
        return [Message.from_text("system", instructions)]
    return []


def request_tokens(fmt: Format, request: JSONObject, tokenizer: Tokenizer = DEFAULT) -> int:
    """Count the prompt a request gives the model: tools, system prompt and conversation."""
    tools = request.get("tools")
    tool_tokens = (
        sum(tokenizer.count_json(tool) for tool in tools) if isinstance(tools, list) else 0
    )
    conversation = parse_conversation(fmt, request)
    messages = parse_system(fmt, request) + conversation[last_compaction(conversation) :]
    return tool_tokens + tokenizer.count_messages(messages)
