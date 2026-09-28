"""Calls recorded by the fake clients, and helpers that read them in the neutral model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.messages import Compaction, Format, JSONObject, JSONValue, Message
from artiik.testing.errors import FakeAPIError
from artiik.testing.tokens import count_json, count_messages


@dataclass(frozen=True)
class RecordedCall:
    """One call to a fake client: the request as sent, and the response or the error."""

    index: int
    endpoint: str
    format: Format
    request: JSONObject
    response: JSONObject | None = None
    error: FakeAPIError | None = None

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
        return parse_conversation(self.format, self.request)

    def system(self) -> list[Message]:
        """The request's system prompt, if it's sent outside the messages."""
        return parse_system(self.format, self.request)

    def tokens(self) -> int:
        """The size of the prompt the model reads, counted with :mod:`artiik.testing.tokens`."""
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


def last_compaction(conversation: Sequence[Message]) -> int:
    """The position of the latest compaction block or item, or 0 when there is none.

    The model reads the conversation from there on: the Responses API ignores
    the input before the latest compaction item, and the Messages API requires
    the compaction block to come first.
    """
    for position in range(len(conversation) - 1, -1, -1):
        if any(isinstance(block, Compaction) for block in conversation[position].blocks):
            return position
    return 0


def request_tokens(fmt: Format, request: JSONObject) -> int:
    """Count the prompt a request gives the model: tools, system prompt and conversation."""
    tools = request.get("tools")
    tool_tokens = sum(count_json(tool) for tool in tools) if isinstance(tools, list) else 0
    conversation = parse_conversation(fmt, request)
    messages = parse_system(fmt, request) + conversation[last_compaction(conversation) :]
    return tool_tokens + count_messages(messages)
