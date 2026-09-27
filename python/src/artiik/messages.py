"""The provider-neutral conversation model.

A conversation is a sequence of :class:`Message` objects, each holding content
blocks. The modules in :mod:`artiik.formats` convert each provider's wire format
to and from this model without losing anything:

- fields artiik doesn't model are kept in ``extra`` and emitted again unchanged;
- blocks artiik doesn't understand are kept whole as :class:`Opaque`;
- blocks that must go back exactly as received (signed compaction blocks,
  encrypted reasoning) are copied, never rebuilt.

Messages and blocks are immutable. Change them with :func:`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, TypeAlias

JSONValue: TypeAlias = "bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None"
"""A value that survives a JSON round trip."""

JSONObject: TypeAlias = "dict[str, JSONValue]"
"""A JSON object."""

Role = Literal["system", "developer", "user", "assistant", "tool"]
"""Who a message comes from. ``tool`` is used by the OpenAI formats for tool results."""


class Format(StrEnum):
    """A provider wire format that artiik converts to and from."""

    ANTHROPIC_MESSAGES = "anthropic-messages"
    OPENAI_RESPONSES = "openai-responses"
    OPENAI_CHAT = "openai-chat"


def _empty_object() -> JSONObject:
    return {}


@dataclass(frozen=True, kw_only=True, slots=True)
class _BlockBase:
    origin: Format | None = None
    """The format the block was parsed from, or ``None`` for blocks artiik creates."""

    extra: JSONObject = field(default_factory=_empty_object, hash=False)
    """Fields of the original block that artiik doesn't model.

    They're emitted again, unchanged, when the block goes back to its ``origin``
    format, and dropped when it's converted to another format.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class Text(_BlockBase):
    """Plain text."""

    text: str


@dataclass(frozen=True, kw_only=True, slots=True)
class Image(_BlockBase):
    """An image. ``source`` keeps the provider's own description of where the image is."""

    source: JSONObject = field(hash=False)


@dataclass(frozen=True, kw_only=True, slots=True)
class Document(_BlockBase):
    """A document, such as a PDF. ``source`` keeps the provider's own description of it."""

    source: JSONObject = field(hash=False)


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolUse(_BlockBase):
    """A call from the model to a tool.

    ``input`` holds the parsed arguments. The OpenAI formats send arguments as a
    JSON string; that string is kept verbatim in ``raw_arguments`` and sent back
    as long as it still parses to ``input``. ``input`` is ``None`` when the model
    produced arguments that aren't valid JSON.
    """

    id: str
    name: str
    input: JSONValue = field(default=None, hash=False)
    raw_arguments: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolResult(_BlockBase):
    """The result of a tool call, answering the :class:`ToolUse` whose ``id`` is ``tool_use_id``.

    ``content`` is a string or nested blocks, or ``None`` when the provider
    allowed it to be left out.
    """

    tool_use_id: str
    content: str | tuple[Block, ...] | None = None
    is_error: bool | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class Thinking(_BlockBase):
    """Visible model reasoning, with the signature that lets it be sent back."""

    thinking: str
    signature: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class RedactedThinking(_BlockBase):
    """Encrypted model reasoning, sent back as received."""

    data: str


@dataclass(frozen=True, kw_only=True, slots=True)
class Compaction(_BlockBase):
    """A provider compaction block or item, kept exactly as received.

    Providers reject a compaction block that was modified, so ``data`` is only
    ever copied. :attr:`summary` exposes the summary text when the provider
    returns it readable.
    """

    data: JSONObject = field(hash=False)

    @property
    def summary(self) -> str | None:
        """The readable summary, or ``None`` when the provider keeps it opaque."""
        content = self.data.get("content")
        return content if isinstance(content, str) else None


@dataclass(frozen=True, kw_only=True, slots=True)
class Opaque(_BlockBase):
    """A block or item artiik doesn't model, kept exactly as received."""

    data: JSONObject = field(hash=False)

    @property
    def type(self) -> str | None:
        """The provider's ``type`` field, when there is one."""
        kind = self.data.get("type")
        return kind if isinstance(kind, str) else None


Block: TypeAlias = (
    Text
    | Image
    | Document
    | ToolUse
    | ToolResult
    | Thinking
    | RedactedThinking
    | Compaction
    | Opaque
)
"""Any content block."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Message:
    """One entry in a conversation.

    ``text_shorthand`` records that the provider sent the content as a plain
    string, so it's sent back the same way. ``bare_item`` marks an entry that
    isn't a message envelope but a single item holding one block: an OpenAI
    Responses item such as ``function_call`` or ``reasoning``, or a legacy Chat
    Completions message artiik keeps whole.
    """

    role: Role
    blocks: tuple[Block, ...] = ()
    origin: Format | None = None
    extra: JSONObject = field(default_factory=_empty_object, hash=False)
    text_shorthand: bool = False
    bare_item: bool = False

    @classmethod
    def from_text(cls, role: Role, text: str) -> Message:
        """Build a message holding a single text block."""
        return cls(role=role, blocks=(Text(text=text),))

    @property
    def text(self) -> str:
        """The concatenated text of the message's :class:`Text` blocks."""
        return "".join(block.text for block in self.blocks if isinstance(block, Text))

    @property
    def tool_uses(self) -> tuple[ToolUse, ...]:
        """The tool calls in this message."""
        return tuple(block for block in self.blocks if isinstance(block, ToolUse))

    @property
    def tool_results(self) -> tuple[ToolResult, ...]:
        """The tool results in this message."""
        return tuple(block for block in self.blocks if isinstance(block, ToolResult))
