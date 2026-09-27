"""Anthropic Messages API format.

Covers the ``system`` and ``messages`` parameters of a request, and the
``content`` of a response. Block types artiik doesn't model (server tool use,
MCP blocks, search results, container uploads, future types) are kept as
:class:`~artiik.messages.Opaque`.

Reference: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

from collections.abc import Iterable

from artiik.errors import FormatError
from artiik.formats._json import (
    check_same_format,
    clone,
    expect_list,
    expect_object,
    get_object,
    get_str,
    start_block,
    to_json,
    to_object,
    without,
)
from artiik.messages import (
    Block,
    Compaction,
    Document,
    Format,
    Image,
    JSONObject,
    JSONValue,
    Message,
    Opaque,
    RedactedThinking,
    Role,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)

FORMAT = Format.ANTHROPIC_MESSAGES

_ROLES: dict[str, Role] = {"user": "user", "assistant": "assistant", "system": "system"}


def parse_messages(messages: Iterable[object]) -> list[Message]:
    """Parse the ``messages`` parameter of a request."""
    return [parse_message(message, f"messages[{index}]") for index, message in enumerate(messages)]


def parse_message(message: object, path: str = "message") -> Message:
    """Parse one message."""
    data = to_object(message, path)
    role_name = get_str(data, "role", path)
    role = _ROLES.get(role_name)
    if role is None:
        raise FormatError(f"{path}.role: unsupported role {role_name!r}")
    if "content" not in data:
        raise FormatError(f"{path}.content: missing")
    content = data["content"]
    extra = without(data, "role", "content")
    if isinstance(content, str):
        return Message(
            role=role,
            blocks=(Text(text=content, origin=FORMAT),),
            origin=FORMAT,
            extra=extra,
            text_shorthand=True,
        )
    return Message(
        role=role,
        blocks=_parse_blocks(content, f"{path}.content"),
        origin=FORMAT,
        extra=extra,
    )


def parse_system(system: object, path: str = "system") -> Message:
    """Parse the ``system`` parameter, a string or a list of text blocks."""
    data = to_json(system, path)
    if isinstance(data, str):
        return Message(
            role="system",
            blocks=(Text(text=data, origin=FORMAT),),
            origin=FORMAT,
            text_shorthand=True,
        )
    return Message(role="system", blocks=_parse_blocks(data, path), origin=FORMAT)


def parse_response(response: object, path: str = "response") -> Message:
    """Parse a response into the assistant message to append to the history.

    Only ``content`` is part of the message. Fields such as ``usage`` and
    ``stop_reason`` describe the call and are read elsewhere.
    """
    data = to_object(response, path)
    if "content" not in data:
        raise FormatError(f"{path}.content: missing")
    return Message(
        role="assistant",
        blocks=_parse_blocks(data["content"], f"{path}.content"),
        origin=FORMAT,
    )


def dump_messages(messages: Iterable[Message]) -> list[JSONObject]:
    """Build the ``messages`` parameter of a request."""
    return [dump_message(message, f"messages[{index}]") for index, message in enumerate(messages)]


def dump_message(message: Message, path: str = "message") -> JSONObject:
    """Build one message."""
    if message.role not in _ROLES:
        raise FormatError(f"{path}: role {message.role!r} can't be sent in {FORMAT.value}")
    if message.bare_item:
        raise FormatError(f"{path}: a bare item from {_origin_name(message)} can't be sent here")
    result: JSONObject = {"role": message.role}
    if message.origin in (None, FORMAT):
        result.update(clone(message.extra))
    result["content"] = _dump_content(message, f"{path}.content")
    return result


def dump_system(message: Message, path: str = "system") -> str | list[JSONObject]:
    """Build the ``system`` parameter."""
    content = _dump_content(message, path)
    if isinstance(content, str):
        return content
    return [expect_object(block, path) for block in content]


def _dump_content(message: Message, path: str) -> str | list[JSONValue]:
    blocks = message.blocks
    if (
        message.text_shorthand
        and len(blocks) == 1
        and isinstance(blocks[0], Text)
        and not blocks[0].extra
    ):
        return blocks[0].text
    return [_dump_block(block, f"{path}[{index}]") for index, block in enumerate(blocks)]


def _parse_blocks(value: JSONValue, path: str) -> tuple[Block, ...]:
    items = expect_list(value, path)
    return tuple(_parse_block(item, f"{path}[{index}]") for index, item in enumerate(items))


def _parse_block(value: JSONValue, path: str) -> Block:
    data = expect_object(value, path)
    match data.get("type"):
        case "text":
            return Text(
                text=get_str(data, "text", path), origin=FORMAT, extra=without(data, "text")
            )
        case "image":
            return Image(
                source=get_object(data, "source", path),
                origin=FORMAT,
                extra=without(data, "source"),
            )
        case "document":
            return Document(
                source=get_object(data, "source", path),
                origin=FORMAT,
                extra=without(data, "source"),
            )
        case "tool_use":
            if "input" not in data:
                raise FormatError(f"{path}.input: missing")
            return ToolUse(
                id=get_str(data, "id", path),
                name=get_str(data, "name", path),
                input=data["input"],
                origin=FORMAT,
                extra=without(data, "id", "name", "input"),
            )
        case "tool_result":
            return _parse_tool_result(data, path)
        case "thinking":
            signature = data.get("signature")
            modeled = ("thinking", "signature") if isinstance(signature, str) else ("thinking",)
            return Thinking(
                thinking=get_str(data, "thinking", path),
                signature=signature if isinstance(signature, str) else None,
                origin=FORMAT,
                extra=without(data, *modeled),
            )
        case "redacted_thinking":
            return RedactedThinking(
                data=get_str(data, "data", path),
                origin=FORMAT,
                extra=without(data, "data"),
            )
        case "compaction":
            return Compaction(data=data, origin=FORMAT)
        case _:
            return Opaque(data=data, origin=FORMAT)


def _parse_tool_result(data: JSONObject, path: str) -> ToolResult:
    modeled = ["tool_use_id"]
    content: str | tuple[Block, ...] | None = None
    raw_content = data.get("content")
    if isinstance(raw_content, str):
        content = raw_content
        modeled.append("content")
    elif isinstance(raw_content, list):
        content = _parse_blocks(raw_content, f"{path}.content")
        modeled.append("content")
    is_error = data.get("is_error")
    if isinstance(is_error, bool):
        modeled.append("is_error")
    return ToolResult(
        tool_use_id=get_str(data, "tool_use_id", path),
        content=content,
        is_error=is_error if isinstance(is_error, bool) else None,
        origin=FORMAT,
        extra=without(data, *modeled),
    )


def _dump_block(block: Block, path: str) -> JSONObject:
    match block:
        case Text():
            result = start_block(block, FORMAT, "text")
            result["text"] = block.text
            return result
        case Image() | Document():
            check_same_format(block, FORMAT, path)
            result = start_block(block, FORMAT, "image" if isinstance(block, Image) else "document")
            result["source"] = clone(block.source)
            return result
        case ToolUse():
            if block.origin not in (None, FORMAT) and block.input is None:
                raise FormatError(f"{path}: tool call {block.id!r} has arguments that aren't JSON")
            result = start_block(block, FORMAT, "tool_use")
            result["id"] = block.id
            result["name"] = block.name
            result["input"] = to_json(block.input, f"{path}.input")
            return result
        case ToolResult():
            result = start_block(block, FORMAT, "tool_result")
            result["tool_use_id"] = block.tool_use_id
            if isinstance(block.content, str):
                result["content"] = block.content
            elif block.content is not None:
                result["content"] = [
                    _dump_block(item, f"{path}.content[{index}]")
                    for index, item in enumerate(block.content)
                ]
            if block.is_error is not None:
                result["is_error"] = block.is_error
            return result
        case Thinking():
            check_same_format(block, FORMAT, path)
            result = start_block(block, FORMAT, "thinking")
            result["thinking"] = block.thinking
            if block.signature is not None:
                result["signature"] = block.signature
            return result
        case RedactedThinking():
            check_same_format(block, FORMAT, path)
            result = start_block(block, FORMAT, "redacted_thinking")
            result["data"] = block.data
            return result
        case Compaction() | Opaque():
            if block.origin is not FORMAT:
                raise FormatError(
                    f"{path}: a {type(block).__name__} block from "
                    f"{block.origin.value if block.origin else 'artiik'} can't be sent in "
                    f"{FORMAT.value}"
                )
            return clone(block.data)


def _origin_name(message: Message) -> str:
    return message.origin.value if message.origin is not None else "artiik"
