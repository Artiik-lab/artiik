"""OpenAI Responses API format: the ``input`` items of a request and the ``output`` of a response.

Message items (with or without ``"type": "message"``) become messages. Every
other item becomes a *bare item*: a message holding one block. ``function_call``
and ``function_call_output`` items become :class:`~artiik.messages.ToolUse` and
:class:`~artiik.messages.ToolResult`; compaction items become
:class:`~artiik.messages.Compaction`; the rest (reasoning, built-in tool calls,
item references, future types) are kept as :class:`~artiik.messages.Opaque`.

Reference: https://platform.openai.com/docs/api-reference/responses/create
"""

from __future__ import annotations

from collections.abc import Iterable

from artiik.errors import FormatError
from artiik.formats._json import (
    check_same_format,
    clone,
    dump_arguments,
    expect_list,
    expect_object,
    get_str,
    parse_arguments,
    start_block,
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
    Role,
    Text,
    ToolResult,
    ToolUse,
)

FORMAT = Format.OPENAI_RESPONSES

_MESSAGE_ROLES: dict[str, Role] = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "developer": "developer",
}


def parse_items(items: Iterable[object], path: str = "input") -> list[Message]:
    """Parse a list of items: the ``input`` of a request or the ``output`` of a response."""
    return [parse_item(item, f"{path}[{index}]") for index, item in enumerate(items)]


def parse_item(item: object, path: str = "item") -> Message:
    """Parse one item."""
    data = to_object(item, path)
    kind = data.get("type")
    if kind is None or kind == "message":
        return _parse_message(data, path)
    if kind == "function_call":
        arguments = get_str(data, "arguments", path)
        tool_use = ToolUse(
            id=get_str(data, "call_id", path),
            name=get_str(data, "name", path),
            input=parse_arguments(arguments),
            raw_arguments=arguments,
            origin=FORMAT,
            extra=without(data, "call_id", "name", "arguments"),
        )
        return _bare(tool_use, "assistant")
    if kind == "function_call_output":
        return _bare(_parse_output(data, path), "tool")
    if kind == "compaction":
        return _bare(Compaction(data=data, origin=FORMAT), "assistant")
    return _bare(Opaque(data=data, origin=FORMAT), "assistant")


def dump_items(messages: Iterable[Message]) -> list[JSONObject]:
    """Build a list of items for the ``input`` of a request."""
    return [dump_item(message, f"input[{index}]") for index, message in enumerate(messages)]


def dump_item(message: Message, path: str = "item") -> JSONObject:
    """Build one item."""
    if message.bare_item:
        if len(message.blocks) != 1:
            raise FormatError(f"{path}: a bare item holds exactly one block")
        return _dump_bare(message.blocks[0], path)
    if message.role == "tool":
        raise FormatError(f"{path}: tool results are sent as function_call_output items")
    result: JSONObject = clone(message.extra) if message.origin in (None, FORMAT) else {}
    result["role"] = message.role
    blocks = message.blocks
    if (
        message.text_shorthand
        and len(blocks) == 1
        and isinstance(blocks[0], Text)
        and not blocks[0].extra
    ):
        result["content"] = blocks[0].text
    else:
        result["content"] = [
            _dump_part(block, message.role, f"{path}.content[{index}]")
            for index, block in enumerate(blocks)
        ]
    return result


def _bare(block: Block, role: Role) -> Message:
    return Message(role=role, blocks=(block,), origin=FORMAT, bare_item=True)


def _parse_message(data: JSONObject, path: str) -> Message:
    role_name = get_str(data, "role", path)
    role = _MESSAGE_ROLES.get(role_name)
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
        blocks=_parse_parts(content, f"{path}.content"),
        origin=FORMAT,
        extra=extra,
    )


def _parse_output(data: JSONObject, path: str) -> ToolResult:
    if "output" not in data:
        raise FormatError(f"{path}.output: missing")
    output = data["output"]
    content = output if isinstance(output, str) else _parse_parts(output, f"{path}.output")
    return ToolResult(
        tool_use_id=get_str(data, "call_id", path),
        content=content,
        origin=FORMAT,
        extra=without(data, "call_id", "output"),
    )


def _parse_parts(value: JSONValue, path: str) -> tuple[Block, ...]:
    items = expect_list(value, path)
    return tuple(_parse_part(item, f"{path}[{index}]") for index, item in enumerate(items))


def _parse_part(value: JSONValue, path: str) -> Block:
    data = expect_object(value, path)
    kind = data.get("type")
    match kind:
        case "input_text" | "output_text":
            return Text(
                text=get_str(data, "text", path), origin=FORMAT, extra=without(data, "text")
            )
        case "input_image":
            return Image(source=without(data, "type"), origin=FORMAT, extra={"type": kind})
        case "input_file":
            return Document(source=without(data, "type"), origin=FORMAT, extra={"type": kind})
        case _:
            return Opaque(data=data, origin=FORMAT)


def _dump_bare(block: Block, path: str) -> JSONObject:
    match block:
        case ToolUse():
            result = start_block(block, FORMAT, "function_call")
            result["call_id"] = block.id
            result["name"] = block.name
            result["arguments"] = dump_arguments(block.raw_arguments, block.input)
            return result
        case ToolResult():
            result = start_block(block, FORMAT, "function_call_output")
            result["call_id"] = block.tool_use_id
            if isinstance(block.content, str):
                result["output"] = block.content
            elif block.content is None:
                result["output"] = ""
            else:
                result["output"] = [
                    _dump_part(item, "tool", f"{path}.output[{index}]")
                    for index, item in enumerate(block.content)
                ]
            return result
        case Compaction() | Opaque():
            _check_kept_whole(block, path)
            return clone(block.data)
        case _:
            raise FormatError(
                f"{path}: a {type(block).__name__} block can't be sent as a Responses item"
            )


def _dump_part(block: Block, role: Role, path: str) -> JSONObject:
    match block:
        case Text():
            text_type = "output_text" if role == "assistant" else "input_text"
            result = start_block(block, FORMAT, text_type)
            result["text"] = block.text
            if block.origin is None and result["type"] == "output_text":
                result["annotations"] = []
            return result
        case Image() | Document():
            check_same_format(block, FORMAT, path)
            default = "input_image" if isinstance(block, Image) else "input_file"
            result = start_block(block, FORMAT, default)
            result.update(clone(block.source))
            return result
        case Opaque():
            _check_kept_whole(block, path)
            return clone(block.data)
        case _:
            raise FormatError(
                f"{path}: a {type(block).__name__} block can't be sent inside a Responses message"
            )


def _check_kept_whole(block: Compaction | Opaque, path: str) -> None:
    if block.origin is not FORMAT:
        source = block.origin.value if block.origin is not None else "artiik"
        raise FormatError(
            f"{path}: a {type(block).__name__} block from {source} can't be sent in {FORMAT.value}"
        )
