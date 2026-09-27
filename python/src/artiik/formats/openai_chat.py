"""OpenAI Chat Completions format: the ``messages`` of a request and ``choices[].message``.

Also used by OpenAI-compatible servers (local models, gateways). An assistant
message's ``tool_calls`` become :class:`~artiik.messages.ToolUse` blocks after
its content blocks; a ``tool`` message becomes a message holding one
:class:`~artiik.messages.ToolResult`. Legacy function-calling messages (the
``function`` role and ``function_call``) are kept whole as bare items.

Reference: https://platform.openai.com/docs/api-reference/chat/create
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
    get_object,
    get_str,
    parse_arguments,
    start_block,
    to_object,
    without,
)
from artiik.messages import (
    Block,
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

FORMAT = Format.OPENAI_CHAT

_ROLES: dict[str, Role] = {
    "system": "system",
    "developer": "developer",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


def parse_messages(messages: Iterable[object]) -> list[Message]:
    """Parse the ``messages`` parameter of a request."""
    return [parse_message(message, f"messages[{index}]") for index, message in enumerate(messages)]


def parse_message(message: object, path: str = "message") -> Message:
    """Parse one message, including ``choices[].message`` from a response."""
    data = to_object(message, path)
    role_name = get_str(data, "role", path)
    if role_name == "function" or "function_call" in data:
        role: Role = "tool" if role_name == "function" else _role(role_name, path)
        return Message(
            role=role,
            blocks=(Opaque(data=data, origin=FORMAT),),
            origin=FORMAT,
            bare_item=True,
        )
    role = _role(role_name, path)
    if role == "tool":
        return _parse_tool_message(data, path)
    modeled = ["role"]
    blocks: list[Block] = []
    content = data.get("content")
    if isinstance(content, str):
        blocks.append(Text(text=content, origin=FORMAT))
        modeled.append("content")
    elif isinstance(content, list):
        blocks.extend(_parse_parts(content, f"{path}.content"))
        modeled.append("content")
    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        blocks.extend(
            _parse_tool_call(call, f"{path}.tool_calls[{index}]")
            for index, call in enumerate(tool_calls)
        )
        modeled.append("tool_calls")
    return Message(
        role=role,
        blocks=tuple(blocks),
        origin=FORMAT,
        extra=without(data, *modeled),
        text_shorthand=isinstance(content, str),
    )


def dump_messages(messages: Iterable[Message]) -> list[JSONObject]:
    """Build the ``messages`` parameter of a request."""
    return [dump_message(message, f"messages[{index}]") for index, message in enumerate(messages)]


def dump_message(message: Message, path: str = "message") -> JSONObject:
    """Build one message."""
    if message.bare_item:
        block = message.blocks[0] if len(message.blocks) == 1 else None
        if not isinstance(block, Opaque) or block.origin is not FORMAT:
            raise FormatError(f"{path}: this bare item can't be sent in {FORMAT.value}")
        return clone(block.data)
    result: JSONObject = {"role": message.role}
    if message.origin in (None, FORMAT):
        result.update(clone(message.extra))
    if message.role == "tool":
        return _dump_tool_message(message, result, path)
    content_blocks = [block for block in message.blocks if not isinstance(block, ToolUse)]
    tool_uses = [block for block in message.blocks if isinstance(block, ToolUse)]
    if content_blocks:
        first = content_blocks[0]
        if (
            message.text_shorthand
            and len(content_blocks) == 1
            and isinstance(first, Text)
            and not first.extra
        ):
            result["content"] = first.text
        else:
            result["content"] = [
                _dump_part(block, f"{path}.content[{index}]")
                for index, block in enumerate(content_blocks)
            ]
    if tool_uses:
        result["tool_calls"] = [
            _dump_tool_call(block, f"{path}.tool_calls[{index}]")
            for index, block in enumerate(tool_uses)
        ]
    return result


def _role(name: str, path: str) -> Role:
    role = _ROLES.get(name)
    if role is None:
        raise FormatError(f"{path}.role: unsupported role {name!r}")
    return role


def _parse_tool_message(data: JSONObject, path: str) -> Message:
    if "content" not in data:
        raise FormatError(f"{path}.content: missing")
    raw_content = data["content"]
    content: str | tuple[Block, ...]
    if isinstance(raw_content, str):
        content = raw_content
    else:
        content = _parse_parts(raw_content, f"{path}.content")
    result = ToolResult(
        tool_use_id=get_str(data, "tool_call_id", path),
        content=content,
        origin=FORMAT,
    )
    return Message(
        role="tool",
        blocks=(result,),
        origin=FORMAT,
        extra=without(data, "role", "tool_call_id", "content"),
    )


def _dump_tool_message(message: Message, result: JSONObject, path: str) -> JSONObject:
    if len(message.blocks) != 1 or not isinstance(message.blocks[0], ToolResult):
        raise FormatError(f"{path}: a tool message holds exactly one ToolResult block")
    tool_result = message.blocks[0]
    result["tool_call_id"] = tool_result.tool_use_id
    if isinstance(tool_result.content, str):
        result["content"] = tool_result.content
    elif tool_result.content is None:
        result["content"] = ""
    else:
        result["content"] = [
            _dump_part(block, f"{path}.content[{index}]")
            for index, block in enumerate(tool_result.content)
        ]
    return result


def _parse_tool_call(value: JSONValue, path: str) -> ToolUse:
    data = expect_object(value, path)
    function = get_object(data, "function", path)
    arguments = get_str(function, "arguments", f"{path}.function")
    extra = without(data, "id", "function")
    function_extra = without(function, "name", "arguments")
    if function_extra:
        extra["function"] = function_extra
    return ToolUse(
        id=get_str(data, "id", path),
        name=get_str(function, "name", f"{path}.function"),
        input=parse_arguments(arguments),
        raw_arguments=arguments,
        origin=FORMAT,
        extra=extra,
    )


def _dump_tool_call(block: ToolUse, path: str) -> JSONObject:
    extra = clone(block.extra) if block.origin in (None, FORMAT) else {}
    function_extra = extra.pop("function", None)
    result: JSONObject = {"id": block.id}
    result.update(extra)
    result.setdefault("type", "function")
    function: JSONObject = {}
    if isinstance(function_extra, dict):
        function.update(function_extra)
    function["name"] = block.name
    function["arguments"] = dump_arguments(block.raw_arguments, block.input)
    result["function"] = function
    return result


def _parse_parts(value: JSONValue, path: str) -> tuple[Block, ...]:
    items = expect_list(value, path)
    return tuple(_parse_part(item, f"{path}[{index}]") for index, item in enumerate(items))


def _parse_part(value: JSONValue, path: str) -> Block:
    data = expect_object(value, path)
    match data.get("type"):
        case "text":
            return Text(
                text=get_str(data, "text", path), origin=FORMAT, extra=without(data, "text")
            )
        case "image_url":
            return Image(
                source=get_object(data, "image_url", path),
                origin=FORMAT,
                extra=without(data, "image_url"),
            )
        case "file":
            return Document(
                source=get_object(data, "file", path),
                origin=FORMAT,
                extra=without(data, "file"),
            )
        case _:
            return Opaque(data=data, origin=FORMAT)


def _dump_part(block: Block, path: str) -> JSONObject:
    match block:
        case Text():
            result = start_block(block, FORMAT, "text")
            result["text"] = block.text
            return result
        case Image():
            check_same_format(block, FORMAT, path)
            result = start_block(block, FORMAT, "image_url")
            result["image_url"] = clone(block.source)
            return result
        case Document():
            check_same_format(block, FORMAT, path)
            result = start_block(block, FORMAT, "file")
            result["file"] = clone(block.source)
            return result
        case Opaque():
            if block.origin is not FORMAT:
                source = block.origin.value if block.origin is not None else "artiik"
                raise FormatError(f"{path}: an Opaque block from {source} can't be sent here")
            return clone(block.data)
        case _:
            raise FormatError(
                f"{path}: a {type(block).__name__} block can't be sent as a Chat content part"
            )
