"""Behavior of the conversation model and the format converters beyond plain round trips."""

import ast
import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

from artiik import FormatError
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.formats._json import to_json
from artiik.messages import (
    Compaction,
    Format,
    JSONObject,
    Message,
    Opaque,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "artiik"


def test_message_helpers() -> None:
    message = Message(
        role="assistant",
        blocks=(
            Text(text="Checking. "),
            ToolUse(id="t1", name="lookup", input={"q": "x"}),
            Text(text="Done."),
        ),
    )
    assert message.text == "Checking. Done."
    assert [tool.id for tool in message.tool_uses] == ["t1"]
    assert message.tool_results == ()
    assert Message.from_text("user", "hi").text == "hi"


def test_messages_are_frozen_and_hashable() -> None:
    message = Message.from_text("user", "hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.role = "assistant"  # type: ignore[misc]
    assert hash(message) == hash(Message.from_text("user", "hi"))


def test_parsing_copies_the_input() -> None:
    cache_control: dict[str, str] = {}
    block: dict[str, object] = {"type": "text", "text": "hi", "cache_control": cache_control}
    messages = anthropic_messages.parse_messages([{"role": "user", "content": [block]}])
    block["text"] = "changed"
    cache_control["type"] = "ephemeral"
    assert anthropic_messages.dump_messages(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {}}]}
    ]


def test_dumping_returns_copies() -> None:
    messages = anthropic_messages.parse_messages(
        [{"role": "user", "content": [{"type": "hologram", "data": {"x": 1}}]}]
    )
    dumped = anthropic_messages.dump_messages(messages)
    dumped[0]["role"] = "assistant"
    assert anthropic_messages.dump_messages(messages)[0]["role"] == "user"
    block = messages[0].blocks[0]
    assert isinstance(block, Opaque)
    assert block.data == {"type": "hologram", "data": {"x": 1}}


def test_changing_a_block_changes_the_output() -> None:
    messages = anthropic_messages.parse_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "4,000 lines of logs",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
    )
    result = messages[0].blocks[0]
    assert isinstance(result, ToolResult)
    stub = dataclasses.replace(result, content="[cleared]")
    cleared = dataclasses.replace(messages[0], blocks=(stub,))
    assert anthropic_messages.dump_message(cleared) == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "cache_control": {"type": "ephemeral"},
                "tool_use_id": "t1",
                "content": "[cleared]",
            }
        ],
    }


@pytest.mark.parametrize(
    ("dump", "expected"),
    [
        (
            anthropic_messages.dump_message,
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ),
        (
            openai_responses.dump_item,
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ),
        (
            openai_chat.dump_message,
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ),
    ],
)
def test_blocks_artiik_creates_use_each_formats_default_shape(
    dump: Callable[[Message], JSONObject], expected: JSONObject
) -> None:
    assert dump(Message.from_text("user", "hi")) == expected


def test_assistant_text_in_responses_is_output_text() -> None:
    assert openai_responses.dump_item(Message.from_text("assistant", "ok")) == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
    }


def test_tool_arguments_are_kept_verbatim_until_they_change() -> None:
    raw = '{"city":  "Paris"}'
    item = {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": raw}
    message = openai_responses.parse_item(item)
    tool = message.blocks[0]
    assert isinstance(tool, ToolUse)
    assert tool.input == {"city": "Paris"}
    assert openai_responses.dump_item(message)["arguments"] == raw

    changed = dataclasses.replace(tool, input={"city": "Rome"})
    rebuilt = openai_responses.dump_item(dataclasses.replace(message, blocks=(changed,)))
    assert rebuilt["arguments"] == '{"city":"Rome"}'


def test_invalid_tool_arguments_are_kept_and_parse_to_none() -> None:
    message = openai_chat.parse_message(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
            ],
        }
    )
    tool = message.blocks[0]
    assert isinstance(tool, ToolUse)
    assert tool.input is None
    assert tool.raw_arguments == "{oops"
    assert openai_chat.dump_message(message)["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
    ]


def test_raw_arguments_survive_between_the_openai_formats() -> None:
    message = openai_chat.parse_message(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
            ],
        }
    )
    as_item = dataclasses.replace(message, bare_item=True)
    assert openai_responses.dump_item(as_item) == {
        "type": "function_call",
        "call_id": "c1",
        "name": "f",
        "arguments": "{oops",
    }


def test_text_converts_across_formats_without_provider_fields() -> None:
    [message] = anthropic_messages.parse_messages(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            }
        ]
    )
    assert openai_chat.dump_message(message) == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }


def test_format_bound_blocks_refuse_to_cross_formats() -> None:
    opaque = Message(
        role="assistant",
        blocks=(Opaque(data={"type": "web_search_call"}, origin=Format.OPENAI_RESPONSES),),
    )
    with pytest.raises(FormatError, match="Opaque block from openai-responses"):
        anthropic_messages.dump_message(opaque)

    thinking = Message(
        role="assistant",
        blocks=(Thinking(thinking="...", signature="sig", origin=Format.ANTHROPIC_MESSAGES),),
    )
    with pytest.raises(FormatError, match="can't be sent inside a Responses message"):
        openai_responses.dump_item(thinking)

    compaction = Message(
        role="assistant",
        blocks=(Compaction(data={"type": "compaction"}, origin=Format.OPENAI_RESPONSES),),
    )
    with pytest.raises(FormatError, match="Compaction block from openai-responses"):
        anthropic_messages.dump_message(compaction)


def test_errors_name_the_path() -> None:
    with pytest.raises(FormatError, match=r"^messages\[1\]\.role: missing"):
        anthropic_messages.parse_messages([{"role": "user", "content": "a"}, {"content": "b"}])
    with pytest.raises(FormatError, match=r"messages\[0\]\.content\[0\]\.input: missing"):
        anthropic_messages.parse_messages(
            [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n"}]}]
        )
    with pytest.raises(FormatError, match="model_dump"):
        to_json(object(), "messages[0]")
    with pytest.raises(FormatError, match=r"unsupported role 'tool'"):
        anthropic_messages.parse_message({"role": "tool", "content": "x"})


def test_compaction_summaries() -> None:
    [message] = anthropic_messages.parse_messages(
        [
            {
                "role": "assistant",
                "content": [{"type": "compaction", "content": "Summary.", "signature": "s"}],
            }
        ]
    )
    block = message.blocks[0]
    assert isinstance(block, Compaction)
    assert block.summary == "Summary."

    item = openai_responses.parse_item(
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "gAAAA"}
    )
    opaque_compaction = item.blocks[0]
    assert isinstance(opaque_compaction, Compaction)
    assert opaque_compaction.summary is None
    assert item.bare_item


def test_opaque_blocks_expose_their_type() -> None:
    assert Opaque(data={"type": "reasoning"}).type == "reasoning"
    assert Opaque(data={}).type is None


def test_the_core_does_not_import_provider_sdks() -> None:
    for path in SOURCE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert not roots & {"anthropic", "openai"}, f"{path} imports a provider SDK"
