"""Every spec fixture must survive parse → dump unchanged."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.messages import Block, Compaction, JSONObject, JSONValue, Message, Opaque, ToolResult

SPEC_DIR = Path(__file__).resolve().parents[2] / "spec" / "fixtures"
FIXTURES = sorted(SPEC_DIR.glob("*/*.json")) if SPEC_DIR.exists() else []
REQUIRED_COVERAGE = {
    "parallel-tool-calls",
    "thinking",
    "images",
    "anthropic-compaction",
    "openai-compaction",
}

pytestmark = pytest.mark.skipif(not FIXTURES, reason="the spec fixtures ship with the repository")


def _load(path: Path) -> JSONObject:
    return cast(JSONObject, json.loads(path.read_text(encoding="utf-8")))


def _entries(fixture: JSONObject, key: str) -> list[object]:
    return cast(list[object], fixture[key])


def _round_trip(fixture: JSONObject) -> tuple[list[Message], JSONValue, list[JSONObject]]:
    """Parse and dump a fixture's conversation: (messages, original, dumped)."""
    match fixture["format"]:
        case "anthropic-messages":
            messages = anthropic_messages.parse_messages(_entries(fixture, "messages"))
            return messages, fixture["messages"], anthropic_messages.dump_messages(messages)
        case "openai-responses":
            messages = openai_responses.parse_items(_entries(fixture, "input"))
            return messages, fixture["input"], openai_responses.dump_items(messages)
        case "openai-chat":
            messages = openai_chat.parse_messages(_entries(fixture, "messages"))
            return messages, fixture["messages"], openai_chat.dump_messages(messages)
        case other:
            raise AssertionError(f"unknown format {other!r}")


def _walk(blocks: tuple[Block, ...]) -> Iterator[Block]:
    for block in blocks:
        yield block
        if isinstance(block, ToolResult) and isinstance(block.content, tuple):
            yield from _walk(block.content)


def _ids(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


@pytest.mark.parametrize("path", FIXTURES, ids=_ids)
def test_fixture_round_trips(path: Path) -> None:
    _, original, dumped = _round_trip(_load(path))
    assert dumped == original


@pytest.mark.parametrize("path", FIXTURES, ids=_ids)
def test_blocks_kept_whole_are_byte_identical(path: Path) -> None:
    messages, original, dumped = _round_trip(_load(path))
    original_text = json.dumps(original, ensure_ascii=False)
    dumped_text = json.dumps(dumped, ensure_ascii=False)
    for message in messages:
        for block in _walk(message.blocks):
            if isinstance(block, Compaction | Opaque):
                serialized = json.dumps(block.data, ensure_ascii=False)
                assert serialized in original_text
                assert serialized in dumped_text


@pytest.mark.parametrize(
    "path",
    [path for path in FIXTURES if "system" in _load(path)],
    ids=_ids,
)
def test_system_round_trips(path: Path) -> None:
    fixture = _load(path)
    system = anthropic_messages.parse_system(fixture["system"])
    assert anthropic_messages.dump_system(system) == fixture["system"]


def test_corpus_covers_the_required_cases() -> None:
    fixtures = [_load(path) for path in FIXTURES]
    covered: set[str] = set()
    for fixture in fixtures:
        covered.update(cast(list[str], fixture["covers"]))
    assert len(fixtures) >= 30
    assert covered >= REQUIRED_COVERAGE
    assert {str(fixture["format"]) for fixture in fixtures} == {
        "anthropic-messages",
        "openai-responses",
        "openai-chat",
    }
