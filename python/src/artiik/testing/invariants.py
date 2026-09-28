"""Tier 0 invariants: properties every request artiik sends must have.

Each checker reads the calls a fake client recorded and returns the violations
it finds; an empty list means the invariant holds. The invariants:

- **budget:** no request (other than a compaction request) exceeds the budget;
- **tool pairs:** every tool call in a request has its result, and every result
  answers a call;
- **append-only:** between compactions, a request only appends to the previous
  one; the cache-keying parameters and the earlier entries don't change, so
  the prompt cache keeps working;
- **pins:** every pinned text is present in every request from the moment it's
  pinned, outside compaction summaries;
- **kept whole:** blocks and items that artiik can't rebuild (compaction,
  thinking, reasoning, and any type artiik doesn't model) go back byte for
  byte as the provider returned them;
- **scopes:** memory lookups never return memories from another scope.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from artiik.messages import Compaction, Format, JSONObject, JSONValue, Message, ToolResult, ToolUse
from artiik.testing.recording import RecordedCall, entries, last_compaction, parse_system

_SIGNED: dict[Format, frozenset[str]] = {
    Format.ANTHROPIC_MESSAGES: frozenset({"compaction", "thinking", "redacted_thinking"}),
    Format.OPENAI_RESPONSES: frozenset({"compaction", "reasoning"}),
}
"""Types the provider signs or encrypts: a changed copy is always an error."""

_REBUILT: dict[Format, frozenset[str]] = {
    Format.ANTHROPIC_MESSAGES: frozenset({"text", "image", "document", "tool_use", "tool_result"}),
    Format.OPENAI_RESPONSES: frozenset({"message", "function_call", "function_call_output"}),
}
"""Types artiik models and rebuilds. The provider's other types are kept whole."""

_CACHE_KEYS = ("model", "system", "instructions", "tools", "tool_choice", "thinking")
"""Request parameters whose change invalidates the prompt cache."""


@dataclass(frozen=True)
class Violation:
    """A broken invariant: which one, on which recorded call, and what went wrong."""

    invariant: str
    call: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.invariant}] call {self.call}: {self.detail}"


@dataclass(frozen=True)
class Recall:
    """One memory lookup: the scope asked for, and the scopes of the memories it returned."""

    call: int
    scope: Mapping[str, str]
    returned: Sequence[Mapping[str, str]]


def check_budget(calls: Iterable[RecordedCall], budget: int) -> list[Violation]:
    """No request other than a compaction request may exceed ``budget`` tokens."""
    violations: list[Violation] = []
    for call in calls:
        if call.is_compaction:
            continue
        tokens = call.tokens()
        if tokens > budget:
            violations.append(Violation("budget", call.index, f"{tokens} tokens > budget {budget}"))
    return violations


def check_tool_pairs(calls: Iterable[RecordedCall]) -> list[Violation]:
    """Every tool call in a request has its result, and every result answers an earlier call."""
    violations: list[Violation] = []
    for call in calls:
        seen: set[str] = set()
        answered: set[str] = set()
        for message in call.conversation():
            for block in message.blocks:
                if isinstance(block, ToolUse):
                    seen.add(block.id)
                elif isinstance(block, ToolResult):
                    if block.tool_use_id not in seen:
                        violations.append(
                            Violation(
                                "tool-pairs",
                                call.index,
                                f"result for {block.tool_use_id} answers no earlier call",
                            )
                        )
                    answered.add(block.tool_use_id)
        for missing in sorted(seen - answered):
            violations.append(Violation("tool-pairs", call.index, f"call {missing} has no result"))
    return violations


def check_append_only(
    calls: Iterable[RecordedCall], *, allowed: Collection[int] = ()
) -> list[Violation]:
    """Between compactions, each successful request only appends to the previous one.

    Nothing already sent changes: not the parameters the prompt cache depends
    on (model, system prompt, tools, tool choice, thinking) and not the earlier
    entries, so every cache breakpoint of the previous request still hits.

    A request that brings a new compaction block or item may rewrite the
    history. ``allowed`` lists other calls that may, such as the calls that
    follow a tool-output clearing batch. ``cache_control`` markers are ignored:
    moving a cache breakpoint doesn't change the prompt.
    """
    violations: list[Violation] = []
    previous: RecordedCall | None = None
    for call in calls:
        if not call.ok or call.is_compaction:
            continue
        if previous is not None and call.index not in allowed and not _compacted(previous, call):
            detail = _first_rewrite(previous, call)
            if detail is not None:
                violations.append(Violation("append-only", call.index, detail))
        previous = call
    return violations


def check_pins(calls: Iterable[RecordedCall], pins: Mapping[str, int]) -> list[Violation]:
    """Each pinned text appears in every request from call ``pins[text]`` on.

    Only the text the model reads directly counts: the system prompt and text
    blocks, not tool results or compaction summaries.
    """
    violations: list[Violation] = []
    for call in calls:
        if not call.ok or call.is_compaction:
            continue
        text = _visible_text(call)
        for pin, first_call in pins.items():
            if call.index >= first_call and pin not in text:
                violations.append(Violation("pins", call.index, f"pin {pin!r} is missing"))
    return violations


def check_kept_whole(calls: Iterable[RecordedCall]) -> list[Violation]:
    """Blocks artiik can't rebuild go back exactly as the provider returned them.

    That covers the types the provider signs or encrypts (compaction, thinking,
    reasoning), and any type the provider returned that artiik doesn't model,
    such as server tool blocks. Only ``cache_control`` may be added. Chat
    Completions has no such blocks.
    """
    violations: list[Violation] = []
    returned: set[str] = set()
    kinds: dict[Format, set[str]] = {fmt: set(signed) for fmt, signed in _SIGNED.items()}
    for call in calls:
        if call.format not in kinds:
            continue
        for position, entry in _entries_of_kinds(call, kinds[call.format]):
            if _canonical(entry) not in returned:
                violations.append(
                    Violation(
                        "kept-whole",
                        call.index,
                        f"{entry.get('type')} at {position} isn't one the provider returned: "
                        "it was changed or made up",
                    )
                )
        for entry in _returned_entries(call):
            kind = entry.get("type")
            if isinstance(kind, str) and kind not in _REBUILT[call.format]:
                kinds[call.format].add(kind)
                returned.add(_canonical(entry))
    return violations


def check_scopes(recalls: Iterable[Recall]) -> list[Violation]:
    """No lookup returns a memory outside the scope it asked for.

    A memory is visible to a lookup when every key of the memory's scope has
    the same value in the lookup's scope. A memory with an empty scope is
    visible everywhere.
    """
    violations: list[Violation] = []
    for recall in recalls:
        for memory in recall.returned:
            if any(recall.scope.get(key) != value for key, value in memory.items()):
                violations.append(
                    Violation(
                        "scopes",
                        recall.call,
                        f"memory scoped {dict(memory)} returned for scope {dict(recall.scope)}",
                    )
                )
    return violations


def check_all(
    calls: Sequence[RecordedCall],
    *,
    budget: int | None = None,
    pins: Mapping[str, int] | None = None,
    allowed_rewrites: Collection[int] = (),
    recalls: Iterable[Recall] = (),
) -> list[Violation]:
    """Run every checker that applies."""
    violations = check_tool_pairs(calls)
    violations += check_append_only(calls, allowed=allowed_rewrites)
    violations += check_kept_whole(calls)
    violations += check_scopes(recalls)
    if budget is not None:
        violations += check_budget(calls, budget)
    if pins is not None:
        violations += check_pins(calls, pins)
    return violations


def assert_holds(violations: Sequence[Violation]) -> None:
    """Fail with a readable list when there are violations."""
    if violations:
        raise AssertionError(
            f"{len(violations)} invariant violation(s):\n"
            + "\n".join(f"  {violation}" for violation in violations)
        )


def _strip(value: JSONValue) -> JSONValue:
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _canonical(value: JSONValue) -> str:
    return json.dumps(_strip(value), sort_keys=True, ensure_ascii=False)


def _compactions(call: RecordedCall) -> set[str]:
    found: set[str] = set()
    for message in call.conversation():
        for block in message.blocks:
            if isinstance(block, Compaction):
                found.add(_canonical(block.data))
    return found


def _compacted(previous: RecordedCall, current: RecordedCall) -> bool:
    return bool(_compactions(current) - _compactions(previous))


def _first_rewrite(previous: RecordedCall, current: RecordedCall) -> str | None:
    for key in _CACHE_KEYS:
        if _strip(previous.request.get(key)) != _strip(current.request.get(key)):
            return f"the {key} parameter changed"
    before = [_strip(entry) for entry in entries(previous.format, previous.request)]
    after = [_strip(entry) for entry in entries(current.format, current.request)]
    if len(after) < len(before):
        return f"the history shrank from {len(before)} to {len(after)} entries"
    for position, (old, new) in enumerate(zip(before, after, strict=False)):
        if old != new:
            return f"entry {position} was rewritten"
    return None


def _visible_text(call: RecordedCall) -> str:
    conversation = call.conversation()
    visible = conversation[last_compaction(conversation) :]
    messages: list[Message] = parse_system(call.format, call.request) + visible
    return "\n".join(message.text for message in messages)


def _entries_of_kinds(call: RecordedCall, kinds: Collection[str]) -> list[tuple[str, JSONObject]]:
    """The request's blocks (or Responses items) whose type is in ``kinds``, with their paths."""
    found: list[tuple[str, JSONObject]] = []
    for position, entry in enumerate(entries(call.format, call.request)):
        if not isinstance(entry, dict):
            continue
        if call.format is Format.OPENAI_RESPONSES:
            if entry.get("type") in kinds:
                found.append((f"input[{position}]", entry))
            continue
        content = entry.get("content")
        if isinstance(content, list):
            for index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") in kinds:
                    found.append((f"messages[{position}].content[{index}]", block))
    return found


def _returned_entries(call: RecordedCall) -> list[JSONObject]:
    """The blocks (or Responses items) of the call's response."""
    response = call.response or {}
    key = "output" if call.format is Format.OPENAI_RESPONSES else "content"
    raw = response.get(key)
    items = cast(list[JSONValue], raw) if isinstance(raw, list) else []
    return [item for item in items if isinstance(item, dict)]
