"""The structure of a conversation: user turns, the steps inside them, and what never moves.

The guard and compaction both cut conversations, and both must cut between
whole pieces so a tool call never loses its result. A conversation splits
into units:

- an **anchor**: a compaction block or item, which stays where it is, with
  the results of any tool calls in the same message;
- an **instruction**: a system or developer message;
- a **user** unit: a user message that starts a turn;
- a **step**: a model reply together with the tool results that answer it.

A **turn** is a user unit and the units after it, up to the next user unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from artiik.messages import Compaction, Format, Message, ToolResult
from artiik.validation import is_turn_start

Kind = Literal["anchor", "instruction", "user", "step"]


@dataclass(frozen=True)
class Unit:
    """A piece of a conversation that is kept or dropped whole: its kind and message positions."""

    kind: Kind
    indices: tuple[int, ...]


def segment(api: Format, history: Sequence[Message]) -> list[Unit]:
    """Split a conversation into units, in order."""
    units: list[Unit] = []
    index = 0
    while index < len(history):
        message = history[index]
        if is_anchor(api, message):
            end = index + 1
            if message.tool_uses:
                # A reply that compacted at a threshold can call tools after its block.
                while end < len(history) and is_answer(api, history[end]):
                    end += 1
            units.append(Unit("anchor", tuple(range(index, end))))
            index = end
        elif not message.bare_item and message.role in ("system", "developer"):
            units.append(Unit("instruction", (index,)))
            index += 1
        elif is_turn_start(message):
            units.append(Unit("user", (index,)))
            index += 1
        else:
            end = index
            while end < len(history) and is_call_side(history[end]):
                end += 1
            while end < len(history) and is_answer(api, history[end]):
                end += 1
            end = max(end, index + 1)
            units.append(Unit("step", tuple(range(index, end))))
            index = end
    return units


def group(units: Sequence[Unit]) -> list[list[Unit]]:
    """Group units into turns. Units before the first user unit form a turn of their own."""
    turns: list[list[Unit]] = []
    for unit in units:
        if unit.kind == "anchor":
            continue
        if unit.kind == "user" or not turns:
            turns.append([])
        turns[-1].append(unit)
    return turns


def is_anchor(api: Format, message: Message) -> bool:
    """Whether a message holds the compaction block or item the model reads from."""
    if not message.blocks:
        return False
    if api is Format.ANTHROPIC_MESSAGES:
        return isinstance(message.blocks[0], Compaction)
    return any(isinstance(block, Compaction) for block in message.blocks)


def is_call_side(message: Message) -> bool:
    """Whether a message is part of a model reply: what a step starts with."""
    if message.role == "assistant":
        return not any(isinstance(block, ToolResult) for block in message.blocks)
    return False


def is_answer(api: Format, message: Message) -> bool:
    """Whether a message carries tool results: what a step ends with."""
    if api is Format.ANTHROPIC_MESSAGES:
        return message.role == "user" and bool(message.tool_results)
    return message.role == "tool"
