"""The budget guard, the last resort that keeps a request under its token budget.

When a request would go over its budget, the guard drops the oldest parts of
the conversation until it fits a lower target, so it doesn't fire again on
the next call: whole user turns first, then the oldest tool steps of the
current turn. A step is a model reply together with the tool results that
answer it, so a tool call never loses its result. System and developer
messages are kept; when their turn goes, they move to just after the first
user turn that remains. A compaction block or item stays where it is.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from artiik.errors import BudgetError
from artiik.messages import Compaction, Format, Message, ToolResult
from artiik.tokens import Tally
from artiik.validation import is_turn_start, visible_start

_Kind = Literal["anchor", "instruction", "user", "step"]


@dataclass(frozen=True)
class Trim:
    """The history after the guard, with what it dropped."""

    messages: tuple[Message, ...]
    tallies: tuple[Tally, ...]
    size: int
    """The estimated size of the request after trimming."""

    dropped_turns: int
    dropped_steps: int
    dropped_messages: int
    moved_messages: int
    """System and developer messages moved because their turn was dropped."""


@dataclass(frozen=True)
class _Unit:
    kind: _Kind
    indices: tuple[int, ...]


def trim(
    api: Format,
    history: Sequence[Message],
    tallies: Sequence[Tally],
    *,
    size: Callable[[Tally], int],
    target: int,
    limit: int,
) -> Trim:
    """Drop the oldest turns, then the oldest steps of the current turn, until the request fits.

    ``size`` turns the tally of the visible history into the estimated size of
    the whole request. The guard stops dropping once that size is at most
    ``target``, and raises :class:`~artiik.errors.BudgetError` if it can't get
    it to ``limit`` or below.
    """
    start = visible_start(api, history)
    units = _units(api, history, start)
    turns = _turns(units)
    current = sum(tallies[start:], Tally())
    dropped: set[int] = set()
    moved: list[int] = []
    dropped_turns = dropped_steps = 0

    def drop(unit: _Unit) -> None:
        nonlocal current
        for index in unit.indices:
            dropped.add(index)
            current = current - tallies[index]

    for turn in turns[:-1]:
        if size(current) <= target:
            break
        if not _droppable(turn):
            continue
        for unit in turn:
            if unit.kind == "instruction":
                moved.extend(unit.indices)
            else:
                drop(unit)
        dropped_turns += 1
    if turns and size(current) > target:
        steps = [unit for unit in turns[-1] if unit.kind == "step"]
        for unit in steps[:-1]:
            if size(current) <= target:
                break
            drop(unit)
            dropped_steps += 1
    final = size(current)
    if final > limit:
        raise BudgetError(final, limit)
    order = _order(start, units, turns, dropped, moved)
    return Trim(
        messages=tuple(history[index] for index in order),
        tallies=tuple(tallies[index] for index in order),
        size=final,
        dropped_turns=dropped_turns,
        dropped_steps=dropped_steps,
        dropped_messages=len(dropped),
        moved_messages=len(moved),
    )


def _units(api: Format, history: Sequence[Message], start: int) -> list[_Unit]:
    units: list[_Unit] = []
    index = start
    while index < len(history):
        message = history[index]
        if index == start and _is_anchor(api, message):
            units.append(_Unit("anchor", (index,)))
            index += 1
        elif not message.bare_item and message.role in ("system", "developer"):
            units.append(_Unit("instruction", (index,)))
            index += 1
        elif is_turn_start(message):
            units.append(_Unit("user", (index,)))
            index += 1
        else:
            end = index
            while end < len(history) and _is_call_side(history[end]):
                end += 1
            while end < len(history) and _is_answer(api, history[end]):
                end += 1
            end = max(end, index + 1)
            units.append(_Unit("step", tuple(range(index, end))))
            index = end
    return units


def _turns(units: Sequence[_Unit]) -> list[list[_Unit]]:
    turns: list[list[_Unit]] = []
    for unit in units:
        if unit.kind == "anchor":
            continue
        if unit.kind == "user" or not turns:
            turns.append([])
        turns[-1].append(unit)
    return turns


def _order(
    start: int,
    units: Sequence[_Unit],
    turns: Sequence[Sequence[_Unit]],
    dropped: set[int],
    moved: Sequence[int],
) -> list[int]:
    order = list(range(start))
    for unit in units:
        if unit.kind == "anchor":
            order.extend(unit.indices)
    placed = not moved
    for turn in turns:
        droppable = _droppable(turn)
        if droppable and all(unit.indices[0] in dropped for unit in droppable):
            continue
        for unit in turn:
            if unit.kind != "instruction" and unit.indices[0] in dropped:
                continue
            order.extend(unit.indices)
            if not placed and unit.kind == "user":
                order.extend(moved)
                placed = True
    if not placed:
        order.extend(moved)
    return order


def _droppable(turn: Sequence[_Unit]) -> list[_Unit]:
    return [unit for unit in turn if unit.kind != "instruction"]


def _is_anchor(api: Format, message: Message) -> bool:
    if not message.blocks:
        return False
    if api is Format.ANTHROPIC_MESSAGES:
        return isinstance(message.blocks[0], Compaction)
    return any(isinstance(block, Compaction) for block in message.blocks)


def _is_call_side(message: Message) -> bool:
    if message.role == "assistant":
        return not any(isinstance(block, ToolResult) for block in message.blocks)
    return False


def _is_answer(api: Format, message: Message) -> bool:
    if api is Format.ANTHROPIC_MESSAGES:
        return message.role == "user" and bool(message.tool_results)
    return message.role == "tool"
