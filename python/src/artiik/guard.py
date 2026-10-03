"""The budget guard, the last resort that keeps a request under its token budget.

When a request would go over its budget, the guard drops the oldest parts of
the conversation until it fits a lower target, so it doesn't fire again on
the next call: whole user turns first, then the oldest tool steps of the
current turn. A step is a model reply together with the tool results that
answer it, so a tool call never loses its result. System and developer
messages are kept; when their turn goes, they move to just after the first
user turn that remains and is followed by a reply (or to the end), where
Anthropic accepts a system message. A compaction block or item stays where it
is.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from artiik.errors import BudgetError
from artiik.messages import Format, Message
from artiik.tokens import Tally
from artiik.turns import Unit, group, segment


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


def trim(
    api: Format,
    history: Sequence[Message],
    tallies: Sequence[Tally],
    *,
    size: Callable[[Tally], int],
    target: int,
    limit: int,
    instruction: Callable[[Message], bool] | None = None,
) -> Trim:
    """Drop the oldest turns, then the oldest steps of the current turn, until the request fits.

    ``size`` turns the tally of the history into the estimated size of the
    whole request. The guard stops dropping once that size is at most
    ``target``, and raises :class:`~artiik.errors.BudgetError` if it can't get
    it to ``limit`` or below. ``instruction`` marks more messages to keep as
    instructions, such as pin messages sent with the user role.
    """
    units = segment(api, history, instruction)
    turns = group(units)
    current = sum(tallies, Tally())
    dropped: set[int] = set()
    moved: list[int] = []
    dropped_turns = dropped_steps = 0

    def drop(unit: Unit) -> None:
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
    order = _order(units, turns, dropped, moved)
    return Trim(
        messages=tuple(history[index] for index in order),
        tallies=tuple(tallies[index] for index in order),
        size=final,
        dropped_turns=dropped_turns,
        dropped_steps=dropped_steps,
        dropped_messages=len(dropped),
        moved_messages=len(moved),
    )


def _order(
    units: Sequence[Unit],
    turns: Sequence[Sequence[Unit]],
    dropped: set[int],
    moved: Sequence[int],
) -> list[int]:
    """The positions of the messages that stay, in order, with the moved instructions placed.

    The moved instructions go after the first user turn followed by something
    other than another user turn, or at the end.
    """
    gone = {
        unit.indices[0]
        for turn in turns
        if (droppable := _droppable(turn)) and all(unit.indices[0] in dropped for unit in droppable)
        for unit in turn
    }
    kept = [unit for unit in units if not _left_out(unit, gone, dropped)]
    slot = len(kept)
    for position, unit in enumerate(kept):
        following = kept[position + 1] if position + 1 < len(kept) else None
        if unit.kind == "user" and (following is None or following.kind != "user"):
            slot = position + 1
            break
    before = [index for unit in kept[:slot] for index in unit.indices]
    after = [index for unit in kept[slot:] for index in unit.indices]
    return [*before, *moved, *after]


def _left_out(unit: Unit, gone: set[int], dropped: set[int]) -> bool:
    first = unit.indices[0]
    return first in gone or (unit.kind in ("user", "step") and first in dropped)


def _droppable(turn: Sequence[Unit]) -> list[Unit]:
    return [unit for unit in turn if unit.kind != "instruction"]
