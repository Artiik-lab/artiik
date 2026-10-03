"""What a :class:`~artiik.context.Context` did, event by event."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from artiik.messages import JSONObject, JSONValue


@dataclass(frozen=True)
class TraceEvent:
    """One thing artiik did or saw, tied to the request it happened on.

    ``kind`` is ``prepare``, ``record``, ``guard``, ``clearing``, ``fetch``,
    ``compaction`` or ``pins``.
    ``request`` numbers a context's ``prepare`` calls from 0, counting any that
    raised; a ``record`` event carries the number of the request it answers,
    and an event between requests the number of the next one. ``data`` holds
    the details, as JSON.
    """

    kind: str
    request: int
    data: JSONObject


class Trace:
    """The events of one context, in order.

    ``sink``, when given, also receives each event as it happens, for example
    to write it to a file.
    """

    def __init__(self, sink: Callable[[TraceEvent], None] | None = None) -> None:
        self.events: list[TraceEvent] = []
        self._sink = sink

    def emit(self, kind: str, request: int, /, **data: JSONValue) -> TraceEvent:
        """Record an event."""
        event = TraceEvent(kind=kind, request=request, data=dict(data))
        self.events.append(event)
        if self._sink is not None:
            self._sink(event)
        return event

    def of(self, kind: str) -> list[TraceEvent]:
        """The events of one kind, in order."""
        return [event for event in self.events if event.kind == kind]
