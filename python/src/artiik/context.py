"""The managed conversation of one agent: ``prepare`` before each provider call, ``record`` after.

This is integration level 2, the documented default. The context owns the
history: messages go in with :meth:`Context.add`, :meth:`Context.prepare`
returns the provider call's keyword arguments, and :meth:`Context.record`
takes the response::

    import anthropic
    import artiik

    client = anthropic.Anthropic()
    ctx = artiik.Context(
        "anthropic-messages",
        model=MODEL,
        budget=150_000,
        system=SYSTEM,
        tools=TOOLS,
        params={"max_tokens": 4096},
    )
    ctx.add({"role": "user", "content": task})
    while True:
        response = client.messages.create(**ctx.prepare())
        ctx.record(response)
        calls = ctx.pending_tool_calls()
        if not calls:
            break
        # run_tool returns a tool_result block for the call.
        ctx.add({"role": "user", "content": [run_tool(call) for call in calls]})

Messages go in and out in the provider's own format, so the rest of the loop
doesn't change.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from artiik.compaction import (
    ANTHROPIC_BETA,
    ANTHROPIC_THRESHOLD_BETA,
    CompactionJob,
    CompactionResult,
    Compactor,
    add_beta,
)
from artiik.errors import BudgetError
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.formats._json import expect_list, expect_object, plain, to_json, to_object
from artiik.guard import Trim, trim
from artiik.messages import (
    Compaction,
    Format,
    JSONObject,
    JSONValue,
    Message,
    Opaque,
    Role,
    ToolUse,
    last_compaction,
)
from artiik.pins import (
    PIN_BUDGET,
    USER_LABEL,
    Ledger,
    Pin,
    PinKind,
    PinSource,
    Restatement,
    approx_tokens,
    check_pin,
    kept_in,
    render_pins,
    restatement,
)
from artiik.tokens import (
    Estimator,
    Tally,
    TokenCounter,
    tally_blocks,
    tally_json,
    tally_message,
)
from artiik.trace import Trace
from artiik.turns import Unit, group, segment
from artiik.usage import Usage, read_usage
from artiik.validation import problems, validate

logger = logging.getLogger("artiik")

SAFETY = 1.05
"""Estimated tokens are scaled up by this factor before they're checked against the budget."""

COUNT_MARGIN = 0.1
"""With an exact counter, a request estimated within this fraction of the budget is counted."""

_HISTORY_KEY: Mapping[Format, str] = {
    Format.ANTHROPIC_MESSAGES: "messages",
    Format.OPENAI_RESPONSES: "input",
    Format.OPENAI_CHAT: "messages",
}
_SYSTEM_KEY: Mapping[Format, str] = {
    Format.ANTHROPIC_MESSAGES: "system",
    Format.OPENAI_RESPONSES: "instructions",
    Format.OPENAI_CHAT: "system",
}
COMPACT_AT = 0.75
"""Without ``compact_at``, a context with a budget compacts at this fraction of it."""

MAX_BREAKPOINTS = 4
"""The most cache breakpoints an Anthropic request may carry."""

_GUARD_ROUNDS = 3
_COMPACTION_PAUSE = 3
"""Requests to wait, after a compaction that didn't work, before trying again."""

_PIN_ROLES: Mapping[Format, tuple[Role, ...]] = {
    Format.ANTHROPIC_MESSAGES: ("system", "user"),
    Format.OPENAI_RESPONSES: ("developer", "system", "user"),
    Format.OPENAI_CHAT: ("developer", "system", "user"),
}
"""The roles pin messages can take in each format; the first is the default."""


@dataclass(frozen=True)
class _Outgoing:
    """Messages artiik adds at the end of the next request, and what they deliver.

    They join the history when the reply is recorded, so a request that fails
    leaves nothing behind in the wrong place.
    """

    messages: tuple[Message, ...] = ()
    own: tuple[Message, ...] = ()
    """The pin messages among them."""
    pins: tuple[str, ...] = ()
    """The new pins they deliver."""
    retractions: tuple[str, ...] = ()
    """The pins they say no longer apply."""
    carried: int = 0
    """How many instructions from a summarized part they send again."""
    restated: bool = False
    """Whether they restate everything after a compaction."""
    restatement: Restatement | None = None


@dataclass(frozen=True)
class _Restate:
    """A compaction happened: restate the pins and the ledger after the next user turn."""

    summary: str | None
    """The summary's text, to check the pins against, when it's readable."""


@dataclass(frozen=True)
class _Sent:
    """The request the context prepared last, to learn from its usage."""

    request: int
    entries: int
    fixed: Tally
    total: Tally
    model: str
    outgoing: _Outgoing


@dataclass(frozen=True)
class _Base:
    """A provider count of the conversation so far, to build the next estimate on.

    ``extra`` is the part of the count made of messages that weren't in the
    history yet, such as a pending pin message.
    """

    entries: int
    tokens: int
    fixed: Tally
    model: str
    extra: Tally = dataclasses.field(default_factory=Tally)


class Context:
    """The conversation artiik manages for one agent.

    - ``api`` is the provider format: ``anthropic-messages``,
      ``openai-responses`` or ``openai-chat``.
    - ``model``, ``system`` and ``tools`` go into every request; ``params``
      holds other request parameters, such as ``max_tokens``. The system prompt
      becomes ``instructions`` for Responses, and a leading system message for
      Chat Completions.
    - ``budget`` caps the prompt of every request, in tokens. When a request
      would go over, the guard drops the oldest turns until it fits
      ``trim_to`` of the budget, and logs what it did.
    - ``compaction`` summarizes the older history when a request would pass
      ``compact_at`` tokens, three quarters of the budget by default. See
      :mod:`artiik.compaction` for the strategies.
    - Pins (:meth:`pin`) and the ``ledger`` of the files and IDs the tools
      used are restated after every compaction, in ``pin_role`` messages: a
      system message for Anthropic and a developer message for OpenAI by
      default, or ``"user"`` for models without them. ``pin_budget`` caps
      their size, in tokens. See :mod:`artiik.pins`.
    - ``estimator`` sizes requests before they're sent; ``counter`` makes the
      sizes exact near the budget, at the cost of a count per request.
    - ``trace`` records what the context does.
    """

    def __init__(
        self,
        api: Format | str,
        *,
        model: str | None = None,
        budget: int | None = None,
        system: object = None,
        tools: Sequence[object] | None = None,
        params: Mapping[str, Any] | None = None,
        estimator: Estimator | None = None,
        counter: TokenCounter | None = None,
        trim_to: float = 0.8,
        compaction: Compactor | None = None,
        compact_at: int | None = None,
        pin_role: Literal["system", "developer", "user"] | None = None,
        pin_budget: int = PIN_BUDGET,
        ledger: Ledger | None = None,
        trace: Trace | None = None,
    ) -> None:
        self.api = _api(api)
        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be a positive number of tokens, got {budget}")
        if not 0 < trim_to <= 1:
            raise ValueError(f"trim_to must be above 0 and at most 1, got {trim_to}")
        _check_params(self.api, params or {})
        if pin_budget <= 0:
            raise ValueError(f"pin_budget must be a positive number of tokens, got {pin_budget}")
        self.compaction = compaction
        self.compact_at = _compact_at(self.api, compaction, compact_at, budget)
        self.pin_role: Role = _pin_role(self.api, pin_role)
        self.pin_budget = pin_budget
        self.ledger = ledger if ledger is not None else Ledger()
        self.model = model
        self.budget = budget
        self.system = _system(self.api, system)
        self.tools = _tools(tools)
        self.params: dict[str, Any] = dict(params or {})
        self.estimator = estimator if estimator is not None else Estimator()
        self.counter = counter
        self.trim_to = trim_to
        self.trace = trace if trace is not None else Trace()
        self.last_usage: Usage | None = None
        self._history: list[Message] = []
        self._tallies: list[Tally] = []
        self._requests = 0
        self._sent: _Sent | None = None
        self._base: _Base | None = None
        self._compaction_paused_until: int | None = None
        self._pins: dict[str, Pin] = {}
        self._pin_count = 0
        self._startup: tuple[Pin, ...] | None = None
        """The pins in the system prompt, fixed at the first request."""
        self._new: list[Pin] = []
        self._retracting: list[Pin] = []
        self._retired: list[Pin] = []
        """Unpinned pins that stay in the system prompt."""
        self._restate: _Restate | None = None
        self._carried: list[Message] = []
        self._own: dict[int, Message] = {}
        """The pin messages in the history, by identity."""
        self._outgoing = _Outgoing()

    @property
    def history(self) -> tuple[Message, ...]:
        """The managed conversation, in the neutral model.

        Messages artiik adds at the end of a request, such as pin messages,
        join it when the reply is recorded.
        """
        return tuple(self._history)

    def add(self, *entries: object) -> None:
        """Append messages to the conversation.

        Each entry is a message in the provider's format (a dict or an SDK
        object), a list of them, or a :class:`~artiik.messages.Message`.
        Responses items such as ``function_call_output`` count as messages.
        """
        for entry in entries:
            if isinstance(entry, Message):
                self._append(entry)
            elif isinstance(entry, list | tuple):
                self.add(*cast("Sequence[object]", entry))
            else:
                self._append(_parse_entry(self.api, plain(entry), f"entry {len(self._history)}"))

    def prepare(self, **params: Any) -> dict[str, Any]:
        """Build the keyword arguments of the next provider call.

        ``params`` are request parameters, merged over the context's own. The
        conversation itself can't be passed here; add messages with :meth:`add`.
        The context checks that the conversation is a valid request, raising
        :class:`~artiik.errors.ValidationError` if it isn't, compacts it past
        ``compact_at``, and lets the guard trim it when it would go over the
        budget. Pin messages and restatements go at the end, after the latest
        user turn.
        """
        _check_params(self.api, params)
        merged = {**self.params, **params}
        model = merged.pop("model", self.model)
        if not isinstance(model, str) or not model:
            raise TypeError("prepare() needs a model: pass model= to the Context or to prepare()")
        if self._startup is None:
            self._startup = tuple(self._pins.values())
        system = self._with_pins(_system(self.api, merged.pop(_SYSTEM_KEY[self.api], self.system)))
        tools = _tools(merged.pop("tools", self.tools))
        request = self._requests
        self._requests += 1
        validate(self.api, self._history)
        fixed = self._fixed(system, tools)
        self._outgoing = self._next_outgoing()
        size = self._size(fixed, model)
        if self._should_compact(size, request):
            self._compact(request, model, (system, tools, merged), fixed, size)
            self._outgoing = self._next_outgoing()
            size = self._size(fixed, model)
        if self.budget is not None:
            size = self._fit(request, size, fixed, model, (system, tools, merged))
        outgoing = self._outgoing
        self._sent = _Sent(
            request=request,
            entries=len(self._history),
            fixed=fixed,
            total=fixed + sum(self._tallies, Tally()) + _tally(outgoing.messages),
            model=model,
            outgoing=outgoing,
        )
        self._trace_outgoing(request, outgoing)
        self.trace.emit(
            "prepare",
            request,
            model=model,
            messages=len(self._history),
            estimated_tokens=size,
            budget=self.budget,
        )
        return self._build(model, system, tools, merged)

    def record(self, response: object) -> Usage:
        """Append the provider's reply to the conversation and read its usage.

        ``response`` is what the call returned: an SDK object or its dict. The
        reply is kept exactly as the provider sent it, which signed blocks
        require. The usage calibrates the estimator and anchors the next
        estimate.
        """
        data = to_object(plain(response), "response")
        replies = _replies(self.api, data)
        usage = read_usage(self.api, data)
        sent, self._sent = self._sent, None
        in_place = sent is not None and len(self._history) == sent.entries
        # Without a reply, the messages artiik added stay pending: a system message
        # can't stand before the next user turn.
        committed = sent is not None and bool(replies)
        if sent is not None and replies:
            self._commit(sent.outgoing)
        compacted = any(
            isinstance(block, Compaction) for reply in replies for block in reply.blocks
        )
        if sent is not None and committed and in_place and usage.input_tokens > 0 and not compacted:
            self.estimator.observe(sent.total, usage.input_tokens, api=self.api, model=sent.model)
            self._base = _Base(
                entries=len(self._history),
                tokens=usage.input_tokens,
                fixed=sent.fixed,
                model=sent.model,
            )
        else:
            self._base = None
        for reply in replies:
            self._append(reply)
        if compacted:
            self._drop_compacted(sent.request if sent is not None else self._requests - 1, replies)
        self.last_usage = usage
        self.trace.emit(
            "record",
            sent.request if sent is not None else self._requests - 1,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            stop_reason=_stop_reason(self.api, data),
            messages=len(replies),
        )
        return usage

    def estimate(self, **params: Any) -> int:
        """The estimated size of the next request's prompt, in tokens, as the guard sees it.

        ``params`` can override ``model``, the system prompt and ``tools``, as
        in :meth:`prepare`.
        """
        model = params.get("model", self.model)
        if not isinstance(model, str) or not model:
            raise TypeError("estimate() needs a model: pass model= to the Context or here")
        system = self._with_pins(_system(self.api, params.get(_SYSTEM_KEY[self.api], self.system)))
        tools = _tools(params.get("tools", self.tools))
        self._outgoing = self._next_outgoing()
        return self._size(self._fixed(system, tools), model)

    def compact(self, **params: Any) -> CompactionResult | None:
        """Compact now, whatever the conversation's size.

        ``params`` can override ``model``, the system prompt and ``tools``, as
        in :meth:`prepare`. Returns the outcome, or ``None`` when there was
        nothing to summarize. Only strategies that compact on the client side
        can be forced.
        """
        if self.compaction is None or not self.compaction.active:
            raise ValueError("compact() needs a compaction strategy that runs on the client side")
        _check_params(self.api, params)
        merged = {**self.params, **params}
        model = merged.pop("model", self.model)
        if not isinstance(model, str) or not model:
            raise TypeError("compact() needs a model: pass model= to the Context or here")
        system = self._with_pins(_system(self.api, merged.pop(_SYSTEM_KEY[self.api], self.system)))
        tools = _tools(merged.pop("tools", self.tools))
        validate(self.api, self._history)
        fixed = self._fixed(system, tools)
        self._outgoing = self._next_outgoing()
        return self._compact(
            self._requests, model, (system, tools, merged), fixed, self._size(fixed, model)
        )

    def pin(
        self, text: str, *, kind: PinKind = "constraint", source: PinSource = "operator"
    ) -> Pin:
        """Keep a text in front of the model, through every compaction.

        ``kind`` is ``constraint``, ``decision``, ``goal`` or ``fact``;
        ``source`` is ``operator``, ``user`` or ``tool``. A pin set before the
        first request goes at the end of the system prompt. A later one goes in
        a message after the latest user turn on the next request, so the
        cached prefix doesn't change. After every compaction, one message
        restates all the active pins and the ledger. Returns the pin; pass it
        or its ``id`` to :meth:`unpin`.
        """
        check_pin(text, kind, source)
        self._pin_count += 1
        pin = Pin(id=f"pin-{self._pin_count}", text=text.strip(), kind=kind, source=source)
        self._pins[pin.id] = pin
        if self._startup is not None:
            self._new.append(pin)
        self.trace.emit(
            "pins",
            self._requests,
            action="pin",
            id=pin.id,
            kind=kind,
            source=source,
            placement="message" if self._startup is not None else "system prompt",
        )
        size = approx_tokens(render_pins(tuple(self._pins.values()), "Pinned context:"))
        if size > self.pin_budget:
            self.trace.emit(
                "pins", self._requests, action="over budget", tokens=size, budget=self.pin_budget
            )
            logger.warning(
                "artiik pins: the active pins take about %d tokens, over the pin budget of %d; "
                "they're all kept, so keep pins short",
                size,
                self.pin_budget,
            )
        return pin

    def unpin(self, pin: Pin | str) -> None:
        """Retire a pin: later restatements leave it out.

        If the model has already seen it, the next request says it no longer
        applies. A pin in the system prompt stays there, because the system
        prompt doesn't change after the first request, so restatements list it
        as no longer in force.
        """
        key = pin.id if isinstance(pin, Pin) else pin
        found = self._pins.pop(key, None)
        if found is None:
            raise ValueError(f"no active pin {key!r}")
        if found in self._new:
            self._new.remove(found)
        elif self._startup is not None:
            self._retracting.append(found)
            if found in self._startup:
                self._retired.append(found)
        self.trace.emit("pins", self._requests, action="unpin", id=found.id)

    def pins(self) -> tuple[Pin, ...]:
        """The active pins, oldest first."""
        return tuple(self._pins.values())

    def pending_tool_calls(self) -> list[ToolUse]:
        """The tool calls in the conversation that have no result yet, oldest first."""
        answered = {
            result.tool_use_id for message in self._history for result in message.tool_results
        }
        return [
            use for message in self._history for use in message.tool_uses if use.id not in answered
        ]

    def _append(self, message: Message) -> None:
        self._history.append(message)
        self._tallies.append(tally_message(message))
        for call in message.tool_uses:
            self.ledger.observe(call)

    def _fixed(self, system: JSONValue, tools: list[JSONValue] | None) -> Tally:
        tally = Tally()
        if system is not None:
            match self.api:
                case Format.ANTHROPIC_MESSAGES:
                    tally += tally_blocks(anthropic_messages.parse_system(system).blocks)
                case Format.OPENAI_RESPONSES:
                    tally += tally_message(Message.from_text("system", str(system)))
                case Format.OPENAI_CHAT:
                    system_message = {"role": "system", "content": system}
                    tally += tally_message(openai_chat.parse_message(system_message, "system"))
        for tool in tools or ():
            tally += tally_json(tool)
        return tally

    def _size(self, fixed: Tally, model: str) -> int:
        """A conservative size of the request: the last count plus an estimate of what's new."""
        base = self._base
        extra = _tally(self._outgoing.messages)
        if (
            base is not None
            and base.model == model
            and base.fixed == fixed
            and base.entries <= len(self._history)
        ):
            added = sum(self._tallies[base.entries :], Tally()) + extra - base.extra
            return base.tokens + self._scaled(added, model)
        return self._scaled(fixed + sum(self._tallies, Tally()) + extra, model)

    def _scaled(self, tally: Tally, model: str) -> int:
        return math.ceil(self.estimator.estimate(tally, api=self.api, model=model) * SAFETY)

    def _should_compact(self, size: int, request: int) -> bool:
        return (
            self.compaction is not None
            and self.compaction.active
            and self.compact_at is not None
            and size > self.compact_at
            and (self._compaction_paused_until is None or request >= self._compaction_paused_until)
        )

    def _compact(
        self,
        request: int,
        model: str,
        parts: tuple[JSONValue, list[JSONValue] | None, dict[str, Any]],
        fixed: Tally,
        size: int,
    ) -> CompactionResult | None:
        """Summarize the history before the part kept word for word, and swap the summary in."""
        compaction = cast(Compactor, self.compaction)
        name = type(compaction).__name__
        plan = self._plan(model, fixed, keeps_users=compaction.keeps_user_messages)
        if plan is None:
            self.trace.emit(
                "compaction",
                request,
                strategy=name,
                outcome="nothing to summarize",
                tokens_before=size,
            )
            return None
        cut, boundaries = plan
        system, tools, params = parts
        job = CompactionJob(
            api=self.api,
            model=model,
            messages=tuple(self._history[:cut]),
            boundaries=boundaries,
            system=system,
            tools=tuple(tools) if tools is not None else None,
            params=params,
        )
        result = compaction.compact(job)
        usage = result.usage
        details: dict[str, JSONValue] = {
            "strategy": name,
            "outcome": result.outcome,
            "attempts": result.attempts,
            "tokens_before": size,
            "input_tokens": usage.input_tokens if usage is not None else 0,
            "output_tokens": usage.output_tokens if usage is not None else 0,
        }
        if result.messages is None:
            self._failed(request, details, retry=result.retry)
            return result
        summarized = result.summarized if result.summarized is not None else len(job.messages)
        replacement = list(result.messages)
        kept = self._history[summarized:]
        history = replacement + kept
        found = problems(self.api, history)
        if found:
            # A strategy's bug: keep the conversation as it was rather than send it broken.
            details["outcome"] = "invalid result"
            details["problems"] = list(found)
            self._failed(request, details, retry=False, found=found)
            return result
        carried = self._release(self._history[:summarized])
        self._history = history
        self._tallies = [tally_message(message) for message in replacement] + self._tallies[
            summarized:
        ]
        self._base = None
        self._restate = _Restate(summary=result.summary)
        self._outgoing = self._next_outgoing()
        after = self._size(fixed, model)
        compact_at = cast(int, self.compact_at)
        self._compaction_paused_until = request + _COMPACTION_PAUSE if after > compact_at else None
        self.trace.emit(
            "compaction",
            request,
            **details,
            summarized_messages=summarized,
            kept_messages=len(kept),
            moved_messages=carried,
            tokens_after=after,
        )
        logger.info(
            "artiik compaction: %s summarized %d messages before request %d (about %d tokens, "
            "now about %d)",
            name,
            summarized,
            request,
            size,
            after,
        )
        if after > compact_at:
            logger.warning(
                "artiik compaction: request %d is still above compact_at (%d tokens) after "
                "compaction; the next try waits %d requests",
                request,
                compact_at,
                _COMPACTION_PAUSE,
            )
        return result

    def _failed(
        self,
        request: int,
        details: dict[str, JSONValue],
        *,
        retry: bool,
        found: Sequence[str] = (),
    ) -> None:
        """Trace and log a compaction that didn't happen, and wait before trying again."""
        self._compaction_paused_until = request + _COMPACTION_PAUSE if retry else sys.maxsize
        self.trace.emit("compaction", request, **details)
        logger.log(
            logging.ERROR if found else logging.WARNING,
            "artiik compaction: %s didn't compact before request %d: %s%s",
            details["strategy"],
            request,
            details["outcome"],
            "".join(f"; {problem}" for problem in found),
        )

    def _plan(
        self, model: str, fixed: Tally, *, keeps_users: bool
    ) -> tuple[int, tuple[int, ...]] | None:
        """Where to cut: keep the current turn, or its latest steps when it's too long.

        Returns the position of the first message kept word for word, and the
        positions in the summarized part where a shorter summary could end.
        The summarized part is cut shorter if its compaction request wouldn't
        fit the budget. ``keeps_users`` is for a strategy that keeps user
        messages word for word: a part made only of them has nothing for it to
        summarize.
        """
        units = segment(self.api, self._history, self._is_own)
        turns = group(units)
        if not turns:
            return None
        limit = cast(int, self.compact_at) // 2
        last = turns[-1]

        def size_of(indices: Sequence[int]) -> int:
            return self._scaled(sum((self._tallies[index] for index in indices), Tally()), model)

        steps = [unit for unit in last if unit.kind == "step"]
        whole = [index for unit in last for index in unit.indices]
        if last[0].kind == "user" and (not steps or size_of(whole) <= limit):
            cut = last[0].indices[0]
        elif steps:
            cut = steps[-1].indices[0]
            kept = size_of(steps[-1].indices)
            for unit in reversed(steps[:-1]):
                kept += size_of(unit.indices)
                if kept > limit:
                    break
                cut = unit.indices[0]
        else:
            cut = last[0].indices[0]
        summarized = [unit for unit in units if unit.indices[0] < cut]
        if self.budget is not None:
            summarized = self._fitting(summarized, fixed, model)
        # An instruction must not open the kept part: it would follow the summary
        # directly, where Anthropic takes no system message. Cut before its turn.
        while summarized and units[len(summarized)].kind == "instruction":
            summarized.pop()
        untouched = {"anchor", "instruction", "user"} if keeps_users else {"anchor", "instruction"}
        if all(unit.kind in untouched for unit in summarized):
            return None
        boundaries = tuple(
            unit.indices[0] for unit in summarized[1:] if unit.kind in ("user", "step")
        )
        return summarized[-1].indices[-1] + 1, boundaries

    def _fitting(self, units: Sequence[Unit], fixed: Tally, model: str) -> list[Unit]:
        """The longest run of units from the first whose compaction request fits the budget."""
        budget = cast(int, self.budget)
        total = fixed
        for position, unit in enumerate(units):
            total = total + sum((self._tallies[index] for index in unit.indices), Tally())
            if self._scaled(total, model) > budget:
                return list(units[:position])
        return list(units)

    def _drop_compacted(self, request: int, replies: Sequence[Message]) -> None:
        """Drop what a compaction block or item that came back in a reply replaced."""
        before = len(self._history)
        blocks = [
            block for reply in replies for block in reply.blocks if isinstance(block, Compaction)
        ]
        if self.api is Format.OPENAI_RESPONSES:
            start = last_compaction(self._history)
            self._release(self._history[:start])
            self._history = self._history[start:]
            self._tallies = self._tallies[start:]
        else:
            position = max(
                index
                for index, message in enumerate(self._history)
                if any(isinstance(block, Compaction) for block in message.blocks)
            )
            message = self._history[position]
            first = max(
                index for index, block in enumerate(message.blocks) if isinstance(block, Compaction)
            )
            if first:
                message = dataclasses.replace(message, blocks=message.blocks[first:])
            self._release(self._history[:position])
            self._history = [message, *self._history[position + 1 :]]
            self._tallies = [tally_message(message), *self._tallies[position + 1 :]]
        self._base = None
        self._restate = _Restate(summary=blocks[-1].summary if blocks else None)
        self.trace.emit(
            "compaction",
            request,
            strategy="provider",
            outcome="compacted",
            summarized_messages=before - len(self._history),
        )

    def _release(self, messages: Sequence[Message]) -> int:
        """Let go of messages a summary replaces, and return how many instructions are carried.

        Pin messages are restated after the summary. Other system and developer
        messages are sent again, after the next user turn, because a summary
        doesn't keep instructions in force.
        """
        carried = 0
        for message in messages:
            if self._is_own(message):
                del self._own[id(message)]
            elif _is_instruction(message):
                self._carried.append(message)
                carried += 1
        return carried

    def _is_own(self, message: Message) -> bool:
        """Whether a message is a pin message artiik wrote."""
        return self._own.get(id(message)) is message

    def _with_pins(self, system: JSONValue) -> JSONValue:
        """The system prompt with the pins set before the first request at its end."""
        pins = self._startup if self._startup is not None else tuple(self._pins.values())
        if not pins:
            return system
        text = render_pins(pins, "Pinned context:")
        if system is None:
            return text
        if isinstance(system, str):
            return f"{system}\n\n{text}"
        block: JSONObject = {"type": "text", "text": text}
        return [*cast("list[JSONValue]", system), block]

    def _next_outgoing(self) -> _Outgoing:
        """The messages to add at the end of the next request, once it ends with a user turn.

        Anthropic takes a system message only after a user turn (tool results
        count), and its compaction docs place restated instructions right after
        the next one, so that no message lands between kept turns.
        """
        if not self._ends_with_user_turn():
            return _Outgoing()
        messages = list(self._carried)
        restated: Restatement | None = None
        if self._restate is not None:
            restated = restatement(
                tuple(self._pins.values()), tuple(self._retired), self.ledger, self.pin_budget
            )
            text = restated.text if restated is not None else ""
        else:
            sections: list[str] = []
            if self._new:
                sections.append(render_pins(self._new, "Pinned context:"))
            if self._retracting:
                sections.append(render_pins(self._retracting, "No longer in force:"))
            text = "\n".join(sections)
        own: tuple[Message, ...] = ()
        if text:
            own = (self._pin_message(text),)
            messages.extend(own)
        return _Outgoing(
            messages=tuple(messages),
            own=own,
            pins=tuple(pin.id for pin in self._new),
            retractions=tuple(pin.id for pin in self._retracting),
            carried=len(self._carried),
            restated=self._restate is not None,
            restatement=restated,
        )

    def _pin_message(self, text: str) -> Message:
        if self.pin_role == "user":
            text = f"{USER_LABEL}\n{text}"
        return Message.from_text(self.pin_role, text)

    def _ends_with_user_turn(self) -> bool:
        """Whether the history ends with a user turn, past any system and pin messages."""
        for message in reversed(self._history):
            if self._is_own(message) or _is_instruction(message):
                continue
            return message.role in ("user", "tool")
        return False

    def _commit(self, outgoing: _Outgoing) -> None:
        """Add the messages a recorded request ended with to the history."""
        for message in outgoing.messages:
            self._append(message)
        for message in outgoing.own:
            self._own[id(message)] = message
        delivered = set(outgoing.pins)
        self._new = [pin for pin in self._new if pin.id not in delivered]
        retracted = set(outgoing.retractions)
        self._retracting = [pin for pin in self._retracting if pin.id not in retracted]
        del self._carried[: outgoing.carried]
        if outgoing.restated:
            self._restate = None

    def _trace_outgoing(self, request: int, outgoing: _Outgoing) -> None:
        """Trace the pins a request delivers, with a receipt after a compaction."""
        if outgoing.restatement is not None:
            self._receipt(request, outgoing.restatement)
        elif not outgoing.restated and (outgoing.pins or outgoing.retractions):
            self.trace.emit(
                "pins",
                request,
                action="send",
                pins=list(outgoing.pins),
                retractions=list(outgoing.retractions),
            )

    def _receipt(self, request: int, restated: Restatement) -> None:
        """Check the pins against the summary, when it's readable, and say what was restated."""
        summary = self._restate.summary if self._restate is not None else None
        pins = tuple(self._pins.values())
        kept = kept_in(summary, pins) if summary is not None else None
        self.trace.emit(
            "pins",
            request,
            action="restate",
            pins=len(pins),
            kept=len(kept) if kept is not None else None,
            missing=[pin.id for pin in pins if pin not in kept] if kept is not None else None,
            retired=restated.retired,
            ledger=restated.ledger,
            tokens=restated.tokens,
            over_budget=restated.over_budget,
        )
        if pins and kept is not None:
            logger.info(
                "artiik pins: summary kept %d/%d pins; restated all %d before request %d",
                len(kept),
                len(pins),
                len(pins),
                request,
            )
        elif pins:
            logger.info(
                "artiik pins: the summary isn't readable text; restated all %d pins before "
                "request %d",
                len(pins),
                request,
            )
        if restated.over_budget:
            logger.warning(
                "artiik pins: the restatement takes about %d tokens, over the pin budget of %d",
                restated.tokens,
                self.pin_budget,
            )

    def _fit(
        self,
        request: int,
        size: int,
        fixed: Tally,
        model: str,
        parts: tuple[JSONValue, list[JSONValue] | None, dict[str, Any]],
    ) -> int:
        """Keep the request within the budget: count it exactly near the limit, trim it if over."""
        budget = cast(int, self.budget)
        if self.counter is not None and size > budget * (1 - COUNT_MARGIN):
            size = self._count(fixed, model, parts)
        if size <= budget:
            return size
        before = size
        trims: list[Trim] = []
        extra = _tally(self._outgoing.messages)
        for _ in range(_GUARD_ROUNDS):
            try:
                result = trim(
                    self.api,
                    self._history,
                    self._tallies,
                    size=lambda view: self._scaled(fixed + view + extra, model),
                    target=math.floor(budget * self.trim_to),
                    limit=budget,
                    instruction=self._is_own,
                )
            except BudgetError as error:
                self._guard_event(request, trims, before, error.needed, fits=False)
                raise
            trims.append(result)
            self._history = list(result.messages)
            self._tallies = list(result.tallies)
            self._base = None
            size = result.size
            if self.counter is not None:
                size = self._count(fixed, model, parts)
            if size <= budget:
                break
        if size > budget:
            self._guard_event(request, trims, before, size, fits=False)
            raise BudgetError(size, budget)
        self._guard_event(request, trims, before, size, fits=True)
        validate(self.api, self._history)
        self._outgoing = self._next_outgoing()
        return size

    def _count(
        self,
        fixed: Tally,
        model: str,
        parts: tuple[JSONValue, list[JSONValue] | None, dict[str, Any]],
    ) -> int:
        """Count the request exactly, and learn from the count."""
        counter = cast(TokenCounter, self.counter)
        system, tools, params = parts
        exact = counter.count(self.api, self._build(model, system, tools, params))
        extra = _tally(self._outgoing.messages)
        total = fixed + sum(self._tallies, Tally()) + extra
        self.estimator.observe(total, exact, api=self.api, model=model)
        self._base = _Base(
            entries=len(self._history), tokens=exact, fixed=fixed, model=model, extra=extra
        )
        return exact

    def _guard_event(
        self, request: int, trims: Sequence[Trim], before: int, after: int, *, fits: bool
    ) -> None:
        turns = sum(result.dropped_turns for result in trims)
        steps = sum(result.dropped_steps for result in trims)
        dropped = sum(result.dropped_messages for result in trims)
        moved = sum(result.moved_messages for result in trims)
        self.trace.emit(
            "guard",
            request,
            fits=fits,
            dropped_turns=turns,
            dropped_steps=steps,
            dropped_messages=dropped,
            moved_messages=moved,
            tokens_before=before,
            tokens_after=after,
            budget=self.budget,
        )
        if fits:
            logger.warning(
                "artiik guard: request %d was over its budget of %s tokens (about %d); dropped "
                "%d turns and %d steps (%d messages), now about %d tokens",
                request,
                self.budget,
                before,
                turns,
                steps,
                dropped,
                after,
            )
        else:
            logger.error(
                "artiik guard: request %d can't fit its budget of %s tokens: about %d tokens "
                "remain after dropping %d turns and %d steps",
                request,
                self.budget,
                after,
                turns,
                steps,
            )

    def _build(
        self,
        model: str,
        system: JSONValue,
        tools: list[JSONValue] | None,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"model": model}
        system = copy.deepcopy(system)
        conversation = [*self._history, *self._outgoing.messages]
        match self.api:
            case Format.ANTHROPIC_MESSAGES:
                if system is not None:
                    request["system"] = system
                if tools is not None:
                    request["tools"] = copy.deepcopy(tools)
                request["messages"] = anthropic_messages.dump_messages(conversation)
            case Format.OPENAI_RESPONSES:
                if system is not None:
                    request["instructions"] = system
                if tools is not None:
                    request["tools"] = copy.deepcopy(tools)
                request["input"] = openai_responses.dump_items(conversation)
            case Format.OPENAI_CHAT:
                if tools is not None:
                    request["tools"] = copy.deepcopy(tools)
                prefix: list[JSONObject] = (
                    [{"role": "system", "content": system}] if system is not None else []
                )
                request["messages"] = prefix + openai_chat.dump_messages(conversation)
        request.update(copy.deepcopy(dict(params)))
        if self.compaction is not None:
            self.compaction.configure(self.api, request, self.compact_at or 0)
        if self.api is Format.ANTHROPIC_MESSAGES:
            self._mark_compaction(request)
        return request

    def _mark_compaction(self, request: dict[str, Any]) -> None:
        """Send the beta headers the history's compaction blocks need, and cache after the block.

        Anthropic wants the beta header on every request that carries a
        compaction block, and a ``cache_control`` breakpoint on the block
        caches the summary.
        """
        blocks = [
            block
            for message in self._history
            for block in message.blocks
            if isinstance(block, Compaction)
        ]
        for block in blocks:
            add_beta(
                request, ANTHROPIC_BETA if "signature" in block.data else ANTHROPIC_THRESHOLD_BETA
            )
        messages = request.get("messages")
        if not blocks or not isinstance(messages, list) or not messages:
            return
        first = cast("list[JSONObject]", messages)[0].get("content")
        if not isinstance(first, list) or not first or not isinstance(first[0], dict):
            return
        block = first[0]
        if (
            block.get("type") == "compaction"
            and "cache_control" not in block
            and _breakpoints(request) < MAX_BREAKPOINTS
        ):
            block["cache_control"] = {"type": "ephemeral"}


def _api(api: Format | str) -> Format:
    try:
        return Format(api)
    except ValueError:
        names = ", ".join(fmt.value for fmt in Format)
        raise ValueError(f"unknown api {api!r}; use one of: {names}") from None


def _check_params(api: Format, params: Mapping[str, object]) -> None:
    key = _HISTORY_KEY[api]
    if key in params:
        raise TypeError(f"the context owns the conversation: add messages with add(), not {key}=")
    if "previous_response_id" in params:
        raise TypeError("the context sends the whole conversation; drop previous_response_id")


def _system(api: Format, system: object) -> JSONValue:
    if system is None:
        return None
    value = to_json(plain(system), "system")
    if api is Format.OPENAI_RESPONSES and not isinstance(value, str):
        raise TypeError("the Responses API takes the system prompt as a string (instructions)")
    if not isinstance(value, str | list):
        raise TypeError("the system prompt must be a string or a list of content blocks")
    return value


def _tools(tools: object) -> list[JSONValue] | None:
    if tools is None:
        return None
    if not isinstance(tools, list | tuple):
        raise TypeError("tools must be a list of tool definitions")
    items = cast("Sequence[object]", tools)
    return [to_json(plain(tool), f"tools[{index}]") for index, tool in enumerate(items)]


def _parse_entry(api: Format, entry: object, path: str) -> Message:
    match api:
        case Format.ANTHROPIC_MESSAGES:
            return anthropic_messages.parse_message(entry, path)
        case Format.OPENAI_RESPONSES:
            return openai_responses.parse_item(entry, path)
        case Format.OPENAI_CHAT:
            return openai_chat.parse_message(entry, path)


def _replies(api: Format, data: JSONObject) -> list[Message]:
    """The messages to append from a response: its reply, kept exactly as sent."""
    match api:
        case Format.ANTHROPIC_MESSAGES:
            reply = anthropic_messages.parse_response(data)
            return [reply] if reply.blocks else []
        case Format.OPENAI_RESPONSES:
            items = openai_responses.parse_items(
                expect_list(data.get("output"), "response.output"), "output"
            )
            while items and _is_reasoning(items[-1]):
                items.pop()
            return items
        case Format.OPENAI_CHAT:
            choices = expect_list(data.get("choices"), "response.choices")
            if not choices:
                return []
            choice = expect_object(choices[0], "response.choices[0]")
            reply = openai_chat.parse_message(choice.get("message"), "choices[0].message")
            return [reply] if reply.blocks else []


def _is_reasoning(item: Message) -> bool:
    return (
        item.bare_item
        and bool(item.blocks)
        and isinstance(item.blocks[0], Opaque)
        and item.blocks[0].type == "reasoning"
    )


def _stop_reason(api: Format, data: JSONObject) -> JSONValue:
    match api:
        case Format.ANTHROPIC_MESSAGES:
            return data.get("stop_reason")
        case Format.OPENAI_RESPONSES:
            details = data.get("incomplete_details")
            if isinstance(details, dict):
                return details.get("reason")
            return data.get("status")
        case Format.OPENAI_CHAT:
            choices = data.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                return choices[0].get("finish_reason")
            return None


def _compact_at(
    api: Format, compaction: Compactor | None, compact_at: int | None, budget: int | None
) -> int | None:
    """Check the compaction settings, and work out the threshold."""
    if compact_at is not None and compact_at <= 0:
        raise ValueError(f"compact_at must be a positive number of tokens, got {compact_at}")
    if compact_at is not None and budget is not None and compact_at > budget:
        raise ValueError(f"compact_at ({compact_at}) can't be above the budget ({budget})")
    if compaction is None:
        return compact_at
    if compaction.apis is not None and api not in compaction.apis:
        raise ValueError(f"{type(compaction).__name__} doesn't work with the {api.value} API")
    if compact_at is None and budget is not None:
        compact_at = int(budget * COMPACT_AT)
    if compact_at is None and (compaction.active or compaction.threshold is None):
        raise ValueError("compaction needs compact_at, or a budget to work it out from")
    if not compaction.active:
        compaction.configure(api, {}, compact_at or 0)
    return compact_at


def _tally(messages: Sequence[Message]) -> Tally:
    return sum((tally_message(message) for message in messages), Tally())


def _pin_role(api: Format, role: str | None) -> Role:
    allowed = _PIN_ROLES[api]
    if role is None:
        return allowed[0]
    if role not in allowed:
        raise ValueError(
            f"pin_role {role!r} isn't available in the {api.value} API; use one of: "
            + ", ".join(allowed)
        )
    return role


def _is_instruction(message: Message) -> bool:
    return not message.bare_item and message.role in ("system", "developer")


def _breakpoints(request: Mapping[str, Any]) -> int:
    """The cache breakpoints of an Anthropic request."""
    count = 1 if "cache_control" in request else 0
    for key in ("tools", "system", "messages"):
        count += _cache_controls(request.get(key))
    return count


def _cache_controls(value: object) -> int:
    if isinstance(value, Mapping):
        fields = cast("Mapping[str, object]", value)
        own = 1 if "cache_control" in fields else 0
        return own + _cache_controls(fields.get("content"))
    if isinstance(value, list):
        return sum(_cache_controls(item) for item in cast("list[object]", value))
    return 0
