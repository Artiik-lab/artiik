"""The Tier 0 invariants hold for sound strategies, and each one trips on a broken strategy."""

import dataclasses
import time
from collections.abc import Mapping, Sequence

import pytest

from artiik.formats import anthropic_messages
from artiik.messages import (
    Block,
    Compaction,
    Format,
    JSONObject,
    JSONValue,
    Message,
    Opaque,
    Thinking,
    ToolResult,
    last_compaction,
)
from artiik.testing import (
    COMPACTION_BETA,
    Environment,
    FakeAnthropic,
    FakeAPIError,
    FakeOpenAI,
    ForgetfulSummarizer,
    Policy,
    Prepare,
    Recall,
    RecordedCall,
    Reply,
    ToolLoopPolicy,
    Violation,
    assert_holds,
    check_all,
    check_append_only,
    check_budget,
    check_kept_whole,
    check_pins,
    check_scopes,
    check_tool_pairs,
    default_request,
    response_json,
    run_session,
)
from artiik.testing.driver import Create
from artiik.testing.model import is_user_turn
from artiik.testing.recording import request_tokens
from artiik.testing.tokens import count_messages

ANTHROPIC = Format.ANTHROPIC_MESSAGES
RESPONSES = Format.OPENAI_RESPONSES
CHAT = Format.OPENAI_CHAT
PIN = "Never run migrations on the production database."


def client_for(
    fmt: Format, policy: Policy | None = None
) -> tuple[FakeAnthropic | FakeOpenAI, Create]:
    if fmt is ANTHROPIC:
        anthropic = FakeAnthropic(policy=policy)
        return anthropic, anthropic.messages.create
    openai = FakeOpenAI(policy=policy)
    create = openai.responses.create if fmt is RESPONSES else openai.chat.completions.create
    return openai, create


def turns(count: int, first: str = "Check the logs.") -> list[str]:
    return [first, *(f"Turn {index}: check the next log." for index in range(1, count))]


# Strategies


def last_messages(count: int) -> Prepare:
    """Broken: a sliding window that cuts tool calls from their results."""

    def prepare(history: Sequence[Message]) -> Sequence[Message]:
        return history[-count:]

    return prepare


def drop_oldest_turns(fmt: Format, budget: int) -> Prepare:
    """Sound for budget and tool pairs: drop whole turns, oldest first, until the request fits."""
    fixed = request_tokens(fmt, default_request(fmt))

    def prepare(history: Sequence[Message]) -> Sequence[Message]:
        starts = [index for index, message in enumerate(history) if is_user_turn(message)]
        for start in starts:
            if fixed + count_messages(history[start:]) <= budget:
                return history[start:]
        return history[starts[-1] :]

    return prepare


def clear_results(history: Sequence[Message], cleared: set[str]) -> list[Message]:
    """Replace the results of the given tool calls with a placeholder."""

    def clear(block: Block) -> Block:
        if isinstance(block, ToolResult) and block.tool_use_id in cleared:
            return dataclasses.replace(block, content="[cleared]")
        return block

    return [
        dataclasses.replace(message, blocks=tuple(clear(block) for block in message.blocks))
        for message in history
    ]


def results_of(history: Sequence[Message]) -> list[str]:
    return [result.tool_use_id for message in history for result in message.tool_results]


def clear_all_but_last(keep: int) -> Prepare:
    """Broken: clears old tool outputs on every call, so no request extends the previous one."""

    def prepare(history: Sequence[Message]) -> Sequence[Message]:
        results = results_of(history)
        return clear_results(history, set(results[:-keep]))

    return prepare


class BatchClearing:
    """Sound: clears old tool outputs in batches and records the calls that follow a batch."""

    def __init__(self, fake: FakeAnthropic | FakeOpenAI, *, trigger: int, keep: int) -> None:
        self.fake = fake
        self.trigger = trigger
        self.keep = keep
        self.cleared: set[str] = set()
        self.batches: list[int] = []

    def __call__(self, history: Sequence[Message]) -> Sequence[Message]:
        live = [result for result in results_of(history) if result not in self.cleared]
        if len(live) > self.trigger:
            self.cleared.update(live[: -self.keep])
            self.batches.append(len(self.fake.calls))
        return clear_results(history, self.cleared)


class AnthropicCompactor:
    """Compacts with the provider every ``every`` turns, as the compaction docs describe.

    The summary replaces the history before the new user turn. With ``pin``,
    the pin is stated again in a system message right after that turn.
    """

    def __init__(
        self, fake: FakeAnthropic, request: JSONObject, *, every: int, pin: str | None = None
    ) -> None:
        self.fake = fake
        self.request = request
        self.every = every
        self.pin = pin
        self.prefix: list[Message] = []
        self.start = 0
        self.turns = 0

    def __call__(self, history: Sequence[Message]) -> Sequence[Message]:
        if is_user_turn(history[-1]):
            self.turns += 1
            if self.turns > 1 and (self.turns - 1) % self.every == 0:
                self._compact(history)
        return [*self.prefix, *history[self.start :]]

    def _compact(self, history: Sequence[Message]) -> None:
        summarized = [*self.prefix, *history[self.start : -1]]
        response = self.fake.beta.messages.create(
            **self.request,
            messages=anthropic_messages.dump_messages(summarized),
            compaction={"type": "summarize"},
        )
        data = response_json(response)
        if data["stop_reason"] != "compaction":
            return
        self.prefix = [anthropic_messages.parse_response(data), history[-1]]
        if self.pin is not None:
            self.prefix.append(Message.from_text("system", self.pin))
        self.start = len(history)


def restate_after_compaction(pin: str, *, drop_compacted: bool = False) -> Prepare:
    """State the pin again in a developer message right after the latest compaction item.

    With ``drop_compacted``, the input before that item, which the API ignores,
    isn't sent at all.
    """

    def prepare(history: Sequence[Message]) -> Sequence[Message]:
        start = last_compaction(history)
        if not any(isinstance(block, Compaction) for block in history[start].blocks):
            return history
        before = [] if drop_compacted else list(history[:start])
        return [*before, history[start], Message.from_text("developer", pin), *history[start + 1 :]]

    return prepare


class ServerToolAgent:
    """A tool-calling agent that also thinks and searches the web on the provider's side."""

    def __init__(self, fmt: Format) -> None:
        self.fmt = fmt
        self.base = ToolLoopPolicy()

    def reply(self, conversation: Sequence[Message]) -> Reply:
        step = len(conversation)
        raw: tuple[JSONObject, ...]
        if self.fmt is ANTHROPIC:
            raw = (
                {
                    "type": "server_tool_use",
                    "id": f"srvtoolu_{step}",
                    "name": "web_search",
                    "input": {"query": "service status"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": f"srvtoolu_{step}",
                    "content": [],
                },
            )
        else:
            raw = (
                {
                    "id": f"ws_{step}",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "service status"},
                },
            )
        reply = self.base.reply(conversation)
        return dataclasses.replace(reply, thinking=f"Step {step}: check the status.", raw=raw)


def tamper(kind: str) -> Prepare:
    """Broken: changes every block of type ``kind`` before sending it back."""

    def change(block: Block) -> Block:
        if kind == "thinking" and isinstance(block, Thinking):
            return dataclasses.replace(block, thinking=f"{block.thinking} Edited.")
        if isinstance(block, Opaque) and block.type == kind:
            return dataclasses.replace(block, data={**block.data, "citations": None})
        return block

    def prepare(history: Sequence[Message]) -> Sequence[Message]:
        return [
            dataclasses.replace(message, blocks=tuple(change(block) for block in message.blocks))
            for message in history
        ]

    return prepare


class ToyMemory:
    """A memory store, scoped or (broken) not, that records each lookup for the scope check."""

    def __init__(self, *, filter_scope: bool) -> None:
        self.filter_scope = filter_scope
        self.memories: list[tuple[str, Mapping[str, str]]] = []
        self.recalls: list[Recall] = []

    def add(self, text: str, scope: Mapping[str, str]) -> None:
        self.memories.append((text, dict(scope)))

    def recall(self, call: int, scope: Mapping[str, str], k: int) -> list[str]:
        hits = [
            (text, memory_scope)
            for text, memory_scope in self.memories
            if not self.filter_scope
            or all(scope.get(key) == value for key, value in memory_scope.items())
        ][:k]
        self.recalls.append(Recall(call=call, scope=scope, returned=[hit[1] for hit in hits]))
        return [hit[0] for hit in hits]


# Every invariant holds for plain sessions


@pytest.mark.parametrize("fmt", list(Format))
def test_a_plain_session_keeps_every_invariant(fmt: Format) -> None:
    fake, create = client_for(fmt, ToolLoopPolicy(calls_per_step=3))
    run_session(create, fmt, turns(5, f"{PIN} Check the logs."))
    assert_holds(check_all(fake.calls, budget=100_000, pins={PIN: 0}))


# Budget


@pytest.mark.parametrize("fmt", list(Format))
def test_budget_trips_when_the_history_outgrows_it(fmt: Format) -> None:
    budget = 3_000
    environment = Environment(output_tokens=300)
    fake, create = client_for(fmt)
    run_session(create, fmt, turns(10), environment=environment)
    violations = check_budget(fake.calls, budget)
    assert violations
    assert {violation.invariant for violation in violations} == {"budget"}

    fake, create = client_for(fmt)
    run_session(
        create, fmt, turns(10), environment=environment, prepare=drop_oldest_turns(fmt, budget)
    )
    assert check_budget(fake.calls, budget) == []
    assert check_tool_pairs(fake.calls) == []


# Tool pairs


@pytest.mark.parametrize("fmt", list(Format))
def test_tool_pairs_trip_on_a_sliding_window(fmt: Format) -> None:
    fake, create = client_for(fmt)
    with pytest.raises(FakeAPIError):
        run_session(create, fmt, turns(3), prepare=last_messages(3))
    violations = check_tool_pairs(fake.calls)
    assert violations
    assert "answers no earlier call" in violations[0].detail


def test_tool_pairs_trip_on_a_call_without_its_result() -> None:
    request: JSONObject = {
        "model": "fake-model",
        "input": [
            {"role": "user", "content": "Go"},
            {"type": "function_call", "call_id": "c1", "name": "read_log", "arguments": "{}"},
        ],
    }
    calls = [RecordedCall(0, "responses.create", RESPONSES, request)]
    assert check_tool_pairs(calls) == [Violation("tool-pairs", 0, "call c1 has no result")]


# Append-only


@pytest.mark.parametrize("fmt", list(Format))
def test_append_only_trips_when_every_call_rewrites_history(fmt: Format) -> None:
    fake, create = client_for(fmt)
    run_session(create, fmt, turns(3), prepare=clear_all_but_last(1))
    violations = check_append_only(fake.calls)
    assert violations
    assert all(violation.detail.endswith("was rewritten") for violation in violations)


@pytest.mark.parametrize("fmt", list(Format))
def test_append_only_allows_announced_rewrites(fmt: Format) -> None:
    fake, create = client_for(fmt)
    strategy = BatchClearing(fake, trigger=4, keep=1)
    run_session(create, fmt, turns(6), prepare=strategy)
    assert strategy.batches
    flagged = [violation.call for violation in check_append_only(fake.calls)]
    assert flagged == strategy.batches
    assert check_append_only(fake.calls, allowed=strategy.batches) == []


BASE_REQUEST: JSONObject = {
    "model": "fake-model",
    "max_tokens": 100,
    "system": "Be brief.",
    "tools": [{"name": "read_log", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "auto"},
    "thinking": {"type": "adaptive"},
    "messages": [{"role": "user", "content": "Go"}],
}
LONGER: list[JSONValue] = [
    {"role": "user", "content": "Go"},
    {"role": "assistant", "content": "Done."},
    {"role": "user", "content": "Next."},
]


def recorded(index: int, request: JSONObject) -> RecordedCall:
    return RecordedCall(index, "messages.create", ANTHROPIC, request, response={"content": []})


@pytest.mark.parametrize("key", ["model", "system", "tools", "tool_choice", "thinking"])
def test_changing_what_the_cache_depends_on_is_a_rewrite(key: str) -> None:
    changed: JSONObject = {**BASE_REQUEST, key: "changed", "messages": LONGER}
    violations = check_append_only([recorded(0, BASE_REQUEST), recorded(1, changed)])
    assert violations == [Violation("append-only", 1, f"the {key} parameter changed")]


def test_moving_cache_breakpoints_is_not_a_rewrite_but_shrinking_is() -> None:
    marked: JSONObject = {"type": "ephemeral"}
    first: JSONObject = {
        **BASE_REQUEST,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Go", "cache_control": marked}]}
        ],
    }
    second: JSONObject = {
        **BASE_REQUEST,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Go"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Ok", "cache_control": marked}],
            },
        ],
    }
    assert check_append_only([recorded(0, first), recorded(1, second)]) == []
    failed = RecordedCall(
        2,
        "messages.create",
        ANTHROPIC,
        BASE_REQUEST,
        error=FakeAPIError.anthropic(529, "overloaded_error", "Overloaded"),
    )
    assert check_append_only([recorded(0, first), recorded(1, second), failed]) == []
    shrunk = recorded(3, {**BASE_REQUEST, "messages": []})
    assert check_append_only([recorded(1, second), shrunk]) == [
        Violation("append-only", 3, "the history shrank from 2 to 0 entries")
    ]


# Pins


@pytest.mark.parametrize("restate", [False, True])
def test_pins_trip_when_a_compaction_drops_them(restate: bool) -> None:
    fake = FakeAnthropic(summarizer=ForgetfulSummarizer([PIN]))
    request: JSONObject = {**default_request(ANTHROPIC), "betas": [COMPACTION_BETA]}
    compactor = AnthropicCompactor(fake, request, every=2, pin=PIN if restate else None)
    run_session(
        fake.beta.messages.create,
        ANTHROPIC,
        turns(6, f"{PIN} Check the logs."),
        request=request,
        prepare=compactor,
    )
    compactions = [call for call in fake.calls if call.is_compaction]
    assert len(compactions) == 2
    for call in compactions:
        summary = str((call.response or {}).get("content"))
        assert "Check the logs." in summary
        assert PIN not in summary
    violations = check_pins(fake.calls, {PIN: 0})
    if restate:
        assert_holds(check_all(fake.calls, pins={PIN: 0}))
    else:
        assert violations
        assert violations[0].call == compactions[0].index + 1


@pytest.mark.parametrize("restate", [False, True])
def test_pins_after_server_side_compaction(restate: bool) -> None:
    fake = FakeOpenAI(summarizer=ForgetfulSummarizer([PIN]))
    request: JSONObject = {
        **default_request(RESPONSES),
        "context_management": [{"type": "compaction", "compact_threshold": 2_000}],
    }
    run_session(
        fake.responses.create,
        RESPONSES,
        turns(6, f"{PIN} Check the logs."),
        environment=Environment(output_tokens=300),
        request=request,
        prepare=restate_after_compaction(PIN) if restate else None,
    )
    compacted = [
        call.index
        for call in fake.calls
        if any(item.get("type") == "compaction" for item in _output(call))
    ]
    assert compacted
    violations = check_pins(fake.calls, {PIN: 0})
    if restate:
        assert_holds(check_all(fake.calls, pins={PIN: 0}))
    else:
        assert violations
        assert violations[0].call == compacted[0] + 1


def _output(call: RecordedCall) -> list[JSONObject]:
    output = (call.response or {}).get("output")
    return [item for item in output if isinstance(item, dict)] if isinstance(output, list) else []


# Kept whole


@pytest.mark.parametrize("fmt", [ANTHROPIC, RESPONSES])
def test_blocks_kept_whole_go_back_unchanged(fmt: Format) -> None:
    fake, create = client_for(fmt, ServerToolAgent(fmt))
    run_session(create, fmt, turns(3))
    assert_holds(check_all(fake.calls))


@pytest.mark.parametrize(
    ("fmt", "kind", "rejected"),
    [
        pytest.param(ANTHROPIC, "thinking", True, id="anthropic-thinking"),
        pytest.param(ANTHROPIC, "web_search_tool_result", False, id="anthropic-server-tool"),
        pytest.param(RESPONSES, "reasoning", True, id="responses-reasoning"),
        pytest.param(RESPONSES, "web_search_call", False, id="responses-server-tool"),
    ],
)
def test_kept_whole_trips_when_a_block_is_changed(fmt: Format, kind: str, rejected: bool) -> None:
    fake, create = client_for(fmt, ServerToolAgent(fmt))
    if rejected:
        with pytest.raises(FakeAPIError):
            run_session(create, fmt, turns(2), prepare=tamper(kind))
    else:
        run_session(create, fmt, turns(2), prepare=tamper(kind))
    violations = check_kept_whole(fake.calls)
    assert violations
    assert all(violation.detail.startswith(f"{kind} at ") for violation in violations)


def test_compaction_blocks_are_kept_whole() -> None:
    fake = FakeAnthropic()
    request: JSONObject = {**default_request(ANTHROPIC), "betas": [COMPACTION_BETA]}
    compactor = AnthropicCompactor(fake, request, every=1)
    run_session(fake.beta.messages.create, ANTHROPIC, turns(3), request=request, prepare=compactor)
    assert check_kept_whole(fake.calls) == []
    forged: JSONObject = {
        **request,
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "compaction", "content": "Made up.", "signature": "sig"}],
            },
            {"role": "user", "content": "Go on."},
        ],
    }
    calls = [*fake.calls, RecordedCall(len(fake.calls), "beta.messages.create", ANTHROPIC, forged)]
    [violation] = check_kept_whole(calls)
    assert violation.detail.startswith("compaction at messages[0].content[0]")


# Scopes


@pytest.mark.parametrize("filter_scope", [False, True])
def test_scopes_trip_when_memories_leak(filter_scope: bool) -> None:
    memory = ToyMemory(filter_scope=filter_scope)
    memory.add("Ana prefers short answers.", {"user": "ana"})
    memory.add("Bob works on billing.", {"user": "bob"})
    memory.add("Project x deploys on Fridays.", {"user": "ana", "project": "x"})
    memory.add("Everyone uses UTC.", {})
    memory.recall(0, {"user": "ana", "project": "x"}, k=10)
    memory.recall(1, {"user": "ana"}, k=10)
    violations = check_scopes(memory.recalls)
    if filter_scope:
        assert violations == []
    else:
        assert [violation.call for violation in violations] == [0, 1, 1]


def test_a_memory_without_scope_is_visible_everywhere() -> None:
    recall = Recall(call=0, scope={"user": "ana"}, returned=[{}, {"user": "ana"}])
    assert check_scopes([recall]) == []


# Reporting


def test_assert_holds_lists_every_violation() -> None:
    violations = [
        Violation("budget", 3, "5000 tokens > budget 4000"),
        Violation("pins", 4, "pin 'x' is missing"),
    ]
    with pytest.raises(AssertionError) as caught:
        assert_holds(violations)
    assert str(caught.value) == (
        "2 invariant violation(s):\n"
        "  [budget] call 3: 5000 tokens > budget 4000\n"
        "  [pins] call 4: pin 'x' is missing"
    )
    assert_holds([])


# A long session


@pytest.mark.parametrize("fmt", [ANTHROPIC, RESPONSES])
def test_hundreds_of_turns_with_compaction_keep_every_invariant(fmt: Format) -> None:
    started = time.perf_counter()
    environment = Environment(output_tokens=1_000)
    policy = ToolLoopPolicy(calls_per_step=3, steps=2)
    if fmt is ANTHROPIC:
        fake = FakeAnthropic(policy=policy, summarizer=ForgetfulSummarizer([PIN]))
        request: JSONObject = {**default_request(fmt), "betas": [COMPACTION_BETA]}
        prepare: Prepare = AnthropicCompactor(fake, request, every=10, pin=PIN)
        create: Create = fake.beta.messages.create
    else:
        fake = FakeOpenAI(policy=policy, summarizer=ForgetfulSummarizer([PIN]))
        request = {
            **default_request(fmt),
            "context_management": [{"type": "compaction", "compact_threshold": 40_000}],
        }
        prepare = restate_after_compaction(PIN, drop_compacted=True)
        create = fake.responses.create
    run_session(
        create,
        fmt,
        turns(200, f"{PIN} Check the logs."),
        environment=environment,
        request=request,
        prepare=prepare,
    )
    compactions = sum(
        1
        for call in fake.calls
        if call.is_compaction or any(item.get("type") == "compaction" for item in _output(call))
    )
    assert len(environment.calls) >= 200 * 3
    assert compactions >= 10
    assert_holds(check_all(fake.calls, budget=70_000, pins={PIN: 0}))
    assert time.perf_counter() - started < 20
