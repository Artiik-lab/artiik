"""Compaction: every strategy, stop reason and error code, on the fake clients.

The fakes enforce each provider's compaction protocol, so a test only passes
when artiik sends what the API accepts. The long sessions at the end check the
Tier 0 invariants on every request each strategy sent.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import pytest

from artiik import AnthropicTokenCounter, Context, Format, Message
from artiik.compaction import (
    ANTHROPIC_BETA,
    ANTHROPIC_THRESHOLD_BETA,
    DEFAULT_INSTRUCTIONS,
    MAX_INSTRUCTIONS,
    AnthropicCompaction,
    AnthropicThresholdCompaction,
    CompactionJob,
    CompactionResult,
    Compactor,
    OpenAICompaction,
    SummaryCompaction,
    add_beta,
)
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.formats._json import expect_list, expect_object
from artiik.messages import Compaction, JSONObject, JSONValue, ToolUse
from artiik.testing import (
    Environment,
    FakeAnthropic,
    FakeAPIError,
    FakeOpenAI,
    RecordedCall,
    ToolLoopPolicy,
    assert_holds,
    check_all,
    default_request,
    run_context,
)
from artiik.testing.driver import Create

ANTHROPIC = Format.ANTHROPIC_MESSAGES
RESPONSES = Format.OPENAI_RESPONSES
CHAT = Format.OPENAI_CHAT
MODEL = "fake-model"
NO_TOOLS = "\n\nDon't call any tool. Reply with the summary text only."


def context_for(fmt: Format, **options: Any) -> Context:
    base = default_request(fmt)
    params = {key: value for key, value in base.items() if key not in ("model", "tools")}
    params.update(options.pop("params", {}))
    tools = options.pop("tools", base["tools"])
    return Context(fmt, model=MODEL, tools=tools, params=params, **options)


def compacting(fmt: Format, compaction: Compactor, **options: Any) -> Context:
    """A context that compacts at 7,500 tokens, well under its budget."""
    return context_for(fmt, budget=20_000, compact_at=7_500, compaction=compaction, **options)


def user(text: str) -> JSONObject:
    return {"role": "user", "content": text}


def fill(
    ctx: Context,
    *,
    first: int = 0,
    turns: int = 4,
    chars: int = 8_000,
    current: JSONObject | None = None,
) -> Context:
    """Add finished turns with long answers, then the current user turn."""
    for number in range(first, first + turns):
        ctx.add(user(f"Turn {number}: read the logs."))
        ctx.add({"role": "assistant", "content": f"Log {number}: " + "x" * chars})
    ctx.add(current if current is not None else user("Now list the errors."))
    return ctx


def answering() -> ToolLoopPolicy:
    """A model that answers every turn right away."""
    return ToolLoopPolicy(steps=0)


def summarize(messages: list[JSONObject]) -> str:
    return f"{len(messages) - 1} messages about logs."


def turns(count: int) -> list[str]:
    return [f"Turn {index}: check the next log." for index in range(count)]


def compactions(fake: FakeAnthropic | FakeOpenAI) -> list[RecordedCall]:
    return [call for call in fake.calls if call.is_compaction]


def outcomes(ctx: Context) -> list[tuple[int, JSONValue]]:
    return [(event.request, event.data["outcome"]) for event in ctx.trace.of("compaction")]


def tracked(
    ctx: Context, fake: FakeAnthropic | FakeOpenAI, create: Create
) -> tuple[Create, dict[int, int]]:
    """Wrap ``create`` to map the context's request numbers to the fake's call indices."""
    calls: dict[int, int] = {}

    def wrapped(**kwargs: Any) -> object:
        calls[ctx.trace.of("prepare")[-1].request] = len(fake.calls)
        return create(**kwargs)

    return wrapped, calls


def rewrites(ctx: Context, calls: Mapping[int, int]) -> set[int]:
    """The calls whose history the context rewrote on purpose: after the guard or a compaction."""
    return {
        calls[event.request]
        for event in ctx.trace.events
        if event.request in calls
        and (event.kind == "guard" or event.data.get("outcome") == "compacted")
    }


def run_turns(ctx: Context, create: Create, count: int) -> None:
    """Send a request, record the answer and add the next user turn, ``count`` times."""
    for number in range(count):
        ctx.record(create(**ctx.prepare()))
        ctx.add(user(f"Next question {number}."))


# Anthropic compaction on demand


def test_anthropic_compaction_swaps_the_summary_in_and_sends_it_first() -> None:
    fake = FakeAnthropic(policy=answering())
    strategy = AnthropicCompaction(fake)
    ctx = fill(compacting(ANTHROPIC, strategy, system="You read logs."))
    old, current = ctx.history[:-1], ctx.history[-1]
    request = ctx.prepare()
    [call] = fake.calls
    assert call.endpoint == "beta.messages.create"
    assert call.request["betas"] == [ANTHROPIC_BETA]
    assert call.request["compaction"] == {"type": "summarize", "instructions": DEFAULT_INSTRUCTIONS}
    assert call.request["system"] == "You read logs."
    assert call.request["tools"] == default_request(ANTHROPIC)["tools"]
    assert call.request["messages"] == anthropic_messages.dump_messages(old)
    assert call.request["max_tokens"] == 8_192
    # The block comes first and once, and the current turn stays word for word.
    assert [message.role for message in ctx.history] == ["assistant", "user"]
    block = ctx.history[0].blocks[0]
    assert isinstance(block, Compaction)
    assert ctx.history[1] == current
    first, second = request["messages"]
    assert first == {
        "role": "assistant",
        "content": [{**block.data, "cache_control": {"type": "ephemeral"}}],
    }
    assert second == user("Now list the errors.")
    assert request["extra_headers"] == {"anthropic-beta": ANTHROPIC_BETA}
    assert "betas" not in request
    ctx.record(fake.messages.create(**request))
    assert fake.model_requests == [MODEL]
    [event] = ctx.trace.of("compaction")
    assert event.data["strategy"] == "AnthropicCompaction"
    assert event.data["outcome"] == "compacted"
    assert event.data["attempts"] == 1
    assert event.data["summarized_messages"] == len(old)
    assert event.data["kept_messages"] == 1
    assert cast(int, event.data["tokens_after"]) < cast(int, event.data["tokens_before"])
    assert cast(int, event.data["input_tokens"]) > 0
    assert cast(int, event.data["output_tokens"]) > 0


def test_the_betas_list_carries_the_beta_when_there_is_one() -> None:
    fake = FakeAnthropic(policy=answering())
    params = {"betas": ["context-1m-2025-08-07"]}
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake), params=params))
    request = ctx.prepare()
    assert fake.calls[0].request["betas"] == ["context-1m-2025-08-07", ANTHROPIC_BETA]
    assert request["betas"] == ["context-1m-2025-08-07", ANTHROPIC_BETA]
    assert "extra_headers" not in request
    assert ctx.params["betas"] == ["context-1m-2025-08-07"]
    fake.beta.messages.create(**request)


def test_the_capability_check_runs_once_per_model() -> None:
    fake = FakeAnthropic(compaction_models=[MODEL])
    strategy = AnthropicCompaction(fake)
    assert strategy.supports(MODEL)
    assert strategy.supports(MODEL)
    assert not strategy.supports("older-model")
    assert fake.model_requests == [MODEL, "older-model"]
    assert AnthropicCompaction(fake, check_support=False).supports("older-model")
    assert fake.model_requests == [MODEL, "older-model"]


class _Models:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def retrieve(self, model_id: str, **kwargs: object) -> object:
        raise self.error


class _Beta:
    def __init__(self, error: Exception) -> None:
        self.models = _Models(error)


class FailingClient:
    """A client whose Models API fails."""

    def __init__(self, error: Exception) -> None:
        self.beta = _Beta(error)


def test_a_model_the_models_api_cant_describe_counts_as_unsupported() -> None:
    missing = FakeAPIError.anthropic(404, "not_found_error", "model: older-model")
    assert not AnthropicCompaction(FailingClient(missing)).supports("older-model")
    with pytest.raises(RuntimeError, match="offline"):
        AnthropicCompaction(FailingClient(RuntimeError("offline"))).supports(MODEL)


def test_a_model_without_compaction_uses_the_fallback() -> None:
    fake = FakeAnthropic(compaction_models=["another-model"], policy=answering())
    strategy = AnthropicCompaction(fake, fallback=SummaryCompaction(summarize))
    ctx = fill(compacting(ANTHROPIC, strategy))
    request = ctx.prepare()
    assert fake.calls == []
    assert ctx.history[0].text == "Summary of the conversation so far:\n\n8 messages about logs."
    assert outcomes(ctx) == [(0, "compacted")]
    fake.messages.create(**request)


def test_a_model_without_compaction_stops_trying() -> None:
    fake = FakeAnthropic(compaction_models=["another-model"], policy=answering())
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    run_turns(ctx, fake.messages.create, 5)
    assert outcomes(ctx) == [(0, "unsupported")]
    assert fake.model_requests == [MODEL]
    assert compactions(fake) == []


@pytest.mark.parametrize("stop", ["max_tokens", "tool_use", "model_context_window_exceeded"])
def test_a_summary_cut_short_is_retried_once(stop: str) -> None:
    fake = FakeAnthropic(policy=answering(), faults={0: stop})
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    ctx.prepare()
    first, second = (call.request for call in fake.calls)
    [event] = ctx.trace.of("compaction")
    assert event.data["outcome"] == "compacted"
    assert event.data["attempts"] == 2
    assert isinstance(ctx.history[0].blocks[0], Compaction)
    assert ctx.history[-1].text == "Now list the errors."
    instructions = expect_object(second["compaction"], "compaction")["instructions"]
    match stop:
        case "max_tokens":
            assert second["max_tokens"] == 2 * cast(int, first["max_tokens"])
            assert instructions == DEFAULT_INSTRUCTIONS
        case "tool_use":
            assert second["max_tokens"] == first["max_tokens"]
            assert instructions == DEFAULT_INSTRUCTIONS + NO_TOOLS
        case _:
            # An older, shorter part is summarized; the rest stays word for word.
            sent = expect_list(second["messages"], "messages")
            assert len(sent) == 4 < len(expect_list(first["messages"], "messages"))
            assert event.data["summarized_messages"] == 4
            assert [message.text for message in ctx.history[1:3]] == [
                "Turn 2: read the logs.",
                "Log 2: " + "x" * 8_000,
            ]


def test_the_same_stop_twice_gives_up() -> None:
    fake = FakeAnthropic(policy=answering(), faults={0: "max_tokens", 1: "max_tokens"})
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    before = ctx.history
    ctx.prepare()
    [event] = ctx.trace.of("compaction")
    assert event.data["outcome"] == "max_tokens"
    assert event.data["attempts"] == 2
    assert ctx.history == before


def test_a_part_too_long_to_summarize_with_nowhere_shorter_to_cut() -> None:
    fake = FakeAnthropic(policy=answering(), faults={0: "model_context_window_exceeded"})
    ctx = compacting(ANTHROPIC, AnthropicCompaction(fake))
    ctx.add(user("x" * 26_000), user("Now list the errors."))
    ctx.prepare()
    assert len(fake.calls) == 1
    assert outcomes(ctx) == [(0, "model_context_window_exceeded")]


@pytest.mark.parametrize(("stop", "retry"), [("refusal", False), ("end_turn", True)])
def test_a_summary_that_doesnt_come_back_is_skipped(stop: str, retry: bool) -> None:
    fake = FakeAnthropic(policy=answering(), faults={0: stop})
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    run_turns(ctx, fake.messages.create, 5)
    # A refusal won't change on a retry; any other stop waits three requests.
    expected = [(0, stop), (3, "compacted")] if retry else [(0, stop)]
    assert outcomes(ctx) == expected
    assert all(call.ok for call in fake.calls)


def test_a_failed_compaction_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeAnthropic(policy=answering(), faults={0: "refusal"})
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    with caplog.at_level(logging.WARNING, logger="artiik"):
        ctx.prepare()
    [record] = caplog.records
    assert record.getMessage() == (
        "artiik compaction: AnthropicCompaction didn't compact before request 0: refusal"
    )


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (
            FakeAPIError.anthropic(
                529, "overloaded_error", "Compaction is unavailable.", code="compaction_unavailable"
            ),
            "unavailable",
        ),
        (FakeAPIError.anthropic(429, "rate_limit_error", "Too many requests."), "unavailable"),
        (FakeAPIError.anthropic(500, "api_error", "Internal server error."), "unavailable"),
        (
            FakeAPIError.anthropic(
                400,
                "invalid_request_error",
                "There is nothing to summarize.",
                code="compaction_nothing_to_summarize",
            ),
            "nothing to summarize",
        ),
    ],
)
def test_errors_compaction_can_wait_out_are_retried_later(
    error: FakeAPIError, outcome: str
) -> None:
    fake = FakeAnthropic(policy=answering(), faults={0: error})
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    run_turns(ctx, fake.messages.create, 5)
    assert outcomes(ctx) == [(0, outcome), (3, "compacted")]


@pytest.mark.parametrize(
    "error",
    [
        FakeAPIError.anthropic(
            400, "invalid_request_error", "Misplaced.", code="compaction_block_misplaced"
        ),
        FakeAPIError.anthropic(
            400, "invalid_request_error", "Bad signature.", code="compaction_signature_invalid"
        ),
        FakeAPIError.anthropic(
            400, "invalid_request_error", "Changed.", code="compaction_content_mismatch"
        ),
        FakeAPIError.anthropic(400, "invalid_request_error", "messages: Field required"),
        FakeAPIError.anthropic(401, "authentication_error", "invalid x-api-key"),
    ],
)
def test_errors_that_mean_a_wrong_request_are_raised(error: FakeAPIError) -> None:
    fake = FakeAnthropic(faults={0: error})
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    with pytest.raises(FakeAPIError) as caught:
        ctx.prepare()
    assert caught.value is error


def test_errors_from_outside_the_api_are_raised() -> None:
    strategy = AnthropicCompaction(object(), check_support=False)
    ctx = fill(compacting(ANTHROPIC, strategy))
    with pytest.raises(AttributeError):
        ctx.prepare()


def test_the_compaction_request_leaves_out_what_compaction_cant_take() -> None:
    fake = FakeAnthropic(policy=answering())
    params: dict[str, Any] = {
        "max_tokens": 1_024,
        "stop_sequences": ["END"],
        "tool_choice": {"type": "any"},
        "output_config": {
            "format": {"type": "json_schema", "schema": {"type": "object"}},
            "effort": "high",
        },
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
        "temperature": 0.5,
        "metadata": {"user_id": "u1"},
        "betas": ["context-management-2025-06-27"],
    }
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake), params=params))
    request = ctx.prepare()
    sent = fake.calls[0].request
    assert {"stop_sequences", "tool_choice", "context_management"}.isdisjoint(sent)
    assert sent["output_config"] == {"effort": "high"}
    assert sent["temperature"] == 0.5
    assert sent["metadata"] == {"user_id": "u1"}
    assert sent["betas"] == ["context-management-2025-06-27", ANTHROPIC_BETA]
    assert sent["max_tokens"] == 8_192
    # The conversation's own requests keep all of it.
    assert request["stop_sequences"] == ["END"]
    assert request["tool_choice"] == {"type": "any"}
    assert request["betas"] == ["context-management-2025-06-27", ANTHROPIC_BETA]
    fake.beta.messages.create(**request)


def test_an_automatic_tool_choice_stays_in_the_compaction_request() -> None:
    fake = FakeAnthropic(policy=answering())
    params: JSONObject = {"tool_choice": {"type": "auto"}, "output_config": {"format": {}}}
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake), params=params))
    ctx.prepare()
    assert fake.calls[0].request["tool_choice"] == {"type": "auto"}
    assert "output_config" not in fake.calls[0].request


def test_instructions_are_checked_and_sent() -> None:
    assert len(DEFAULT_INSTRUCTIONS) <= MAX_INSTRUCTIONS
    for bad in ["", "  \n", "x" * (MAX_INSTRUCTIONS + 1)]:
        with pytest.raises(ValueError, match="instructions"):
            AnthropicCompaction(FakeAnthropic(), instructions=bad)
        with pytest.raises(ValueError, match="instructions"):
            AnthropicThresholdCompaction(instructions=bad)
        with pytest.raises(ValueError, match="instructions"):
            SummaryCompaction(summarize, instructions=bad)
    fake = FakeAnthropic(policy=answering())
    strategy = AnthropicCompaction(fake, instructions="Keep every error code.", max_tokens=2_000)
    ctx = fill(compacting(ANTHROPIC, strategy))
    ctx.prepare()
    assert fake.calls[0].request["compaction"] == {
        "type": "summarize",
        "instructions": "Keep every error code.",
    }
    assert fake.calls[0].request["max_tokens"] == 2_000


def test_add_beta_follows_how_the_sdk_sends_betas() -> None:
    request: dict[str, Any] = {}
    add_beta(request, "a")
    add_beta(request, "a")
    assert request == {"extra_headers": {"anthropic-beta": "a"}}
    add_beta(request, "b")
    assert request == {"extra_headers": {"anthropic-beta": "a,b"}}
    listed: dict[str, Any] = {"betas": ("a",)}
    add_beta(listed, "b")
    assert listed == {"betas": ["a", "b"]}
    # An anthropic-beta header replaces the list in the SDK, so the beta goes in both.
    headers = {"Anthropic-Beta": "a", "X-Trace": "1"}
    both: dict[str, Any] = {"betas": ["a"], "extra_headers": headers}
    add_beta(both, "b")
    assert both == {"betas": ["a", "b"], "extra_headers": {"Anthropic-Beta": "a,b", "X-Trace": "1"}}
    assert headers == {"Anthropic-Beta": "a", "X-Trace": "1"}


def test_a_header_with_other_betas_gets_the_compaction_beta_too() -> None:
    fake = FakeAnthropic(policy=answering())
    params = {"betas": ["context-1m-2025-08-07"], "extra_headers": {"anthropic-beta": "other"}}
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake), params=params))
    request = ctx.prepare()
    assert fake.calls[0].request["extra_headers"] == {"anthropic-beta": f"other,{ANTHROPIC_BETA}"}
    assert request["extra_headers"] == {"anthropic-beta": f"other,{ANTHROPIC_BETA}"}
    fake.beta.messages.create(**request)


def test_the_block_gets_a_cache_breakpoint_when_one_is_free() -> None:
    cached: JSONObject = {"type": "ephemeral"}
    tool = expect_object(expect_list(default_request(ANTHROPIC)["tools"], "tools")[0], "tool")
    options: dict[str, Any] = {
        "system": [{"type": "text", "text": "You read logs.", "cache_control": cached}],
        "tools": [{**tool, "cache_control": cached}],
        "params": {"cache_control": cached},
    }
    # Three breakpoints so far: the tools, the system prompt and the top-level one.
    fake = FakeAnthropic(policy=answering())
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake), **options))
    request = ctx.prepare()
    assert request["messages"][0]["content"][0]["cache_control"] == cached
    fake.messages.create(**request)
    # A fourth one in the conversation leaves none for the block.
    fake = FakeAnthropic(policy=answering())
    current: JSONObject = {
        "role": "user",
        "content": [{"type": "text", "text": "Now list the errors.", "cache_control": cached}],
    }
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake), **options), current=current)
    request = ctx.prepare()
    assert "cache_control" not in request["messages"][0]["content"][0]
    fake.messages.create(**request)


def test_compacting_again_summarizes_the_last_summary_too() -> None:
    fake = FakeAnthropic(policy=answering())
    ctx = fill(compacting(ANTHROPIC, AnthropicCompaction(fake)))
    ctx.record(fake.messages.create(**ctx.prepare()))
    block = ctx.history[0].blocks[0]
    assert isinstance(block, Compaction)
    fill(ctx, first=4, current=user("Which error came first?"))
    ctx.record(fake.messages.create(**ctx.prepare()))
    first, second = compactions(fake)
    sent = expect_list(second.request["messages"], "messages")
    # The earlier block goes into the summary exactly as it came back.
    assert sent[0] == {"role": "assistant", "content": [block.data]}
    found = [
        candidate
        for message in ctx.history
        for candidate in message.blocks
        if isinstance(candidate, Compaction)
    ]
    assert len(found) == 1
    assert found[0] is ctx.history[0].blocks[0]
    assert found[0] != block
    assert first.ok and second.ok


def test_a_long_current_turn_keeps_its_latest_steps() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(steps=12))
    ctx = compacting(ANTHROPIC, AnthropicCompaction(fake))
    environment = Environment(output_tokens=900)
    run_context(ctx, fake.messages.create, ["Read every log."], environment=environment)
    [event] = ctx.trace.of("compaction")
    assert event.data["outcome"] == "compacted"
    request = fake.calls[-1].request
    first, second = expect_list(request["messages"], "messages")[:2]
    # The summary is followed by the steps kept word for word: two assistant messages in a row.
    assert expect_object(first, "first")["role"] == "assistant"
    assert expect_object(second, "second")["role"] == "assistant"
    assert ctx.pending_tool_calls() == []
    assert all(message.text != "Read every log." for message in ctx.history)
    assert all(call.ok for call in fake.calls)


def test_the_compaction_request_fits_the_budget() -> None:
    fake = FakeAnthropic(policy=answering())
    ctx = fill(
        context_for(ANTHROPIC, budget=10_000, compaction=AnthropicCompaction(fake)),
        chars=4_000,
        current=user("y" * 18_000),
    )
    for step in range(2):
        use: JSONObject = {"type": "tool_use", "id": f"toolu_{step}", "name": "read_log"}
        ctx.add(
            {"role": "assistant", "content": [{**use, "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"toolu_{step}", "content": "ok"}
                ],
            },
        )
    request = ctx.prepare()
    [call] = compactions(fake)
    assert call.tokens() <= 10_000
    # The long user message didn't fit in the summary, so it stays word for word.
    assert ctx.history[1].text == "y" * 18_000
    fake.messages.create(**request)


def test_exact_counts_carry_the_beta_header_a_block_needs() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(calls_per_step=2))
    ctx = context_for(
        ANTHROPIC,
        budget=6_000,
        compact_at=5_800,
        compaction=AnthropicCompaction(fake),
        counter=AnthropicTokenCounter(fake),
    )
    run_context(ctx, fake.messages.create, turns(30), environment=Environment(output_tokens=150))
    assert ctx.trace.of("compaction")
    assert any(
        request.get("extra_headers") == {"anthropic-beta": ANTHROPIC_BETA}
        for request in fake.count_requests
    )


# Anthropic compaction at a threshold


def test_threshold_compaction_asks_the_api_to_compact() -> None:
    ctx = context_for(ANTHROPIC, budget=200_000, compaction=AnthropicThresholdCompaction())
    ctx.add(user("Hi"))
    request = ctx.prepare()
    assert request["context_management"] == {
        "edits": [
            {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150_000}}
        ]
    }
    assert request["extra_headers"] == {"anthropic-beta": ANTHROPIC_THRESHOLD_BETA}
    FakeAnthropic().beta.messages.create(**request)
    # Only the beta endpoint takes context_management.
    with pytest.raises(TypeError, match="context_management"):
        FakeAnthropic().messages.create(**request)


def test_threshold_settings_merge_into_the_request() -> None:
    clear: JSONObject = {"type": "clear_tool_uses_20250919"}
    strategy = AnthropicThresholdCompaction(trigger=60_000, instructions="Keep the error codes.")
    params = {"betas": ["context-management-2025-06-27"], "context_management": {"edits": [clear]}}
    ctx = context_for(ANTHROPIC, budget=200_000, params=params, compaction=strategy)
    ctx.add(user("Hi"))
    request = ctx.prepare()
    edit: JSONObject = {
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": 60_000},
        "instructions": "Keep the error codes.",
    }
    assert request["context_management"] == {"edits": [clear, edit]}
    assert request["betas"] == ["context-management-2025-06-27", ANTHROPIC_THRESHOLD_BETA]
    assert ctx.params["context_management"] == {"edits": [clear]}
    FakeAnthropic().beta.messages.create(**request)


def test_threshold_compaction_needs_a_trigger_of_50_000_tokens() -> None:
    with pytest.raises(ValueError, match="at least 50000"):
        AnthropicThresholdCompaction(trigger=40_000)
    with pytest.raises(ValueError, match="at least 50000"):
        context_for(ANTHROPIC, budget=60_000, compaction=AnthropicThresholdCompaction())
    strategy = AnthropicThresholdCompaction(trigger=50_000)
    assert context_for(ANTHROPIC, budget=60_000, compaction=strategy).compact_at == 45_000
    assert context_for(ANTHROPIC, compaction=strategy).compact_at is None


def test_threshold_compaction_drops_what_the_block_replaced() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(calls_per_step=2))
    ctx = context_for(
        ANTHROPIC, budget=100_000, compact_at=60_000, compaction=AnthropicThresholdCompaction()
    )
    create, calls = tracked(ctx, fake, fake.beta.messages.create)
    run_context(ctx, create, turns(20), environment=Environment(output_tokens=2_000))
    events = ctx.trace.of("compaction")
    assert len(events) >= 2
    assert all(event.data["strategy"] == "provider" for event in events)
    for event in events[:-1]:
        # The next request starts from the block the reply opened with.
        following = fake.calls[calls[event.request + 1]].request
        first = expect_object(expect_list(following["messages"], "messages")[0], "first")
        block = expect_object(expect_list(first["content"], "content")[0], "block")
        assert block["type"] == "compaction"
        assert "signature" not in block
        assert cast(int, event.data["summarized_messages"]) > 0
    assert_holds(check_all(fake.calls, budget=100_000, allowed_rewrites=rewrites(ctx, calls)))


def test_threshold_compaction_cant_be_forced() -> None:
    ctx = context_for(ANTHROPIC, budget=200_000, compaction=AnthropicThresholdCompaction())
    with pytest.raises(ValueError, match="client side"):
        ctx.compact()


# OpenAI Responses compaction


def test_server_side_compaction_is_turned_on_in_every_request() -> None:
    ctx = context_for(RESPONSES, budget=10_000, compaction=OpenAICompaction())
    ctx.add(user("Hi"))
    assert ctx.prepare()["context_management"] == [
        {"type": "compaction", "compact_threshold": 7_500}
    ]
    ctx = context_for(RESPONSES, compaction=OpenAICompaction(threshold=5_000))
    ctx.add(user("Hi"))
    assert ctx.prepare()["context_management"] == [
        {"type": "compaction", "compact_threshold": 5_000}
    ]
    own: list[JSONValue] = [{"type": "compaction", "compact_threshold": 9_000}]
    ctx = context_for(
        RESPONSES,
        budget=10_000,
        params={"context_management": own},
        compaction=OpenAICompaction(),
    )
    ctx.add(user("Hi"))
    assert ctx.prepare()["context_management"] == own


def test_server_side_compaction_drops_the_input_it_replaced() -> None:
    fake = FakeOpenAI(policy=ToolLoopPolicy(calls_per_step=2))
    ctx = context_for(RESPONSES, budget=6_000, compaction=OpenAICompaction())
    create, calls = tracked(ctx, fake, fake.responses.create)
    run_context(ctx, create, turns(30), environment=Environment(output_tokens=150))
    events = ctx.trace.of("compaction")
    assert len(events) >= 2
    for event in events[:-1]:
        following = fake.calls[calls[event.request + 1]].request
        first = expect_object(expect_list(following["input"], "input")[0], "first")
        assert first["type"] == "compaction"
    assert_holds(check_all(fake.calls, budget=6_000, allowed_rewrites=rewrites(ctx, calls)))


def test_the_compact_endpoint_keeps_the_user_messages_and_its_item() -> None:
    fake = FakeOpenAI(policy=answering())
    ctx = fill(compacting(RESPONSES, OpenAICompaction(fake)))
    old = ctx.history[:-1]
    request = ctx.prepare()
    [call] = fake.calls
    assert call.endpoint == "responses.compact"
    assert call.request == {"model": MODEL, "input": openai_responses.dump_items(old)}
    output = expect_list((call.response or {})["output"], "output")
    kinds = [expect_object(item, "item")["type"] for item in output]
    assert kinds == ["message"] * 4 + ["compaction"]
    # The returned items go first, as they are, then the current turn.
    assert request["input"] == [*output, user("Now list the errors.")]
    ctx.record(fake.responses.create(**request))
    # The endpoint keeps user messages as they are, so they alone leave nothing to summarize.
    assert ctx.compact() is None
    assert len(compactions(fake)) == 1
    # Compacting again sends the whole window: the kept user messages and the item.
    fill(ctx, first=4, current=user("Which error came first?"))
    ctx.record(fake.responses.create(**ctx.prepare()))
    second = compactions(fake)[1]
    assert expect_list(second.request["input"], "input")[:5] == output
    users = [message.text for message in ctx.history if message.role == "user"]
    assert users == [
        *(f"Turn {number}: read the logs." for number in range(4)),
        "Now list the errors.",
        *(f"Turn {number}: read the logs." for number in range(4, 8)),
        "Which error came first?",
    ]
    found = [
        index
        for index, message in enumerate(ctx.history)
        if message.blocks and isinstance(message.blocks[0], Compaction)
    ]
    assert found == [9]


def test_the_compact_endpoint_needs_a_user_message() -> None:
    fake = FakeOpenAI(policy=answering())
    ctx = compacting(RESPONSES, OpenAICompaction(fake))
    ctx.add({"role": "developer", "content": "Report errors only."})
    for number in range(4):
        ctx.add({"role": "assistant", "content": f"Log {number}: " + "x" * 8_000})
    ctx.add(user("Now list the errors."))
    request = ctx.prepare()
    assert fake.calls == []
    assert outcomes(ctx) == [(0, "no user message")]
    fake.responses.create(**request)


def test_compact_endpoint_errors() -> None:
    unavailable = FakeAPIError.openai(500, "The server had an error.", error_type="server_error")
    fake = FakeOpenAI(policy=answering(), faults={0: unavailable})
    ctx = fill(compacting(RESPONSES, OpenAICompaction(fake)))
    run_turns(ctx, fake.responses.create, 5)
    assert outcomes(ctx) == [(0, "unavailable"), (3, "compacted")]
    wrong = FakeAPIError.openai(400, "Unknown parameter.", code="unknown_parameter")
    fake = FakeOpenAI(policy=answering(), faults={0: wrong})
    ctx = fill(compacting(RESPONSES, OpenAICompaction(fake)))
    with pytest.raises(FakeAPIError, match="Unknown parameter"):
        ctx.prepare()


def test_server_side_compaction_cant_be_forced() -> None:
    ctx = context_for(RESPONSES, budget=10_000, compaction=OpenAICompaction())
    with pytest.raises(ValueError, match="client side"):
        ctx.compact()


# A summarize callable


def dumped(fmt: Format, messages: tuple[Message, ...]) -> list[JSONObject]:
    match fmt:
        case Format.ANTHROPIC_MESSAGES:
            return anthropic_messages.dump_messages(messages)
        case Format.OPENAI_RESPONSES:
            return openai_responses.dump_items(messages)
        case Format.OPENAI_CHAT:
            return openai_chat.dump_messages(messages)


def fake_for(fmt: Format) -> tuple[FakeAnthropic | FakeOpenAI, Create]:
    if fmt is ANTHROPIC:
        anthropic = FakeAnthropic(policy=answering())
        return anthropic, anthropic.messages.create
    openai = FakeOpenAI(policy=answering())
    create = openai.responses.create if fmt is RESPONSES else openai.chat.completions.create
    return openai, create


@pytest.mark.parametrize("fmt", list(Format))
def test_a_summarize_callable_works_with_any_api(fmt: Format) -> None:
    seen: list[list[JSONObject]] = []

    def summarize_logs(messages: list[JSONObject]) -> str:
        seen.append(messages)
        return " The logs had two errors. "

    fake, create = fake_for(fmt)
    strategy = SummaryCompaction(summarize_logs)
    ctx = fill(compacting(fmt, strategy, system="You read logs."))
    old = ctx.history[:-1]
    request = ctx.prepare()
    # The part to summarize, then the request for the summary.
    [messages] = seen
    assert messages == dumped(fmt, (*old, Message.from_text("user", DEFAULT_INSTRUCTIONS)))
    summary = "Summary of the conversation so far:\n\nThe logs had two errors."
    assert [message.text for message in ctx.history] == [summary, "Now list the errors."]
    [event] = ctx.trace.of("compaction")
    assert event.data["strategy"] == "SummaryCompaction"
    create(**request)
    assert fake.calls[0].ok


def test_a_summary_can_have_its_own_label_and_instructions() -> None:
    seen: list[JSONObject] = []

    def summarize_logs(messages: list[JSONObject]) -> str:
        seen.append(messages[-1])
        return "Two errors."

    strategy = SummaryCompaction(summarize_logs, instructions="Keep the errors.", label="Earlier:")
    ctx = fill(compacting(CHAT, strategy))
    ctx.prepare()
    assert seen == openai_chat.dump_messages([Message.from_text("user", "Keep the errors.")])
    assert ctx.history[0].text == "Earlier:\n\nTwo errors."


def test_an_empty_summary_compacts_nothing() -> None:
    strategy = SummaryCompaction(lambda messages: "  \n")
    ctx = fill(compacting(CHAT, strategy))
    before = ctx.history
    ctx.prepare()
    assert ctx.history == before
    assert outcomes(ctx) == [(0, "empty summary")]


def test_instructions_in_the_summarized_part_are_kept() -> None:
    ctx = compacting(CHAT, SummaryCompaction(summarize))
    ctx.add({"role": "system", "content": "Answer in French."})
    fill(ctx)
    ctx.prepare()
    # The summary covers the instruction too, and the instruction itself stays.
    assert [(message.role, message.text) for message in ctx.history] == [
        ("user", "Summary of the conversation so far:\n\n9 messages about logs."),
        ("user", "Now list the errors."),
        ("system", "Answer in French."),
    ]
    [event] = ctx.trace.of("compaction")
    assert event.data["moved_messages"] == 1
    assert event.data["kept_messages"] == 1


# The context


def test_compact_compacts_now() -> None:
    fake = FakeAnthropic(policy=answering())
    ctx = fill(
        context_for(ANTHROPIC, budget=100_000, compaction=AnthropicCompaction(fake)), chars=500
    )
    ctx.record(fake.messages.create(**ctx.prepare()))
    assert compactions(fake) == []
    result = ctx.compact()
    assert result is not None
    assert result.outcome == "compacted"
    assert isinstance(ctx.history[0].blocks[0], Compaction)
    assert [message.text for message in ctx.history[1:]] == ["Now list the errors.", "Done."]
    fresh = context_for(ANTHROPIC, compact_at=1_000, compaction=AnthropicCompaction(fake))
    fresh.add(user("Hi"))
    assert fresh.compact() is None
    assert outcomes(fresh) == [(0, "nothing to summarize")]


def test_compact_needs_a_strategy() -> None:
    ctx = context_for(ANTHROPIC, budget=10_000)
    with pytest.raises(ValueError, match="client side"):
        ctx.compact()


def test_compaction_settings_are_checked() -> None:
    with pytest.raises(ValueError, match="doesn't work with the openai-chat API"):
        context_for(CHAT, budget=10_000, compaction=AnthropicCompaction(FakeAnthropic()))
    with pytest.raises(ValueError, match="doesn't work with the anthropic-messages API"):
        context_for(ANTHROPIC, budget=10_000, compaction=OpenAICompaction())
    with pytest.raises(ValueError, match="can't be above the budget"):
        context_for(ANTHROPIC, budget=10_000, compact_at=20_000)
    with pytest.raises(ValueError, match="positive"):
        context_for(ANTHROPIC, compact_at=0, compaction=SummaryCompaction(summarize))
    with pytest.raises(ValueError, match="needs compact_at"):
        context_for(ANTHROPIC, compaction=SummaryCompaction(summarize))
    with pytest.raises(ValueError, match="needs compact_at"):
        context_for(RESPONSES, compaction=OpenAICompaction())
    strategy = SummaryCompaction(summarize)
    assert context_for(ANTHROPIC, budget=10_000, compaction=strategy).compact_at == 7_500
    assert context_for(CHAT, compact_at=3_000, compaction=strategy).compact_at == 3_000


class Broken(Compactor):
    """Returns a summary that leaves a tool call without its result."""

    def compact(self, job: CompactionJob) -> CompactionResult:
        call = ToolUse(id="toolu_1", name="read_log", input={})
        return CompactionResult("compacted", messages=(Message(role="assistant", blocks=(call,)),))


def test_a_result_that_would_break_the_conversation_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = fill(compacting(ANTHROPIC, Broken()))
    before = ctx.history
    with caplog.at_level(logging.ERROR, logger="artiik"):
        ctx.prepare()
        ctx.prepare()
    assert ctx.history == before
    [event] = ctx.trace.of("compaction")
    assert event.data["outcome"] == "invalid result"
    assert "tool calls without a tool_result" in caplog.records[0].getMessage()


def test_a_request_still_too_big_after_compacting_waits_before_trying_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeAnthropic(policy=answering())
    ctx = fill(
        compacting(ANTHROPIC, AnthropicCompaction(fake)),
        current=user("y" * 36_000),
    )
    with caplog.at_level(logging.WARNING, logger="artiik"):
        run_turns(ctx, fake.messages.create, 4)
    assert outcomes(ctx) == [(0, "compacted"), (3, "compacted")]
    assert "still above compact_at" in caplog.records[0].getMessage()
    assert all(call.ok for call in fake.calls)


# Tier 0: long sessions with every strategy


STRATEGIES = [
    "anthropic on demand",
    "anthropic threshold",
    "openai server",
    "openai endpoint",
    "summary",
]


def session(name: str) -> tuple[FakeAnthropic | FakeOpenAI, Context, Create, int]:
    """A fake, a context with the strategy, the endpoint to call and the tool output size."""
    policy = ToolLoopPolicy(calls_per_step=2)
    match name:
        case "anthropic on demand":
            anthropic = FakeAnthropic(policy=policy)
            ctx = context_for(ANTHROPIC, budget=6_000, compaction=AnthropicCompaction(anthropic))
            return anthropic, ctx, anthropic.messages.create, 150
        case "anthropic threshold":
            anthropic = FakeAnthropic(policy=policy)
            ctx = context_for(
                ANTHROPIC,
                budget=100_000,
                compact_at=60_000,
                compaction=AnthropicThresholdCompaction(),
            )
            return anthropic, ctx, anthropic.beta.messages.create, 2_000
        case "openai server":
            openai = FakeOpenAI(policy=policy)
            ctx = context_for(RESPONSES, budget=6_000, compaction=OpenAICompaction())
            return openai, ctx, openai.responses.create, 150
        case "openai endpoint":
            openai = FakeOpenAI(policy=policy)
            ctx = context_for(RESPONSES, budget=6_000, compaction=OpenAICompaction(openai))
            return openai, ctx, openai.responses.create, 150
        case _:
            openai = FakeOpenAI(policy=policy)
            ctx = context_for(CHAT, budget=6_000, compaction=SummaryCompaction(summarize))
            return openai, ctx, openai.chat.completions.create, 150


@pytest.mark.parametrize("name", STRATEGIES)
def test_long_sessions_keep_every_invariant(name: str) -> None:
    fake, ctx, create, output_tokens = session(name)
    create, calls = tracked(ctx, fake, create)
    run_context(ctx, create, turns(120), environment=Environment(output_tokens=output_tokens))
    events = ctx.trace.of("compaction")
    assert len(events) >= 10
    assert all(event.data["outcome"] == "compacted" for event in events)
    assert all(call.ok for call in fake.calls)
    budget = cast(int, ctx.budget)
    assert_holds(check_all(fake.calls, budget=budget, allowed_rewrites=rewrites(ctx, calls)))
    if isinstance(fake, FakeAnthropic) and name == "anthropic on demand":
        assert fake.model_requests == [MODEL]
