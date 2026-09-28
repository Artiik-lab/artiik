"""The Context, integration level 2, end to end with the fake clients."""

import dataclasses
import logging
import math
from collections.abc import Sequence
from typing import Any, cast

import pytest

from artiik import (
    AnthropicTokenCounter,
    BudgetError,
    Context,
    Estimator,
    Format,
    Trace,
    TraceEvent,
    ValidationError,
)
from artiik.context import SAFETY
from artiik.messages import JSONObject, JSONValue, Message
from artiik.testing import (
    Environment,
    FakeAnthropic,
    FakeObject,
    FakeOpenAI,
    Policy,
    Reply,
    ScriptedPolicy,
    Tokenizer,
    ToolCall,
    ToolLoopPolicy,
    assert_holds,
    check_append_only,
    check_budget,
    check_kept_whole,
    check_tool_pairs,
    default_request,
    run_context,
)
from artiik.testing.driver import Create
from artiik.tokens import tally_request

ANTHROPIC = Format.ANTHROPIC_MESSAGES
RESPONSES = Format.OPENAI_RESPONSES
CHAT = Format.OPENAI_CHAT
MODEL = "fake-model"
TOOLS = cast(list[JSONValue], default_request(ANTHROPIC)["tools"])


def fake_for(
    fmt: Format, policy: Policy | None = None, tokenizer: Tokenizer | None = None
) -> tuple[FakeAnthropic | FakeOpenAI, Create]:
    if fmt is ANTHROPIC:
        anthropic = FakeAnthropic(policy=policy, tokenizer=tokenizer)
        return anthropic, anthropic.messages.create
    openai = FakeOpenAI(policy=policy, tokenizer=tokenizer)
    create = openai.responses.create if fmt is RESPONSES else openai.chat.completions.create
    return openai, create


def context_for(fmt: Format, **options: Any) -> Context:
    base = default_request(fmt)
    params = {key: value for key, value in base.items() if key not in ("model", "tools")}
    return Context(
        fmt, model=MODEL, tools=cast(list[JSONValue], base["tools"]), params=params, **options
    )


def turns(count: int) -> list[str]:
    return [f"Turn {index}: check the next log." for index in range(count)]


def user(text: str) -> JSONObject:
    return {"role": "user", "content": text}


# Building requests


def test_prepare_builds_an_anthropic_request() -> None:
    ctx = Context(
        ANTHROPIC,
        model=MODEL,
        system="Be brief.",
        tools=TOOLS,
        params={"max_tokens": 100, "temperature": 0},
    )
    ctx.add(user("Hi"))
    assert ctx.prepare(temperature=0.5) == {
        "model": MODEL,
        "system": "Be brief.",
        "tools": TOOLS,
        "messages": [user("Hi")],
        "max_tokens": 100,
        "temperature": 0.5,
    }


def test_prepare_builds_a_responses_request() -> None:
    ctx = Context(RESPONSES, model=MODEL, system="Be brief.", params={"store": False})
    ctx.add(user("Hi"))
    assert ctx.prepare(model="other-model") == {
        "model": "other-model",
        "instructions": "Be brief.",
        "input": [user("Hi")],
        "store": False,
    }


def test_prepare_builds_a_chat_request_with_the_system_prompt_first() -> None:
    ctx = Context(CHAT, model=MODEL, system="Be brief.", tools=TOOLS)
    ctx.add(user("Hi"))
    assert ctx.prepare() == {
        "model": MODEL,
        "tools": TOOLS,
        "messages": [{"role": "system", "content": "Be brief."}, user("Hi")],
    }


def test_prepared_requests_are_copies() -> None:
    ctx = Context(ANTHROPIC, model=MODEL, system=[{"type": "text", "text": "Be brief."}])
    ctx.add(user("Hi"))
    request = ctx.prepare(max_tokens=10)
    request["messages"].append(user("Injected"))
    request["system"][0]["text"] = "Changed."
    assert ctx.prepare(max_tokens=10)["messages"] == [user("Hi")]
    assert ctx.prepare(max_tokens=10)["system"] == [{"type": "text", "text": "Be brief."}]


def test_add_takes_provider_data_sdk_objects_messages_and_lists() -> None:
    ctx = Context(RESPONSES, model=MODEL)
    ctx.add(
        FakeObject(user("First")),
        [
            user("Second"),
            {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{}"},
        ],
        Message.from_text("user", "Third"),
    )
    assert [message.text for message in ctx.history] == ["First", "Second", "", "Third"]
    assert [use.id for use in ctx.pending_tool_calls()] == ["c1"]
    ctx.add({"type": "function_call_output", "call_id": "c1", "output": "ok"})
    assert ctx.pending_tool_calls() == []


@pytest.mark.parametrize(
    ("api", "params", "expected"),
    [
        pytest.param(ANTHROPIC, {"messages": []}, "add messages with add", id="messages"),
        pytest.param(RESPONSES, {"input": []}, "not input=", id="input"),
        pytest.param(RESPONSES, {"previous_response_id": "r"}, "whole conversation", id="previous"),
    ],
)
def test_the_history_isnt_a_request_parameter(
    api: Format, params: dict[str, Any], expected: str
) -> None:
    ctx = Context(api, model=MODEL)
    ctx.add(user("Hi"))
    with pytest.raises(TypeError, match=expected):
        ctx.prepare(**params)
    with pytest.raises(TypeError, match=expected):
        Context(api, model=MODEL, params=params)


def test_context_arguments_are_checked() -> None:
    with pytest.raises(
        ValueError, match="unknown api 'completions'; use one of: anthropic-messages"
    ):
        Context("completions")
    with pytest.raises(ValueError, match="budget"):
        Context(ANTHROPIC, budget=0)
    with pytest.raises(ValueError, match="trim_to"):
        Context(ANTHROPIC, trim_to=1.5)
    with pytest.raises(TypeError, match="instructions"):
        Context(RESPONSES, system=[{"type": "input_text", "text": "Be brief."}])
    with pytest.raises(TypeError, match="tools must be a list"):
        Context(ANTHROPIC, tools=cast(Sequence[object], "ls"))
    ctx = Context(ANTHROPIC)
    ctx.add(user("Hi"))
    with pytest.raises(TypeError, match="needs a model"):
        ctx.prepare()
    with pytest.raises(TypeError, match="needs a model"):
        ctx.estimate()


def test_an_invalid_conversation_is_never_sent() -> None:
    ctx = Context(ANTHROPIC, model=MODEL)
    ctx.add(user("Go"))
    call: JSONObject = {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "."}}
    ctx.add({"role": "assistant", "content": [call]})
    with pytest.raises(ValidationError) as caught:
        ctx.prepare(max_tokens=10)
    assert caught.value.problems == (
        "messages[1]: tool calls without a tool_result in the next message: t1",
    )
    empty = Context(CHAT, model=MODEL)
    with pytest.raises(ValidationError, match="empty"):
        empty.prepare()


# Recording responses


@pytest.mark.parametrize("fmt", [ANTHROPIC, RESPONSES])
def test_signed_blocks_are_recorded_exactly_as_returned(fmt: Format) -> None:
    class Thinker:
        def __init__(self) -> None:
            self.base = ToolLoopPolicy()

        def reply(self, conversation: Sequence[Message]) -> Reply:
            reply = self.base.reply(conversation)
            return dataclasses.replace(reply, thinking=f"Step {len(conversation)}.")

    fake, create = fake_for(fmt, Thinker())
    ctx = context_for(fmt)
    run_context(ctx, create, turns(3))
    assert all(call.ok for call in fake.calls)
    assert check_kept_whole(fake.calls) == []


def test_record_returns_the_usage_and_traces_both_calls() -> None:
    events: list[TraceEvent] = []
    fake = FakeAnthropic(policy=ToolLoopPolicy(steps=0))
    ctx = context_for(ANTHROPIC, trace=Trace(sink=events.append))
    ctx.add(user("Hi"))
    usage = ctx.record(fake.messages.create(**ctx.prepare()))
    assert usage.input_tokens == fake.calls[0].tokens()
    assert ctx.last_usage == usage
    assert [(event.kind, event.request) for event in events] == [("prepare", 0), ("record", 0)]
    assert events[1].data["stop_reason"] == "end_turn"
    assert events[1].data["input_tokens"] == usage.input_tokens
    assert ctx.trace.events == events
    assert [message.text for message in ctx.history] == ["Hi", "Done."]


def test_empty_replies_and_dangling_reasoning_arent_recorded() -> None:
    fake = FakeAnthropic(policy=ScriptedPolicy([Reply(stop_reason="max_tokens")]))
    ctx = context_for(ANTHROPIC)
    ctx.add(user("Hi"))
    ctx.record(fake.messages.create(**ctx.prepare()))
    assert len(ctx.history) == 1
    openai = FakeOpenAI(
        policy=ScriptedPolicy([Reply(thinking="Hmm.")]), faults={0: "max_output_tokens"}
    )
    responses = context_for(RESPONSES)
    responses.add(user("Hi"))
    responses.record(openai.responses.create(**responses.prepare()))
    assert len(responses.history) == 1
    responses.add(user("Again"))
    responses.prepare()


def test_parallel_tool_calls_are_pending_until_answered() -> None:
    calls = (ToolCall(name="ls"), ToolCall(name="cat"))
    fake = FakeAnthropic(policy=ScriptedPolicy([Reply(tool_calls=calls)]))
    ctx = context_for(ANTHROPIC)
    ctx.add(user("Go"))
    ctx.record(fake.messages.create(**ctx.prepare()))
    pending = ctx.pending_tool_calls()
    assert [call.name for call in pending] == ["ls", "cat"]
    ctx.add(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call.id, "content": "ok"} for call in pending
            ],
        }
    )
    assert ctx.pending_tool_calls() == []


# Estimates


@pytest.mark.parametrize("fmt", list(Format))
def test_estimates_build_on_the_last_usage(fmt: Format) -> None:
    fake, create = fake_for(fmt, tokenizer=Tokenizer(chars_per_token=3))
    ctx = context_for(fmt)
    run_context(ctx, create, turns(2), environment=Environment(output_tokens=500))
    ctx.add(user("One more turn."))
    estimate = ctx.estimate()
    ctx.record(create(**ctx.prepare()))
    actual = fake.calls[-1].tokens()
    assert actual <= estimate <= actual * SAFETY + 5


def test_estimates_trust_the_provider_count_over_their_own_guess() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(steps=0))
    ctx = context_for(ANTHROPIC)
    image: JSONObject = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
    ctx.add({"role": "user", "content": [image, {"type": "text", "text": "What is this?"}]})
    guess = ctx.estimate()
    ctx.record(fake.messages.create(**ctx.prepare()))
    counted = fake.calls[-1].tokens()
    assert guess - counted > 500
    ctx.add(user("And this?"))
    estimate = ctx.estimate()
    ctx.record(fake.messages.create(**ctx.prepare()))
    assert fake.calls[-1].tokens() <= estimate <= fake.calls[-1].tokens() + 10


def test_the_first_estimate_is_the_calibrated_full_estimate() -> None:
    ctx = context_for(CHAT, system="Be brief.")
    ctx.add(user("Hi"))
    request = ctx.prepare()
    [prepare] = ctx.trace.of("prepare")
    raw = ctx.estimator.estimate(tally_request(CHAT, request), api=CHAT, model=MODEL)
    assert prepare.data["estimated_tokens"] == math.ceil(raw * SAFETY)


# The guard


@pytest.mark.parametrize("fmt", list(Format))
def test_the_guard_keeps_requests_under_the_budget_and_traces_every_action(fmt: Format) -> None:
    budget = 3_000
    fake, create = fake_for(fmt, ToolLoopPolicy(calls_per_step=2))
    ctx = context_for(fmt, budget=budget)
    run_context(ctx, create, turns(12), environment=Environment(output_tokens=150))
    guards = ctx.trace.of("guard")
    assert guards
    for event in guards:
        assert event.data["fits"] is True
        assert cast(int, event.data["dropped_turns"]) >= 1
        assert cast(int, event.data["tokens_before"]) > budget
        assert cast(int, event.data["tokens_after"]) <= budget * ctx.trim_to
    rewrites = [violation.call for violation in check_append_only(fake.calls)]
    assert rewrites == [event.request for event in guards]
    assert_holds(check_budget(fake.calls, budget) + check_tool_pairs(fake.calls))


def test_the_guard_logs_what_it_drops(caplog: pytest.LogCaptureFixture) -> None:
    fake, create = fake_for(ANTHROPIC)
    ctx = context_for(ANTHROPIC, budget=2_000)
    with caplog.at_level(logging.WARNING, logger="artiik"):
        run_context(ctx, create, turns(8), environment=Environment(output_tokens=150))
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == len(ctx.trace.of("guard"))
    assert messages[0].startswith("artiik guard: request ")
    assert "dropped 1 turns" in messages[0] or "dropped 2 turns" in messages[0]
    assert fake.calls


def test_a_turn_too_big_for_the_budget_raises_budget_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = context_for(ANTHROPIC, budget=500)
    ctx.add(user("x" * 4_000))
    with caplog.at_level(logging.ERROR, logger="artiik"), pytest.raises(BudgetError) as caught:
        ctx.prepare()
    assert caught.value.budget == 500
    assert caught.value.needed > 500
    [event] = ctx.trace.of("guard")
    assert event.data["fits"] is False
    assert "can't fit its budget" in caplog.records[0].getMessage()


def test_exact_counts_take_over_near_the_budget() -> None:
    budget = 3_000
    fake = FakeAnthropic(policy=ToolLoopPolicy(), tokenizer=Tokenizer(chars_per_token=2.5))
    ctx = context_for(ANTHROPIC, budget=budget, counter=AnthropicTokenCounter(fake))
    run_context(ctx, fake.messages.create, turns(10), environment=Environment(output_tokens=150))
    assert fake.count_requests
    assert ctx.trace.of("guard")
    assert_holds(check_budget(fake.calls, budget) + check_tool_pairs(fake.calls))


def test_a_shared_estimator_carries_calibration_to_new_contexts() -> None:
    estimator = Estimator()
    fake = FakeAnthropic(tokenizer=Tokenizer(chars_per_token=2.5))
    run_context(context_for(ANTHROPIC, estimator=estimator), fake.messages.create, turns(2))
    fresh = context_for(ANTHROPIC, estimator=estimator)
    assert fresh.estimator.ratio(ANTHROPIC, MODEL) == pytest.approx(4 / 2.5, rel=0.05)


# The acceptance session


@pytest.mark.parametrize("fmt", list(Format))
def test_a_500_turn_session_stays_valid_and_within_the_budget(fmt: Format) -> None:
    budget = 3_000
    fake, create = fake_for(fmt, ToolLoopPolicy(calls_per_step=2), Tokenizer(chars_per_token=3.2))
    ctx = context_for(fmt, budget=budget, system="You read logs and report problems.")
    run_context(ctx, create, turns(500), environment=Environment(output_tokens=60))
    assert len(fake.calls) == 1_500
    assert all(call.ok for call in fake.calls)
    guards = [event.request for event in ctx.trace.of("guard")]
    assert len(guards) >= 50
    assert_holds(
        check_budget(fake.calls, budget)
        + check_tool_pairs(fake.calls)
        + check_kept_whole(fake.calls)
        + check_append_only(fake.calls, allowed=guards)
    )
