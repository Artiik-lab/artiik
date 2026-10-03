"""Live smoke tests: each compaction strategy, pins and clearing, against the real provider API.

Run them by hand after changing provider code (see CONTRIBUTING.md); they cost
a few cents. A test skips unless ``ARTIIK_LIVE=1`` and its provider's settings
are in the environment:

- Anthropic: ``ANTHROPIC_API_KEY``, and ``ARTIIK_ANTHROPIC_MODEL``, a model
  with compaction on demand;
- OpenAI: ``OPENAI_API_KEY``, and ``ARTIIK_OPENAI_MODEL``, a model with
  Responses compaction.

``ARTIIK_LIVE_HEAVY=1`` also lets Anthropic compact at its threshold for real,
which sends about 60,000 input tokens.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from artiik import (
    FETCH_TOOL,
    AnthropicClearing,
    AnthropicCompaction,
    AnthropicThresholdCompaction,
    Clearing,
    Compaction,
    Context,
    Format,
    MemoryStore,
    OpenAICompaction,
    SummaryCompaction,
)
from artiik.clearing import answer

SYSTEM = "You help ship releases. Answer in one short sentence."
LOGS = "".join(f"{index:05d} deploy-worker: health check passed\n" for index in range(300))
FACTS = (
    "Keep these for later: the release is on 14 October, the deploy key is at "
    "vault path kv/deploy, and staging runs on stg-3. Here is today's worker log:\n" + LOGS
)


def setting(*names: str) -> list[str]:
    """The values of the environment variables, or a skip when the test can't run."""
    if os.environ.get("ARTIIK_LIVE") != "1":
        pytest.skip("live tests run only with ARTIIK_LIVE=1")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip("set " + ", ".join(missing))
    return [os.environ[name] for name in names]


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def talk(ctx: Context, create: Any, *questions: str) -> None:
    """Ask each question in turn, recording the answers."""
    for question in questions:
        ctx.add(user(question))
        ctx.record(create(**ctx.prepare()))


def answered(ctx: Context) -> str:
    last = ctx.history[-1]
    assert last.role == "assistant", last
    return last.text


# Anthropic


def test_anthropic_compaction_on_demand() -> None:
    anthropic = pytest.importorskip("anthropic")
    _, model = setting("ANTHROPIC_API_KEY", "ARTIIK_ANTHROPIC_MODEL")
    client = anthropic.Anthropic()
    strategy = AnthropicCompaction(client)
    if not strategy.supports(model):
        pytest.skip(f"{model} doesn't support compaction on demand")
    ctx = Context(
        "anthropic-messages",
        model=model,
        system=SYSTEM,
        budget=100_000,
        compaction=strategy,
        params={"max_tokens": 300},
    )
    talk(ctx, client.messages.create, FACTS, "Is staging healthy?")
    ctx.add(user("When is the release?"))
    result = ctx.compact()
    assert result is not None and result.outcome == "compacted", result
    assert isinstance(ctx.history[0].blocks[0], Compaction)
    # The block goes first on the next request, with the beta header.
    ctx.record(client.messages.create(**ctx.prepare()))
    print("after compacting:", answered(ctx))
    # Compacting again, on a conversation that starts with a block.
    ctx.add(user("Where is the deploy key?"))
    result = ctx.compact()
    assert result is not None and result.outcome == "compacted", result
    ctx.record(client.messages.create(**ctx.prepare()))
    print("after compacting again:", answered(ctx))


def test_anthropic_threshold_compaction_is_accepted() -> None:
    anthropic = pytest.importorskip("anthropic")
    _, model = setting("ANTHROPIC_API_KEY", "ARTIIK_ANTHROPIC_MODEL")
    client = anthropic.Anthropic()
    ctx = Context(
        "anthropic-messages",
        model=model,
        system=SYSTEM,
        compaction=AnthropicThresholdCompaction(trigger=50_000),
        params={"max_tokens": 300},
    )
    talk(ctx, client.beta.messages.create, "Say hello.")
    answered(ctx)
    if os.environ.get("ARTIIK_LIVE_HEAVY") != "1":
        return
    # Past the trigger, the API compacts inside the request.
    talk(ctx, client.beta.messages.create, FACTS + LOGS * 20 + "\nIs staging healthy?")
    assert any(event.data["strategy"] == "provider" for event in ctx.trace.of("compaction"))
    assert isinstance(ctx.history[0].blocks[0], Compaction)
    talk(ctx, client.beta.messages.create, "When is the release?")
    print("after compacting:", answered(ctx))


def test_anthropic_pins() -> None:
    anthropic = pytest.importorskip("anthropic")
    _, model = setting("ANTHROPIC_API_KEY", "ARTIIK_ANTHROPIC_MODEL")
    client = anthropic.Anthropic()
    strategy = AnthropicCompaction(client)
    if not strategy.supports(model):
        pytest.skip(f"{model} doesn't support compaction on demand")
    ctx = Context(
        "anthropic-messages",
        model=model,
        system=SYSTEM,
        compact_at=100_000,
        compaction=strategy,
        params={"max_tokens": 300},
    )
    # Before the first request, a pin goes in the system prompt; after it, in a system message.
    ctx.pin("The release is on 14 October.", kind="fact")
    talk(ctx, client.messages.create, FACTS)
    ctx.pin("Answer in French.")
    talk(ctx, client.messages.create, "Is staging healthy?")
    print("with a pin message:", answered(ctx))
    ctx.add(user("When is the release?"))
    result = ctx.compact()
    assert result is not None and result.outcome == "compacted", result
    # The restatement goes in a system message after the user turn.
    ctx.record(client.messages.create(**ctx.prepare()))
    print("after the restatement:", answered(ctx))


# OpenAI


def test_openai_server_side_compaction() -> None:
    openai = pytest.importorskip("openai")
    _, model = setting("OPENAI_API_KEY", "ARTIIK_OPENAI_MODEL")
    client = openai.OpenAI()
    ctx = Context(
        "openai-responses",
        model=model,
        system=SYSTEM,
        compaction=OpenAICompaction(threshold=2_000),
        params={"max_output_tokens": 300},
    )
    talk(ctx, client.responses.create, FACTS + "\nIs staging healthy?")
    events = ctx.trace.of("compaction")
    assert events, "the server didn't compact: check compact_threshold"
    talk(ctx, client.responses.create, "When is the release?")
    print("after compacting:", answered(ctx))


def test_openai_compact_endpoint() -> None:
    openai = pytest.importorskip("openai")
    _, model = setting("OPENAI_API_KEY", "ARTIIK_OPENAI_MODEL")
    client = openai.OpenAI()
    ctx = Context(
        "openai-responses",
        model=model,
        system=SYSTEM,
        compact_at=100_000,
        compaction=OpenAICompaction(client),
        params={"max_output_tokens": 300},
    )
    talk(ctx, client.responses.create, FACTS, "Is staging healthy?")
    ctx.add(user("When is the release?"))
    result = ctx.compact()
    assert result is not None and result.outcome == "compacted", result
    # The user messages come back word for word, then one compaction item.
    assert isinstance(ctx.history[-2].blocks[0], Compaction)
    ctx.record(client.responses.create(**ctx.prepare()))
    print("after compacting:", answered(ctx))
    ctx.add(user("Where is the deploy key?"))
    result = ctx.compact()
    assert result is not None and result.outcome == "compacted", result
    ctx.record(client.responses.create(**ctx.prepare()))
    print("after compacting again:", answered(ctx))


def test_openai_pins() -> None:
    openai = pytest.importorskip("openai")
    _, model = setting("OPENAI_API_KEY", "ARTIIK_OPENAI_MODEL")
    client = openai.OpenAI()
    ctx = Context(
        "openai-responses",
        model=model,
        system=SYSTEM,
        compact_at=100_000,
        compaction=OpenAICompaction(client),
        params={"max_output_tokens": 300},
    )
    ctx.pin("The release is on 14 October.", kind="fact")
    talk(ctx, client.responses.create, FACTS)
    ctx.pin("Answer in French.")
    talk(ctx, client.responses.create, "Is staging healthy?")
    ctx.add(user("When is the release?"))
    result = ctx.compact()
    assert result is not None and result.outcome == "compacted", result
    ctx.record(client.responses.create(**ctx.prepare()))
    print("after the restatement:", answered(ctx))


def test_chat_completions_with_a_summarize_callable() -> None:
    openai = pytest.importorskip("openai")
    _, model = setting("OPENAI_API_KEY", "ARTIIK_OPENAI_MODEL")
    client = openai.OpenAI()

    def summarize(messages: list[dict[str, Any]]) -> str:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    ctx = Context(
        "openai-chat",
        model=model,
        system=SYSTEM,
        compact_at=100_000,
        compaction=SummaryCompaction(summarize),
    )
    talk(ctx, client.chat.completions.create, FACTS, "Is staging healthy?")
    ctx.add(user("When is the release?"))
    result = ctx.compact()
    assert result is not None and result.outcome == "compacted", result
    print("summary:", result.summary)
    ctx.record(client.chat.completions.create(**ctx.prepare()))
    print("after compacting:", answered(ctx))


# Clearing


ROUNDS_QUESTION = "Read the logs of workers 1, 2 and 3, then tell me which one failed."
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"worker": {"type": "integer"}},
    "required": ["worker"],
    "additionalProperties": False,
}
READ_LOG: dict[str, dict[str, Any]] = {
    "anthropic-messages": {
        "name": "read_log",
        "description": "Read a worker's log.",
        "input_schema": SCHEMA,
    },
    "openai-responses": {
        "type": "function",
        "name": "read_log",
        "description": "Read a worker's log.",
        "parameters": SCHEMA,
        "strict": True,
    },
    "openai-chat": {
        "type": "function",
        "function": {
            "name": "read_log",
            "description": "Read a worker's log.",
            "parameters": SCHEMA,
        },
    },
}


def worker_log(worker: int) -> str:
    """About 2,000 tokens of log; worker 2 runs out of disk."""
    return "".join(
        f"{index:05d} worker-{worker}: "
        + ("disk full\n" if (worker, index) == (2, 150) else "request ok\n")
        for index in range(300)
    )


def rounds(api: str) -> list[dict[str, Any]]:
    """A user turn and three rounds of read_log calls with their outputs."""
    entries: list[dict[str, Any]] = [user(ROUNDS_QUESTION)]
    for worker in (1, 2, 3):
        call_id = f"call_live_{worker}"
        arguments = {"worker": worker}
        output = worker_log(worker)
        match api:
            case "anthropic-messages":
                use = {"type": "tool_use", "id": call_id, "name": "read_log", "input": arguments}
                entries.append({"role": "assistant", "content": [use]})
            case "openai-responses":
                entries.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "read_log",
                        "arguments": json.dumps(arguments),
                    }
                )
            case _:
                function = {"name": "read_log", "arguments": json.dumps(arguments)}
                entries.append(
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": call_id, "type": "function", "function": function}],
                    }
                )
        entries.extend(grouped(api, [answer(Format(api), call_id, output)]))
    return entries


def grouped(api: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool results as the history takes them: one user message for Anthropic."""
    return [{"role": "user", "content": results}] if api == "anthropic-messages" else results


def converse(ctx: Context, create: Any) -> None:
    """Send the conversation, answering read_log and artiik_fetch, until the model answers."""
    for _ in range(5):
        ctx.record(create(**ctx.prepare()))
        calls = ctx.pending_tool_calls()
        if not calls:
            return
        results = [
            ctx.fetch_result(call)
            if call.name == FETCH_TOOL
            else answer(ctx.api, call.id, worker_log(dict(call.input or {}).get("worker", 1)))
            for call in calls
        ]
        ctx.add(*grouped(ctx.api.value, results))
    raise AssertionError("the model kept calling tools")


@pytest.mark.parametrize("api", ["anthropic-messages", "openai-responses", "openai-chat"])
def test_clearing_with_fetch(api: str) -> None:
    if api == "anthropic-messages":
        anthropic = pytest.importorskip("anthropic")
        _, model = setting("ANTHROPIC_API_KEY", "ARTIIK_ANTHROPIC_MODEL")
        create = anthropic.Anthropic().messages.create
        params: dict[str, Any] = {"max_tokens": 300}
    else:
        openai = pytest.importorskip("openai")
        _, model = setting("OPENAI_API_KEY", "ARTIIK_OPENAI_MODEL")
        client = openai.OpenAI()
        if api == "openai-responses":
            create = client.responses.create
            params = {"max_output_tokens": 300}
        else:
            create = client.chat.completions.create
            params = {}
    ctx = Context(
        api,
        model=model,
        system=SYSTEM,
        tools=[READ_LOG[api]],
        clearing=Clearing(trigger=2_000, keep=1, clear_at_least=0, store=MemoryStore()),
        params=params,
    )
    ctx.add(*rounds(api))
    # The logs of workers 1 and 2 become stubs; worker 3's, not read yet, stays.
    converse(ctx, create)
    cleared = ctx.trace.of("clearing")
    assert [event.data["results"] for event in cleared[:1]] == [2], cleared
    print("with stubs:", answered(ctx))
    ctx.add(user(f"Use {FETCH_TOOL} to read worker 1's log again, then quote its last line."))
    converse(ctx, create)
    print("fetches:", [event.data for event in ctx.trace.of("fetch")])
    print("after reading back:", answered(ctx))


def test_anthropic_server_side_clearing() -> None:
    anthropic = pytest.importorskip("anthropic")
    _, model = setting("ANTHROPIC_API_KEY", "ARTIIK_ANTHROPIC_MODEL")
    client = anthropic.Anthropic()
    ctx = Context(
        "anthropic-messages",
        model=model,
        system=SYSTEM,
        tools=[READ_LOG["anthropic-messages"]],
        clearing=AnthropicClearing(trigger=2_000, keep=1, clear_at_least=0),
        params={"max_tokens": 300},
    )
    ctx.add(*rounds("anthropic-messages"))
    # context_management goes through the beta endpoint, with its beta header.
    ctx.record(client.beta.messages.create(**ctx.prepare()))
    applied = [event.data for event in ctx.trace.of("clearing")]
    assert applied, "the API didn't report clearing anything"
    assert applied[0]["edit"] == "clear_tool_uses_20250919"
    print("applied:", applied)
    print("after clearing:", answered(ctx))
