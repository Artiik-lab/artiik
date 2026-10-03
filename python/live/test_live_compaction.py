"""Live smoke tests: each compaction strategy against the real provider API.

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

import os
from typing import Any

import pytest

from artiik import (
    AnthropicCompaction,
    AnthropicThresholdCompaction,
    Compaction,
    Context,
    OpenAICompaction,
    SummaryCompaction,
)

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
