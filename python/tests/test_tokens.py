"""Token accounting: usage, estimates that calibrate themselves, and exact counters."""

import importlib
from types import ModuleType
from typing import Any, cast

import pytest

from artiik import AnthropicTokenCounter, Context, Estimator, Format, TiktokenCounter, Usage
from artiik.messages import (
    Compaction,
    Document,
    Image,
    JSONObject,
    JSONValue,
    Message,
    Opaque,
    RedactedThinking,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)
from artiik.testing import (
    COMPACTION_BETA,
    Environment,
    FakeAnthropic,
    FakeOpenAI,
    Tokenizer,
    ToolLoopPolicy,
    default_request,
    run_context,
)
from artiik.testing.driver import Create
from artiik.tokens import (
    BLOCK_TOKENS,
    MEDIA_TOKENS,
    MESSAGE_TOKENS,
    Tally,
    tally_block,
    tally_json,
    tally_message,
    tally_request,
)
from artiik.usage import read_usage

ANTHROPIC = Format.ANTHROPIC_MESSAGES
RESPONSES = Format.OPENAI_RESPONSES
CHAT = Format.OPENAI_CHAT
MODEL = "fake-model"


def session(
    fmt: Format, tokenizer: Tokenizer, turns: int, **context: Any
) -> tuple[Context, FakeAnthropic | FakeOpenAI]:
    """Run a Context session against a fake whose tokenizer is ``tokenizer``."""
    base = default_request(fmt)
    fake: FakeAnthropic | FakeOpenAI
    create: Create
    if fmt is ANTHROPIC:
        fake = FakeAnthropic(policy=ToolLoopPolicy(calls_per_step=2), tokenizer=tokenizer)
        create = fake.messages.create
    else:
        fake = FakeOpenAI(policy=ToolLoopPolicy(calls_per_step=2), tokenizer=tokenizer)
        create = fake.responses.create if fmt is RESPONSES else fake.chat.completions.create
    params = {key: value for key, value in base.items() if key not in ("model", "tools")}
    ctx = Context(
        fmt,
        model=MODEL,
        system="You read logs and report problems.",
        tools=cast(list[JSONValue], base["tools"]),
        params=params,
        **context,
    )
    texts = [f"Turn {index}: check the next log for errors." for index in range(turns)]
    run_context(ctx, create, texts, environment=Environment(output_tokens=300))
    return ctx, fake


# Tallies


def test_tallies_count_characters_fixed_tokens_and_media() -> None:
    assert tally_block(Text(text="abcd")) == Tally(chars=4, fixed=BLOCK_TOKENS)
    assert tally_block(Image(source={"type": "url"})) == Tally(fixed=BLOCK_TOKENS, media=1)
    assert tally_block(Document(source={"type": "url"})) == Tally(fixed=BLOCK_TOKENS, media=1)
    assert tally_block(ToolUse(id="t", name="ls", input={"a": 1})) == Tally(
        chars=len("ls") + len('{"a":1}'), fixed=BLOCK_TOKENS
    )
    assert tally_block(ToolUse(id="t", name="ls", input={"a": 1}, raw_arguments='{ "a": 1 }')) == (
        Tally(chars=len("ls") + len('{ "a": 1 }'), fixed=BLOCK_TOKENS)
    )
    assert tally_block(ToolResult(tool_use_id="t", content="ok")) == Tally(chars=2, fixed=1)
    assert tally_block(ToolResult(tool_use_id="t")) == Tally(fixed=BLOCK_TOKENS)
    nested = ToolResult(tool_use_id="t", content=(Text(text="ab"), Image(source={})))
    assert tally_block(nested) == Tally(chars=2, fixed=3 * BLOCK_TOKENS, media=1)
    assert tally_block(Thinking(thinking="hmm", signature="s")) == Tally(chars=3, fixed=1)
    assert tally_block(RedactedThinking(data="xyz")) == Tally(chars=3, fixed=1)
    readable = Compaction(data={"type": "compaction", "content": "Summary."})
    assert tally_block(readable) == Tally(chars=len("Summary."), fixed=BLOCK_TOKENS)
    opaque = Compaction(data={"type": "compaction", "encrypted_content": "x"})
    assert tally_block(opaque).chars == len('{"type":"compaction","encrypted_content":"x"}')
    assert tally_block(Opaque(data={"type": "x"})).chars == len('{"type":"x"}')
    message = Message(role="user", blocks=(Text(text="abcd"), Text(text="ef")))
    assert tally_message(message) == Tally(chars=6, fixed=MESSAGE_TOKENS + 2 * BLOCK_TOKENS)
    assert tally_json({"a": 1}) == Tally(chars=len('{"a":1}'))
    assert Tally(chars=5, fixed=2, media=1) - Tally(chars=1, fixed=1) == Tally(4, 1, 1)


def test_tally_request_counts_what_the_model_reads() -> None:
    tool: JSONObject = {"name": "ls", "input_schema": {"type": "object"}}
    request: JSONObject = {
        "model": MODEL,
        "system": "Be brief.",
        "tools": [tool],
        "messages": [{"role": "user", "content": "Hi"}],
    }
    expected = (
        tally_json(tool)
        + Tally(chars=len("Be brief."), fixed=BLOCK_TOKENS)
        + Tally(chars=2, fixed=MESSAGE_TOKENS + BLOCK_TOKENS)
    )
    assert tally_request(ANTHROPIC, request) == expected
    compacted: JSONObject = {
        "model": MODEL,
        "instructions": "Be brief.",
        "input": [
            {"role": "user", "content": "Long ago."},
            {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
            {"id": "cmp_1", "type": "compaction", "encrypted_content": "x"},
            {"role": "user", "content": "Hi"},
        ],
    }
    item = Compaction(data={"id": "cmp_1", "type": "compaction", "encrypted_content": "x"})
    # The compaction item stands in for the items before it, except the user messages.
    assert tally_request(RESPONSES, compacted) == (
        tally_message(Message.from_text("system", "Be brief."))
        + tally_message(Message.from_text("user", "Long ago."))
        + tally_message(Message(role="assistant", blocks=(item,)))
        + tally_message(Message.from_text("user", "Hi"))
    )


# The estimator


def test_the_estimator_starts_high_and_follows_the_provider() -> None:
    estimator = Estimator()
    tally = Tally(chars=4_000, fixed=10)
    assert estimator.estimate(tally, api=ANTHROPIC, model=MODEL) == 1_250 + 10
    assert estimator.estimate(tally, api=CHAT, model=MODEL) == 1_100 + 10
    estimator.observe(tally, 1_500 + 10, api=ANTHROPIC, model=MODEL)
    assert estimator.ratio(ANTHROPIC, MODEL) == pytest.approx(1.5)
    assert estimator.estimate(tally, api=ANTHROPIC, model=MODEL) == 1_510
    assert estimator.ratio(ANTHROPIC, "other-model") == Estimator.START[ANTHROPIC]
    assert estimator.ratio(CHAT, MODEL) == Estimator.START[CHAT]


def test_recent_calls_weigh_most() -> None:
    estimator = Estimator(decay=0.7)
    tally = Tally(chars=4_000)
    for _ in range(5):
        estimator.observe(tally, 1_000, api=CHAT, model=MODEL)
    for _ in range(6):
        estimator.observe(tally, 2_000, api=CHAT, model=MODEL)
    assert 1.8 < estimator.ratio(CHAT, MODEL) < 2.0


def test_the_estimator_ignores_calls_it_cant_learn_from() -> None:
    estimator = Estimator()
    estimator.observe(Tally(fixed=10, media=1), 5_000, api=CHAT, model=MODEL)
    estimator.observe(Tally(chars=400, fixed=100), 50, api=CHAT, model=MODEL)
    assert estimator.ratio(CHAT, MODEL) == Estimator.START[CHAT]
    estimator.observe(Tally(chars=4), 1_000_000, api=CHAT, model=MODEL)
    assert estimator.ratio(CHAT, MODEL) == Estimator.MAX_RATIO
    with pytest.raises(ValueError, match="decay"):
        Estimator(decay=1)


def test_media_has_a_flat_cost_per_api() -> None:
    estimator = Estimator()
    tally = Tally(fixed=1, media=2)
    assert estimator.estimate(tally, api=ANTHROPIC, model=MODEL) == 1 + 2 * MEDIA_TOKENS[ANTHROPIC]
    assert estimator.estimate(tally, api=RESPONSES, model=MODEL) == 1 + 2 * MEDIA_TOKENS[RESPONSES]


@pytest.mark.parametrize("fmt", list(Format))
@pytest.mark.parametrize(
    "tokenizer",
    [
        pytest.param(Tokenizer(chars_per_token=2.8), id="denser-text"),
        pytest.param(Tokenizer(chars_per_token=2.6, message_overhead=6), id="more-overhead"),
        pytest.param(Tokenizer(chars_per_token=5, message_overhead=2), id="lighter-text"),
    ],
)
def test_after_calibration_estimates_are_within_ten_percent(
    fmt: Format, tokenizer: Tokenizer
) -> None:
    _, fake = session(fmt, tokenizer, turns=5)
    estimator = Estimator()
    last = fake.calls[-1]
    guess = estimator.estimate(tally_request(fmt, last.request), api=fmt, model=MODEL)
    assert abs(guess - last.tokens()) > 0.1 * last.tokens()
    errors: list[float] = []
    for index, call in enumerate(fake.calls):
        tally = tally_request(fmt, call.request)
        if index >= 3:
            estimate = estimator.estimate(tally, api=fmt, model=MODEL)
            errors.append(abs(estimate - call.tokens()) / call.tokens())
        estimator.observe(tally, call.tokens(), api=fmt, model=MODEL)
    assert len(errors) == len(fake.calls) - 3
    assert max(errors) <= 0.10


@pytest.mark.parametrize("fmt", list(Format))
def test_a_context_calibrates_from_every_response(fmt: Format) -> None:
    ctx, fake = session(fmt, Tokenizer(chars_per_token=2.8), turns=3)
    expected = 4 / 2.8
    assert ctx.estimator.ratio(fmt, MODEL) == pytest.approx(expected, rel=0.05)
    last = fake.calls[-1]
    estimate = ctx.estimator.estimate(tally_request(fmt, last.request), api=fmt, model=MODEL)
    assert estimate == pytest.approx(last.tokens(), rel=0.05)


# Usage


def test_anthropic_usage_adds_up_the_cache() -> None:
    usage: JSONObject = {
        "input_tokens": 10,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "output_tokens": 5,
    }
    assert read_usage(ANTHROPIC, {"usage": usage}) == Usage(
        input_tokens=130, output_tokens=5, cache_read_tokens=100, cache_write_tokens=20, raw=usage
    )


def test_anthropic_compaction_usage_comes_from_the_iterations() -> None:
    usage: JSONObject = {
        "input_tokens": 0,
        "output_tokens": 0,
        "iterations": [{"type": "compaction", "input_tokens": 144, "output_tokens": 276}],
    }
    read = read_usage(ANTHROPIC, {"usage": usage})
    assert (read.input_tokens, read.output_tokens) == (144, 276)


def test_openai_usage() -> None:
    responses: JSONObject = {
        "input_tokens": 300,
        "input_tokens_details": {"cached_tokens": 256},
        "output_tokens": 40,
        "output_tokens_details": {"reasoning_tokens": 12},
    }
    assert read_usage(RESPONSES, {"usage": responses}) == Usage(
        input_tokens=300,
        output_tokens=40,
        cache_read_tokens=256,
        reasoning_tokens=12,
        raw=responses,
    )
    chat: JSONObject = {
        "prompt_tokens": 300,
        "completion_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 128},
        "completion_tokens_details": {"reasoning_tokens": 7},
    }
    assert read_usage(CHAT, {"usage": chat}) == Usage(
        input_tokens=300, output_tokens=40, cache_read_tokens=128, reasoning_tokens=7, raw=chat
    )


@pytest.mark.parametrize("fmt", list(Format))
def test_missing_usage_reads_as_zero(fmt: Format) -> None:
    assert read_usage(fmt, {}) == Usage()
    assert read_usage(fmt, {"usage": None}) == Usage()


def test_recorded_usage_matches_what_the_fake_counted() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(steps=0))
    ctx = Context(
        ANTHROPIC, model=MODEL, params={"max_tokens": 100, "cache_control": {"type": "ephemeral"}}
    )
    for text in ("First question.", "Second question."):
        ctx.add({"role": "user", "content": text})
        usage = ctx.record(fake.messages.create(**ctx.prepare()))
        assert usage.input_tokens == fake.calls[-1].tokens()
    assert ctx.last_usage is not None
    assert ctx.last_usage.cache_read_tokens > 0


# Exact counters


def test_the_anthropic_counter_matches_what_the_api_counts() -> None:
    fake = FakeAnthropic(tokenizer=Tokenizer(chars_per_token=3), policy=ToolLoopPolicy(steps=0))
    counter = AnthropicTokenCounter(fake)
    request: JSONObject = {
        **default_request(ANTHROPIC),
        "system": "Be brief.",
        "temperature": 0.5,
        "messages": [{"role": "user", "content": "How many tokens is this?"}],
    }
    count = counter.count(ANTHROPIC, request)
    fake.messages.create(**request)
    assert count == fake.calls[0].tokens()
    assert set(fake.count_requests[0]) == {"model", "system", "tools", "messages"}
    beta = {**request, "betas": [COMPACTION_BETA]}
    assert counter.count(ANTHROPIC, beta) == count
    assert fake.count_requests[1]["betas"] == [COMPACTION_BETA]
    # The beta header can come through extra_headers, as a compaction block needs.
    headers: JSONObject = {**request, "extra_headers": {"anthropic-beta": COMPACTION_BETA}}
    assert counter.count(ANTHROPIC, headers) == count
    assert fake.count_requests[2]["extra_headers"] == {"anthropic-beta": COMPACTION_BETA}
    # Only the beta endpoint takes context_management; the plain one would raise.
    managed: JSONObject = {**headers, "context_management": {"edits": []}}
    assert counter.count(ANTHROPIC, managed) == count
    assert fake.count_requests[3]["context_management"] == {"edits": []}
    with pytest.raises(ValueError, match="anthropic-messages"):
        counter.count(CHAT, request)


class WordEncoding:
    """Stands in for a tiktoken encoding: one token per word."""

    def encode_ordinary(self, text: str) -> list[int]:
        return [0] * len(text.split())


def test_the_tiktoken_counter_counts_text_and_framing() -> None:
    counter = TiktokenCounter(WordEncoding())
    chat: JSONObject = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Be very brief."},
            {"role": "user", "content": "Hello there", "name": "ana"},
        ],
    }
    reply, message = TiktokenCounter.REPLY_TOKENS, TiktokenCounter.MESSAGE_TOKENS
    assert counter.count(CHAT, chat) == reply + (message + 1 + 3) + (message + 1 + 1 + 1 + 2)
    responses: JSONObject = {
        "model": MODEL,
        "instructions": "Be brief.",
        "tools": [{"type": "function", "name": "ls"}],
        "input": [
            {"role": "user", "content": "List files"},
            {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{} {}"},
            {"type": "function_call_output", "call_id": "c1", "output": "a b c"},
        ],
    }
    tools = 1
    expected = reply + (message + 1 + 2) + (message + 1 + 2) + (message + 1 + 1 + 2)
    expected += (message + 1 + 3) + tools
    assert counter.count(RESPONSES, responses) == expected
    with pytest.raises(ValueError, match="AnthropicTokenCounter"):
        counter.count(ANTHROPIC, chat)


def test_the_tiktoken_counter_explains_how_to_install_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> ModuleType:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"pip install 'artiik\[tiktoken\]'"):
        TiktokenCounter()


def test_the_tiktoken_counter_with_the_real_encoding() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    try:
        encoding = tiktoken.get_encoding("o200k_base")
    except Exception as error:  # the encoding is downloaded on first use
        pytest.skip(f"the o200k_base encoding isn't available: {error}")
    text = "Context management for AI agents."
    request: JSONObject = {"model": MODEL, "messages": [{"role": "user", "content": text}]}
    words = len(encoding.encode_ordinary(text))
    role = len(encoding.encode_ordinary("user"))
    assert TiktokenCounter().count(CHAT, request) == 3 + 3 + role + words
