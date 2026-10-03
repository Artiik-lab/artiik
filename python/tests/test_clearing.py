"""Tool-output clearing: stubs in rare batches, originals kept byte for byte, and Anthropic's."""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from artiik import (
    FETCH_TOOL,
    AnthropicClearing,
    AnthropicCompaction,
    AnthropicThresholdCompaction,
    AnthropicTokenCounter,
    Clearing,
    Context,
    FileStore,
    Format,
    MemoryStore,
    Message,
    OpenAICompaction,
    SummaryCompaction,
    ToolResult,
)
from artiik.clearing import (
    ANTHROPIC_BETA,
    ID_LENGTH,
    Offload,
    answer,
    candidates,
    default_trigger,
    fetch_tool,
    original,
    stub,
)
from artiik.formats import anthropic_messages
from artiik.messages import JSONObject, JSONValue, Text, Thinking, ToolUse
from artiik.store import check_path
from artiik.testing import (
    Environment,
    FakeAnthropic,
    FakeOpenAI,
    Reply,
    ToolCall,
    ToolLoopPolicy,
    assert_holds,
    check_all,
    check_append_only,
    check_tool_pairs,
    default_request,
    run_context,
)
from artiik.testing.driver import Create
from artiik.testing.recording import RecordedCall, entries
from artiik.validation import validate

ANTHROPIC = Format.ANTHROPIC_MESSAGES
RESPONSES = Format.OPENAI_RESPONSES
CHAT = Format.OPENAI_CHAT
MODEL = "fake-model"
STUB_ID = re.compile(r'artiik_fetch\("([0-9a-f]+)"\)')
BIG = "log line ok\n" * 400
TRICKY = 'naïve café, 日本語, 🚀\n\ttabs, "quotes", back\\slash, nul \x00, end. ' * 20


def context_for(fmt: Format, **options: Any) -> Context:
    base = default_request(fmt)
    params = {key: value for key, value in base.items() if key not in ("model", "tools")}
    params.update(options.pop("params", {}))
    tools = cast(list[JSONValue], base["tools"])
    return Context(fmt, model=MODEL, tools=tools, params=params, **options)


def fake_for(fmt: Format, **options: Any) -> tuple[FakeAnthropic | FakeOpenAI, Create]:
    if fmt is ANTHROPIC:
        anthropic = FakeAnthropic(**options)
        return anthropic, anthropic.messages.create
    openai = FakeOpenAI(**options)
    return openai, openai.responses.create if fmt is RESPONSES else openai.chat.completions.create


def clearing(**options: Any) -> Clearing:
    """A clearing strategy with a store in memory, so the tests write no files."""
    options.setdefault("store", MemoryStore())
    return Clearing(**options)


def user(text: str) -> JSONObject:
    return {"role": "user", "content": text}


def grouped(fmt: Format, answers: Sequence[JSONObject]) -> list[JSONObject]:
    """Tool results as the history takes them: one user message for Anthropic."""
    if fmt is ANTHROPIC:
        return [{"role": "user", "content": list[JSONValue](answers)}]
    return list(answers)


def results(fmt: Format, *pairs: tuple[str, JSONValue]) -> list[JSONObject]:
    """Tool results in the provider's format, for calls by id."""
    return grouped(fmt, [answer(fmt, call_id, content) for call_id, content in pairs])


def raw_results(fmt: Format, request: JSONObject) -> dict[str, JSONValue]:
    """The tool results of a request as sent, by call id, exactly as JSON."""
    found: dict[str, JSONValue] = {}
    for entry in entries(fmt, request):
        if not isinstance(entry, dict):
            continue
        if fmt is RESPONSES and entry.get("type") == "function_call_output":
            found[cast(str, entry["call_id"])] = entry["output"]
        elif fmt is CHAT and entry.get("role") == "tool":
            found[cast(str, entry["tool_call_id"])] = entry["content"]
        elif fmt is ANTHROPIC and isinstance(entry.get("content"), list):
            for block in cast(list[JSONValue], entry["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    found[cast(str, block["tool_use_id"])] = block.get("content")
    return found


def tool_names(fmt: Format, request: JSONObject) -> list[str]:
    tools = cast(list[JSONObject], request.get("tools", []))
    if fmt is CHAT:
        return [cast(str, cast(JSONObject, tool["function"])["name"]) for tool in tools]
    return [cast(str, tool["name"]) for tool in tools]


def batches(ctx: Context) -> list[JSONObject]:
    return [
        event.data for event in ctx.trace.of("clearing") if event.data.get("outcome") == "cleared"
    ]


def stub_ids(conversation: Sequence[Message]) -> list[str]:
    return [
        found
        for message in conversation
        for result in message.tool_results
        if isinstance(result.content, str)
        for found in STUB_ID.findall(result.content)
    ]


def fetched_ids(conversation: Sequence[Message]) -> set[str]:
    return {
        cast(str, cast(JSONObject, use.input)["id"])
        for message in conversation
        for use in message.tool_uses
        if use.name == FETCH_TOOL and isinstance(use.input, dict)
    }


class ReadingPolicy:
    """A tool-calling agent that reads one cleared output back every ``every`` replies."""

    def __init__(self, *, every: int = 6, **options: Any) -> None:
        self.base = ToolLoopPolicy(**options)
        self.every = every
        self.replies = 0

    def reply(self, conversation: Sequence[Message]) -> Reply:
        self.replies += 1
        reply = self.base.reply(conversation)
        if not reply.tool_calls or self.replies % self.every:
            return reply
        asked = fetched_ids(conversation)
        fresh = [found for found in stub_ids(conversation) if found not in asked]
        if not fresh:
            return reply
        return Reply(tool_calls=(ToolCall(name=FETCH_TOOL, arguments={"id": fresh[0]}),))


class Steps:
    """Answers with one function of the conversation per call, in order."""

    def __init__(self, *steps: Callable[[Sequence[Message]], Reply]) -> None:
        self.steps = list(steps)

    def reply(self, conversation: Sequence[Message]) -> Reply:
        return self.steps.pop(0)(conversation)


def calls(*names: str) -> Callable[[Sequence[Message]], Reply]:
    tools = tuple(ToolCall(name=name, arguments={"n": n}) for n, name in enumerate(names))
    return lambda _: Reply(tool_calls=tools)


def done(_: Sequence[Message]) -> Reply:
    return Reply(text="Done.")


def fetch_a_stub(conversation: Sequence[Message]) -> Reply:
    found = stub_ids(conversation)
    assert found, "no stub to read back"
    return Reply(tool_calls=(ToolCall(name=FETCH_TOOL, arguments={"id": found[0]}),))


def drive(ctx: Context, create: Create, outputs: Sequence[Sequence[JSONValue]]) -> list[JSONObject]:
    """One user turn whose tool rounds get ``outputs``; returns the requests sent.

    The context answers ``artiik_fetch`` calls; their place in ``outputs`` is ignored.
    """
    requests: list[JSONObject] = []
    ctx.add(user("Go."))
    for round_outputs in [*outputs, None]:
        request = ctx.prepare()
        requests.append(request)
        ctx.record(create(**request))
        pending = ctx.pending_tool_calls()
        if round_outputs is None:
            assert not pending
            return requests
        answers = [
            ctx.fetch_result(call) if call.name == FETCH_TOOL else answer(ctx.api, call.id, output)
            for call, output in zip(pending, round_outputs, strict=True)
        ]
        ctx.add(*grouped(ctx.api, answers))
    return requests


# The store


@pytest.mark.parametrize("kind", ["memory", "files"])
def test_a_store_keeps_files_by_relative_path(kind: str, tmp_path: Path) -> None:
    store = MemoryStore() if kind == "memory" else FileStore(tmp_path / "store")
    assert store.read("offload/a1b2.json") is None
    store.write("offload/a1b2.json", b"first")
    store.write("offload/a1b2.json", b"second")
    store.write("notes.md", "caf\u00e9".encode())
    assert store.read("offload/a1b2.json") == b"second"
    assert store.read("notes.md") == "caf\u00e9".encode()
    if isinstance(store, FileStore):
        offload = tmp_path / "store" / "offload"
        assert (offload / "a1b2.json").read_bytes() == b"second"
        # A write replaces the file in one step and leaves no temporary file behind.
        assert [path.name for path in offload.iterdir()] == ["a1b2.json"]


@pytest.mark.parametrize(
    "path",
    ["", "/etc/passwd", "../outside", "offload/../../x", "a//b", "a/./b", "a\\b", "C:/x", "a/"],
)
def test_store_paths_stay_inside_the_store(path: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid store path"):
        check_path(path)
    with pytest.raises(ValueError, match="invalid store path"):
        FileStore(tmp_path).write(path, b"x")
    with pytest.raises(ValueError, match="invalid store path"):
        MemoryStore().read(path)
    assert list(tmp_path.iterdir()) == []


def test_a_file_store_defaults_to_artiik_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = FileStore()
    monkeypatch.chdir("/")
    store.write("offload/x.json", b"{}")
    assert (tmp_path / ".artiik" / "offload" / "x.json").read_bytes() == b"{}"


def test_a_failed_write_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(tmp_path)

    def broken(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", broken)
    with pytest.raises(OSError, match="disk full"):
        store.write("offload/x.json", b"{}")
    assert list((tmp_path / "offload").iterdir()) == []


# The parts of clearing


def test_clearing_settings_are_checked() -> None:
    with pytest.raises(ValueError, match="trigger"):
        Clearing(trigger=0)
    with pytest.raises(ValueError, match="keep"):
        Clearing(keep=-1)
    with pytest.raises(ValueError, match="clear_at_least"):
        Clearing(clear_at_least=-1)
    with pytest.raises(ValueError, match="min_tokens"):
        Clearing(min_tokens=-1)
    with pytest.raises(TypeError, match="list of tool names"):
        Clearing(exclude="search")
    with pytest.raises(TypeError, match="tool names"):
        Clearing(exclude=[""])
    with pytest.raises(ValueError, match="not both"):
        AnthropicClearing(trigger=1_000, trigger_tool_uses=5)
    with pytest.raises(ValueError, match="trigger_tool_uses"):
        AnthropicClearing(trigger_tool_uses=0)
    with pytest.raises(TypeError, match="clear_inputs"):
        AnthropicClearing(clear_inputs="search")
    assert isinstance(Clearing().store, FileStore)


def test_a_stub_says_which_tool_its_id_and_its_size() -> None:
    assert (
        stub("search", "3f2a", 4_812, fetch=True)
        == '[cleared: tool "search", id 3f2a, 4,812 tokens. Call artiik_fetch("3f2a") to read it]'
    )
    assert stub("search", "3f2a", 4_812, fetch=False) == (
        '[cleared: tool "search", id 3f2a, 4,812 tokens]'
    )
    assert stub('say "hi"', "3f2a", 12, fetch=False).startswith('[cleared: tool "say \\"hi\\""')


def test_the_fetch_tool_is_defined_in_each_format() -> None:
    schema: JSONObject = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The id in the [cleared: ...] note, such as 3f2a.",
            }
        },
        "required": ["id"],
        "additionalProperties": False,
    }
    anthropic = fetch_tool(ANTHROPIC)
    assert set(anthropic) == {"name", "description", "input_schema"}
    assert anthropic["name"] == FETCH_TOOL
    assert anthropic["input_schema"] == schema
    responses = fetch_tool(RESPONSES)
    assert set(responses) == {"type", "name", "description", "parameters", "strict"}
    assert (responses["type"], responses["name"]) == ("function", FETCH_TOOL)
    assert responses["parameters"] == schema
    assert responses["strict"] is True
    chat = fetch_tool(CHAT)
    function = cast(JSONObject, chat["function"])
    assert chat["type"] == "function"
    assert set(function) == {"name", "description", "parameters"}
    assert (function["name"], function["parameters"]) == (FETCH_TOOL, schema)
    # Each call returns its own copy.
    cast(JSONObject, anthropic["input_schema"])["required"] = []
    assert cast(JSONObject, fetch_tool(ANTHROPIC)["input_schema"])["required"] == ["id"]


def test_the_default_trigger_comes_before_compaction() -> None:
    assert default_trigger(None, None) is None
    assert default_trigger(100_000, None) == 50_000
    assert default_trigger(None, 60_000) == 40_000
    assert default_trigger(100_000, 90_000) == 50_000
    assert default_trigger(100_000, 60_000) == 40_000
    assert default_trigger(1, None) == 1


def anthropic_rounds(*rounds: Sequence[tuple[str, str]]) -> list[Message]:
    """A user turn, rounds of tool calls ``(id, name)`` with their results, and an answer."""
    history = [anthropic_messages.parse_message(user("Go."))]
    for number, round_calls in enumerate(rounds):
        uses: list[JSONValue] = [
            {"type": "tool_use", "id": call_id, "name": name, "input": {}}
            for call_id, name in round_calls
        ]
        history.append(anthropic_messages.parse_message({"role": "assistant", "content": uses}))
        history.extend(
            anthropic_messages.parse_message(entry)
            for entry in results(
                ANTHROPIC, *((call_id, f"out {number}") for call_id, _ in round_calls)
            )
        )
    history.append(anthropic_messages.parse_message({"role": "assistant", "content": "Done."}))
    return history


def ids_of(found: Sequence[Any]) -> list[str]:
    return [candidate.result.tool_use_id for candidate in found]


def test_candidates_leave_the_latest_and_the_excluded_results() -> None:
    history = anthropic_rounds(
        [("a", "search"), ("b", "read")], [("c", "search")], [("d", "memory")], [("e", "read")]
    )
    found, held = candidates(ANTHROPIC, history, keep=2, exclude=("memory",), cleared={"b"})
    assert ids_of(found) == ["a", "c"]
    assert [candidate.tool for candidate in found] == ["search", "search"]
    assert held == 0
    found, _ = candidates(ANTHROPIC, history, keep=0, exclude=(), cleared=())
    assert ids_of(found) == ["a", "b", "c", "d", "e"]
    assert (found[1].message, found[1].block) == (2, 1)
    found, _ = candidates(ANTHROPIC, history, keep=9, exclude=(), cleared=())
    assert found == []


def test_candidates_leave_the_results_the_model_has_not_read() -> None:
    history = anthropic_rounds([("a", "search")], [("b", "search")])[:-1]
    found, _ = candidates(ANTHROPIC, history, keep=0, exclude=(), cleared=())
    assert ids_of(found) == ["a"]


def test_candidates_hold_anthropic_results_before_a_thinking_block() -> None:
    history = anthropic_rounds([("a", "search")], [("b", "search")], [("c", "search")])
    thinking = Thinking(thinking="Hmm.", signature="sig")
    history[3] = Message(role="assistant", blocks=(thinking, *history[3].blocks))
    found, held = candidates(ANTHROPIC, history, keep=0, exclude=(), cleared=())
    # "a" comes before the thinking block, "b" and "c" after it.
    assert ids_of(found) == ["b", "c"]
    assert held == 1
    # Already cleared, a result isn't counted again.
    found, held = candidates(ANTHROPIC, history, keep=0, exclude=(), cleared={"a"})
    assert held == 0
    # The other APIs don't tie reasoning to the history before it.
    found, held = candidates(RESPONSES, history, keep=0, exclude=(), cleared=())
    assert (ids_of(found), held) == (["a", "b", "c"], 0)


def test_candidates_skip_empty_results_and_unknown_calls() -> None:
    history = anthropic_rounds([("a", "search"), ("b", "search")], [("c", "search")])
    history[2] = Message(
        role="user",
        blocks=(ToolResult(tool_use_id="a", content=""), ToolResult(tool_use_id="b", content=None)),
    )
    found, _ = candidates(ANTHROPIC, history, keep=0, exclude=(), cleared=())
    assert ids_of(found) == ["c"]
    no_calls = [
        Message(role="assistant", blocks=(Text(text="?"),)) if m.tool_uses else m for m in history
    ]
    found, _ = candidates(ANTHROPIC, no_calls, keep=0, exclude=(), cleared=())
    assert found == []


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(TRICKY, id="text"),
        pytest.param("lone surrogate \ud800 stays", id="surrogate"),
        pytest.param(
            [
                {"type": "text", "text": TRICKY},
                {"type": "image", "source": {"type": "url", "url": "x"}},
            ],
            id="blocks",
        ),
        pytest.param([{"type": "text", "text": "1.0 and 1e400", "score": 0.1}], id="numbers"),
    ],
)
def test_the_offload_gives_back_the_original_byte_for_byte(content: JSONValue) -> None:
    store = MemoryStore()
    offload = Offload(store)
    entry = offload.put(ANTHROPIC, tool="search", tool_use_id="toolu_1", content=content, tokens=9)
    assert re.fullmatch(r"[0-9a-f]{4}", entry.id)
    assert entry.path == f"offload/{entry.id}.json"
    back = offload.get(entry.id)
    assert back == content
    assert json.dumps(back) == json.dumps(content)
    data = store.read(entry.path)
    assert data is not None
    record = json.loads(data.decode("ascii"))
    assert {key: record[key] for key in ("id", "tool", "tool_use_id", "api", "tokens")} == {
        "id": entry.id,
        "tool": "search",
        "tool_use_id": "toolu_1",
        "api": "anthropic-messages",
        "tokens": 9,
    }


def test_an_offload_id_grows_when_it_is_taken() -> None:
    store = MemoryStore()
    entry = Offload(store).put(ANTHROPIC, tool="s", tool_use_id="toolu_1", content="x", tokens=1)
    # The same output of the same call, cleared in another conversation, shares the file.
    again = Offload(store).put(ANTHROPIC, tool="s", tool_use_id="toolu_1", content="x", tokens=1)
    assert again.id == entry.id
    # When another file holds the id, the next one is a character longer.
    store.write(entry.path, b"someone else's")
    other = Offload(store)
    taken = other.put(ANTHROPIC, tool="s", tool_use_id="toolu_1", content="x", tokens=1)
    assert taken.id.startswith(entry.id)
    assert len(taken.id) == ID_LENGTH + 1
    assert other.get(taken.id) == "x"
    assert store.read(entry.path) == b"someone else's"


def test_an_offload_only_reads_back_what_it_cleared() -> None:
    store = MemoryStore()
    mine = Offload(store)
    theirs = Offload(store).put(
        ANTHROPIC, tool="s", tool_use_id="toolu_1", content="secret", tokens=1
    )
    with pytest.raises(KeyError, match="no cleared tool output"):
        mine.get(theirs.id)
    kept = mine.put(ANTHROPIC, tool="s", tool_use_id="toolu_2", content="x", tokens=1)
    # A file that another call's output replaced isn't given back as this one.
    forged = {"id": kept.id, "tool": "s", "tool_use_id": "toolu_3", "content": "forged"}
    store.write(kept.path, json.dumps(forged).encode())
    with pytest.raises(KeyError, match="no longer has"):
        mine.get(kept.id)
    store.write(kept.path, b"[]")
    with pytest.raises(KeyError, match="no longer has"):
        mine.get(kept.id)
    assert [item.id for item in mine.entries] == [kept.id]


def test_original_is_the_content_the_provider_receives() -> None:
    blocks: list[JSONValue] = [{"type": "text", "text": "a", "citations": None}]
    anthropic = Context(ANTHROPIC, model=MODEL)
    anthropic.add(results(ANTHROPIC, ("t", blocks)))
    assert original(ANTHROPIC, anthropic.history[0].tool_results[0]) == blocks
    parts: list[JSONValue] = [{"type": "input_text", "text": "a"}]
    responses = Context(RESPONSES, model=MODEL)
    responses.add(results(RESPONSES, ("t", parts)))
    assert original(RESPONSES, responses.history[0].tool_results[0]) == parts
    chat = Context(CHAT, model=MODEL)
    chat.add(results(CHAT, ("t", "plain")))
    assert original(CHAT, chat.history[0].tool_results[0]) == "plain"


# Clearing in a context


def test_clearing_settings_follow_the_budget_and_compaction() -> None:
    ctx = context_for(ANTHROPIC, budget=100_000, clearing=clearing())
    assert (ctx.clear_at, ctx.clear_at_least) == (50_000, 12_500)
    summary = SummaryCompaction(lambda _: "S")
    ctx = context_for(
        ANTHROPIC,
        budget=100_000,
        compact_at=60_000,
        compaction=summary,
        clearing=clearing(clear_at_least=0),
    )
    assert (ctx.clear_at, ctx.clear_at_least) == (40_000, 0)
    # A provider strategy's own threshold is where compaction happens.
    ctx = context_for(
        ANTHROPIC,
        budget=200_000,
        compaction=AnthropicThresholdCompaction(trigger=90_000),
        clearing=AnthropicClearing(),
    )
    assert ctx.clear_at == 60_000
    ctx = context_for(ANTHROPIC, clearing=AnthropicClearing())
    assert (ctx.clear_at, ctx.clear_at_least) == (None, None)
    ctx = context_for(ANTHROPIC, budget=100_000, clearing=AnthropicClearing(trigger_tool_uses=20))
    assert (ctx.clear_at, ctx.clear_at_least) == (None, None)
    with pytest.raises(ValueError, match="needs a trigger"):
        context_for(ANTHROPIC, clearing=clearing())
    with pytest.raises(ValueError, match="above the budget"):
        context_for(ANTHROPIC, budget=10_000, clearing=clearing(trigger=20_000))
    with pytest.raises(ValueError, match="compaction would always come first"):
        context_for(
            ANTHROPIC, budget=100_000, compaction=summary, clearing=clearing(trigger=80_000)
        )
    with pytest.raises(ValueError, match="anthropic-messages"):
        context_for(RESPONSES, budget=10_000, clearing=AnthropicClearing())


@pytest.mark.parametrize("fmt", list(Format))
def test_the_fetch_tool_goes_in_every_request_from_the_first(fmt: Format) -> None:
    ctx = context_for(fmt, budget=50_000, clearing=clearing())
    ctx.add(user("Hello."))
    first = ctx.prepare()
    assert tool_names(fmt, first) == ["read_log", FETCH_TOOL]
    assert cast(list[JSONValue], first["tools"])[-1] == fetch_tool(fmt)
    # The estimate counts it too.
    bare = context_for(fmt, budget=50_000)
    bare.add(user("Hello."))
    assert ctx.estimate() > bare.estimate()
    # A tool of that name from the caller isn't added twice.
    assert tool_names(fmt, ctx.prepare(tools=[fetch_tool(fmt)])) == [FETCH_TOOL]
    quiet = context_for(fmt, budget=50_000, clearing=clearing(fetch=False))
    quiet.add(user("Hello."))
    assert tool_names(fmt, quiet.prepare()) == ["read_log"]
    native = context_for(ANTHROPIC, budget=50_000, clearing=AnthropicClearing())
    native.add(user("Hello."))
    assert tool_names(ANTHROPIC, native.prepare()) == ["read_log"]


@pytest.mark.parametrize("fmt", list(Format))
def test_a_batch_clears_every_old_result_into_a_stub_and_keeps_the_original(
    fmt: Format,
) -> None:
    store = MemoryStore()
    policy = Steps(calls("search", "search"), calls("read"), calls("read"), done)
    fake, create = fake_for(fmt, policy=policy)
    ctx = context_for(fmt, budget=100_000, clearing=Clearing(trigger=5_000, keep=1, store=store))
    requests = drive(ctx, create, [[BIG, BIG], [BIG], [BIG]])
    events = batches(ctx)
    # The batch runs when the fourth request passes the trigger. The latest result
    # stays: the model hasn't read it yet.
    assert [event.request for event in ctx.trace.of("clearing")] == [3]
    assert events[0]["results"] == 3
    sent = list(raw_results(fmt, requests[3]).values())
    assert sent[3] == BIG
    ids: list[str] = []
    for content, tool in zip(sent[:3], ["search", "search", "read"], strict=True):
        assert isinstance(content, str)
        assert re.fullmatch(
            rf'\[cleared: tool "{tool}", id ([0-9a-f]{{4}}), [\d,]+ tokens\. '
            r'Call artiik_fetch\("\1"\) to read it\]',
            content,
        )
        ids.extend(STUB_ID.findall(content))
    assert ids == events[0]["ids"]
    for found in ids:
        assert ctx.fetch(found) == BIG
        assert ctx.fetch(f" {found.upper()} ") == BIG
        call = ToolUse(id="call_x", name=FETCH_TOOL, input={"id": found.upper()})
        assert ctx.fetch_result(call) == answer(fmt, "call_x", BIG)
        assert store.read(f"offload/{found}.json") is not None
    event = events[0]
    assert cast(int, event["tokens_after"]) < cast(int, event["tokens_before"])
    assert cast(int, event["tokens_saved"]) > 3 * 1_000
    # The last request appends to the cleared one.
    assert_holds(check_all(fake.calls, allowed_rewrites={3}))


def test_a_batch_waits_until_it_saves_enough() -> None:
    policy = Steps(calls("read"), calls("read"), calls("read"), calls("read"), done)
    fake, create = fake_for(ANTHROPIC, policy=policy)
    ctx = context_for(
        ANTHROPIC,
        budget=100_000,
        clearing=clearing(trigger=1_000, keep=0, clear_at_least=2_000),
    )
    drive(ctx, create, [[BIG], [BIG], [BIG], [BIG]])
    # A read result is about 1,200 tokens: the batch waits for two.
    assert [event.request for event in ctx.trace.of("clearing")] == [3]
    assert batches(ctx)[0]["results"] == 2
    assert_holds(check_all(fake.calls, allowed_rewrites={3}))


def test_small_and_excluded_results_stay() -> None:
    policy = Steps(calls("memory", "ping", "read"), calls("read"), done)
    _, create = fake_for(ANTHROPIC, policy=policy)
    ctx = context_for(
        ANTHROPIC,
        budget=100_000,
        clearing=clearing(trigger=1_000, keep=0, exclude=["memory"], clear_at_least=0),
    )
    requests = drive(ctx, create, [[BIG, "pong", BIG], [BIG]])
    last = list(raw_results(ANTHROPIC, requests[-1]).values())
    assert last[0] == BIG
    assert last[1] == "pong"
    assert isinstance(last[2], str) and last[2].startswith('[cleared: tool "read"')
    assert last[3] == BIG


@pytest.mark.parametrize(("min_tokens", "cleared"), [(100, False), (0, True)])
def test_results_under_min_tokens_stay(min_tokens: int, cleared: bool) -> None:
    medium = "result line\n" * 20  # About 60 tokens: more than a stub, under 100.
    policy = Steps(calls("read"), calls("read"), done)
    _, create = fake_for(ANTHROPIC, policy=policy)
    ctx = context_for(
        ANTHROPIC,
        budget=100_000,
        clearing=clearing(trigger=100, keep=0, clear_at_least=0, min_tokens=min_tokens),
    )
    requests = drive(ctx, create, [[medium], [medium]])
    first = next(iter(raw_results(ANTHROPIC, requests[-1]).values()))
    assert (first != medium) is cleared


def test_without_fetch_the_stubs_dont_offer_it_and_the_originals_are_kept() -> None:
    store = MemoryStore()
    policy = Steps(calls("read"), calls("read"), done)
    _, create = fake_for(CHAT, policy=policy)
    ctx = context_for(
        CHAT, budget=100_000, clearing=Clearing(trigger=1_000, fetch=False, keep=0, store=store)
    )
    requests = drive(ctx, create, [[BIG], [BIG]])
    content = next(iter(raw_results(CHAT, requests[-1]).values()))
    assert isinstance(content, str)
    found = re.fullmatch(r'\[cleared: tool "read", id ([0-9a-f]{4}), [\d,]+ tokens\]', content)
    assert found is not None
    assert ctx.fetch(found.group(1)) == BIG
    assert store.read(f"offload/{found.group(1)}.json") is not None


def tricky_contents(fmt: Format) -> list[JSONValue]:
    match fmt:
        case Format.ANTHROPIC_MESSAGES:
            image: JSONObject = {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
            }
            search: JSONObject = {
                "type": "search_result",
                "source": "https://example.com/a",
                "title": "A",
                "content": [{"type": "text", "text": TRICKY}],
                "citations": {"enabled": True},
            }
            return [TRICKY, [{"type": "text", "text": TRICKY}, image, search]]
        case Format.OPENAI_RESPONSES:
            picture: JSONObject = {
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBORw0KGgo=",
                "detail": "auto",
            }
            return [TRICKY, [{"type": "input_text", "text": TRICKY}, picture]]
        case Format.OPENAI_CHAT:
            return [TRICKY, [{"type": "text", "text": TRICKY}]]


@pytest.mark.parametrize(
    ("fmt", "kind"), [(fmt, kind) for fmt in Format for kind in ("text", "parts")]
)
def test_artiik_fetch_gives_the_model_the_original_byte_for_byte(fmt: Format, kind: str) -> None:
    content = tricky_contents(fmt)[0 if kind == "text" else 1]
    policy = Steps(calls("read"), calls("read"), fetch_a_stub, done)
    fake, create = fake_for(fmt, policy=policy)
    ctx = context_for(fmt, budget=100_000, clearing=clearing(trigger=200, keep=1))
    requests = drive(ctx, create, [[content], ["ok"], [None]])
    first_call = next(iter(raw_results(fmt, requests[1])))
    sent = raw_results(fmt, requests[1])[first_call]
    # The second request carries the original; the third, a stub; the fourth, the
    # original again, as the result of artiik_fetch.
    assert json.dumps(sent, ensure_ascii=False) == json.dumps(content, ensure_ascii=False)
    stubbed = raw_results(fmt, requests[2])[first_call]
    assert isinstance(stubbed, str) and STUB_ID.search(stubbed)
    fetched = list(raw_results(fmt, requests[3]).values())[-1]
    assert json.dumps(fetched, ensure_ascii=False) == json.dumps(sent, ensure_ascii=False)
    assert [event.data["found"] for event in ctx.trace.of("fetch")] == [True]
    assert_holds(check_all(fake.calls, allowed_rewrites={2}))


@pytest.mark.parametrize("fmt", list(Format))
def test_artiik_fetch_answers_a_bad_id_with_an_error_the_model_can_read(fmt: Format) -> None:
    ctx = context_for(fmt, budget=100_000, clearing=clearing())
    unknown = ctx.fetch_result(ToolUse(id="call_1", name=FETCH_TOOL, input={"id": "beef"}))
    missing = ctx.fetch_result(ToolUse(id="call_2", name=FETCH_TOOL, input={"path": "x"}))
    assert unknown == answer(fmt, "call_1", "No cleared tool output has the id 'beef'.", error=True)
    assert missing == answer(fmt, "call_2", 'artiik_fetch needs an "id" argument.', error=True)
    if fmt is ANTHROPIC:
        assert unknown["is_error"] is True
    assert [event.data for event in ctx.trace.of("fetch")] == [
        {"id": "beef", "found": False},
        {"id": None, "found": False},
    ]
    with pytest.raises(ValueError, match="isn't an artiik_fetch call"):
        ctx.fetch_result(ToolUse(id="call_3", name="read_log", input={}))
    with pytest.raises(KeyError, match="no cleared tool output"):
        ctx.fetch("beef")
    with pytest.raises(KeyError, match="doesn't clear"):
        context_for(fmt).fetch("beef")


def test_clearing_comes_before_compaction_and_can_spare_it() -> None:
    summaries: list[str] = []

    def summarize(messages: list[JSONObject]) -> str:
        summaries.append(json.dumps(messages))
        return "Summary."

    policy = Steps(calls("read"), calls("read"), calls("read"), done)
    _, create = fake_for(CHAT, policy=policy)
    ctx = context_for(
        CHAT,
        budget=20_000,
        compact_at=4_000,
        compaction=SummaryCompaction(summarize),
        clearing=clearing(trigger=2_500, keep=0, clear_at_least=0),
    )
    drive(ctx, create, [[BIG], [BIG], [BIG]])
    # Each request passes compact_at only with the new result in it, and clearing
    # the older ones brings it back under: nothing is summarized.
    assert len(batches(ctx)) == 2
    assert ctx.trace.of("compaction") == []
    assert summaries == []


def test_past_the_budget_a_small_batch_still_waits_and_the_guard_makes_room() -> None:
    policy = Steps(calls("read"), calls("read"), calls("read"), done)
    fake, create = fake_for(ANTHROPIC, policy=policy)
    ctx = context_for(
        ANTHROPIC,
        budget=3_500,
        clearing=clearing(trigger=1_000, keep=0, clear_at_least=1_000_000),
    )
    drive(ctx, create, [[BIG], [BIG], [BIG]])
    # Clearing one result at a time would change the prompt on every request.
    assert ctx.trace.of("clearing") == []
    assert [event.request for event in ctx.trace.of("guard")] == [3]
    assert_holds(check_all(fake.calls, budget=3_500, allowed_rewrites={3}))


def thinking(base: Callable[[Sequence[Message]], Reply]) -> Callable[[Sequence[Message]], Reply]:
    def step(conversation: Sequence[Message]) -> Reply:
        reply = base(conversation)
        return Reply(text=reply.text, tool_calls=reply.tool_calls, thinking="Let me think.")

    return step


def test_results_before_a_thinking_block_stay_with_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    steps = [calls("read"), calls("read"), calls("read"), calls("read"), done]
    policy = Steps(*(thinking(step) for step in steps))
    # This fake rejects a thinking block whose earlier turns changed.
    fake, create = fake_for(ANTHROPIC, policy=policy, check_thinking_prefix=True)
    ctx = context_for(ANTHROPIC, budget=100_000, clearing=clearing(trigger=1_000, keep=0))
    with caplog.at_level(logging.WARNING, logger="artiik"):
        drive(ctx, create, [[BIG], [BIG], [BIG], [BIG]])
    assert batches(ctx) == []
    held = [event.data for event in ctx.trace.of("clearing")]
    assert held == [{"strategy": "Clearing", "outcome": "held", "results": 1}]
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert "before a thinking block" in warnings[0]
    assert "AnthropicClearing" in warnings[0]
    assert_holds(check_all(fake.calls))


def test_reasoning_items_dont_hold_clearing_back() -> None:
    steps = [calls("read"), calls("read"), calls("read"), done]
    fake, create = fake_for(RESPONSES, policy=Steps(*(thinking(step) for step in steps)))
    ctx = context_for(RESPONSES, budget=100_000, clearing=clearing(trigger=1_000, keep=0))
    drive(ctx, create, [[BIG], [BIG], [BIG]])
    assert batches(ctx)
    rewrites = {event.request for event in ctx.trace.of("clearing")}
    assert_holds(check_all(fake.calls, allowed_rewrites=rewrites))


def test_a_batch_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    policy = Steps(calls("read"), calls("read"), done)
    _, create = fake_for(ANTHROPIC, policy=policy)
    ctx = context_for(ANTHROPIC, budget=100_000, clearing=clearing(trigger=1_000, keep=0))
    with caplog.at_level(logging.INFO, logger="artiik"):
        drive(ctx, create, [[BIG], [BIG]])
    logged = [record.getMessage() for record in caplog.records if "clearing" in record.getMessage()]
    assert len(logged) == 1
    assert re.fullmatch(
        r"artiik clearing: cleared 1 tool results before request 2 \(about \d+ tokens, now "
        r"about \d+\)",
        logged[0],
    )


@pytest.mark.parametrize("fmt", list(Format))
def test_tool_pairs_stay_intact_after_any_clearing(fmt: Format) -> None:
    generator = random.Random(18)
    for _ in range(12):
        policy = ToolLoopPolicy(
            calls_per_step=generator.randint(1, 4), steps=generator.randint(1, 4)
        )
        fake, create = fake_for(fmt, policy=policy)
        ctx = context_for(
            fmt,
            budget=60_000,
            clearing=clearing(
                trigger=generator.randint(500, 8_000),
                keep=generator.randint(0, 5),
                clear_at_least=generator.choice([0, 500, 3_000]),
                min_tokens=generator.choice([0, 100, 400]),
                exclude=generator.choice([(), ("read_log",)]),
            ),
        )
        environment = Environment(output_tokens=generator.randint(20, 600))
        for turn in range(6):
            ctx.add(user(f"Turn {turn}."))
            for _ in range(20):
                request = ctx.prepare()
                # Whatever clearing just did, the history is a valid request.
                validate(fmt, ctx.history)
                ctx.record(create(**request))
                pending = ctx.pending_tool_calls()
                if not pending:
                    break
                ctx.add(
                    *grouped(
                        fmt,
                        [
                            answer(fmt, call.id, environment.run(call.name, call.input))
                            for call in pending
                        ],
                    )
                )
        assert_holds(check_tool_pairs(fake.calls))
        uses = [use.id for message in ctx.history for use in message.tool_uses]
        answered = [
            result.tool_use_id for message in ctx.history for result in message.tool_results
        ]
        assert sorted(uses) == sorted(answered)


# Long sessions: the Tier 0 checks


SETUPS = [
    "anthropic",
    "responses",
    "chat",
    "anthropic with compaction",
    "responses with compaction",
    "chat with compaction",
]


def setup(name: str) -> tuple[FakeAnthropic | FakeOpenAI, Context, Create]:
    policy = ReadingPolicy(calls_per_step=2, steps=3)
    compacting = name.endswith("with compaction")
    budget = 12_000 if compacting else 24_000
    store = MemoryStore()
    match name.split()[0]:
        case "anthropic":
            anthropic = FakeAnthropic(policy=policy)
            strategy = AnthropicCompaction(anthropic) if compacting else None
            ctx = context_for(
                ANTHROPIC, budget=budget, compaction=strategy, clearing=Clearing(store=store)
            )
            return anthropic, ctx, anthropic.messages.create
        case "responses":
            openai = FakeOpenAI(policy=policy)
            strategy = OpenAICompaction(openai) if compacting else None
            ctx = context_for(
                RESPONSES, budget=budget, compaction=strategy, clearing=Clearing(store=store)
            )
            return openai, ctx, openai.responses.create
        case _:
            openai = FakeOpenAI(policy=policy)
            strategy = (
                SummaryCompaction(lambda _: "Summary of the work so far.") if compacting else None
            )
            ctx = context_for(
                CHAT, budget=budget, compaction=strategy, clearing=Clearing(store=store)
            )
            return openai, ctx, openai.chat.completions.create


def originals_and_fetches(
    fmt: Format, calls_made: Sequence[RecordedCall]
) -> list[tuple[JSONValue, JSONValue]]:
    """Each artiik_fetch result in the requests, with the original it should give back."""
    first: dict[str, JSONValue] = {}
    stub_of: dict[str, str] = {}
    pairs: list[tuple[JSONValue, JSONValue]] = []
    seen: set[str] = set()
    for call in calls_made:
        if not call.ok or call.is_compaction:
            continue
        conversation = call.conversation()
        fetches = {
            use.id: cast(str, cast(JSONObject, use.input)["id"])
            for message in conversation
            for use in message.tool_uses
            if use.name == FETCH_TOOL and isinstance(use.input, dict)
        }
        for call_id, content in raw_results(fmt, call.request).items():
            if isinstance(content, str) and (found := STUB_ID.search(content)):
                stub_of[found.group(1)] = call_id
            elif call_id not in first:
                first[call_id] = content
            if call_id in fetches and call_id not in seen:
                seen.add(call_id)
                pairs.append((first[stub_of[fetches[call_id]]], content))
    return pairs


@pytest.mark.parametrize("name", SETUPS)
def test_long_sessions_change_the_prompt_prefix_once_per_batch(name: str) -> None:
    fake, ctx, create = setup(name)
    calls_made: dict[int, int] = {}

    def tracked(**kwargs: Any) -> object:
        calls_made[ctx.trace.of("prepare")[-1].request] = len(fake.calls)
        return create(**kwargs)

    run_context(
        ctx, tracked, [f"Turn {n}." for n in range(40)], environment=Environment(output_tokens=150)
    )
    events = batches(ctx)
    batch_calls = {calls_made[event.request] for event in ctx.trace.of("clearing")}
    others = {
        calls_made[event.request]
        for event in ctx.trace.events
        if event.kind == "guard" or event.data.get("outcome") == "compacted"
    }
    assert len(events) >= 3
    assert len(events) <= len(fake.calls) // 8
    if name.endswith("with compaction"):
        assert others
    budget = cast(int, ctx.budget)
    assert_holds(check_all(fake.calls, budget=budget, allowed_rewrites=batch_calls | others))
    # The prompt prefix changes on the request of each batch, and nowhere else.
    rewrites = {violation.call for violation in check_append_only(fake.calls)}
    assert batch_calls - others <= rewrites
    assert rewrites <= batch_calls | others
    # The model read cleared outputs back, each exactly as it was first sent.
    pairs = originals_and_fetches(ctx.api, fake.calls)
    assert len(pairs) >= 2
    for before, after in pairs:
        assert json.dumps(after, ensure_ascii=False) == json.dumps(before, ensure_ascii=False)
    assert all(event.data["found"] for event in ctx.trace.of("fetch"))


# Anthropic's server-side clearing


def test_the_anthropic_edit_has_the_documented_shape() -> None:
    documented: JSONObject = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 30000},
        "keep": {"type": "tool_uses", "value": 3},
        "clear_at_least": {"type": "input_tokens", "value": 5000},
        "exclude_tools": ["web_search"],
    }
    strategy = AnthropicClearing(
        trigger=30_000, keep=3, clear_at_least=5_000, exclude=["web_search"]
    )
    assert strategy.edit(trigger=30_000, clear_at_least=5_000) == documented
    assert list(strategy.edit(trigger=30_000, clear_at_least=5_000)) == list(documented)
    assert AnthropicClearing().edit() == {
        "type": "clear_tool_uses_20250919",
        "keep": {"type": "tool_uses", "value": 3},
    }
    uses = AnthropicClearing(trigger_tool_uses=12, clear_inputs=True).edit(trigger=99)
    assert uses["trigger"] == {"type": "tool_uses", "value": 12}
    assert uses["clear_tool_inputs"] is True
    assert AnthropicClearing(clear_inputs=["bash"]).edit()["clear_tool_inputs"] == ["bash"]
    # Through a context, the trigger and the least to clear come from the budget.
    ctx = context_for(ANTHROPIC, budget=60_000, clearing=strategy)
    ctx.add(user("Hello."))
    sent = ctx.prepare()
    assert sent["context_management"] == {"edits": [documented]}
    assert sent["extra_headers"] == {"anthropic-beta": ANTHROPIC_BETA}
    ctx = context_for(ANTHROPIC, budget=60_000, clearing=AnthropicClearing(keep=5))
    ctx.add(user("Hello."))
    edit = cast(list[JSONObject], ctx.prepare(betas=[])["context_management"]["edits"])[0]
    assert edit["trigger"] == {"type": "input_tokens", "value": 30_000}
    assert edit["clear_at_least"] == {"type": "input_tokens", "value": 7_500}


def test_the_anthropic_edit_joins_the_edits_already_there() -> None:
    strategy = AnthropicClearing()
    thinking_edit: JSONObject = {"type": "clear_thinking_20251015"}
    request: dict[str, Any] = {"context_management": {"edits": [thinking_edit]}, "betas": ["x"]}
    strategy.configure(ANTHROPIC, request)
    assert [edit["type"] for edit in request["context_management"]["edits"]] == [
        "clear_thinking_20251015",
        "clear_tool_uses_20250919",
    ]
    assert request["betas"] == ["x", ANTHROPIC_BETA]
    own: JSONObject = {
        "type": "clear_tool_uses_20250919",
        "keep": {"type": "tool_uses", "value": 9},
    }
    request = {"context_management": {"edits": [own]}}
    strategy.configure(ANTHROPIC, request)
    assert request["context_management"]["edits"] == [own]
    with pytest.raises(ValueError, match="anthropic-messages"):
        strategy.configure(CHAT, {})


def test_anthropic_clearing_goes_before_threshold_compaction() -> None:
    ctx = context_for(
        ANTHROPIC,
        budget=200_000,
        compaction=AnthropicThresholdCompaction(),
        clearing=AnthropicClearing(),
    )
    ctx.add(user("Hello."))
    sent = ctx.prepare()
    edits = cast(list[JSONObject], cast(JSONObject, sent["context_management"])["edits"])
    assert [edit["type"] for edit in edits] == ["clear_tool_uses_20250919", "compact_20260112"]
    assert sent["extra_headers"] == {"anthropic-beta": f"{ANTHROPIC_BETA},compact-2026-01-12"}


def test_anthropic_clearing_needs_the_beta_endpoint() -> None:
    fake = FakeAnthropic()
    ctx = context_for(ANTHROPIC, budget=60_000, clearing=AnthropicClearing())
    ctx.add(user("Hello."))
    with pytest.raises(TypeError, match="context_management"):
        fake.messages.create(**ctx.prepare())
    ctx.record(fake.beta.messages.create(**ctx.prepare()))


def test_anthropic_clearing_leaves_the_history_alone_and_traces_what_the_api_cleared() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(calls_per_step=2, steps=3))
    ctx = context_for(
        ANTHROPIC,
        budget=60_000,
        clearing=AnthropicClearing(keep=2),
        params={"cache_control": {"type": "ephemeral"}},
    )
    run_context(
        ctx,
        fake.beta.messages.create,
        [f"Turn {n}." for n in range(30)],
        environment=Environment(output_tokens=400),
    )
    applied = [
        cast(JSONObject, call.response)["context_management"]
        for call in fake.calls
        if cast(JSONObject, call.response).get("context_management") != {"applied_edits": []}
    ]
    events = [event.data for event in ctx.trace.of("clearing")]
    assert len(events) == len(applied) > 10
    for event, response in zip(events, applied, strict=True):
        edit = cast(list[JSONObject], cast(JSONObject, response)["applied_edits"])[0]
        assert event == {
            "strategy": "provider",
            "edit": "clear_tool_uses_20250919",
            "cleared_tool_uses": edit["cleared_tool_uses"],
            "cleared_input_tokens": edit["cleared_input_tokens"],
        }
    # artiik sends the whole history every time: the requests only ever append.
    assert_holds(check_all(fake.calls, budget=60_000))
    assert ctx.trace.of("guard") == []
    assert max(call.tokens() for call in fake.calls) <= 30_000


def test_responses_with_applied_edits_dont_calibrate_the_estimator() -> None:
    ctx = context_for(ANTHROPIC, budget=60_000, clearing=AnthropicClearing())
    ratio = ctx.estimator.ratio(ANTHROPIC, MODEL)
    edited: JSONObject = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "Hi."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 300, "output_tokens": 2},
        "context_management": {
            "applied_edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "cleared_tool_uses": 4,
                    "cleared_input_tokens": 9_000,
                }
            ]
        },
    }
    ctx.add(user("Hello " * 300))
    full = ctx.estimate()
    ctx.prepare()
    ctx.record(edited)
    assert ctx.estimator.ratio(ANTHROPIC, MODEL) == ratio
    # The count of what the model read still anchors the next estimate.
    assert ctx.estimate() < full
    ctx.add(user("Again " * 300))
    ctx.prepare()
    ctx.record(
        {
            **edited,
            "usage": {"input_tokens": 1_500, "output_tokens": 2},
            "context_management": {"applied_edits": []},
        }
    )
    assert ctx.estimator.ratio(ANTHROPIC, MODEL) != ratio


def test_exact_counts_with_anthropic_clearing_dont_calibrate_the_estimator() -> None:
    fake = FakeAnthropic()
    ctx = context_for(
        ANTHROPIC,
        budget=8_000,
        counter=AnthropicTokenCounter(fake),
        clearing=AnthropicClearing(trigger=2_000, keep=0, clear_at_least=0),
    )
    ratio = ctx.estimator.ratio(ANTHROPIC, MODEL)
    history: list[JSONObject] = [user("Go.")]
    for number in range(5):
        history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{number}", "name": "read_log", "input": {}}
                ],
            }
        )
        history.extend(results(ANTHROPIC, (f"t{number}", BIG)))
    ctx.add(history)
    sent = ctx.prepare()
    # The full history is over the budget, but the count is of what's left after clearing.
    assert fake.count_requests
    assert ctx.trace.of("guard") == []
    assert ctx.estimator.ratio(ANTHROPIC, MODEL) == ratio
    ctx.record(fake.beta.messages.create(**sent))
    assert fake.calls[-1].tokens() <= 8_000
