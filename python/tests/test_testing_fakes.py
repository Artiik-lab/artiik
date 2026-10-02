"""The fake clients check requests the way the provider APIs do, and answer like them."""

from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import cast

import pytest

from artiik.formats import openai_responses
from artiik.formats._json import expect_list, expect_object
from artiik.messages import (
    Compaction,
    Format,
    Image,
    JSONObject,
    JSONValue,
    Message,
    Text,
)
from artiik.testing import (
    COMPACTION_BETA,
    DigestSummarizer,
    Environment,
    FakeAnthropic,
    FakeAPIError,
    FakeObject,
    FakeOpenAI,
    ForgetfulSummarizer,
    Reply,
    ScriptedPolicy,
    ToolCall,
    ToolLoopPolicy,
    assert_holds,
    check_all,
    default_request,
    response_json,
    run_session,
)
from artiik.testing.caching import AnthropicCache, AnthropicUsage, OpenAICache, Unit
from artiik.testing.driver import Create
from artiik.testing.fake_anthropic import THRESHOLD_BETA, THRESHOLD_MINIMUM
from artiik.testing.tokens import (
    BLOCK_OVERHEAD,
    MEDIA_TOKENS,
    MESSAGE_OVERHEAD,
    count_json,
    count_message,
    count_text,
)

MODEL = "fake-model"


def user(text: str) -> JSONObject:
    return {"role": "user", "content": text}


def assistant(content: JSONValue) -> JSONObject:
    return {"role": "assistant", "content": content}


def objects(data: JSONObject, key: str) -> list[JSONObject]:
    return [expect_object(item, key) for item in expect_list(data[key], key)]


def usage_of(data: JSONObject) -> JSONObject:
    return expect_object(data["usage"], "usage")


def replies(*items: Reply) -> ScriptedPolicy:
    return ScriptedPolicy(items)


def client_for(fmt: Format) -> tuple[FakeAnthropic | FakeOpenAI, Create]:
    if fmt is Format.ANTHROPIC_MESSAGES:
        anthropic = FakeAnthropic()
        return anthropic, anthropic.messages.create
    openai = FakeOpenAI()
    create = (
        openai.responses.create
        if fmt is Format.OPENAI_RESPONSES
        else openai.chat.completions.create
    )
    return openai, create


# Anthropic Messages


def test_anthropic_answers_like_the_messages_api() -> None:
    fake = FakeAnthropic(policy=replies(Reply(text="Hello.")))
    response = fake.messages.create(model=MODEL, max_tokens=100, messages=[user("Hi")])
    assert isinstance(response, FakeObject)
    assert response.stop_reason == "end_turn"
    data = response_json(response)
    assert data["id"] == "msg_0000"
    assert objects(data, "content") == [{"type": "text", "text": "Hello."}]
    usage = usage_of(data)
    assert set(usage) == {
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
    }
    assert usage["input_tokens"] == fake.calls[0].tokens()
    assert fake.requests == [{"model": MODEL, "max_tokens": 100, "messages": [user("Hi")]}]


def test_anthropic_tool_calls_get_ids_and_the_tool_use_stop_reason() -> None:
    calls = (ToolCall(name="grep", arguments={"pattern": "error"}), ToolCall(name="ls"))
    fake = FakeAnthropic(policy=replies(Reply(text="Looking.", tool_calls=calls)))
    data = response_json(fake.messages.create(model=MODEL, max_tokens=100, messages=[user("Go")]))
    assert data["stop_reason"] == "tool_use"
    assert objects(data, "content") == [
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "toolu_0000_0", "name": "grep", "input": {"pattern": "error"}},
        {"type": "tool_use", "id": "toolu_0000_1", "name": "ls", "input": {}},
    ]


TOOL_USE = assistant([{"type": "tool_use", "id": "t1", "name": "ls", "input": {}}])
TOOL_RESULT: JSONObject = {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param([user("Go"), TOOL_USE], "without tool_result blocks", id="unanswered"),
        pytest.param(
            [user("Go"), TOOL_USE, user("And?")], "without tool_result blocks", id="not-next"
        ),
        pytest.param(
            [user("Go"), {"role": "user", "content": [TOOL_RESULT]}],
            "unexpected tool_use_id",
            id="orphaned-result",
        ),
        pytest.param(
            [
                user("Go"),
                TOOL_USE,
                {"role": "user", "content": [{"type": "text", "text": "x"}, TOOL_RESULT]},
            ],
            "must come before any other content",
            id="result-after-text",
        ),
    ],
)
def test_anthropic_rejects_broken_tool_pairs(messages: list[JSONObject], expected: str) -> None:
    fake = FakeAnthropic()
    with pytest.raises(FakeAPIError, match=expected) as caught:
        fake.messages.create(model=MODEL, max_tokens=100, messages=messages)
    assert caught.value.status_code == 400
    assert fake.calls[0].error is caught.value


@pytest.mark.parametrize("missing", ["model", "max_tokens", "messages"])
def test_anthropic_requires_model_max_tokens_and_messages(missing: str) -> None:
    request: JSONObject = {"model": MODEL, "max_tokens": 10, "messages": [user("x")]}
    del request[missing]
    with pytest.raises(FakeAPIError, match=f"{missing}: Field required"):
        FakeAnthropic().messages.create(**request)


def test_anthropic_allows_four_cache_breakpoints() -> None:
    fake = FakeAnthropic(policy=replies(Reply(text="ok")))
    marked: JSONObject = {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
    messages = [{"role": "user", "content": [marked] * 4}]
    fake.messages.create(model=MODEL, max_tokens=10, messages=messages)
    with pytest.raises(FakeAPIError, match="A maximum of 4 blocks with cache_control"):
        fake.messages.create(
            model=MODEL, max_tokens=10, messages=messages, cache_control={"type": "ephemeral"}
        )
    with pytest.raises(FakeAPIError, match="Found 5"):
        fake.messages.create(model=MODEL, max_tokens=10, messages=messages, system=[marked])


def test_anthropic_rejects_prompts_over_the_context_window() -> None:
    fake = FakeAnthropic(context_window=50)
    with pytest.raises(FakeAPIError, match="prompt is too long"):
        fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x" * 400)])


def test_anthropic_betas_need_the_beta_endpoint() -> None:
    with pytest.raises(TypeError, match="betas"):
        FakeAnthropic().messages.create(
            model=MODEL, max_tokens=10, messages=[user("x")], betas=[COMPACTION_BETA]
        )


@pytest.mark.parametrize("name", ["compaction", "context_management", "mcp_servers"])
def test_anthropic_beta_parameters_need_the_beta_endpoint(name: str) -> None:
    # Like the SDK, whose plain methods don't take these keyword arguments.
    fake = FakeAnthropic()
    extra: dict[str, JSONValue] = {name: {}}
    with pytest.raises(TypeError, match=name):
        fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x")], **extra)
    with pytest.raises(TypeError, match=name):
        fake.messages.count_tokens(model=MODEL, messages=[user("x")], **extra)
    with pytest.raises(TypeError, match="betas"):
        fake.models.retrieve(MODEL, betas=[COMPACTION_BETA])


@pytest.mark.parametrize(
    "stop_reason",
    ["max_tokens", "stop_sequence", "pause_turn", "refusal", "model_context_window_exceeded"],
)
def test_anthropic_faults_can_set_any_stop_reason(stop_reason: str) -> None:
    fake = FakeAnthropic(policy=replies(Reply(text="Partial")), faults={0: stop_reason})
    data = response_json(fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x")]))
    assert data["stop_reason"] == stop_reason


def test_anthropic_faults_raise_errors_and_the_call_is_still_recorded() -> None:
    overloaded = FakeAPIError.anthropic(529, "overloaded_error", "Overloaded")
    fake = FakeAnthropic(policy=replies(Reply(text="ok")), faults={0: overloaded})
    with pytest.raises(FakeAPIError) as caught:
        fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x")])
    assert caught.value is overloaded
    assert caught.value.body == {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    assert not fake.calls[0].ok
    retry = response_json(fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x")]))
    assert retry["id"] == "msg_0001"


def test_thinking_blocks_must_come_back_unchanged() -> None:
    reply = Reply(thinking="Read the log first.", tool_calls=(ToolCall(name="read_log"),))
    fake = FakeAnthropic(policy=replies(reply, Reply(text="Done.")))
    data = response_json(fake.messages.create(model=MODEL, max_tokens=100, messages=[user("Go")]))
    thinking, tool_use = objects(data, "content")
    assert thinking["type"] == "thinking"
    assert thinking["thinking"] == "Read the log first."
    assert isinstance(thinking["signature"], str)
    result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "ok"}],
    }
    fake.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[user("Go"), assistant([thinking, tool_use]), result],
    )
    edited = {**thinking, "thinking": "Read the log."}
    with pytest.raises(FakeAPIError, match="Invalid `signature` in `thinking` block"):
        fake.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[user("Go"), assistant([edited, tool_use]), result],
        )


def test_raw_blocks_are_returned_as_given() -> None:
    server_call: JSONObject = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "status"},
    }
    fake = FakeAnthropic(policy=replies(Reply(raw=(server_call,), text="Found it.")))
    data = response_json(fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x")]))
    assert objects(data, "content") == [server_call, {"type": "text", "text": "Found it."}]


# Anthropic on-demand compaction

HISTORY: list[JSONObject] = [
    user("Name the entities of a recipe app."),
    assistant("Recipe, Ingredient and Step."),
    user("Now the fields of Recipe."),
]


def compaction_request(messages: Sequence[JSONObject], **extra: JSONValue) -> JSONObject:
    return {
        "model": MODEL,
        "max_tokens": 4096,
        "betas": [COMPACTION_BETA],
        "messages": list(messages),
        "compaction": {"type": "summarize"},
        **extra,
    }


def compact(fake: FakeAnthropic) -> JSONObject:
    data = response_json(fake.beta.messages.create(**compaction_request(HISTORY)))
    assert data["stop_reason"] == "compaction"
    [block] = objects(data, "content")
    return block


def test_compaction_returns_a_signed_block_and_counts_usage_in_iterations() -> None:
    fake = FakeAnthropic()
    data = response_json(fake.beta.messages.create(**compaction_request(HISTORY)))
    assert data["stop_reason"] == "compaction"
    [block] = objects(data, "content")
    assert set(block) == {"type", "content", "signature"}
    assert block["type"] == "compaction"
    summary = block["content"]
    assert isinstance(summary, str)
    assert "Name the entities of a recipe app." in summary
    usage = usage_of(data)
    assert (usage["input_tokens"], usage["output_tokens"]) == (0, 0)
    assert objects(usage, "iterations") == [
        {
            "type": "compaction",
            "input_tokens": fake.calls[0].tokens(),
            "output_tokens": count_text(summary),
        }
    ]
    assert fake.calls[0].is_compaction


def test_the_block_goes_first_on_every_later_request() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(steps=0))
    block = compact(fake)
    fake.beta.messages.create(
        model=MODEL,
        max_tokens=100,
        betas=[COMPACTION_BETA],
        messages=[assistant([block]), user("Continue.")],
    )
    marked = {**block, "cache_control": {"type": "ephemeral"}}
    fake.beta.messages.create(
        model=MODEL,
        max_tokens=100,
        betas=[COMPACTION_BETA],
        messages=[{"role": "user", "content": [marked, {"type": "text", "text": "Continue."}]}],
    )
    fake.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[assistant([block]), user("Continue.")],
        extra_headers={"anthropic-beta": f"another-beta, {COMPACTION_BETA}"},
    )
    assert all(call.ok for call in fake.calls)


def misplaced(block: JSONObject) -> list[JSONObject]:
    return [user("Earlier."), assistant([block])]


def twice(block: JSONObject) -> list[JSONObject]:
    return [assistant([block, block])]


def edited_content(block: JSONObject) -> list[JSONObject]:
    return [assistant([{**block, "content": f"{block['content']} And more."}])]


def edited_signature(block: JSONObject) -> list[JSONObject]:
    return [assistant([{**block, "signature": "sig_forged"}])]


def with_null_fields(block: JSONObject) -> list[JSONObject]:
    return [assistant([{**block, "citations": None}])]


@pytest.mark.parametrize(
    ("send_back", "code", "expected"),
    [
        pytest.param(misplaced, "compaction_block_misplaced", "must come first", id="misplaced"),
        pytest.param(twice, None, "only one compaction block", id="twice"),
        pytest.param(edited_content, "compaction_content_mismatch", "match", id="edited-content"),
        pytest.param(edited_signature, "compaction_signature_invalid", "invalid", id="forged"),
        pytest.param(with_null_fields, None, "citations: Extra inputs", id="null-fields"),
    ],
)
def test_compaction_blocks_must_come_back_first_and_unchanged(
    send_back: Callable[[JSONObject], list[JSONObject]], code: str | None, expected: str
) -> None:
    fake = FakeAnthropic()
    block = compact(fake)
    with pytest.raises(FakeAPIError, match=expected) as caught:
        fake.beta.messages.create(
            model=MODEL, max_tokens=100, betas=[COMPACTION_BETA], messages=send_back(block)
        )
    assert caught.value.code == code


def test_compaction_blocks_need_the_beta() -> None:
    fake = FakeAnthropic()
    block = compact(fake)
    with pytest.raises(FakeAPIError, match="'compaction' is not one of the expected"):
        fake.messages.create(model=MODEL, max_tokens=100, messages=[assistant([block])])


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        pytest.param({"betas": []}, "requires anthropic-beta", id="no-beta"),
        pytest.param({"context_management": {"edits": []}}, "context_management", id="context"),
        pytest.param({"stop_sequences": ["END"]}, "stop_sequences", id="stop-sequences"),
        pytest.param({"tool_choice": {"type": "any"}}, "tool_choice", id="tool-choice-any"),
        pytest.param(
            {"tool_choice": {"type": "tool", "name": "ls"}}, "tool_choice", id="tool-choice-tool"
        ),
        pytest.param({"output_config": {"format": {"type": "json"}}}, "format", id="format"),
        pytest.param({"compaction": {"type": "truncate"}}, "summarize", id="type"),
        pytest.param({"compaction": {"instructions": "  "}}, "non-blank", id="blank"),
        pytest.param({"compaction": {"instructions": "x" * 16_385}}, "16384", id="long"),
        pytest.param({"compaction": "summarize"}, "dictionary", id="not-an-object"),
    ],
)
def test_compaction_requests_are_checked(extra: JSONObject, expected: str) -> None:
    with pytest.raises(FakeAPIError, match=expected):
        FakeAnthropic().beta.messages.create(**compaction_request(HISTORY, **extra))


def test_compaction_accepts_instructions_and_harmless_tool_choices() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.instructions: list[str | None] = []

        def summarize(self, conversation: Sequence[Message], instructions: str | None) -> str:
            self.instructions.append(instructions)
            return "Summary."

    recorder = Recorder()
    fake = FakeAnthropic(summarizer=recorder)
    fake.beta.messages.create(**compaction_request(HISTORY, tool_choice={"type": "auto"}))
    fake.beta.messages.create(
        **compaction_request(HISTORY, compaction={"type": "summarize", "instructions": "Keep ids."})
    )
    assert recorder.instructions == [None, "Keep ids."]


def test_compaction_is_refused_for_unsupported_models_and_empty_conversations() -> None:
    fake = FakeAnthropic(compaction_models={"large-model"})
    with pytest.raises(FakeAPIError, match="does not support compaction"):
        fake.beta.messages.create(**compaction_request(HISTORY))
    with pytest.raises(FakeAPIError, match="nothing to summarize") as caught:
        FakeAnthropic().beta.messages.create(**compaction_request([]))
    assert caught.value.code == "compaction_nothing_to_summarize"


@pytest.mark.parametrize(
    "stop_reason",
    ["max_tokens", "model_context_window_exceeded", "tool_use", "refusal", "end_turn"],
)
def test_a_failed_summary_is_a_200_with_empty_content(stop_reason: str) -> None:
    fake = FakeAnthropic(faults={0: stop_reason})
    data = response_json(fake.beta.messages.create(**compaction_request(HISTORY)))
    assert data["stop_reason"] == stop_reason
    assert data["content"] == []
    [iteration] = objects(usage_of(data), "iterations")
    assert iteration["output_tokens"] == 0


def test_compaction_unavailable_is_a_529_with_an_error_code() -> None:
    error = FakeAPIError.anthropic(
        529, "overloaded_error", "Compaction is unavailable.", code="compaction_unavailable"
    )
    fake = FakeAnthropic(faults={0: error})
    with pytest.raises(FakeAPIError) as caught:
        fake.beta.messages.create(**compaction_request(HISTORY))
    assert caught.value.status_code == 529
    assert caught.value.body == {
        "type": "error",
        "error": {
            "type": "overloaded_error",
            "message": "Compaction is unavailable.",
            "details": {"error_code": "compaction_unavailable"},
        },
    }


# Anthropic prompt caching


def test_an_anthropic_beta_header_replaces_the_betas_list() -> None:
    # The SDK sends betas as the anthropic-beta header, and extra_headers wins.
    fake = FakeAnthropic()
    block = compact(fake)
    with pytest.raises(FakeAPIError, match="'compaction' is not one of the expected"):
        fake.beta.messages.create(
            model=MODEL,
            max_tokens=100,
            betas=[COMPACTION_BETA],
            messages=[assistant([block]), user("Continue.")],
            extra_headers={"Anthropic-Beta": "another-beta"},
        )


def test_the_models_api_reports_compaction_support_on_the_beta() -> None:
    fake = FakeAnthropic(compaction_models={"large-model"})
    large = response_json(fake.beta.models.retrieve("large-model", betas=[COMPACTION_BETA]))
    capabilities = expect_object(large["capabilities"], "capabilities")
    assert capabilities["compaction"] == {"supported": True, "summarize": {"supported": True}}
    small = response_json(fake.beta.models.retrieve("small-model", betas=[COMPACTION_BETA]))
    capabilities = expect_object(small["capabilities"], "capabilities")
    assert capabilities["compaction"] == {"supported": False, "summarize": {"supported": False}}
    plain = response_json(fake.models.retrieve("large-model"))
    assert plain["capabilities"] is None
    assert plain["max_input_tokens"] == fake.context_window
    assert fake.model_requests == ["large-model", "small-model", "large-model"]


def threshold_request(
    messages: Sequence[JSONObject], trigger: int = THRESHOLD_MINIMUM, **extra: JSONValue
) -> JSONObject:
    edit: JSONObject = {
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": trigger},
    }
    return {
        "model": MODEL,
        "max_tokens": 1_000,
        "betas": [THRESHOLD_BETA],
        "messages": list(messages),
        "context_management": {"edits": [edit]},
        **extra,
    }


LONG_HISTORY: list[JSONObject] = [user("Read this log: " + "x" * 220_000), assistant("Read.")]


def test_threshold_compaction_compacts_past_its_trigger() -> None:
    fake = FakeAnthropic(policy=replies(Reply(text="Done."), Reply(text="Fine.")))
    data = response_json(
        fake.beta.messages.create(**threshold_request([*LONG_HISTORY, user("Go on.")]))
    )
    block, text = objects(data, "content")
    assert set(block) == {"type", "content"}
    assert block["type"] == "compaction"
    assert text == {"type": "text", "text": "Done."}
    compaction, message = objects(usage_of(data), "iterations")
    assert compaction["type"] == "compaction"
    assert cast(int, compaction["input_tokens"]) > THRESHOLD_MINIMUM
    # The reply read the summary, and the top-level usage covers only that.
    assert message["type"] == "message"
    assert usage_of(data)["input_tokens"] == message["input_tokens"]
    assert cast(int, message["input_tokens"]) < 1_000
    # Sent back after the content it replaced, the block is where the model reads from.
    reply = assistant([block, text])
    fake.beta.messages.create(**threshold_request([*LONG_HISTORY, reply, user("Next.")]))
    assert fake.calls[1].tokens() < 1_000


def test_threshold_compaction_waits_for_its_trigger() -> None:
    fake = FakeAnthropic(policy=replies(Reply(text="Done.")))
    data = response_json(fake.beta.messages.create(**threshold_request(HISTORY)))
    assert [block["type"] for block in objects(data, "content")] == ["text"]
    assert [item["type"] for item in objects(usage_of(data), "iterations")] == ["message"]


@pytest.mark.parametrize(
    ("request_for", "expected"),
    [
        pytest.param(
            lambda: threshold_request(HISTORY, betas=[]), "requires anthropic-beta", id="no-beta"
        ),
        pytest.param(lambda: threshold_request(HISTORY, trigger=40_000), "50000", id="low"),
        pytest.param(
            lambda: threshold_request(
                [assistant([{"type": "compaction", "content": "Made up."}]), user("Go on.")]
            ),
            "doesn't match one the API returned",
            id="made-up-block",
        ),
    ],
)
def test_threshold_requests_are_checked(
    request_for: Callable[[], JSONObject], expected: str
) -> None:
    with pytest.raises(FakeAPIError, match=expected):
        FakeAnthropic().beta.messages.create(**request_for())


def test_threshold_compaction_cant_run_on_a_signed_block() -> None:
    fake = FakeAnthropic()
    block = compact(fake)
    request = threshold_request(
        [assistant([block]), user("Go on.")], betas=[THRESHOLD_BETA, COMPACTION_BETA]
    )
    with pytest.raises(FakeAPIError, match="signed compaction block"):
        fake.beta.messages.create(**request)
    with pytest.raises(FakeAPIError, match="does not support compact_20260112"):
        FakeAnthropic(compaction_models={"large-model"}).beta.messages.create(
            **threshold_request(HISTORY)
        )


def test_append_only_sessions_read_everything_the_last_call_cached() -> None:
    fake = FakeAnthropic()
    request: JSONObject = {
        **default_request(Format.ANTHROPIC_MESSAGES),
        "cache_control": {"type": "ephemeral"},
    }
    run_session(fake.messages.create, Format.ANTHROPIC_MESSAGES, ["a", "b", "c"], request=request)
    usages = [usage_of(call.response) for call in fake.calls if call.response is not None]
    assert usages[0]["cache_read_input_tokens"] == 0
    assert usages[0]["cache_creation_input_tokens"] == fake.calls[0].tokens()
    for before, after in pairwise(usages):
        written = cast(int, before["cache_read_input_tokens"]) + cast(
            int, before["cache_creation_input_tokens"]
        )
        assert after["cache_read_input_tokens"] == written
        assert after["input_tokens"] == 0


def test_a_rewrite_loses_the_cache() -> None:
    fake = FakeAnthropic(policy=ScriptedPolicy([Reply(text="ok")] * 2))
    cached = {"type": "ephemeral"}
    messages = [user("First question."), assistant("Answer."), user("Second question.")]
    fake.messages.create(model=MODEL, max_tokens=10, messages=messages, cache_control=cached)
    edited = [user("First question, edited."), *messages[1:]]
    data = response_json(
        fake.messages.create(model=MODEL, max_tokens=10, messages=edited, cache_control=cached)
    )
    assert usage_of(data)["cache_read_input_tokens"] == 0


def test_the_cache_lookup_only_looks_twenty_blocks_back() -> None:
    cached: JSONObject = {"type": "ephemeral"}
    first = [user("Start.")]
    blocks: list[JSONValue] = [{"type": "text", "text": f"part {index}"} for index in range(25)]
    midway: list[JSONValue] = [
        *blocks[:12],
        {"type": "text", "text": "part 12", "cache_control": cached},
    ]
    for content, expected_read in [(blocks, False), ([*midway, *blocks[13:]], True)]:
        fake = FakeAnthropic(policy=ScriptedPolicy([Reply(text="ok")] * 2))
        fake.messages.create(model=MODEL, max_tokens=10, messages=first, cache_control=cached)
        longer = [*first, assistant(content), user("Go on.")]
        data = response_json(
            fake.messages.create(model=MODEL, max_tokens=10, messages=longer, cache_control=cached)
        )
        read = fake.calls[0].tokens() if expected_read else 0
        assert usage_of(data)["cache_read_input_tokens"] == read


# OpenAI Responses


def test_responses_answers_like_the_responses_api() -> None:
    reply = Reply(text="Checking.", tool_calls=(ToolCall(name="read_log", arguments={"step": 0}),))
    fake = FakeOpenAI(policy=replies(reply))
    data = response_json(fake.responses.create(model=MODEL, input="Go"))
    assert (data["object"], data["status"]) == ("response", "completed")
    message, call = objects(data, "output")
    assert (message["type"], message["role"]) == ("message", "assistant")
    assert call == {
        "id": "fc_0000_0",
        "type": "function_call",
        "call_id": "call_0000_0",
        "name": "read_log",
        "arguments": '{"step":0}',
        "status": "completed",
    }
    usage = usage_of(data)
    assert usage["input_tokens"] == fake.calls[0].tokens()
    assert usage["total_tokens"] == fake.calls[0].tokens() + cast(int, usage["output_tokens"])


FUNCTION_CALL: JSONObject = {
    "type": "function_call",
    "call_id": "c1",
    "name": "read_log",
    "arguments": "{}",
}
FUNCTION_OUTPUT: JSONObject = {"type": "function_call_output", "call_id": "c1", "output": "ok"}


def test_responses_function_calls_and_outputs_must_pair_up() -> None:
    fake = FakeOpenAI(policy=replies(Reply(text="ok")))
    with pytest.raises(FakeAPIError, match="No tool call found for function call output"):
        fake.responses.create(model=MODEL, input=[user("Go"), FUNCTION_OUTPUT])
    with pytest.raises(FakeAPIError, match="No tool output found for function call c1"):
        fake.responses.create(model=MODEL, input=[user("Go"), FUNCTION_CALL])
    fake.responses.create(model=MODEL, input=[user("Go"), FUNCTION_CALL, FUNCTION_OUTPUT])


def test_responses_compact_server_side_past_the_threshold() -> None:
    fake = FakeOpenAI(policy=ScriptedPolicy([Reply(text="ok")] * 2))
    settings: list[JSONValue] = [{"type": "compaction", "compact_threshold": 200}]
    long_input = [user("x" * 2_000)]
    data = response_json(
        fake.responses.create(model=MODEL, input=long_input, context_management=settings)
    )
    compaction, message = objects(data, "output")
    assert set(compaction) == {"id", "type", "encrypted_content"}
    assert compaction["type"] == "compaction"
    assert cast(int, usage_of(data)["input_tokens"]) < 200
    # The docs allow dropping the input before the compaction item.
    carried = [compaction, message, user("Next.")]
    data = response_json(
        fake.responses.create(model=MODEL, input=carried, context_management=settings)
    )
    assert [item["type"] for item in objects(data, "output")] == ["message"]
    assert usage_of(data)["input_tokens"] == fake.calls[1].tokens() < 200
    tampered = {**compaction, "encrypted_content": "AAAA"}
    with pytest.raises(FakeAPIError, match="Invalid compaction item"):
        fake.responses.create(model=MODEL, input=[tampered, user("Next.")])


def test_responses_cache_the_prompt_the_model_read_after_compacting() -> None:
    class LongSummaries:
        def summarize(self, conversation: Sequence[Message], instructions: str | None) -> str:
            return "Summary. " * 1_000

    fake = FakeOpenAI(policy=ScriptedPolicy([Reply(text="ok")] * 2), summarizer=LongSummaries())
    settings: list[JSONValue] = [{"type": "compaction", "compact_threshold": 200}]
    data = response_json(
        fake.responses.create(model=MODEL, input=[user("x" * 2_000)], context_management=settings)
    )
    compaction, message = objects(data, "output")
    data = response_json(
        fake.responses.create(model=MODEL, input=[compaction, message, user("Next.")])
    )
    details = expect_object(usage_of(data)["input_tokens_details"], "details")
    assert cast(int, details["cached_tokens"]) >= 1_024


def test_responses_compact_endpoint() -> None:
    fake = FakeOpenAI(policy=replies(Reply(text="ok"), Reply(text="ok")))
    call: JSONObject = {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{}"}
    output: JSONObject = {"type": "function_call_output", "call_id": "c1", "output": "a.txt"}
    window = [user("Plan the release."), call, output, user("Ship it.")]
    data = response_json(fake.responses.compact(model=MODEL, input=window))
    assert data["object"] == "response.compaction"
    # The user messages come back word for word, then one compaction item.
    first, second, item = objects(data, "output")
    assert first == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Plan the release."}],
    }
    assert second["content"] == [{"type": "input_text", "text": "Ship it."}]
    assert item["type"] == "compaction"
    assert fake.calls[0].is_compaction
    fake.responses.create(model=MODEL, input=[first, second, item, user("Go on.")])
    # The model reads the user messages before the item, and nothing else before it.
    read = openai_responses.parse_items([first, second, item, user("Go on.")])
    assert fake.calls[1].tokens() == sum(count_message(message) for message in read)
    unanswered: JSONObject = {**call, "call_id": "stale"}
    fake.responses.create(model=MODEL, input=[unanswered, first, second, item, user("Go on.")])
    with pytest.raises(FakeAPIError, match="at least one user message"):
        fake.responses.compact(model=MODEL, input=[call, output])


def test_reasoning_items_must_come_back_unchanged() -> None:
    reply = Reply(thinking="Read the log first.", tool_calls=(ToolCall(name="read_log"),))
    fake = FakeOpenAI(policy=replies(reply, Reply(text="Done.")))
    data = response_json(fake.responses.create(model=MODEL, input="Go"))
    reasoning, call = objects(data, "output")
    assert reasoning["type"] == "reasoning"
    details = expect_object(usage_of(data)["output_tokens_details"], "details")
    assert details["reasoning_tokens"] == count_text("Read the log first.")
    output: JSONObject = {
        "type": "function_call_output",
        "call_id": call["call_id"],
        "output": "ok",
    }
    fake.responses.create(model=MODEL, input=[user("Go"), reasoning, call, output])
    tampered = {**reasoning, "encrypted_content": "dGFtcGVyZWQ="}
    with pytest.raises(FakeAPIError, match="could not be verified"):
        fake.responses.create(model=MODEL, input=[user("Go"), tampered, call, output])


def test_responses_needs_the_whole_input() -> None:
    with pytest.raises(FakeAPIError, match="previous_response_id") as caught:
        FakeOpenAI().responses.create(model=MODEL, input="Go", previous_response_id="resp_1")
    assert caught.value.body == {
        "error": {
            "message": caught.value.message,
            "type": "invalid_request_error",
            "param": "previous_response_id",
            "code": None,
        }
    }


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_responses_can_be_incomplete(reason: str) -> None:
    fake = FakeOpenAI(policy=replies(Reply(text="Partial")), faults={0: reason})
    data = response_json(fake.responses.create(model=MODEL, input="Go"))
    assert data["status"] == "incomplete"
    assert data["incomplete_details"] == {"reason": reason}


@pytest.mark.parametrize("fmt", [Format.OPENAI_RESPONSES, Format.OPENAI_CHAT])
def test_openai_rejects_input_over_the_context_window(fmt: Format) -> None:
    fake = FakeOpenAI(context_window=50)
    with pytest.raises(FakeAPIError) as caught:
        if fmt is Format.OPENAI_RESPONSES:
            fake.responses.create(model=MODEL, input="x" * 400)
        else:
            fake.chat.completions.create(model=MODEL, messages=[user("x" * 400)])
    assert caught.value.code == "context_length_exceeded"


# OpenAI Chat Completions


def test_chat_answers_like_chat_completions() -> None:
    reply = Reply(tool_calls=(ToolCall(name="read_log", arguments={"step": 1}),))
    fake = FakeOpenAI(policy=replies(reply))
    data = response_json(fake.chat.completions.create(model=MODEL, messages=[user("Go")]))
    [choice] = objects(data, "choices")
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"] == {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "tool_calls": [
            {
                "id": "call_0000_0",
                "type": "function",
                "function": {"name": "read_log", "arguments": '{"step":1}'},
            }
        ],
    }
    assert usage_of(data)["prompt_tokens"] == fake.calls[0].tokens()


ASSISTANT_CALL: JSONObject = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read_log", "arguments": "{}"}}
    ],
}
TOOL_MESSAGE: JSONObject = {"role": "tool", "tool_call_id": "c1", "content": "ok"}


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param([user("Go"), TOOL_MESSAGE], "preceeding message", id="orphaned"),
        pytest.param([user("Go"), ASSISTANT_CALL], "did not have response messages: c1", id="end"),
        pytest.param(
            [user("Go"), ASSISTANT_CALL, user("Next")],
            "did not have response messages: c1",
            id="interrupted",
        ),
    ],
)
def test_chat_tool_messages_must_answer_the_calls(
    messages: list[JSONObject], expected: str
) -> None:
    fake = FakeOpenAI()
    with pytest.raises(FakeAPIError, match=expected):
        fake.chat.completions.create(model=MODEL, messages=messages)


def test_chat_keeps_reasoning_hidden_and_can_stop_early() -> None:
    reply = Reply(text="Done.", thinking="Hidden.", raw=({"type": "server_block"},))
    fake = FakeOpenAI(policy=replies(reply), faults={0: "length"})
    data = response_json(fake.chat.completions.create(model=MODEL, messages=[user("Go")]))
    [choice] = objects(data, "choices")
    assert choice["message"] == {"role": "assistant", "content": "Done.", "refusal": None}
    assert choice["finish_reason"] == "length"


def test_openai_caches_prefixes_from_1024_tokens_in_128_token_steps() -> None:
    fake = FakeOpenAI(policy=ScriptedPolicy([Reply(text="ok")] * 4))
    short = [user("Hello.")]
    fake.chat.completions.create(model=MODEL, messages=short)
    data = response_json(
        fake.chat.completions.create(model=MODEL, messages=[*short, assistant("ok"), user("More.")])
    )
    details = expect_object(usage_of(data)["prompt_tokens_details"], "details")
    assert details["cached_tokens"] == 0
    long = [user("x" * 8_000)]
    fake.chat.completions.create(model=MODEL, messages=long)
    data = response_json(
        fake.chat.completions.create(model=MODEL, messages=[*long, assistant("ok"), user("More.")])
    )
    details = expect_object(usage_of(data)["prompt_tokens_details"], "details")
    assert details["cached_tokens"] == fake.calls[2].tokens() // 128 * 128


# Building blocks


def test_fake_objects_read_like_sdk_objects() -> None:
    data: JSONObject = {"content": [{"type": "text", "text": "Hi."}], "usage": {"input_tokens": 3}}
    response = FakeObject(data)
    [block] = cast(list[FakeObject], response.content)
    assert block.text == "Hi."
    assert cast(FakeObject, response["usage"]).input_tokens == 3
    with pytest.raises(AttributeError):
        _ = response.missing
    with pytest.raises(AttributeError, match="read-only"):
        response.content = []
    dumped = response.model_dump()
    dumped["usage"] = None
    assert response.to_dict() == data
    assert response == FakeObject(data)


def test_environment_outputs_are_deterministic_and_sized() -> None:
    environment = Environment(output_tokens=50)
    first = environment.run("read_log", {"step": 0})
    assert count_text(first) == 50
    assert environment.run("read_log", {"step": 0}) == first
    assert environment.run("read_log", {"step": 1}) != first
    assert [name for name, _ in environment.calls] == ["read_log"] * 3


def test_tool_loop_policy_makes_parallel_calls_then_answers() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(calls_per_step=3, steps=2, answer="All read."))
    environment = Environment(output_tokens=10)
    history = run_session(
        fake.messages.create,
        Format.ANTHROPIC_MESSAGES,
        ["Read the logs."],
        environment=environment,
    )
    assert [len(message.tool_uses) for message in history] == [0, 3, 0, 3, 0, 0]
    assert history[-1].text == "All read."
    assert len(environment.calls) == 6


def test_scripted_policy_fails_when_the_script_runs_out() -> None:
    fake = FakeAnthropic(policy=replies(Reply(text="Only one.")))
    fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x")])
    with pytest.raises(AssertionError, match="only 1 replies"):
        fake.messages.create(model=MODEL, max_tokens=10, messages=[user("x")])


def test_digest_summaries_carry_earlier_summaries_forward() -> None:
    first = DigestSummarizer().summarize([Message.from_text("user", "Never touch prod.")], None)
    block = Compaction(data={"type": "compaction", "content": first, "signature": "sig"})
    second = DigestSummarizer().summarize(
        [Message(role="assistant", blocks=(block,)), Message.from_text("user", "Next.")], None
    )
    assert "Never touch prod." in second
    assert "user: Next." in second


def test_forgetful_summaries_drop_the_given_phrases() -> None:
    conversation = [Message.from_text("user", "Never touch prod. Fix the login bug.")]
    summary = ForgetfulSummarizer(["Never touch prod."]).summarize(conversation, None)
    assert "Never touch prod." not in summary
    assert "Fix the login bug." in summary


@pytest.mark.parametrize("fmt", list(Format))
def test_run_session_drives_every_format(fmt: Format) -> None:
    fake, create = client_for(fmt)
    history = run_session(create, fmt, ["One.", "Two."])
    assert len(fake.calls) == 6
    assert history[-1].text == "Done."
    assert_holds(check_all(fake.calls))


def test_run_session_stops_runaway_turns() -> None:
    fake = FakeAnthropic(policy=ToolLoopPolicy(steps=100))
    with pytest.raises(AssertionError, match="within 5 steps"):
        run_session(fake.messages.create, Format.ANTHROPIC_MESSAGES, ["Go"], max_steps=5)


def test_response_json_prefers_the_fields_the_api_returned() -> None:
    class SDKObject:
        def to_dict(self) -> JSONObject:
            return {"type": "text", "text": "Hi."}

        def model_dump(self) -> JSONObject:
            return {"type": "text", "text": "Hi.", "citations": None}

    assert response_json(SDKObject()) == {"type": "text", "text": "Hi."}
    assert response_json({"type": "text", "text": "Hi."}) == {"type": "text", "text": "Hi."}


def test_anthropic_cache_reads_the_longest_written_prefix() -> None:
    cache = AnthropicCache()
    first = [Unit("a", 10), Unit("b", 10, breakpoint=True)]
    assert cache.account(first) == AnthropicUsage(cache_read=0, cache_write=20, uncached=0)
    second = [*first, Unit("c", 5), Unit("d", 5, breakpoint=True), Unit("e", 7)]
    assert cache.account(second) == AnthropicUsage(cache_read=20, cache_write=10, uncached=7)
    plain = [Unit("a", 10), Unit("b", 10)]
    assert cache.account(plain) == AnthropicUsage(cache_read=0, cache_write=0, uncached=20)


def test_openai_cache_needs_1024_tokens_and_counts_in_128_token_steps() -> None:
    cache = OpenAICache()
    assert cache.account([Unit("a", 1_000)]) == 0
    assert cache.account([Unit("a", 1_000), Unit("b", 500)]) == 0
    assert cache.account([Unit("a", 1_000), Unit("b", 500), Unit("c", 1)]) == 1_408


def test_token_counts() -> None:
    assert [count_text(text) for text in ("", "abcd", "abcde")] == [0, 1, 2]
    assert count_json({"a": 1}) == count_text('{"a":1}')
    message = Message(role="user", blocks=(Text(text="abcd"), Image(source={"type": "url"})))
    assert count_message(message) == MESSAGE_OVERHEAD + 1 + MEDIA_TOKENS + 2 * BLOCK_OVERHEAD
