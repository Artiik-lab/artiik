"""Conversations are checked the way each provider API checks requests."""

import pytest

from artiik import Format, ValidationError
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.messages import JSONObject, JSONValue, Message
from artiik.validation import problems, validate

ANTHROPIC = Format.ANTHROPIC_MESSAGES
RESPONSES = Format.OPENAI_RESPONSES
CHAT = Format.OPENAI_CHAT


def anthropic(*messages: JSONObject) -> list[Message]:
    return anthropic_messages.parse_messages(messages)


def responses(*items: JSONObject) -> list[Message]:
    return openai_responses.parse_items(items)


def chat(*messages: JSONObject) -> list[Message]:
    return openai_chat.parse_messages(messages)


def user(text: str) -> JSONObject:
    return {"role": "user", "content": text}


def tool_use(*ids: str) -> JSONObject:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": id_, "name": "ls", "input": {}} for id_ in ids],
    }


def results(*ids: str) -> JSONObject:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": id_, "content": "ok"} for id_ in ids],
    }


def message(role: str, *blocks: JSONObject) -> JSONObject:
    content: list[JSONValue] = list(blocks)
    return {"role": role, "content": content}


COMPACTION: JSONObject = {"type": "compaction", "content": "Summary.", "signature": "sig"}
TOOL_USE: JSONObject = {"type": "tool_use", "id": "a", "name": "ls", "input": {"path": "."}}
TOOL_RESULT: JSONObject = {"type": "tool_result", "tool_use_id": "a", "content": "ok"}
THINKING: JSONObject = {"type": "thinking", "thinking": "x", "signature": "s"}


# Anthropic Messages


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([user("Hi")], id="one-turn"),
        pytest.param([user("Go"), tool_use("a", "b"), results("a", "b")], id="parallel-calls"),
        pytest.param(
            [user("Go"), tool_use("a"), results("a"), {"role": "assistant", "content": "Done."}],
            id="finished-turn",
        ),
        pytest.param(
            [user("Plan it."), {"role": "system", "content": "Never touch prod."}],
            id="mid-conversation-system",
        ),
        pytest.param(
            [{"role": "assistant", "content": [COMPACTION]}, user("Go on.")],
            id="compaction-first",
        ),
        pytest.param(
            [{"role": "user", "content": [COMPACTION, {"type": "text", "text": "Go on."}]}],
            id="compaction-in-user-message",
        ),
        pytest.param([user("Hi"), message("assistant")], id="empty-prefill"),
    ],
)
def test_valid_anthropic_conversations(messages: list[JSONObject]) -> None:
    assert problems(ANTHROPIC, anthropic(*messages)) == []


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [user("Go"), tool_use("a", "b"), results("a")],
            "messages[1]: tool calls without a tool_result in the next message: b",
            id="parallel-call-unanswered",
        ),
        pytest.param(
            [user("Go"), tool_use("a")],
            "messages[1]: tool calls without a tool_result in the next message: a",
            id="last-call-unanswered",
        ),
        pytest.param(
            [user("Go"), tool_use("a"), user("Wait."), results("a")],
            "messages[1]: tool calls without a tool_result in the next message: a",
            id="result-not-next",
        ),
        pytest.param(
            [user("Go"), results("x")],
            "messages[1]: tool_result x answers no tool_use in the previous message",
            id="orphaned-result",
        ),
        pytest.param(
            [
                user("Go"),
                tool_use("a"),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here."},
                        {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
                    ],
                },
            ],
            "messages[2]: tool_result blocks must come before any other content",
            id="result-after-text",
        ),
        pytest.param(
            [user("Go"), tool_use("a"), results("a", "a")],
            "messages[2]: tool call id a is used more than once",
            id="duplicate-result",
        ),
        pytest.param(
            [user("Go"), tool_use("a"), results("a"), tool_use("a"), results("a")],
            "messages: tool call id a is used more than once",
            id="duplicate-call-id",
        ),
        pytest.param(
            [message("user", TOOL_USE)],
            "messages[0].content[0]: tool_use only goes in an assistant message",
            id="tool-use-from-user",
        ),
        pytest.param(
            [user("Go"), message("assistant", TOOL_RESULT)],
            "messages[1].content[0]: tool_result only goes in a user message",
            id="result-from-assistant",
        ),
        pytest.param(
            [message("user", THINKING)],
            "messages[0].content[0]: thinking only goes in an assistant message",
            id="thinking-from-user",
        ),
        pytest.param(
            [user("Go"), message("assistant", COMPACTION)],
            "messages[1].content[0]: a compaction block must be the first block of the first "
            "message",
            id="compaction-not-first",
        ),
        pytest.param(
            [{"role": "assistant", "content": [COMPACTION, COMPACTION]}],
            "messages: 2 compaction blocks; send only the newest",
            id="two-compaction-blocks",
        ),
        pytest.param(
            [message("user"), user("Go")],
            "messages[0]: the content is empty",
            id="empty-content",
        ),
    ],
)
def test_anthropic_problems(messages: list[JSONObject], expected: str) -> None:
    assert expected in problems(ANTHROPIC, anthropic(*messages))


def test_anthropic_rejects_roles_and_items_from_other_formats() -> None:
    [developer] = chat({"role": "developer", "content": "Be brief."})
    [call] = responses({"type": "function_call", "call_id": "c", "name": "ls", "arguments": "{}"})
    found = problems(ANTHROPIC, [*anthropic(user("Go")), developer, call])
    assert found == [
        "messages[1]: role 'developer' can't be sent in the Messages API",
        "messages[2]: role 'assistant' can't be sent in the Messages API",
    ]


# OpenAI Responses

CALL: JSONObject = {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{}"}
OUTPUT: JSONObject = {"type": "function_call_output", "call_id": "c1", "output": "ok"}
REASONING: JSONObject = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "x"}
COMPACTION_ITEM: JSONObject = {"id": "cmp_1", "type": "compaction", "encrypted_content": "x"}


@pytest.mark.parametrize(
    "items",
    [
        pytest.param([user("Go"), CALL, OUTPUT], id="call-and-output"),
        pytest.param(
            [
                user("Go"),
                REASONING,
                CALL,
                {**CALL, "call_id": "c2"},
                OUTPUT,
                {**OUTPUT, "call_id": "c2"},
            ],
            id="reasoning-and-parallel-calls",
        ),
        pytest.param(
            [
                user("Go"),
                REASONING,
                {"role": "assistant", "content": [{"type": "output_text", "text": "Done."}]},
            ],
            id="reasoning-before-a-message",
        ),
        pytest.param(
            [user("Go"), OUTPUT, COMPACTION_ITEM, user("Next.")], id="ignored-before-compaction"
        ),
        pytest.param([{"role": "developer", "content": "Be brief."}, user("Go")], id="developer"),
    ],
)
def test_valid_responses_input(items: list[JSONObject]) -> None:
    assert problems(RESPONSES, responses(*items)) == []


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param(
            [user("Go"), OUTPUT],
            "input[1]: function_call_output c1 answers no earlier function_call",
            id="orphaned-output",
        ),
        pytest.param(
            [user("Go"), CALL],
            "input: function calls without a function_call_output: c1",
            id="call-without-output",
        ),
        pytest.param(
            [user("Go"), CALL, OUTPUT, OUTPUT],
            "input[3]: a second function_call_output for c1",
            id="second-output",
        ),
        pytest.param(
            [user("Go"), CALL, CALL, OUTPUT],
            "input[2]: call_id c1 is used by two function calls",
            id="duplicate-call",
        ),
        pytest.param(
            [user("Go"), REASONING],
            "input[1]: a reasoning item must be followed by the item it came with",
            id="dangling-reasoning",
        ),
        pytest.param(
            [user("Go"), REASONING, user("Next.")],
            "input[1]: a reasoning item must be followed by the item it came with",
            id="reasoning-before-user",
        ),
        pytest.param(
            [COMPACTION_ITEM, OUTPUT],
            "input[1]: function_call_output c1 answers no earlier function_call",
            id="call-hidden-by-compaction",
        ),
    ],
)
def test_responses_problems(items: list[JSONObject], expected: str) -> None:
    assert expected in problems(RESPONSES, responses(*items))


def test_responses_rejects_tool_role_messages() -> None:
    [tool] = chat({"role": "tool", "tool_call_id": "c1", "content": "ok"})
    assert problems(RESPONSES, [*responses(user("Go")), tool]) == [
        "input[1]: role 'tool' isn't a message role; tool results are function_call_output items"
    ]


# OpenAI Chat Completions

ASSISTANT_CALLS: JSONObject = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "ls", "arguments": "{}"}},
    ],
}


def tool(call_id: str) -> JSONObject:
    return {"role": "tool", "tool_call_id": call_id, "content": "ok"}


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([user("Go"), ASSISTANT_CALLS, tool("c1"), tool("c2")], id="parallel-calls"),
        pytest.param(
            [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": "Go", "name": "ana"},
            ],
            id="developer-and-name",
        ),
        pytest.param(
            [user("Go"), ASSISTANT_CALLS, tool("c2"), tool("c1"), user("Thanks.")],
            id="any-order",
        ),
    ],
)
def test_valid_chat_conversations(messages: list[JSONObject]) -> None:
    assert problems(CHAT, chat(*messages)) == []


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [user("Go"), tool("c1")],
            "messages[1]: tool message c1 answers no tool call of the assistant message before it",
            id="orphaned-tool-message",
        ),
        pytest.param(
            [user("Go"), ASSISTANT_CALLS, tool("c1"), user("Next.")],
            "messages[1]: tool calls without a tool message right after: c2",
            id="interrupted",
        ),
        pytest.param(
            [user("Go"), ASSISTANT_CALLS, tool("c1")],
            "messages[1]: tool calls without a tool message right after: c2",
            id="unanswered-at-the-end",
        ),
        pytest.param(
            [user("Go"), ASSISTANT_CALLS, tool("c1"), tool("c1"), tool("c2")],
            "messages[3]: a second tool message for c1",
            id="second-tool-message",
        ),
        pytest.param(
            [
                user("Go"),
                ASSISTANT_CALLS,
                tool("c1"),
                tool("c2"),
                ASSISTANT_CALLS,
                tool("c1"),
                tool("c2"),
            ],
            "messages: tool call id c1 is used more than once",
            id="duplicate-call-id",
        ),
    ],
)
def test_chat_problems(messages: list[JSONObject], expected: str) -> None:
    assert expected in problems(CHAT, chat(*messages))


# Errors


@pytest.mark.parametrize("api", list(Format))
def test_an_empty_conversation_is_a_problem(api: Format) -> None:
    assert problems(api, []) == ["the conversation is empty"]


def test_validate_lists_every_problem() -> None:
    with pytest.raises(ValidationError) as caught:
        validate(ANTHROPIC, anthropic(user("Go"), tool_use("a"), user("Wait.")))
    assert caught.value.problems == (
        "messages[1]: tool calls without a tool_result in the next message: a",
    )
    assert str(caught.value) == (
        "the conversation can't be sent as it is:\n"
        "- messages[1]: tool calls without a tool_result in the next message: a"
    )
    validate(ANTHROPIC, anthropic(user("Go")))
