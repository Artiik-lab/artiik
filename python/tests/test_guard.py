"""The guard drops the oldest turns, then the oldest steps, and never splits a tool pair."""

import pytest

from artiik import BudgetError, Format
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.guard import Trim, trim
from artiik.messages import JSONObject, JSONValue, Message
from artiik.tokens import Tally, tally_message
from artiik.validation import is_turn_start, problems

ANTHROPIC = Format.ANTHROPIC_MESSAGES
RESPONSES = Format.OPENAI_RESPONSES
CHAT = Format.OPENAI_CHAT


def size(view: Tally) -> int:
    return view.chars + view.fixed


def cut(api: Format, history: list[Message], *, target: int, limit: int | None = None) -> Trim:
    tallies = [tally_message(message) for message in history]
    total = size(sum(tallies, Tally()))
    return trim(
        api, history, tallies, size=size, target=target, limit=total if limit is None else limit
    )


def total(history: list[Message]) -> int:
    return size(sum((tally_message(message) for message in history), Tally()))


def texts(messages: tuple[Message, ...]) -> list[str]:
    return [message.text or message.role for message in messages]


def anthropic_turn(number: int, steps: int = 1, answer: bool = True) -> list[JSONObject]:
    messages: list[JSONObject] = [{"role": "user", "content": f"Turn {number}."}]
    for step in range(steps):
        call = f"t{number}{step}"
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": call, "name": "ls", "input": {}}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call, "content": "ok"}],
            }
        )
    if answer:
        messages.append({"role": "assistant", "content": f"Answer {number}."})
    return messages


def anthropic(*turns: list[JSONObject]) -> list[Message]:
    return anthropic_messages.parse_messages([message for turn in turns for message in turn])


def test_the_oldest_whole_turns_go_first() -> None:
    history = anthropic(*(anthropic_turn(number) for number in range(5)))
    result = cut(ANTHROPIC, history, target=total(history) // 2)
    assert result.dropped_turns == 3
    assert result.dropped_steps == 0
    assert result.dropped_messages == 12
    assert texts(result.messages)[0] == "Turn 3."
    assert len(result.messages) == 8
    assert result.size == total(list(result.messages))
    assert problems(ANTHROPIC, result.messages) == []


def test_the_guard_stops_as_soon_as_the_request_fits() -> None:
    history = anthropic(*(anthropic_turn(number) for number in range(5)))
    result = cut(ANTHROPIC, history, target=total(history) * 9 // 10)
    assert result.dropped_turns == 1
    assert texts(result.messages)[0] == "Turn 1."


def test_nothing_is_dropped_when_the_request_already_fits() -> None:
    history = anthropic(anthropic_turn(0), anthropic_turn(1))
    result = cut(ANTHROPIC, history, target=total(history))
    assert result.messages == tuple(history)
    assert (result.dropped_turns, result.dropped_steps, result.dropped_messages) == (0, 0, 0)


def test_then_the_oldest_steps_of_the_current_turn_go() -> None:
    history = anthropic(anthropic_turn(0, steps=6, answer=False))
    result = cut(ANTHROPIC, history, target=1)
    assert result.dropped_turns == 0
    assert result.dropped_steps == 5
    assert [message.role for message in result.messages] == ["user", "assistant", "user"]
    assert texts(result.messages)[0] == "Turn 0."
    assert [use.id for use in result.messages[1].tool_uses] == ["t05"]
    assert problems(ANTHROPIC, result.messages) == []


def test_a_request_that_cant_fit_raises_budget_error() -> None:
    history = anthropic(anthropic_turn(0), anthropic_turn(1, steps=3, answer=False))
    smallest = total(anthropic(anthropic_turn(1, steps=0, answer=False))) + total(
        anthropic_messages.parse_messages(anthropic_turn(1, steps=3, answer=False)[-2:])
    )
    with pytest.raises(BudgetError) as caught:
        cut(ANTHROPIC, history, target=1, limit=smallest - 1)
    assert caught.value.needed == smallest
    assert caught.value.budget == smallest - 1
    assert "raise the budget" in str(caught.value)
    assert cut(ANTHROPIC, history, target=1, limit=smallest).size == smallest


def test_system_messages_move_after_the_first_user_turn_that_remains() -> None:
    turns = [anthropic_turn(number) for number in range(4)]
    turns[1].insert(1, {"role": "system", "content": "Never touch prod."})
    history = anthropic(*turns)
    remaining = anthropic(turns[1][1:2], turns[2], turns[3])
    result = cut(ANTHROPIC, history, target=total(remaining))
    assert result.dropped_turns == 2
    assert result.moved_messages == 1
    assert texts(result.messages)[:2] == ["Turn 2.", "Never touch prod."]
    assert problems(ANTHROPIC, result.messages) == []


def test_a_leading_instruction_stays_first_and_isnt_a_turn() -> None:
    history = openai_chat.parse_messages(
        [
            {"role": "developer", "content": "Answer in French."},
            *(
                message
                for number in range(3)
                for message in (
                    {"role": "user", "content": f"Turn {number}."},
                    {"role": "assistant", "content": f"Answer {number}."},
                )
            ),
        ]
    )
    result = cut(CHAT, history, target=total(history) * 3 // 4)
    assert result.dropped_turns == 1
    assert result.moved_messages == 0
    assert texts(result.messages) == [
        "Answer in French.",
        "Turn 1.",
        "Answer 1.",
        "Turn 2.",
        "Answer 2.",
    ]


def test_an_anthropic_compaction_block_stays_first() -> None:
    compaction: JSONObject = {"type": "compaction", "content": "Summary.", "signature": "sig"}
    history = anthropic(
        [{"role": "assistant", "content": [compaction]}],
        *(anthropic_turn(number) for number in range(4)),
    )
    result = cut(ANTHROPIC, history, target=total(history) // 2)
    assert result.dropped_turns >= 1
    assert result.messages[0] == history[0]
    assert problems(ANTHROPIC, result.messages) == []


def responses_turns(count: int) -> list[JSONObject]:
    items: list[JSONObject] = []
    for number in range(count):
        turn: list[JSONObject] = [
            {"role": "user", "content": f"Turn {number}."},
            {"type": "reasoning", "id": f"rs_{number}", "summary": [], "encrypted_content": "x"},
            {"type": "function_call", "call_id": f"c{number}", "name": "ls", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{number}", "output": "ok"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"Answer {number}."}],
            },
        ]
        items.extend(turn)
    return items


def test_responses_keep_the_compaction_item_and_reasoning_with_its_calls() -> None:
    items: list[JSONObject] = [
        {"role": "user", "content": "Long ago."},
        {"type": "function_call", "call_id": "old", "name": "ls", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "old", "output": "ok"},
        {"id": "cmp_1", "type": "compaction", "encrypted_content": "x"},
        *responses_turns(4),
    ]
    history = openai_responses.parse_items(items)
    result = cut(RESPONSES, history, target=total([history[3], *history[-10:]]))
    # What came before the compaction item is the oldest turn, so it goes first.
    assert result.dropped_turns == 3
    assert result.messages[0] == history[3]
    assert texts(result.messages)[1] == "Turn 2."
    assert result.size == total(list(result.messages))
    assert problems(RESPONSES, result.messages) == []


def test_a_compaction_item_after_kept_user_messages_stays_in_place() -> None:
    # /responses/compact returns the user messages, then the compaction item.
    items: list[JSONObject] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "A."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "B."}]},
        {"id": "cmp_1", "type": "compaction", "encrypted_content": "x"},
        *responses_turns(3),
    ]
    history = openai_responses.parse_items(items)
    result = cut(RESPONSES, history, target=total(history[2:]) - 1)
    assert result.dropped_turns == 3
    assert result.messages[0] == history[2]
    assert texts(result.messages)[1] == "Turn 1."
    assert problems(RESPONSES, result.messages) == []


def test_a_moved_instruction_goes_where_anthropic_takes_a_system_message() -> None:
    history = anthropic(
        [
            {"role": "user", "content": "Turn 0."},
            {"role": "system", "content": "Never touch prod."},
            {"role": "assistant", "content": "Answer 0."},
        ],
        # A turn left without a reply: a system message can't follow it.
        [{"role": "user", "content": "Turn 1."}],
        anthropic_turn(2),
    )
    # The instruction stays, so the target counts it.
    trimmed = cut(ANTHROPIC, history, target=total(history[1:2] + history[3:]))
    assert trimmed.dropped_turns == 1
    assert texts(trimmed.messages)[:3] == ["Turn 1.", "Turn 2.", "Never touch prod."]
    assert problems(ANTHROPIC, trimmed.messages) == []


def test_a_compaction_block_stays_with_the_results_of_its_calls() -> None:
    # A reply that compacted at a threshold can call tools after its block.
    block: JSONObject = {"type": "compaction", "content": "Summary."}
    call: JSONObject = {"type": "tool_use", "id": "t0", "name": "ls", "input": {}}
    result: JSONObject = {"type": "tool_result", "tool_use_id": "t0", "content": "ok"}
    history = anthropic(
        [{"role": "assistant", "content": [block, call]}, {"role": "user", "content": [result]}],
        *(anthropic_turn(number) for number in range(3)),
    )
    trimmed = cut(ANTHROPIC, history, target=total(history[:2] + history[-4:]))
    assert trimmed.messages[:2] == tuple(history[:2])
    assert trimmed.dropped_turns == 2
    assert problems(ANTHROPIC, trimmed.messages) == []


def chat_step(step: int) -> list[JSONObject]:
    calls: list[JSONValue] = [
        {"id": f"c{step}{n}", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
        for n in range(2)
    ]
    assistant: JSONObject = {"role": "assistant", "content": None, "tool_calls": calls}
    return [
        assistant,
        *({"role": "tool", "tool_call_id": f"c{step}{n}", "content": "ok"} for n in range(2)),
    ]


def responses_step(step: int) -> list[JSONObject]:
    calls: list[JSONObject] = [
        {"type": "function_call", "call_id": f"c{step}{n}", "name": "ls", "arguments": "{}"}
        for n in range(2)
    ]
    outputs: list[JSONObject] = [
        {"type": "function_call_output", "call_id": f"c{step}{n}", "output": "ok"} for n in range(2)
    ]
    return calls + outputs


@pytest.mark.parametrize(
    ("api", "per_step"),
    [pytest.param(CHAT, 3, id="chat"), pytest.param(RESPONSES, 4, id="responses")],
)
def test_tool_results_stay_with_their_calls_in_the_openai_formats(
    api: Format, per_step: int
) -> None:
    entries: list[JSONObject] = [{"role": "user", "content": "Go."}]
    for step in range(5):
        entries.extend(chat_step(step) if api is CHAT else responses_step(step))
    history = (
        openai_chat.parse_messages(entries)
        if api is CHAT
        else openai_responses.parse_items(entries)
    )
    result = cut(api, history, target=1)
    assert result.dropped_steps == 4
    assert len(result.messages) == 1 + per_step
    assert [use.id for message in result.messages for use in message.tool_uses] == ["c40", "c41"]
    assert problems(api, result.messages) == []


def test_a_turn_starts_with_a_user_message_that_carries_no_tool_results() -> None:
    plain, carrier, empty = anthropic_messages.parse_messages(
        [
            {"role": "user", "content": "Next task."},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                    {"type": "text", "text": "Also check the logs."},
                ],
            },
            {"role": "user", "content": []},
        ]
    )
    [output] = openai_responses.parse_items(
        [{"type": "function_call_output", "call_id": "c", "output": "ok"}]
    )
    assert is_turn_start(plain)
    assert not is_turn_start(carrier)
    assert not is_turn_start(empty)
    assert not is_turn_start(output)
    assert not is_turn_start(Message.from_text("assistant", "Done."))
