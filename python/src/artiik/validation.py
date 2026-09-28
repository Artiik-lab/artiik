"""Checks that a conversation is a valid request for its provider, before it's sent.

The providers reject requests whose tool calls and results don't pair up, or
whose special blocks sit in the wrong place. artiik checks the same rules
before every call, so a mistake shows up as a clear error in the agent rather
than a 400 from the API. The rules:

- **Tool pairs.** Every tool call is answered, parallel calls included, and
  every result answers a call. Anthropic wants the results in the message
  right after the call, before any other content. Chat Completions wants the
  ``tool`` messages right after the assistant message that made the calls.
- **Roles.** Each API has its own set, and some blocks belong to one role:
  tool calls and thinking to the assistant, Anthropic tool results to the user.
- **Compaction.** Anthropic takes at most one compaction block, as the first
  block of the first message. The Responses API reads the input from the
  latest compaction item on, so only that part is checked.
- **Reasoning.** A Responses reasoning item must be followed by the item it
  came with.

A tool call left without its result also blocks a compaction request, so the
same check covers compaction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from artiik.errors import ValidationError
from artiik.messages import (
    Compaction,
    Format,
    Message,
    Opaque,
    RedactedThinking,
    Thinking,
    ToolResult,
    ToolUse,
    last_compaction,
)

_ROLES: dict[Format, frozenset[str]] = {
    Format.ANTHROPIC_MESSAGES: frozenset({"user", "assistant", "system"}),
    Format.OPENAI_RESPONSES: frozenset({"user", "assistant", "system", "developer"}),
    Format.OPENAI_CHAT: frozenset({"user", "assistant", "system", "developer", "tool"}),
}


def validate(api: Format, conversation: Sequence[Message]) -> None:
    """Raise :class:`~artiik.errors.ValidationError` listing every problem, if there are any."""
    found = problems(api, conversation)
    if found:
        raise ValidationError(tuple(found))


def problems(api: Format, conversation: Sequence[Message]) -> list[str]:
    """Every reason the conversation can't be sent as it is; empty when it can."""
    if not conversation:
        return ["the conversation is empty"]
    match api:
        case Format.ANTHROPIC_MESSAGES:
            return _anthropic(conversation)
        case Format.OPENAI_RESPONSES:
            return _responses(conversation)
        case Format.OPENAI_CHAT:
            return _chat(conversation)


def visible_start(api: Format, conversation: Sequence[Message]) -> int:
    """Where the part of the conversation the model reads begins.

    The Responses API ignores the input before the latest compaction item; the
    other formats send everything.
    """
    return last_compaction(conversation) if api is Format.OPENAI_RESPONSES else 0


def is_turn_start(message: Message) -> bool:
    """Whether a message starts a user turn: a user message that isn't carrying tool results."""
    return (
        message.role == "user"
        and not message.bare_item
        and bool(message.blocks)
        and not any(isinstance(block, ToolResult) for block in message.blocks)
    )


def _anthropic(messages: Sequence[Message]) -> list[str]:
    found: list[str] = []
    compactions = [
        (index, position)
        for index, message in enumerate(messages)
        for position, block in enumerate(message.blocks)
        if isinstance(block, Compaction)
    ]
    if len(compactions) > 1:
        found.append(f"messages: {len(compactions)} compaction blocks; send only the newest")
    found.extend(
        f"messages[{index}].content[{position}]: a compaction block must be the first block "
        "of the first message"
        for index, position in compactions
        if (index, position) != (0, 0)
    )
    found.extend(_duplicates("messages", [use.id for m in messages for use in m.tool_uses]))
    last = len(messages) - 1
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if message.bare_item or message.role not in _ROLES[Format.ANTHROPIC_MESSAGES]:
            found.append(f"{path}: role {message.role!r} can't be sent in the Messages API")
            continue
        if not message.blocks and not (index == last and message.role == "assistant"):
            found.append(f"{path}: the content is empty")
        for position, block in enumerate(message.blocks):
            if (
                isinstance(block, ToolUse | Thinking | RedactedThinking)
                and message.role != "assistant"
            ):
                found.append(
                    f"{path}.content[{position}]: {_name(block)} only goes in an assistant message"
                )
            if isinstance(block, ToolResult) and message.role != "user":
                found.append(f"{path}.content[{position}]: tool_result only goes in a user message")
        if message.role == "assistant" and message.tool_uses:
            following = messages[index + 1] if index < last else None
            answered = (
                {result.tool_use_id for result in following.tool_results}
                if following is not None and following.role == "user"
                else set[str]()
            )
            missing = [use.id for use in message.tool_uses if use.id not in answered]
            if missing:
                found.append(
                    f"{path}: tool calls without a tool_result in the next message: "
                    + ", ".join(missing)
                )
        if message.role == "user" and message.tool_results:
            found.extend(_anthropic_results(messages, index))
    return found


def _anthropic_results(messages: Sequence[Message], index: int) -> list[str]:
    found: list[str] = []
    path = f"messages[{index}]"
    message = messages[index]
    previous = messages[index - 1] if index > 0 else None
    known = (
        {use.id for use in previous.tool_uses}
        if previous is not None and previous.role == "assistant"
        else set[str]()
    )
    found.extend(
        f"{path}: tool_result {result.tool_use_id} answers no tool_use in the previous message"
        for result in message.tool_results
        if result.tool_use_id not in known
    )
    found.extend(_duplicates(path, [result.tool_use_id for result in message.tool_results]))
    seen_other = False
    for block in message.blocks:
        if not isinstance(block, ToolResult):
            seen_other = True
        elif seen_other:
            found.append(f"{path}: tool_result blocks must come before any other content")
            break
    return found


def _responses(items: Sequence[Message]) -> list[str]:
    found: list[str] = []
    calls: list[str] = []
    answered: set[str] = set()
    for index in range(last_compaction(items), len(items)):
        item = items[index]
        path = f"input[{index}]"
        if not item.bare_item:
            if item.role not in _ROLES[Format.OPENAI_RESPONSES]:
                found.append(
                    f"{path}: role {item.role!r} isn't a message role; tool results are "
                    "function_call_output items"
                )
            continue
        block = item.blocks[0] if item.blocks else None
        if isinstance(block, ToolUse):
            if block.id in calls:
                found.append(f"{path}: call_id {block.id} is used by two function calls")
            calls.append(block.id)
        elif isinstance(block, ToolResult):
            if block.tool_use_id not in calls:
                found.append(
                    f"{path}: function_call_output {block.tool_use_id} answers no earlier "
                    "function_call"
                )
            elif block.tool_use_id in answered:
                found.append(f"{path}: a second function_call_output for {block.tool_use_id}")
            answered.add(block.tool_use_id)
        elif isinstance(block, Opaque) and block.type == "reasoning":
            following = items[index + 1] if index + 1 < len(items) else None
            if following is None or not _comes_with_reasoning(following):
                found.append(f"{path}: a reasoning item must be followed by the item it came with")
    missing = [call for call in calls if call not in answered]
    if missing:
        found.append("input: function calls without a function_call_output: " + ", ".join(missing))
    return found


def _comes_with_reasoning(item: Message) -> bool:
    if not item.bare_item:
        return item.role == "assistant"
    return bool(item.blocks) and not isinstance(item.blocks[0], ToolResult)


def _chat(messages: Sequence[Message]) -> list[str]:
    found: list[str] = []
    expected: set[str] = set()
    pending: list[str] = []
    owner = 0
    found.extend(_duplicates("messages", [use.id for m in messages for use in m.tool_uses]))
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if message.role not in _ROLES[Format.OPENAI_CHAT]:
            found.append(f"{path}: role {message.role!r} isn't valid in Chat Completions")
        if message.role == "tool" and not message.bare_item:
            result = message.tool_results[0] if message.tool_results else None
            if result is None:
                found.append(f"{path}: a tool message needs a tool_call_id")
            elif result.tool_use_id not in expected:
                found.append(
                    f"{path}: tool message {result.tool_use_id} answers no tool call of the "
                    "assistant message before it"
                )
            elif result.tool_use_id not in pending:
                found.append(f"{path}: a second tool message for {result.tool_use_id}")
            else:
                pending.remove(result.tool_use_id)
            continue
        if pending:
            found.append(_unanswered(owner, pending))
        expected = set()
        pending = []
        if message.role == "assistant" and message.tool_uses:
            owner = index
            pending = [use.id for use in message.tool_uses]
            expected = set(pending)
    if pending:
        found.append(_unanswered(owner, pending))
    return found


def _unanswered(owner: int, pending: Iterable[str]) -> str:
    return f"messages[{owner}]: tool calls without a tool message right after: " + ", ".join(
        pending
    )


def _duplicates(path: str, ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for item in ids:
        if item in seen and item not in repeated:
            repeated.append(item)
        seen.add(item)
    return [f"{path}: tool call id {item} is used more than once" for item in repeated]


def _name(block: ToolUse | Thinking | RedactedThinking) -> str:
    match block:
        case ToolUse():
            return "tool_use"
        case Thinking():
            return "thinking"
        case RedactedThinking():
            return "redacted_thinking"
