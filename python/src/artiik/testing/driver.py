"""Agent loops for tests and baselines: send the history, run the tools, repeat.

:func:`run_session` keeps the history itself and sends it through an optional
``prepare`` strategy. :func:`run_context` lets an :class:`~artiik.Context`
manage the history, the way an application uses artiik.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from artiik.context import Context
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.formats._json import expect_list, expect_object, plain, to_object
from artiik.messages import Format, JSONObject, JSONValue, Message, ToolResult, ToolUse
from artiik.testing.environment import Environment

Create: TypeAlias = "Callable[..., object]"
"""An endpoint such as ``client.messages.create``: it takes the request as keyword arguments."""

Prepare: TypeAlias = "Callable[[Sequence[Message]], Sequence[Message]]"
"""Turns the full history into the messages sent on the next call."""

_SCHEMA: JSONObject = {"type": "object", "properties": {"step": {"type": "integer"}}}
_DESCRIPTION = "Read a log file."


def default_request(fmt: Format, *, tool_name: str = "read_log") -> JSONObject:
    """Request parameters for :func:`run_session`: a model name and one tool definition."""
    match fmt:
        case Format.ANTHROPIC_MESSAGES:
            tool: JSONObject = {
                "name": tool_name,
                "description": _DESCRIPTION,
                "input_schema": dict(_SCHEMA),
            }
            return {"model": "fake-model", "max_tokens": 1024, "tools": [tool]}
        case Format.OPENAI_RESPONSES:
            tool = {
                "type": "function",
                "name": tool_name,
                "description": _DESCRIPTION,
                "parameters": dict(_SCHEMA),
            }
            return {"model": "fake-model", "store": False, "tools": [tool]}
        case Format.OPENAI_CHAT:
            function: JSONObject = {
                "name": tool_name,
                "description": _DESCRIPTION,
                "parameters": dict(_SCHEMA),
            }
            return {"model": "fake-model", "tools": [{"type": "function", "function": function}]}


def run_session(
    create: Create,
    fmt: Format,
    turns: Sequence[str],
    *,
    environment: Environment | None = None,
    request: Mapping[str, JSONValue] | None = None,
    prepare: Prepare | None = None,
    max_steps: int = 20,
) -> list[Message]:
    """Run a conversation and return its full history.

    For each user turn, the loop sends the history to ``create``, runs the
    tools the model calls in ``environment``, and repeats until the model
    answers without calling a tool. ``create`` is an endpoint in ``fmt``, such
    as ``fake.messages.create`` or ``fake.responses.create``; SDK clients work
    too. ``request`` holds the other parameters (default:
    :func:`default_request`). ``prepare`` turns the full history into what's
    sent on each call; the default sends everything. Tests pass strategies
    through it, including broken ones, to prove that each invariant can fail.
    """
    tools = environment if environment is not None else Environment()
    base: JSONObject = dict(request) if request is not None else default_request(fmt)
    history: list[Message] = []
    for text in turns:
        history.append(Message.from_text("user", text))
        for _ in range(max_steps):
            sent = list(prepare(history)) if prepare is not None else history
            new = _send(create, fmt, base, sent)
            history.extend(new)
            calls = [use for message in new for use in message.tool_uses]
            if not calls:
                break
            history.extend(_results(fmt, calls, tools))
        else:
            raise AssertionError(f"the model didn't finish the turn within {max_steps} steps")
    return history


def run_context(
    context: Context,
    create: Create,
    turns: Sequence[str],
    *,
    environment: Environment | None = None,
    params: Mapping[str, Any] | None = None,
    max_steps: int = 20,
) -> None:
    """Run a conversation through a context, as an application would.

    For each user turn: add it, then ``create(**context.prepare(**params))``,
    ``context.record(response)``, and run the pending tool calls in
    ``environment``, until the model answers without calling a tool.
    """
    tools = environment if environment is not None else Environment()
    for text in turns:
        context.add(Message.from_text("user", text))
        for _ in range(max_steps):
            context.record(create(**context.prepare(**dict(params or {}))))
            calls = context.pending_tool_calls()
            if not calls:
                break
            context.add(*_results(context.api, calls, tools))
        else:
            raise AssertionError(f"the model didn't finish the turn within {max_steps} steps")


def response_json(response: object) -> JSONObject:
    """Read a response as JSON.

    SDK objects go through ``to_dict()``, which keeps only the fields the API
    returned; ``model_dump()`` would add ``null`` fields that some APIs reject
    when the blocks are sent back.
    """
    return to_object(plain(response), "response")


def _send(create: Create, fmt: Format, base: JSONObject, sent: Sequence[Message]) -> list[Message]:
    match fmt:
        case Format.ANTHROPIC_MESSAGES:
            kwargs = {**base, "messages": anthropic_messages.dump_messages(sent)}
            return [anthropic_messages.parse_response(response_json(create(**kwargs)))]
        case Format.OPENAI_RESPONSES:
            kwargs = {**base, "input": openai_responses.dump_items(sent)}
            response = response_json(create(**kwargs))
            output = expect_list(response.get("output"), "response.output")
            return openai_responses.parse_items(output, "output")
        case Format.OPENAI_CHAT:
            kwargs = {**base, "messages": openai_chat.dump_messages(sent)}
            response = response_json(create(**kwargs))
            choices = expect_list(response.get("choices"), "response.choices")
            choice = expect_object(choices[0], "response.choices[0]")
            return [openai_chat.parse_message(choice.get("message"), "choices[0].message")]


def _results(fmt: Format, calls: Sequence[ToolUse], environment: Environment) -> list[Message]:
    results = [
        ToolResult(tool_use_id=call.id, content=environment.run(call.name, call.input))
        for call in calls
    ]
    match fmt:
        case Format.ANTHROPIC_MESSAGES:
            return [Message(role="user", blocks=tuple(results))]
        case Format.OPENAI_RESPONSES:
            return [Message(role="tool", blocks=(result,), bare_item=True) for result in results]
        case Format.OPENAI_CHAT:
            return [Message(role="tool", blocks=(result,)) for result in results]
