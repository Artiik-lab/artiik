"""A fake OpenAI client that enforces the API rules artiik relies on.

- Chat Completions: a ``tool`` message answers a ``tool_calls`` entry of the
  assistant message before it, and every tool call is answered before the
  conversation goes on.
- Responses: every ``function_call_output`` answers an earlier
  ``function_call``, every call has an output, and ``reasoning`` items come
  back exactly as the fake returned them.
- Compaction (OpenAI docs, "Compaction"): ``context_management=[{"type":
  "compaction", "compact_threshold": N}]`` compacts server-side once the input
  passes N tokens, and ``client.responses.compact(model=..., input=...)``
  compacts on request. The guide doesn't document the compaction item's
  fields, so this fake assumes ``{"id", "type": "compaction",
  "encrypted_content"}``. It rejects items that come back modified, and ignores
  the input before the latest compaction item.
- The context window.

Replies come from a :class:`~artiik.testing.model.Policy`, and automatic prompt
caching is simulated.
References: https://platform.openai.com/docs/api-reference/responses and
https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from artiik.errors import FormatError
from artiik.formats import openai_chat, openai_responses
from artiik.formats._json import clone, to_object
from artiik.messages import (
    Compaction,
    Format,
    JSONObject,
    JSONValue,
    Message,
    Opaque,
    ToolResult,
    ToolUse,
)
from artiik.testing.caching import OpenAICache, Unit
from artiik.testing.errors import FakeAPIError
from artiik.testing.model import DigestSummarizer, Policy, Reply, Summarizer, ToolLoopPolicy
from artiik.testing.objects import FakeObject
from artiik.testing.recording import RecordedCall, entries, last_compaction, parse_system
from artiik.testing.tokens import count_json, count_message, count_messages, count_text

_Handler: TypeAlias = "Callable[[JSONObject, int, str | None], JSONObject]"


class FakeOpenAI:
    """A stand-in for ``openai.OpenAI`` with ``responses`` and ``chat.completions``.

    - ``policy`` decides the replies; the default is a small tool-calling agent.
    - ``summarizer`` writes the summaries hidden in compaction items.
    - ``faults`` maps a call index to an error to raise on that call, or to a
      stop reason (``max_output_tokens``, ``length``, ``content_filter``) to
      return instead of the normal one.

    Every call is recorded in ``calls``, including the ones that fail.
    """

    def __init__(
        self,
        *,
        policy: Policy | None = None,
        summarizer: Summarizer | None = None,
        context_window: int = 128_000,
        faults: Mapping[int, FakeAPIError | str] | None = None,
    ) -> None:
        self.policy: Policy = policy if policy is not None else ToolLoopPolicy()
        self.summarizer: Summarizer = summarizer if summarizer is not None else DigestSummarizer()
        self.context_window = context_window
        self.faults = dict(faults or {})
        self.calls: list[RecordedCall] = []
        self.responses = _Responses(self)
        self.chat = _Chat(self)
        self._cache = OpenAICache()
        self._issued: dict[str, JSONObject] = {}

    @property
    def requests(self) -> list[JSONObject]:
        """The requests of every call, in order."""
        return [call.request for call in self.calls]

    def _call(
        self, endpoint: str, fmt: Format, kwargs: Mapping[str, object], handler: _Handler
    ) -> FakeObject:
        index = len(self.calls)
        try:
            request = to_object(dict(kwargs), "request")
        except FormatError as error:
            raise _invalid(str(error)) from error
        fault = self.faults.get(index)
        try:
            if isinstance(fault, FakeAPIError):
                raise fault
            response = handler(request, index, fault)
        except FakeAPIError as error:
            self.calls.append(RecordedCall(index, endpoint, fmt, request, error=error))
            raise
        self.calls.append(RecordedCall(index, endpoint, fmt, request, response=response))
        return FakeObject(clone(response))

    def _responses_create(self, request: JSONObject, index: int, stop: str | None) -> JSONObject:
        model = _model(request)
        if "previous_response_id" in request:
            raise _invalid(
                "previous_response_id isn't supported by the fake; send the whole input.",
                param="previous_response_id",
            )
        items = self._input(request)
        start = last_compaction(items)
        prompt, visible = entries(Format.OPENAI_RESPONSES, request)[start:], items[start:]
        _check_function_pairs(visible)
        instructions = parse_system(Format.OPENAI_RESPONSES, request)
        base_tokens = _tool_tokens(request) + count_messages(instructions)
        tokens = base_tokens + count_messages(visible)
        output: list[JSONValue] = []
        threshold = _compaction_threshold(request)
        if threshold is not None and tokens > threshold:
            item = self._issue_compaction(visible, index)
            output.append(item)
            prompt, visible = [clone(item)], openai_responses.parse_items([item])
            tokens = base_tokens + count_messages(visible)
        if tokens > self.context_window:
            raise _context_exceeded(tokens, self.context_window, "input")
        reply = self.policy.reply(visible)
        reply_items = self._output_items(reply, index)
        output.extend(reply_items)
        stop_reason = stop or reply.stop_reason
        incomplete = stop_reason in ("max_output_tokens", "content_filter")
        cached = self._cache.account(_units(request, prompt, visible))
        output_tokens = (
            reply.output_tokens if reply.output_tokens is not None else count_json(reply_items)
        )
        reasoning_tokens = count_text(reply.thinking) if reply.thinking is not None else 0
        return {
            "id": f"resp_{index:04d}",
            "object": "response",
            "status": "incomplete" if incomplete else "completed",
            "incomplete_details": {"reason": stop_reason} if incomplete else None,
            "model": model,
            "output": output,
            "usage": _responses_usage(tokens, min(cached, tokens), output_tokens, reasoning_tokens),
        }

    def _responses_compact(self, request: JSONObject, index: int, stop: str | None) -> JSONObject:
        model = _model(request)
        items = self._input(request)
        visible = items[last_compaction(items) :]
        _check_function_pairs(visible)
        tokens = _tool_tokens(request) + count_messages(visible)
        if tokens > self.context_window:
            raise _context_exceeded(tokens, self.context_window, "input")
        if not any(message.blocks for message in visible):
            raise _invalid("There is nothing to compact.", param="input")
        item = self._issue_compaction(visible, index)
        return {
            "id": f"resp_{index:04d}",
            "object": "response.compaction",
            "model": model,
            "output": [item],
            "usage": _responses_usage(tokens, 0, count_json(item), 0),
        }

    def _chat_create(self, request: JSONObject, index: int, stop: str | None) -> JSONObject:
        model = _model(request)
        if not isinstance(request.get("messages"), list):
            raise _invalid("Missing required parameter: 'messages'.", param="messages")
        try:
            messages = openai_chat.parse_messages(entries(Format.OPENAI_CHAT, request))
        except FormatError as error:
            raise _invalid(str(error), param="messages") from error
        _check_tool_messages(messages)
        tokens = _tool_tokens(request) + count_messages(messages)
        if tokens > self.context_window:
            raise _context_exceeded(tokens, self.context_window, "messages")
        reply = self.policy.reply(messages)
        message: JSONObject = {"role": "assistant", "content": reply.text or None, "refusal": None}
        if reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{index:04d}_{position}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": _arguments(call.arguments)},
                }
                for position, call in enumerate(reply.tool_calls)
            ]
        finish_reason = stop or reply.stop_reason or ("tool_calls" if reply.tool_calls else "stop")
        cached = self._cache.account(
            _units(request, entries(Format.OPENAI_CHAT, request), messages)
        )
        output_tokens = (
            reply.output_tokens if reply.output_tokens is not None else count_json(message)
        )
        return {
            "id": f"chatcmpl-{index:04d}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": tokens,
                "completion_tokens": output_tokens,
                "total_tokens": tokens + output_tokens,
                "prompt_tokens_details": {"cached_tokens": min(cached, tokens)},
            },
        }

    def _input(self, request: JSONObject) -> list[Message]:
        """Parse the input, and check that compaction and reasoning items come back unchanged."""
        if not isinstance(request.get("input"), str | list):
            raise _invalid("Missing required parameter: 'input'.", param="input")
        try:
            items = openai_responses.parse_items(entries(Format.OPENAI_RESPONSES, request))
        except FormatError as error:
            raise _invalid(str(error), param="input") from error
        for item in items:
            for block in item.blocks:
                if isinstance(block, Compaction) and not self._was_issued(block.data):
                    raise _invalid(
                        "Invalid compaction item: send it back exactly as it was returned.",
                        param="input",
                    )
                if (
                    isinstance(block, Opaque)
                    and block.type == "reasoning"
                    and not self._was_issued(block.data)
                ):
                    raise _invalid(
                        f"The encrypted content for item {block.data.get('id')} could not "
                        "be verified.",
                        param="input",
                    )
        return items

    def _was_issued(self, item: JSONObject) -> bool:
        item_id = item.get("id")
        return isinstance(item_id, str) and self._issued.get(item_id) == item

    def _issue(self, item: JSONObject) -> JSONObject:
        item_id = item["id"]
        assert isinstance(item_id, str)
        self._issued[item_id] = clone(item)
        return item

    def _issue_compaction(self, visible: Sequence[Message], index: int) -> JSONObject:
        summary = self.summarizer.summarize(visible, None)
        return self._issue(
            {
                "id": f"cmp_{index:04d}",
                "type": "compaction",
                "encrypted_content": _encrypt(summary),
            }
        )

    def _output_items(self, reply: Reply, index: int) -> list[JSONValue]:
        items: list[JSONValue] = []
        if reply.thinking is not None:
            reasoning: JSONObject = {
                "id": f"rs_{index:04d}",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": _encrypt(reply.thinking),
            }
            items.append(self._issue(reasoning))
        items.extend(clone(item) for item in reply.raw)
        if reply.text:
            items.append(
                {
                    "id": f"msg_{index:04d}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": reply.text, "annotations": []}],
                }
            )
        for position, call in enumerate(reply.tool_calls):
            items.append(
                {
                    "id": f"fc_{index:04d}_{position}",
                    "type": "function_call",
                    "call_id": f"call_{index:04d}_{position}",
                    "name": call.name,
                    "arguments": _arguments(call.arguments),
                    "status": "completed",
                }
            )
        return items


class _Responses:
    def __init__(self, client: FakeOpenAI) -> None:
        self._client = client

    def create(self, **kwargs: object) -> FakeObject:
        """Create a response, like ``client.responses.create``."""
        client = self._client
        handler = client._responses_create  # pyright: ignore[reportPrivateUsage]
        return client._call("responses.create", Format.OPENAI_RESPONSES, kwargs, handler)  # pyright: ignore[reportPrivateUsage]

    def compact(self, **kwargs: object) -> FakeObject:
        """Compact an input, like ``client.responses.compact``."""
        client = self._client
        handler = client._responses_compact  # pyright: ignore[reportPrivateUsage]
        return client._call("responses.compact", Format.OPENAI_RESPONSES, kwargs, handler)  # pyright: ignore[reportPrivateUsage]


class _Completions:
    def __init__(self, client: FakeOpenAI) -> None:
        self._client = client

    def create(self, **kwargs: object) -> FakeObject:
        """Create a chat completion, like ``client.chat.completions.create``."""
        client = self._client
        handler = client._chat_create  # pyright: ignore[reportPrivateUsage]
        return client._call("chat.completions.create", Format.OPENAI_CHAT, kwargs, handler)  # pyright: ignore[reportPrivateUsage]


class _Chat:
    def __init__(self, client: FakeOpenAI) -> None:
        self.completions = _Completions(client)


def _invalid(message: str, *, param: str | None = None, code: str | None = None) -> FakeAPIError:
    return FakeAPIError.openai(400, message, param=param, code=code)


def _context_exceeded(tokens: int, window: int, param: str) -> FakeAPIError:
    return _invalid(
        f"Your input exceeds the context window of this model: {tokens} tokens > {window}.",
        param=param,
        code="context_length_exceeded",
    )


def _model(request: JSONObject) -> str:
    model = request.get("model")
    if not isinstance(model, str):
        raise _invalid("Missing required parameter: 'model'.", param="model")
    return model


def _encrypt(text: str) -> str:
    """Stand in for encryption: the content is opaque to callers, but the fake can read it."""
    return base64.b64encode(text.encode()).decode()


def _arguments(arguments: JSONObject) -> str:
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _tool_tokens(request: JSONObject) -> int:
    tools = request.get("tools")
    return sum(count_json(tool) for tool in tools) if isinstance(tools, list) else 0


def _compaction_threshold(request: JSONObject) -> int | None:
    settings = request.get("context_management")
    if settings is None:
        return None
    if not isinstance(settings, list):
        raise _invalid("context_management must be a list.", param="context_management")
    for setting in settings:
        if isinstance(setting, dict) and setting.get("type") == "compaction":
            threshold = setting.get("compact_threshold")
            if not isinstance(threshold, int) or isinstance(threshold, bool):
                raise _invalid("compact_threshold must be an integer.", param="context_management")
            return threshold
    return None


def _check_function_pairs(items: Sequence[Message]) -> None:
    calls: set[str] = set()
    answered: set[str] = set()
    for item in items:
        for block in item.blocks:
            if isinstance(block, ToolUse):
                calls.add(block.id)
            elif isinstance(block, ToolResult):
                if block.tool_use_id not in calls:
                    raise _invalid(
                        f"No tool call found for function call output with call_id "
                        f"{block.tool_use_id}.",
                        param="input",
                    )
                answered.add(block.tool_use_id)
    missing = sorted(calls - answered)
    if missing:
        raise _invalid(f"No tool output found for function call {missing[0]}.", param="input")


def _check_tool_messages(messages: Sequence[Message]) -> None:
    expected: set[str] = set()
    pending: set[str] = set()
    for position, message in enumerate(messages):
        if message.role == "tool" and not message.bare_item:
            result = message.tool_results[0]
            if result.tool_use_id not in expected:
                raise _invalid(
                    # "preceeding" is the API's own spelling.
                    "Invalid parameter: messages with role 'tool' must be a response to a "
                    "preceeding message with 'tool_calls'.",
                    param=f"messages.[{position}].role",
                )
            pending.discard(result.tool_use_id)
            continue
        if pending:
            raise _missing_tool_messages(pending)
        expected = {use.id for use in message.tool_uses} if message.role == "assistant" else set()
        pending = set(expected)
    if pending:
        raise _missing_tool_messages(pending)


def _missing_tool_messages(pending: set[str]) -> FakeAPIError:
    return _invalid(
        "An assistant message with 'tool_calls' must be followed by tool messages responding "
        "to each 'tool_call_id'. The following tool_call_ids did not have response messages: "
        + ", ".join(sorted(pending)),
        param="messages",
    )


def _responses_usage(
    input_tokens: int, cached: int, output_tokens: int, reasoning_tokens: int
) -> JSONObject:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _units(
    request: JSONObject, prompt: Sequence[JSONValue], messages: Sequence[Message]
) -> list[Unit]:
    """Split the prompt the model reads into cacheable pieces: instructions, tools, entries.

    ``prompt`` holds the entries as sent and ``messages`` the same entries parsed.
    """
    units: list[Unit] = []
    instructions = request.get("instructions")
    if isinstance(instructions, str):
        units.append(Unit(key=json.dumps(instructions), tokens=count_json(instructions)))
    tools = request.get("tools")
    if isinstance(tools, list):
        units.extend(
            Unit(key=json.dumps(tool, sort_keys=True), tokens=count_json(tool)) for tool in tools
        )
    for raw, message in zip(prompt, messages, strict=True):
        units.append(Unit(key=json.dumps(raw, sort_keys=True), tokens=count_message(message)))
    return units
