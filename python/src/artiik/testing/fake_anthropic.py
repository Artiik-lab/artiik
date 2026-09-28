"""A fake Anthropic client that enforces the Messages API rules artiik relies on.

Every request is checked the way the API checks it:

- each ``tool_use`` is answered by a ``tool_result`` in the next message, and
  tool results come first in that message;
- at most four cache breakpoints, and the context window;
- signed blocks (``thinking`` and ``compaction``) come back exactly as the fake
  returned them, apart from ``cache_control``;
- the on-demand compaction protocol (Anthropic docs, "Compaction on demand",
  beta ``compact-2026-09-04``): the beta header, the parameters that can't be
  combined with ``compaction``, and the returned block sent back first and
  alone on every later request.

Replies come from a :class:`~artiik.testing.model.Policy`, summaries from a
:class:`~artiik.testing.model.Summarizer`, and prompt caching is simulated.
Reference: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence

from artiik.errors import FormatError
from artiik.formats import anthropic_messages
from artiik.formats._json import clone, to_object, without
from artiik.messages import Compaction, Format, JSONObject, JSONValue, Message, ToolResult
from artiik.testing.caching import AnthropicCache, Unit
from artiik.testing.errors import FakeAPIError
from artiik.testing.model import DigestSummarizer, Policy, Reply, Summarizer, ToolLoopPolicy
from artiik.testing.objects import FakeObject
from artiik.testing.recording import RecordedCall
from artiik.testing.tokens import MESSAGE_OVERHEAD, count_block, count_json, count_text

FORMAT = Format.ANTHROPIC_MESSAGES
COMPACTION_BETA = "compact-2026-09-04"
MAX_CACHE_BREAKPOINTS = 4
MAX_INSTRUCTIONS_CHARS = 16_384


class FakeAnthropic:
    """A stand-in for ``anthropic.Anthropic`` with ``messages.create`` and ``beta.messages.create``.

    - ``policy`` decides the replies; the default is a small tool-calling agent.
    - ``summarizer`` writes compaction summaries.
    - ``compaction_models`` lists the models that support on-demand compaction
      (``None`` means all of them).
    - ``faults`` maps a call index to an error to raise on that call, or to a
      stop reason to return instead of the normal answer.

    Every call is recorded in ``calls``, including the ones that fail.
    """

    def __init__(
        self,
        *,
        policy: Policy | None = None,
        summarizer: Summarizer | None = None,
        context_window: int = 200_000,
        compaction_models: Collection[str] | None = None,
        faults: Mapping[int, FakeAPIError | str] | None = None,
        cache_lookback: int = 20,
    ) -> None:
        self.policy: Policy = policy if policy is not None else ToolLoopPolicy()
        self.summarizer: Summarizer = summarizer if summarizer is not None else DigestSummarizer()
        self.context_window = context_window
        self.compaction_models = None if compaction_models is None else frozenset(compaction_models)
        self.faults = dict(faults or {})
        self.calls: list[RecordedCall] = []
        self.messages = _Messages(self, beta=False)
        self.beta = _Beta(self)
        self._cache = AnthropicCache(lookback=cache_lookback)
        self._signed: dict[str, JSONObject] = {}

    @property
    def requests(self) -> list[JSONObject]:
        """The requests of every call, in order."""
        return [call.request for call in self.calls]

    def _create(self, kwargs: Mapping[str, object], *, beta: bool) -> FakeObject:
        index = len(self.calls)
        endpoint = "beta.messages.create" if beta else "messages.create"
        if not beta and "betas" in kwargs:
            raise TypeError("Messages.create() got an unexpected keyword argument 'betas'")
        try:
            request = to_object(dict(kwargs), "request")
        except FormatError as error:
            raise _invalid(str(error)) from error
        fault = self.faults.get(index)
        try:
            if isinstance(fault, FakeAPIError):
                raise fault
            response = self._respond(request, index, stop_override=fault)
        except FakeAPIError as error:
            self.calls.append(RecordedCall(index, endpoint, FORMAT, request, error=error))
            raise
        self.calls.append(RecordedCall(index, endpoint, FORMAT, request, response=response))
        return FakeObject(clone(response))

    def _respond(self, request: JSONObject, index: int, *, stop_override: str | None) -> JSONObject:
        model = request.get("model")
        if not isinstance(model, str):
            raise _invalid("model: Field required")
        if not isinstance(request.get("max_tokens"), int):
            raise _invalid("max_tokens: Field required")
        raw_messages = request.get("messages")
        if not isinstance(raw_messages, list):
            raise _invalid("messages: Field required")
        try:
            messages = anthropic_messages.parse_messages(raw_messages)
            system = (
                anthropic_messages.parse_system(request["system"]) if "system" in request else None
            )
        except FormatError as error:
            raise _invalid(str(error)) from error
        betas = _betas(request)
        self._check_compaction_block(messages, COMPACTION_BETA in betas)
        self._check_thinking(raw_messages)
        _check_cache_breakpoints(request)
        _check_tool_pairs(messages)
        units = _units(request, messages, system)
        tokens = sum(unit.tokens for unit in units)
        if tokens > self.context_window:
            raise _invalid(f"prompt is too long: {tokens} tokens > {self.context_window} maximum")
        if "compaction" in request:
            return self._compact(request, model, messages, betas, tokens, index, stop_override)
        return self._reply(model, messages, units, index, stop_override)

    def _check_compaction_block(self, messages: Sequence[Message], beta_enabled: bool) -> None:
        found = [
            (message_index, block_index, block)
            for message_index, message in enumerate(messages)
            for block_index, block in enumerate(message.blocks)
            if isinstance(block, Compaction)
        ]
        if not found:
            return
        message_index, block_index, block = found[0]
        if not beta_enabled:
            raise _invalid(
                f"messages.{message_index}.content.{block_index}: 'compaction' is not one of "
                "the expected content block types"
            )
        if len(found) > 1:
            raise _invalid("A request can carry only one compaction block. Send the newest one.")
        if (message_index, block_index) != (0, 0):
            raise _invalid(
                "The compaction block must come first in messages; remove the messages it "
                "summarizes.",
                code="compaction_block_misplaced",
            )
        sent = without(block.data, "cache_control")
        signature = sent.get("signature")
        issued = self._signed.get(signature) if isinstance(signature, str) else None
        if issued is None or issued.get("type") != "compaction":
            raise _invalid(
                "The compaction block's signature is invalid.", code="compaction_signature_invalid"
            )
        extra = sorted(set(sent) - set(issued))
        if extra:
            raise _invalid(
                f"messages.0.content.0.compaction.{extra[0]}: Extra inputs are not permitted"
            )
        if sent != issued:
            raise _invalid(
                "The compaction block's content doesn't match its signature.",
                code="compaction_content_mismatch",
            )

    def _check_thinking(self, raw_messages: list[JSONValue]) -> None:
        for message_index, message in enumerate(raw_messages):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for block_index, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "thinking":
                    continue
                signature = block.get("signature")
                issued = self._signed.get(signature) if isinstance(signature, str) else None
                if issued is None or without(block, "cache_control") != issued:
                    raise _invalid(
                        f"messages.{message_index}.content.{block_index}: Invalid `signature` "
                        "in `thinking` block"
                    )

    def _compact(
        self,
        request: JSONObject,
        model: str,
        messages: Sequence[Message],
        betas: set[str],
        tokens: int,
        index: int,
        stop_override: str | None,
    ) -> JSONObject:
        if COMPACTION_BETA not in betas:
            raise _invalid(f"The compaction parameter requires anthropic-beta: {COMPACTION_BETA}")
        instructions = _compaction_instructions(request.get("compaction"))
        if "context_management" in request:
            raise _invalid("compaction can't be combined with context_management on one request")
        if "stop_sequences" in request:
            raise _invalid("stop_sequences can't be sent with compaction")
        tool_choice = request.get("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("any", "tool"):
            raise _invalid("tool_choice of type 'any' or 'tool' can't be sent with compaction")
        output_config = request.get("output_config")
        if isinstance(output_config, dict) and "format" in output_config:
            raise _invalid("output_config.format can't be sent with compaction")
        if self.compaction_models is not None and model not in self.compaction_models:
            raise _invalid(f"model {model} does not support compaction")
        if not any(message.blocks for message in messages):
            raise _invalid("There is nothing to summarize.", code="compaction_nothing_to_summarize")
        iteration: JSONObject = {"type": "compaction", "input_tokens": tokens, "output_tokens": 0}
        usage: JSONObject = {"input_tokens": 0, "output_tokens": 0, "iterations": [iteration]}
        if stop_override is not None:
            return _message(index, model, [], stop_override, usage)
        summary = self.summarizer.summarize(messages, instructions)
        block: JSONObject = {
            "type": "compaction",
            "content": summary,
            "signature": _signature("compaction", index, summary),
        }
        self._sign(block)
        iteration["output_tokens"] = count_text(summary)
        return _message(index, model, [block], "compaction", usage)

    def _reply(
        self,
        model: str,
        messages: Sequence[Message],
        units: Sequence[Unit],
        index: int,
        stop_override: str | None,
    ) -> JSONObject:
        reply = self.policy.reply(messages)
        content = self._content(reply, index)
        stop_reason = (
            stop_override or reply.stop_reason or ("tool_use" if reply.tool_calls else "end_turn")
        )
        cache = self._cache.account(units)
        output_tokens = (
            reply.output_tokens if reply.output_tokens is not None else count_json(content)
        )
        usage: JSONObject = {
            "input_tokens": cache.uncached,
            "cache_creation_input_tokens": cache.cache_write,
            "cache_read_input_tokens": cache.cache_read,
            "output_tokens": output_tokens,
        }
        return _message(index, model, content, stop_reason, usage)

    def _content(self, reply: Reply, index: int) -> list[JSONValue]:
        content: list[JSONValue] = []
        if reply.thinking is not None:
            thinking: JSONObject = {
                "type": "thinking",
                "thinking": reply.thinking,
                "signature": _signature("thinking", index, reply.thinking),
            }
            self._sign(thinking)
            content.append(thinking)
        content.extend(clone(block) for block in reply.raw)
        if reply.text:
            content.append({"type": "text", "text": reply.text})
        for call_index, call in enumerate(reply.tool_calls):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{index:04d}_{call_index}",
                    "name": call.name,
                    "input": clone(call.arguments),
                }
            )
        return content

    def _sign(self, block: JSONObject) -> None:
        signature = block["signature"]
        assert isinstance(signature, str)
        self._signed[signature] = clone(block)


class _Messages:
    def __init__(self, client: FakeAnthropic, *, beta: bool) -> None:
        self._client = client
        self._beta = beta

    def create(self, **kwargs: object) -> FakeObject:
        """Send a request, like ``client.messages.create``."""
        return self._client._create(kwargs, beta=self._beta)  # pyright: ignore[reportPrivateUsage]


class _Beta:
    def __init__(self, client: FakeAnthropic) -> None:
        self.messages = _Messages(client, beta=True)


def _invalid(message: str, *, code: str | None = None) -> FakeAPIError:
    return FakeAPIError.anthropic(400, "invalid_request_error", message, code=code)


def _signature(kind: str, index: int, text: str) -> str:
    return "sig_" + hashlib.sha256(f"{kind}:{index}:{text}".encode()).hexdigest()[:32]


def _message(
    index: int, model: str, content: list[JSONValue], stop_reason: str, usage: JSONObject
) -> JSONObject:
    return {
        "id": f"msg_{index:04d}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _betas(request: JSONObject) -> set[str]:
    betas: set[str] = set()
    raw = request.get("betas")
    if isinstance(raw, list):
        betas.update(item for item in raw if isinstance(item, str))
    headers = request.get("extra_headers")
    if isinstance(headers, dict):
        header = headers.get("anthropic-beta")
        if isinstance(header, str):
            betas.update(part.strip() for part in header.split(",") if part.strip())
    return betas


def _compaction_instructions(config: JSONValue) -> str | None:
    """Check the ``compaction`` parameter and return its instructions, if any."""
    if not isinstance(config, dict):
        raise _invalid("compaction: Input should be a valid dictionary")
    kind = config.get("type", "summarize")
    if kind != "summarize":
        raise _invalid(f"compaction.type: Input should be 'summarize', got {kind!r}")
    instructions = config.get("instructions")
    if instructions is None:
        return None
    if (
        not isinstance(instructions, str)
        or not instructions.strip()
        or len(instructions) > MAX_INSTRUCTIONS_CHARS
    ):
        raise _invalid(
            "compaction.instructions: must be a non-blank string of at most "
            f"{MAX_INSTRUCTIONS_CHARS} characters"
        )
    return instructions


def _check_cache_breakpoints(request: JSONObject) -> None:
    found = 1 if "cache_control" in request else 0
    for key in ("tools", "system", "messages"):
        found += _count_cache_control(request.get(key), top=key == "messages")
    if found > MAX_CACHE_BREAKPOINTS:
        raise _invalid(
            f"A maximum of {MAX_CACHE_BREAKPOINTS} blocks with cache_control may be provided. "
            f"Found {found}."
        )


def _count_cache_control(value: JSONValue, *, top: bool = False) -> int:
    if isinstance(value, list):
        return sum(_count_cache_control(item, top=top) for item in value)
    if isinstance(value, dict):
        if top:
            return _count_cache_control(value.get("content"))
        own = 1 if "cache_control" in value else 0
        return own + _count_cache_control(value.get("content"))
    return 0


def _check_tool_pairs(messages: Sequence[Message]) -> None:
    for index, message in enumerate(messages):
        if message.role == "assistant" and message.tool_uses:
            following = messages[index + 1] if index + 1 < len(messages) else None
            answered: set[str] = (
                {result.tool_use_id for result in following.tool_results}
                if following is not None and following.role == "user"
                else set()
            )
            missing = sorted({use.id for use in message.tool_uses} - answered)
            if missing:
                raise _invalid(
                    f"messages.{index + 1}: tool_use ids were found without tool_result blocks "
                    f"immediately after: {', '.join(missing)}. Each tool_use block must have a "
                    "corresponding tool_result block in the next message."
                )
        if message.role == "user" and message.tool_results:
            previous = messages[index - 1] if index > 0 else None
            known: set[str] = (
                {use.id for use in previous.tool_uses}
                if previous is not None and previous.role == "assistant"
                else set()
            )
            for result in message.tool_results:
                if result.tool_use_id not in known:
                    raise _invalid(
                        f"messages.{index}.content: unexpected tool_use_id found in tool_result "
                        f"blocks: {result.tool_use_id}. Each tool_result block must have a "
                        "corresponding tool_use block in the previous message."
                    )
            seen_other = False
            for block in message.blocks:
                if not isinstance(block, ToolResult):
                    seen_other = True
                elif seen_other:
                    raise _invalid(
                        f"messages.{index}.content: tool_result blocks must come before any "
                        "other content in the message."
                    )


def _units(request: JSONObject, messages: Sequence[Message], system: Message | None) -> list[Unit]:
    """Split a request into cacheable pieces, in cache order: tools, system, messages."""
    units: list[Unit] = []
    tools = request.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            units.append(_unit(tool, count_json(tool)))
    raw_system = request.get("system")
    if system is not None:
        raw_blocks = raw_system if isinstance(raw_system, list) else [raw_system]
        units.extend(_block_units(raw_blocks, system, overhead=0))
    raw_messages = request.get("messages")
    if isinstance(raw_messages, list):
        for raw, message in zip(raw_messages, messages, strict=True):
            content = raw.get("content") if isinstance(raw, dict) else None
            raw_blocks = content if isinstance(content, list) else [content]
            units.extend(_block_units(raw_blocks, message, overhead=MESSAGE_OVERHEAD))
    if "cache_control" in request and units:
        last = units[-1]
        units[-1] = Unit(key=last.key, tokens=last.tokens, breakpoint=True)
    return units


def _block_units(raw_blocks: Sequence[JSONValue], message: Message, *, overhead: int) -> list[Unit]:
    units: list[Unit] = []
    for position, (raw, block) in enumerate(zip(raw_blocks, message.blocks, strict=True)):
        tokens = count_block(block) + (overhead if position == 0 else 0)
        units.append(_unit(raw, tokens))
    if not message.blocks and overhead:
        units.append(Unit(key=json.dumps([message.role]), tokens=overhead))
    return units


def _unit(raw: JSONValue, tokens: int) -> Unit:
    if isinstance(raw, dict):
        key = json.dumps(without(raw, "cache_control"), sort_keys=True, ensure_ascii=False)
        return Unit(key=key, tokens=tokens, breakpoint="cache_control" in raw)
    return Unit(key=json.dumps(raw, ensure_ascii=False), tokens=tokens)
