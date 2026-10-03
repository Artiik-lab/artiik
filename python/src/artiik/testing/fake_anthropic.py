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
  alone on every later request;
- compaction at a token threshold (beta ``compact-2026-01-12``): a
  ``compact_20260112`` edit in ``context_management`` compacts inside an
  ordinary request once the input passes its trigger, and the API drops the
  content before the latest compaction block;
- tool-result clearing (beta ``context-management-2025-06-27``): a
  ``clear_tool_uses_20250919`` edit, checked field by field, clears the
  oldest tool results past its trigger before the model reads them (and
  before a threshold compaction), and the response lists it in
  ``context_management.applied_edits``; ``count_tokens`` counts after it;
- ``role: "system"`` messages inside ``messages`` (Anthropic docs,
  "Mid-conversation system messages"): not first, right after a user turn
  (tool results count), before an assistant turn or last, with no
  ``cache_control``, and only on the models that support them.

``models.retrieve`` reports each model's capabilities, as the Models API does.

Replies come from a :class:`~artiik.testing.model.Policy`, summaries from a
:class:`~artiik.testing.model.Summarizer`, and prompt caching is simulated.
Reference: https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from typing import cast

from artiik.errors import FormatError
from artiik.formats import anthropic_messages
from artiik.formats._json import clone, to_object, without
from artiik.messages import (
    Compaction,
    Format,
    JSONObject,
    JSONValue,
    Message,
    Opaque,
    ToolResult,
    ToolUse,
    visible,
)
from artiik.testing.caching import AnthropicCache, Unit
from artiik.testing.errors import FakeAPIError
from artiik.testing.model import DigestSummarizer, Policy, Reply, Summarizer, ToolLoopPolicy
from artiik.testing.objects import FakeObject
from artiik.testing.recording import RecordedCall
from artiik.testing.tokens import DEFAULT, Tokenizer

FORMAT = Format.ANTHROPIC_MESSAGES
COMPACTION_BETA = "compact-2026-09-04"
THRESHOLD_BETA = "compact-2026-01-12"
THRESHOLD_MINIMUM = 50_000
CLEARING_BETA = "context-management-2025-06-27"
CLEARING_EDIT = "clear_tool_uses_20250919"
CLEARED = "[This tool result was cleared to save context.]"
"""What the fake puts in place of a cleared tool result."""
_CLEARING_FIELDS = frozenset(
    {"type", "trigger", "keep", "clear_at_least", "exclude_tools", "clear_tool_inputs"}
)
_EDIT_TYPES = ("clear_tool_uses_20250919", "clear_thinking_20251015", "compact_20260112")
MAX_CACHE_BREAKPOINTS = 4
MAX_INSTRUCTIONS_CHARS = 16_384
BETA_ONLY = frozenset(
    {"betas", "compaction", "context_management", "fallback_credit_token", "fallbacks"}
    | {"mcp_servers", "speed"}
)
"""Parameters the SDK's ``beta.messages.create`` takes and ``messages.create`` doesn't."""
COUNT_BETA_ONLY = frozenset({"betas", "compaction", "context_management", "mcp_servers", "speed"})
"""The same for ``count_tokens``."""


class FakeAnthropic:
    """A stand-in for ``anthropic.Anthropic`` with ``messages.create`` and ``beta.messages.create``.

    ``messages.count_tokens`` counts a request the way ``create`` would.

    - ``policy`` decides the replies; the default is a small tool-calling agent.
    - ``summarizer`` writes compaction summaries.
    - ``tokenizer`` counts tokens; pass another one to stand in for a model
      family whose tokenizer differs.
    - ``compaction_models`` lists the models that support compaction, on
      demand and at a threshold (``None`` means all of them).
    - ``system_message_models`` lists the models that take ``system``
      messages inside ``messages`` (``None`` means all of them).
    - ``check_thinking_prefix`` rejects a thinking block sent back after
      anything before it changed (the system prompt, the tools, an earlier
      message), as models that check preserved thinking do. Server-side
      clearing doesn't count, and requests with a compaction block aren't
      checked.
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
        system_message_models: Collection[str] | None = None,
        check_thinking_prefix: bool = False,
        faults: Mapping[int, FakeAPIError | str] | None = None,
        cache_lookback: int = 20,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.policy: Policy = policy if policy is not None else ToolLoopPolicy()
        self.summarizer: Summarizer = summarizer if summarizer is not None else DigestSummarizer()
        self.context_window = context_window
        self.compaction_models = None if compaction_models is None else frozenset(compaction_models)
        self.system_message_models = (
            None if system_message_models is None else frozenset(system_message_models)
        )
        self.check_thinking_prefix = check_thinking_prefix
        self.faults = dict(faults or {})
        self.tokenizer = tokenizer if tokenizer is not None else DEFAULT
        self.calls: list[RecordedCall] = []
        self.count_requests: list[JSONObject] = []
        self.model_requests: list[str] = []
        self.messages = _Messages(self, beta=False)
        self.models = _Models(self, beta=False)
        self.beta = _Beta(self)
        self._cache = AnthropicCache(lookback=cache_lookback)
        self._signed: dict[str, JSONObject] = {}
        self._prefixes: dict[str, str] = {}
        """The conversation each thinking block was made after, by signature."""
        self._threshold_blocks: set[str] = set()
        self._counted: int | None = None

    @property
    def requests(self) -> list[JSONObject]:
        """The requests of every call, in order."""
        return [call.request for call in self.calls]

    def _create(self, kwargs: Mapping[str, object], *, beta: bool) -> FakeObject:
        index = len(self.calls)
        endpoint = "beta.messages.create" if beta else "messages.create"
        if not beta:
            _check_keywords("Messages.create", kwargs, BETA_ONLY)
        try:
            request = to_object(dict(kwargs), "request")
        except FormatError as error:
            raise _invalid(str(error)) from error
        fault = self.faults.get(index)
        self._counted = None
        try:
            if isinstance(fault, FakeAPIError):
                raise fault
            response = self._respond(request, index, stop_override=fault)
        except FakeAPIError as error:
            self.calls.append(
                RecordedCall(
                    index, endpoint, FORMAT, request, error=error, prompt_tokens=self._counted
                )
            )
            raise
        self.calls.append(
            RecordedCall(
                index, endpoint, FORMAT, request, response=response, prompt_tokens=self._counted
            )
        )
        return FakeObject(clone(response))

    def _count_tokens(self, kwargs: Mapping[str, object], *, beta: bool) -> FakeObject:
        if not beta:
            _check_keywords("Messages.count_tokens", kwargs, COUNT_BETA_ONLY)
        try:
            request = to_object(dict(kwargs), "request")
        except FormatError as error:
            raise _invalid(str(error)) from error
        self.count_requests.append(request)
        _, messages, system, betas, units = self._read(request, reply=False)
        before = sum(unit.tokens for unit in units)
        _, units, applied = self._clear(request, betas, messages, system, units)
        result: JSONObject = {"input_tokens": sum(unit.tokens for unit in units)}
        if applied is not None:
            result["context_management"] = {"original_input_tokens": before}
        return FakeObject(result)

    def _model_info(self, model: str, kwargs: Mapping[str, object], *, beta: bool) -> FakeObject:
        if not beta:
            _check_keywords("Models.retrieve", kwargs, {"betas"})
        self.model_requests.append(model)
        supported = self.compaction_models is None or model in self.compaction_models
        capabilities: JSONObject = {
            "compaction": {"supported": supported, "summarize": {"supported": supported}},
            "context_management": {
                "supported": True,
                "clear_tool_uses_20250919": {"supported": True},
                "compact_20260112": {"supported": supported},
            },
        }
        return FakeObject(
            {
                "type": "model",
                "id": model,
                "display_name": model,
                "created_at": "2026-01-01T00:00:00Z",
                "capabilities": capabilities if beta else None,
                "max_input_tokens": self.context_window,
                "max_tokens": 64_000,
            }
        )

    def _respond(self, request: JSONObject, index: int, *, stop_override: str | None) -> JSONObject:
        model, messages, system, betas, units = self._read(request, reply=True)
        if "compaction" in request and "context_management" in request:
            raise _invalid("compaction can't be combined with context_management on one request")
        messages, units, applied = self._clear(request, betas, messages, system, units)
        edit = _threshold_edit(request, betas)
        if edit is not None:
            if self.compaction_models is not None and model not in self.compaction_models:
                raise _invalid(f"model {model} does not support compact_20260112")
            response = self._threshold(request, model, messages, system, edit, index, stop_override)
        else:
            tokens = sum(unit.tokens for unit in units)
            self._counted = tokens
            if tokens > self.context_window:
                raise _invalid(
                    f"prompt is too long: {tokens} tokens > {self.context_window} maximum"
                )
            if "compaction" in request:
                return self._compact(request, model, messages, betas, tokens, index, stop_override)
            response = self._reply(model, messages, units, index, stop_override)
        content = response.get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and isinstance(block.get("signature"), str):
                # The conversation as sent, before any server-side edit.
                self._prefixes[cast(str, block["signature"])] = _prefix(
                    request, cast(list[JSONValue], request["messages"])
                )
        if applied is not None:
            response["context_management"] = {"applied_edits": applied}
        return response

    def _clear(
        self,
        request: JSONObject,
        betas: set[str],
        messages: list[Message],
        system: Message | None,
        units: list[Unit],
    ) -> tuple[list[Message], list[Unit], list[JSONValue] | None]:
        """Apply a ``clear_tool_uses_20250919`` edit, as the API does before the model reads.

        Returns the messages and units the model reads, and the applied edits,
        or ``None`` when the request has no clearing edit.
        """
        edit = _clearing_edit(request, betas)
        if edit is None:
            return messages, units, None
        trigger = edit["trigger"]
        keep = edit["keep"]
        assert isinstance(trigger, dict) and isinstance(keep, dict)
        uses = [use for message in messages for use in message.tool_uses]
        tokens = sum(unit.tokens for unit in units)
        measure = tokens if trigger["type"] == "input_tokens" else len(uses)
        limit = trigger["value"]
        assert isinstance(limit, int)
        if measure <= limit:
            return messages, units, []
        kept = keep["value"]
        assert isinstance(kept, int)
        names = edit.get("exclude_tools")
        excluded: set[str] = (
            {name for name in names if isinstance(name, str)} if isinstance(names, list) else set()
        )
        answered = {result.tool_use_id for message in messages for result in message.tool_results}
        targets = [
            use
            for use in uses[: max(len(uses) - kept, 0)]
            if use.name not in excluded and use.id in answered
        ]
        inputs = edit.get("clear_tool_inputs")
        cleared_inputs = {
            use.id
            for use in targets
            if inputs is True or (isinstance(inputs, list) and use.name in inputs)
        }
        if not targets:
            return messages, units, []
        target_ids = {use.id for use in targets}
        raw = clone({"messages": request["messages"]})["messages"]
        assert isinstance(raw, list)
        for raw_message, message in zip(raw, messages, strict=True):
            content = raw_message.get("content") if isinstance(raw_message, dict) else None
            if not isinstance(content, list):
                continue
            for raw_block, block in zip(content, message.blocks, strict=True):
                if not isinstance(raw_block, dict):
                    continue
                if isinstance(block, ToolResult) and block.tool_use_id in target_ids:
                    raw_block["content"] = CLEARED
                elif isinstance(block, ToolUse) and block.id in cleared_inputs:
                    raw_block["input"] = {}
        cleared_request = {**request, "messages": raw}
        cleared = anthropic_messages.parse_messages(raw)
        cleared_units = _units(
            cleared_request,
            cleared,
            system,
            self.tokenizer,
            keyed=_check_cache_breakpoints(request) > 0,
        )
        saved = tokens - sum(unit.tokens for unit in cleared_units)
        least = edit.get("clear_at_least")
        minimum = least.get("value") if isinstance(least, dict) else None
        if isinstance(minimum, int) and saved < minimum:
            return messages, units, []
        applied: JSONObject = {
            "type": CLEARING_EDIT,
            "cleared_tool_uses": len(targets),
            "cleared_input_tokens": saved,
        }
        return cleared, cleared_units, [applied]

    def _read(
        self, request: JSONObject, *, reply: bool
    ) -> tuple[str, list[Message], Message | None, set[str], list[Unit]]:
        """Check a request the way the API does, and split it into cacheable units."""
        model = request.get("model")
        if not isinstance(model, str):
            raise _invalid("model: Field required")
        if reply and not isinstance(request.get("max_tokens"), int):
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
        self._check_compaction_block(messages, betas, threshold=_has_threshold_edit(request))
        self._check_thinking(raw_messages)
        if self.check_thinking_prefix:
            self._check_thinking_prefix(request, raw_messages)
        self._check_system_messages(model, messages)
        cached = _check_cache_breakpoints(request) > 0
        _check_tool_pairs(visible(FORMAT, messages))
        units = _units(request, messages, system, self.tokenizer, keyed=cached)
        return model, messages, system, betas, units

    def _check_compaction_block(
        self, messages: Sequence[Message], betas: set[str], *, threshold: bool
    ) -> None:
        found = [
            (message_index, block_index, block)
            for message_index, message in enumerate(messages)
            for block_index, block in enumerate(message.blocks)
            if isinstance(block, Compaction) and "signature" in block.data
        ]
        self._check_threshold_blocks(messages, betas)
        if not found:
            return
        if threshold:
            raise _invalid(
                "Threshold compaction (compact_20260112) can't run on a request that carries a "
                "signed compaction block."
            )
        message_index, block_index, block = found[0]
        if COMPACTION_BETA not in betas:
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

    def _check_system_messages(self, model: str, messages: Sequence[Message]) -> None:
        """Check where the ``system`` messages inside ``messages`` stand."""
        for index, message in enumerate(messages):
            if message.role != "system":
                continue
            if self.system_message_models is not None and model not in self.system_message_models:
                raise _invalid(
                    f'messages.{index}: model {model} does not support role: "system" '
                    "messages; use the top-level system parameter"
                )
            before = index - 1
            while before > 0 and messages[before].role == "system":
                before -= 1
            previous = messages[before] if index > 0 else None
            following = messages[index + 1] if index + 1 < len(messages) else None
            follows_user = previous is not None and (
                previous.role == "user" or _ends_in_server_tool_result(previous)
            )
            if (
                not follows_user
                or before < 0
                or (following is not None and following.role not in ("assistant", "system"))
            ):
                raise _invalid(
                    f'messages.{index}: a role: "system" message with content must '
                    "immediately follow a user turn (including a user turn carrying tool_result "
                    "blocks) or an assistant turn ending in a server tool result, or be the "
                    "last message"
                )
            for position, block in enumerate(message.blocks):
                if "cache_control" in block.extra:
                    raise _invalid(
                        f"messages.{index}.content.{position}: cache_control is not permitted "
                        'on a role: "system" message'
                    )

    def _check_threshold_blocks(self, messages: Sequence[Message], betas: set[str]) -> None:
        """Check the unsigned blocks that threshold compaction returned."""
        for message_index, message in enumerate(messages):
            for block_index, block in enumerate(message.blocks):
                if not isinstance(block, Compaction) or "signature" in block.data:
                    continue
                path = f"messages.{message_index}.content.{block_index}"
                if THRESHOLD_BETA not in betas:
                    raise _invalid(
                        f"{path}: 'compaction' is not one of the expected content block types"
                    )
                if _canonical(block.data) not in self._threshold_blocks:
                    raise _invalid(
                        f"{path}: the compaction block doesn't match one the API returned.",
                        code="compaction_content_mismatch",
                    )

    def _threshold(
        self,
        request: JSONObject,
        model: str,
        messages: Sequence[Message],
        system: Message | None,
        edit: JSONObject,
        index: int,
        stop_override: str | None,
    ) -> JSONObject:
        """Answer a request that carries a ``compact_20260112`` edit."""
        shown = visible(FORMAT, messages)
        fixed = self._fixed_tokens(request, system)
        tokens = fixed + self.tokenizer.count_messages(shown)
        self._counted = tokens
        if tokens > self.context_window:
            raise _invalid(f"prompt is too long: {tokens} tokens > {self.context_window} maximum")
        iterations: list[JSONValue] = []
        content: list[JSONValue] = []
        reads = shown
        trigger = edit["trigger"]
        assert isinstance(trigger, dict)
        value = trigger["value"]
        assert isinstance(value, int)
        if tokens > value:
            instructions = edit.get("instructions")
            summary = self.summarizer.summarize(
                shown, instructions if isinstance(instructions, str) else None
            )
            block: JSONObject = {"type": "compaction", "content": summary}
            self._threshold_blocks.add(_canonical(block))
            content.append(block)
            iterations.append(
                {
                    "type": "compaction",
                    "input_tokens": tokens,
                    "output_tokens": self.tokenizer.count_text(summary),
                }
            )
            reads = [
                Message(
                    role="assistant", blocks=(Compaction(data=block, origin=FORMAT),), origin=FORMAT
                )
            ]
        reply = self.policy.reply(reads)
        reply_content = self._content(reply, index)
        content.extend(reply_content)
        input_tokens = fixed + self.tokenizer.count_messages(reads)
        output_tokens = (
            reply.output_tokens
            if reply.output_tokens is not None
            else self.tokenizer.count_json(reply_content)
        )
        iterations.append(
            {"type": "message", "input_tokens": input_tokens, "output_tokens": output_tokens}
        )
        stop_reason = (
            stop_override or reply.stop_reason or ("tool_use" if reply.tool_calls else "end_turn")
        )
        usage: JSONObject = {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": output_tokens,
            "iterations": iterations,
        }
        return _message(index, model, content, stop_reason, usage)

    def _fixed_tokens(self, request: JSONObject, system: Message | None) -> int:
        tools = request.get("tools")
        tool_tokens = (
            sum(self.tokenizer.count_json(tool) for tool in tools) if isinstance(tools, list) else 0
        )
        system_tokens = (
            sum(self.tokenizer.count_block(block) for block in system.blocks) if system else 0
        )
        return tool_tokens + system_tokens

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

    def _check_thinking_prefix(self, request: JSONObject, raw_messages: list[JSONValue]) -> None:
        """Reject a thinking block whose conversation changed before it."""
        blocks = [
            (message_index, block_index, block)
            for message_index, message in enumerate(raw_messages)
            if isinstance(message, dict) and isinstance(message.get("content"), list)
            for block_index, block in enumerate(cast(list[JSONValue], message["content"]))
            if isinstance(block, dict)
        ]
        if any(block.get("type") == "compaction" for _, _, block in blocks):
            return
        for message_index, block_index, block in blocks:
            signature = block.get("signature")
            if block.get("type") != "thinking" or not isinstance(signature, str):
                continue
            made_after = self._prefixes.get(signature)
            if made_after is not None and made_after != _prefix(
                request, raw_messages[:message_index]
            ):
                raise _invalid(
                    f"messages.{message_index}.content.{block_index}: Invalid `signature` in "
                    "`thinking` block. The block is bound to a different conversation."
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
        iteration["output_tokens"] = self.tokenizer.count_text(summary)
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
            reply.output_tokens
            if reply.output_tokens is not None
            else self.tokenizer.count_json(content)
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

    def count_tokens(self, **kwargs: object) -> FakeObject:
        """Count a request's input tokens, like ``client.messages.count_tokens``."""
        return self._client._count_tokens(kwargs, beta=self._beta)  # pyright: ignore[reportPrivateUsage]


class _Models:
    def __init__(self, client: FakeAnthropic, *, beta: bool) -> None:
        self._client = client
        self._beta = beta

    def retrieve(self, model_id: str, **kwargs: object) -> FakeObject:
        """Describe a model, like ``client.models.retrieve``; only the beta lists capabilities."""
        return self._client._model_info(model_id, kwargs, beta=self._beta)  # pyright: ignore[reportPrivateUsage]


class _Beta:
    def __init__(self, client: FakeAnthropic) -> None:
        self.messages = _Messages(client, beta=True)
        self.models = _Models(client, beta=True)


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


def _ends_in_server_tool_result(message: Message) -> bool:
    last = message.blocks[-1] if message.role == "assistant" and message.blocks else None
    return isinstance(last, Opaque) and str(last.data.get("type", "")).endswith("_tool_result")


def _check_keywords(method: str, kwargs: Mapping[str, object], beta_only: Collection[str]) -> None:
    for name in kwargs:
        if name in beta_only:
            raise TypeError(f"{method}() got an unexpected keyword argument {name!r}")


def _betas(request: JSONObject) -> set[str]:
    """The betas a request turns on.

    Like the SDK, which sends ``betas`` as the ``anthropic-beta`` header, an
    ``anthropic-beta`` entry in ``extra_headers`` replaces the list.
    """
    headers = request.get("extra_headers")
    if isinstance(headers, dict):
        for name, header in headers.items():
            if name.lower() == "anthropic-beta" and isinstance(header, str):
                return {part.strip() for part in header.split(",") if part.strip()}
    raw = request.get("betas")
    return {item for item in raw if isinstance(item, str)} if isinstance(raw, list) else set()


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


def _check_cache_breakpoints(request: JSONObject) -> int:
    found = 1 if "cache_control" in request else 0
    for key in ("tools", "system", "messages"):
        found += _count_cache_control(request.get(key), top=key == "messages")
    if found > MAX_CACHE_BREAKPOINTS:
        raise _invalid(
            f"A maximum of {MAX_CACHE_BREAKPOINTS} blocks with cache_control may be provided. "
            f"Found {found}."
        )
    return found


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


def _units(
    request: JSONObject,
    messages: Sequence[Message],
    system: Message | None,
    tokenizer: Tokenizer,
    *,
    keyed: bool,
) -> list[Unit]:
    """Split a request into cacheable pieces, in cache order: tools, system, messages.

    Without ``keyed``, the request has no cache breakpoint, so the pieces only
    carry their sizes: serializing them for cache keys would be wasted work.
    """
    units: list[Unit] = []
    tools = request.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            units.append(_unit(tool, tokenizer.count_json(tool), keyed=keyed))
    raw_system = request.get("system")
    if system is not None:
        raw_blocks = raw_system if isinstance(raw_system, list) else [raw_system]
        units.extend(_block_units(raw_blocks, system, overhead=0, tokenizer=tokenizer, keyed=keyed))
    raw_messages = request.get("messages")
    if isinstance(raw_messages, list):
        for raw, message in zip(raw_messages, messages, strict=True):
            content = raw.get("content") if isinstance(raw, dict) else None
            raw_blocks = content if isinstance(content, list) else [content]
            units.extend(
                _block_units(
                    raw_blocks,
                    message,
                    overhead=tokenizer.message_overhead,
                    tokenizer=tokenizer,
                    keyed=keyed,
                )
            )
    if "cache_control" in request and units:
        last = units[-1]
        units[-1] = Unit(key=last.key, tokens=last.tokens, breakpoint=True)
    return units


def _block_units(
    raw_blocks: Sequence[JSONValue],
    message: Message,
    *,
    overhead: int,
    tokenizer: Tokenizer,
    keyed: bool,
) -> list[Unit]:
    units: list[Unit] = []
    for position, (raw, block) in enumerate(zip(raw_blocks, message.blocks, strict=True)):
        tokens = tokenizer.count_block(block) + (overhead if position == 0 else 0)
        units.append(_unit(raw, tokens, keyed=keyed))
    if not message.blocks and overhead:
        units.append(Unit(key=json.dumps([message.role]), tokens=overhead))
    return units


def _unit(raw: JSONValue, tokens: int, *, keyed: bool) -> Unit:
    if not keyed:
        return Unit(key="", tokens=tokens)
    if isinstance(raw, dict):
        key = json.dumps(without(raw, "cache_control"), sort_keys=True, ensure_ascii=False)
        return Unit(key=key, tokens=tokens, breakpoint="cache_control" in raw)
    return Unit(key=json.dumps(raw, ensure_ascii=False), tokens=tokens)


def _has_threshold_edit(request: JSONObject) -> bool:
    settings = request.get("context_management")
    edits = settings.get("edits") if isinstance(settings, dict) else None
    return isinstance(edits, list) and any(
        isinstance(edit, dict) and edit.get("type") == "compact_20260112" for edit in edits
    )


def _threshold_edit(request: JSONObject, betas: set[str]) -> JSONObject | None:
    """The request's ``compact_20260112`` edit, checked, with its trigger filled in."""
    settings = request.get("context_management")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise _invalid("context_management: Input should be a valid dictionary")
    edits = settings.get("edits")
    if not isinstance(edits, list):
        raise _invalid("context_management.edits: Field required")
    for position, edit in enumerate(edits):
        if not isinstance(edit, dict) or edit.get("type") != "compact_20260112":
            continue
        path = f"context_management.edits.{position}"
        if THRESHOLD_BETA not in betas:
            raise _invalid(f"{path}: compact_20260112 requires anthropic-beta: {THRESHOLD_BETA}")
        trigger = edit.get("trigger", {"type": "input_tokens", "value": 150_000})
        if not isinstance(trigger, dict) or trigger.get("type") != "input_tokens":
            raise _invalid(f"{path}.trigger.type: Input should be 'input_tokens'")
        value = trigger.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value < THRESHOLD_MINIMUM:
            raise _invalid(
                f"{path}.trigger.value: Input should be greater than or equal to "
                f"{THRESHOLD_MINIMUM}"
            )
        return {**edit, "trigger": {"type": "input_tokens", "value": value}}
    return None


def _clearing_edit(request: JSONObject, betas: set[str]) -> JSONObject | None:
    """The request's ``clear_tool_uses_20250919`` edit, checked, with its defaults filled in."""
    settings = request.get("context_management")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise _invalid("context_management: Input should be a valid dictionary")
    for key in settings:
        if key != "edits":
            raise _invalid(f"context_management.{key}: Extra inputs are not permitted")
    edits = settings.get("edits")
    if not isinstance(edits, list):
        raise _invalid("context_management.edits: Field required")
    found: JSONObject | None = None
    for position, edit in enumerate(edits):
        path = f"context_management.edits.{position}"
        kind = edit.get("type") if isinstance(edit, dict) else None
        if kind not in _EDIT_TYPES:
            raise _invalid(
                f"{path}: Input tag {kind!r} found using 'type' does not match any of the "
                f"expected tags: {', '.join(repr(tag) for tag in _EDIT_TYPES)}"
            )
        if kind == "clear_thinking_20251015":
            raise _invalid(f"{path}: the fake doesn't simulate clear_thinking_20251015")
        if kind != CLEARING_EDIT:
            continue
        assert isinstance(edit, dict)
        if CLEARING_BETA not in betas:
            raise _invalid(f"{path}: {CLEARING_EDIT} requires anthropic-beta: {CLEARING_BETA}")
        if found is not None:
            raise _invalid(f"{path}: only one {CLEARING_EDIT} edit is allowed")
        found = _check_clearing_edit(edit, path)
    return found


def _check_clearing_edit(edit: JSONObject, path: str) -> JSONObject:
    """Check a ``clear_tool_uses_20250919`` edit field by field, and fill in the defaults."""
    for key in edit:
        if key not in _CLEARING_FIELDS:
            raise _invalid(f"{path}.{key}: Extra inputs are not permitted")
    trigger = edit.get("trigger", {"type": "input_tokens", "value": 100_000})
    _check_amount(trigger, f"{path}.trigger", ("input_tokens", "tool_uses"), minimum=1)
    keep = edit.get("keep", {"type": "tool_uses", "value": 3})
    _check_amount(keep, f"{path}.keep", ("tool_uses",), minimum=0)
    least = edit.get("clear_at_least")
    if least is not None:
        _check_amount(least, f"{path}.clear_at_least", ("input_tokens",), minimum=0)
    excluded = edit.get("exclude_tools")
    if excluded is not None and not _is_names(excluded):
        raise _invalid(f"{path}.exclude_tools: Input should be a valid list of strings")
    inputs = edit.get("clear_tool_inputs")
    if inputs is not None and not isinstance(inputs, bool) and not _is_names(inputs):
        raise _invalid(
            f"{path}.clear_tool_inputs: Input should be a valid boolean or a list of strings"
        )
    return {**edit, "trigger": trigger, "keep": keep}


def _check_amount(value: JSONValue, path: str, kinds: Sequence[str], *, minimum: int) -> None:
    """Check a ``{"type": ..., "value": ...}`` object such as a trigger."""
    if not isinstance(value, dict):
        raise _invalid(f"{path}: Input should be a valid dictionary")
    for key in value:
        if key not in ("type", "value"):
            raise _invalid(f"{path}.{key}: Extra inputs are not permitted")
    if value.get("type") not in kinds:
        expected = " or ".join(repr(kind) for kind in kinds)
        raise _invalid(f"{path}.type: Input should be {expected}")
    amount = value.get("value")
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < minimum:
        raise _invalid(f"{path}.value: Input should be an integer of at least {minimum}")


def _is_names(value: JSONValue) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _prefix(request: JSONObject, messages: Sequence[JSONValue]) -> str:
    """What a thinking block is bound to: the system prompt, the tools and the messages before it.

    ``cache_control`` markers and earlier thinking blocks don't count.
    """
    return json.dumps(
        _bindable([request.get("system"), request.get("tools"), list(messages)]),
        sort_keys=True,
        ensure_ascii=False,
    )


def _bindable(value: JSONValue) -> JSONValue:
    if isinstance(value, dict):
        return {key: _bindable(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [
            _bindable(item)
            for item in value
            if not (
                isinstance(item, dict) and item.get("type") in ("thinking", "redacted_thinking")
            )
        ]
    return value


def _canonical(block: JSONObject) -> str:
    return json.dumps(without(block, "cache_control"), sort_keys=True, ensure_ascii=False)
