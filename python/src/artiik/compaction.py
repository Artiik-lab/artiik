"""Compaction: replace the older part of a conversation with a summary.

A :class:`~artiik.Context` compacts when a request would pass ``compact_at``
tokens. It keeps the current turn word for word, summarizes everything before
it, and only cuts between whole turns or steps, so a tool call never loses its
result. If compaction fails, the context carries on without it, and the guard
still keeps each request within the budget.

The strategies:

- :class:`AnthropicCompaction`: Anthropic's compaction on demand (beta
  ``compact-2026-09-04``). artiik sends the summarization request, swaps the
  returned block in, and sends it first on every later request.
- :class:`AnthropicThresholdCompaction`: Anthropic's compaction at a token
  threshold (beta ``compact-2026-01-12``). The API compacts inside an ordinary
  request; artiik drops what the block replaced.
- :class:`OpenAICompaction`: OpenAI Responses compaction, either server-side at
  a threshold or through ``/responses/compact``.
- :class:`SummaryCompaction`: a ``summarize`` callable you supply, for Chat
  Completions, OpenAI-compatible servers, or any model without native
  compaction.

References:

- https://docs.anthropic.com/en/api/messages: the ``compaction`` parameter,
  ``context_management`` and the compaction block;
- https://docs.anthropic.com/en/api/models: ``capabilities.compaction``;
- https://developers.openai.com/api/docs/guides/compaction: Responses
  compaction, on the server and through ``/responses/compact``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.formats._json import expect_list, plain, to_object
from artiik.messages import Compaction, Format, JSONObject, JSONValue, Message
from artiik.usage import Usage, read_usage

ANTHROPIC_BETA = "compact-2026-09-04"
"""The beta header of Anthropic's compaction on demand."""

ANTHROPIC_THRESHOLD_BETA = "compact-2026-01-12"
"""The beta header of Anthropic's compaction at a token threshold."""

ANTHROPIC_THRESHOLD_MINIMUM = 50_000
"""The lowest trigger Anthropic accepts for threshold compaction, in input tokens."""

MAX_INSTRUCTIONS = 16_384
"""The longest summarization prompt Anthropic accepts, in characters."""

DEFAULT_INSTRUCTIONS = """\
Write a summary of the conversation above. It replaces those messages, and \
the agent will continue the task from your summary alone, so keep everything \
it still needs:

1. Goals: what the user wants, and the task in progress.
2. Constraints: every rule, limit and preference the user or the system \
stated. Keep their wording.
3. Decisions: what was decided, and why.
4. State of the work: what is done, what is in progress, what remains, and \
the open questions.
5. Artifact trail: every file, path, URL, identifier, command and resource \
that was created, read or changed, and what happened to it.
6. Facts and results from tool outputs that later steps rely on.

Leave out greetings, plans that were dropped, and tool output that no longer \
matters. Write plain text in short sections. Don't call tools: reply with the \
summary only."""
"""artiik's summarization prompt. It keeps what agents lose most often in a summary."""

_NO_TOOLS = "\n\nDon't call any tool. Reply with the summary text only."

_LEFT_OUT = ("context_management", "stop_sequences", "stream", "max_tokens", "betas", "compaction")
"""Request parameters an Anthropic compaction request leaves out or sets itself."""


@dataclass(frozen=True)
class CompactionJob:
    """What a strategy needs to summarize part of a conversation.

    ``messages`` is the part to summarize, oldest first. ``boundaries`` lists
    the positions in it where the part could end instead, without splitting
    a tool call from its result, for a strategy that has to summarize less.
    ``system``, ``tools`` and ``params`` are the conversation's own request
    settings.
    """

    api: Format
    model: str
    messages: tuple[Message, ...]
    boundaries: tuple[int, ...]
    system: JSONValue
    tools: tuple[JSONValue, ...] | None
    params: Mapping[str, Any]


@dataclass(frozen=True)
class CompactionResult:
    """The outcome of one compaction attempt.

    - ``outcome`` is ``compacted``, or why it didn't happen: a stop reason
      such as ``max_tokens`` or ``refusal``, or ``unavailable``,
      ``unsupported``, ``nothing to summarize``.
    - ``messages`` replace the first ``summarized`` messages of the job when
      the attempt worked; ``summarized`` defaults to all of them.
    - ``retry`` says whether trying again later may work.
    """

    outcome: str
    messages: tuple[Message, ...] | None = None
    summarized: int | None = None
    summary: str | None = None
    usage: Usage | None = None
    attempts: int = 1
    retry: bool = True


class Compactor:
    """A compaction strategy.

    An *active* strategy compacts on the client side: the context calls
    :meth:`compact` when a request would pass ``compact_at``. A strategy that
    isn't active asks the provider to compact instead, through the request
    parameters it adds in :meth:`configure`.
    """

    active: bool = True
    apis: frozenset[Format] | None = None
    """The APIs the strategy works with; ``None`` for any."""

    threshold: int | None = None
    """The strategy's own compaction threshold, when it has one."""

    keeps_user_messages: bool = False
    """Whether the strategy keeps user messages word for word instead of summarizing them."""

    def compact(self, job: CompactionJob) -> CompactionResult:
        """Summarize ``job.messages``."""
        raise NotImplementedError

    def configure(self, api: Format, request: dict[str, Any], threshold: int) -> None:
        """Add the request parameters that turn on the provider's compaction."""


class AnthropicCompaction(Compactor):
    """Anthropic's compaction on demand (beta ``compact-2026-09-04``).

    ``client`` is an ``anthropic.Anthropic`` client. The strategy checks once
    per model, through ``client.beta.models.retrieve``, that the model
    supports it (``capabilities.compaction``), and then summarizes with
    ``client.beta.messages.create``, sending the conversation's system prompt
    and tools as the API asks. Models without support use ``fallback``, when
    there is one.

    A summary that doesn't come back is handled per stop reason:
    ``max_tokens`` is retried with twice the ``max_tokens``, ``tool_use`` with
    instructions that forbid tools, and ``model_context_window_exceeded`` on
    an older, shorter part of the conversation. ``refusal`` and ``end_turn``
    give up for now. A ``compaction_unavailable`` error is reported as
    ``unavailable``, to retry later.
    """

    apis = frozenset({Format.ANTHROPIC_MESSAGES})

    def __init__(
        self,
        client: Any,
        *,
        instructions: str | None = None,
        max_tokens: int = 8_192,
        fallback: Compactor | None = None,
        check_support: bool = True,
    ) -> None:
        self.instructions = _instructions(instructions)
        self.max_tokens = max_tokens
        self.fallback = fallback
        self.check_support = check_support
        self._client = client
        self._support: dict[str, bool] = {}

    def supports(self, model: str) -> bool:
        """Whether a model supports compaction on demand, asked once and remembered."""
        if not self.check_support:
            return True
        if model not in self._support:
            self._support[model] = self._ask(model)
        return self._support[model]

    def compact(self, job: CompactionJob) -> CompactionResult:
        """Summarize the job's messages with a compaction request."""
        if job.api is not Format.ANTHROPIC_MESSAGES:
            raise ValueError("AnthropicCompaction works with the anthropic-messages API")
        if not self.supports(job.model):
            if self.fallback is not None:
                return self.fallback.compact(job)
            return CompactionResult("unsupported", retry=False)
        instructions = self.instructions
        max_tokens = self.max_tokens
        summarized = len(job.messages)
        retried: set[str] = set()
        attempts = 0
        while True:
            attempts += 1
            request = self._request(job, job.messages[:summarized], instructions, max_tokens)
            try:
                response = to_object(
                    plain(self._client.beta.messages.create(**request)), "response"
                )
            except Exception as error:
                failure = _api_failure(error, attempts)
                if failure is None:
                    raise
                return failure
            usage = read_usage(Format.ANTHROPIC_MESSAGES, response)
            stop = response.get("stop_reason")
            reply = anthropic_messages.parse_response(response)
            blocks = [block for block in reply.blocks if isinstance(block, Compaction)]
            if stop == "compaction" and blocks:
                message = Message(
                    role="assistant", blocks=(blocks[0],), origin=Format.ANTHROPIC_MESSAGES
                )
                return CompactionResult(
                    "compacted",
                    messages=(message,),
                    summarized=summarized,
                    summary=blocks[0].summary,
                    usage=usage,
                    attempts=attempts,
                )
            if stop == "max_tokens" and stop not in retried:
                max_tokens *= 2
            elif stop == "tool_use" and stop not in retried:
                instructions += _NO_TOOLS
            elif stop == "model_context_window_exceeded" and stop not in retried:
                shorter = _shorter(job.boundaries, summarized)
                if shorter is None:
                    return CompactionResult(str(stop), usage=usage, attempts=attempts)
                summarized = shorter
            else:
                return CompactionResult(
                    str(stop), usage=usage, attempts=attempts, retry=stop != "refusal"
                )
            retried.add(str(stop))

    def _ask(self, model: str) -> bool:
        try:
            info = to_object(
                plain(self._client.beta.models.retrieve(model, betas=[ANTHROPIC_BETA])), "model"
            )
        except Exception as error:
            if _status(error) is None:
                raise
            return False
        capabilities = info.get("capabilities")
        compaction = capabilities.get("compaction") if isinstance(capabilities, dict) else None
        if not isinstance(compaction, dict) or compaction.get("supported") is not True:
            return False
        summarize = compaction.get("summarize")
        return not isinstance(summarize, dict) or summarize.get("supported") is True

    def _request(
        self,
        job: CompactionJob,
        messages: Sequence[Message],
        instructions: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        request = {key: value for key, value in job.params.items() if key not in _LEFT_OUT}
        if _kind(request.get("tool_choice")) in ("any", "tool"):
            del request["tool_choice"]
        output_config: object = request.get("output_config")
        if isinstance(output_config, Mapping):
            config = cast("Mapping[str, object]", output_config)
            if "format" in config:
                rest = {key: value for key, value in config.items() if key != "format"}
                if rest:
                    request["output_config"] = rest
                else:
                    del request["output_config"]
        betas: object = job.params.get("betas")
        request["model"] = job.model
        request["max_tokens"] = max_tokens
        request["betas"] = (
            [beta for beta in cast("Sequence[object]", betas) if isinstance(beta, str)]
            if isinstance(betas, list | tuple)
            else []
        )
        add_beta(request, ANTHROPIC_BETA)
        if job.system is not None:
            request["system"] = job.system
        if job.tools is not None:
            request["tools"] = list(job.tools)
        request["messages"] = anthropic_messages.dump_messages(messages)
        request["compaction"] = {"type": "summarize", "instructions": instructions}
        return request


class AnthropicThresholdCompaction(Compactor):
    """Anthropic's compaction at a token threshold (beta ``compact-2026-01-12``).

    The context adds a ``compact_20260112`` edit to ``context_management`` on
    every request, triggered at ``trigger`` input tokens (default: the
    context's ``compact_at``; at least 50,000). When the API compacts, the
    reply starts with a compaction block, and the context drops everything
    the block replaced. Only ``client.beta.messages.create`` takes
    ``context_management``, so send the requests there. It can't be combined
    with :class:`AnthropicCompaction` in one conversation.
    """

    active = False
    apis = frozenset({Format.ANTHROPIC_MESSAGES})

    def __init__(self, *, trigger: int | None = None, instructions: str | None = None) -> None:
        if trigger is not None and trigger < ANTHROPIC_THRESHOLD_MINIMUM:
            raise ValueError(
                f"Anthropic's threshold compaction needs a trigger of at least "
                f"{ANTHROPIC_THRESHOLD_MINIMUM} tokens, got {trigger}"
            )
        self.threshold = trigger
        self.instructions = None if instructions is None else _instructions(instructions)

    def configure(self, api: Format, request: dict[str, Any], threshold: int) -> None:
        """Add the ``compact_20260112`` edit and its beta header."""
        if api is not Format.ANTHROPIC_MESSAGES:
            raise ValueError("AnthropicThresholdCompaction works with the anthropic-messages API")
        value = self.threshold if self.threshold is not None else threshold
        if value < ANTHROPIC_THRESHOLD_MINIMUM:
            raise ValueError(
                f"Anthropic's threshold compaction needs a trigger of at least "
                f"{ANTHROPIC_THRESHOLD_MINIMUM} tokens; compact_at is {value}"
            )
        edit: JSONObject = {
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": value},
        }
        if self.instructions is not None:
            edit["instructions"] = self.instructions
        settings: object = request.get("context_management")
        edits = (
            cast("Mapping[str, object]", settings).get("edits")
            if isinstance(settings, Mapping)
            else None
        )
        if isinstance(edits, list):
            items = cast("list[object]", edits)
            if not any(_kind(item) == "compact_20260112" for item in items):
                items.append(edit)
        else:
            request["context_management"] = {"edits": [edit]}
        add_beta(request, ANTHROPIC_THRESHOLD_BETA)


class OpenAICompaction(Compactor):
    """OpenAI Responses compaction.

    Without a client, the server compacts: the context adds
    ``context_management=[{"type": "compaction", "compact_threshold": N}]`` to
    every request, with ``threshold`` or the context's ``compact_at`` as N, and
    drops the input before the compaction item that comes back. With
    ``client``, an ``openai.OpenAI`` client, the context compacts through
    ``client.responses.compact`` at ``compact_at`` instead. The endpoint
    returns the user messages word for word, then one compaction item for
    everything else, and the context sends those items as they are.
    """

    apis = frozenset({Format.OPENAI_RESPONSES})
    keeps_user_messages = True

    def __init__(self, client: Any = None, *, threshold: int | None = None) -> None:
        self.active = client is not None
        self.threshold = threshold
        self._client = client

    def compact(self, job: CompactionJob) -> CompactionResult:
        """Compact the job's messages through ``/responses/compact``."""
        if job.api is not Format.OPENAI_RESPONSES:
            raise ValueError("OpenAICompaction works with the openai-responses API")
        if self._client is None:
            raise ValueError("compacting on request needs a client: OpenAICompaction(client)")
        if not any(message.role == "user" and not message.bare_item for message in job.messages):
            # The endpoint needs a user message in the input.
            return CompactionResult("no user message")
        request = {"model": job.model, "input": openai_responses.dump_items(job.messages)}
        try:
            response = to_object(plain(self._client.responses.compact(**request)), "response")
        except Exception as error:
            failure = _api_failure(error, 1)
            if failure is None:
                raise
            return failure
        output = openai_responses.parse_items(
            expect_list(response.get("output"), "response.output"), "output"
        )
        if not output:
            return CompactionResult("empty output", usage=read_usage(job.api, response))
        return CompactionResult(
            "compacted", messages=tuple(output), usage=read_usage(job.api, response)
        )

    def configure(self, api: Format, request: dict[str, Any], threshold: int) -> None:
        """Turn on server-side compaction, unless the strategy compacts on request."""
        if self.active:
            return
        if api is not Format.OPENAI_RESPONSES:
            raise ValueError("OpenAICompaction works with the openai-responses API")
        value = self.threshold if self.threshold is not None else threshold
        setting: JSONObject = {"type": "compaction", "compact_threshold": value}
        settings: object = request.get("context_management")
        if isinstance(settings, list):
            items = cast("list[object]", settings)
            if not any(_kind(item) == "compaction" for item in items):
                items.append(setting)
        else:
            request["context_management"] = [setting]


class SummaryCompaction(Compactor):
    """Compaction through a summarize callable you supply, for any API.

    ``summarize(messages)`` receives the part to summarize in the context's
    provider format, followed by a user message that asks for the summary
    (``instructions``, artiik's by default). It returns the summary text,
    typically by sending those messages to your own model. The summary comes
    back as a user message that starts with ``label``.
    """

    def __init__(
        self,
        summarize: Callable[[list[JSONObject]], str],
        *,
        instructions: str | None = None,
        label: str = "Summary of the conversation so far:",
    ) -> None:
        self.summarize = summarize
        self.instructions = _instructions(instructions)
        self.label = label

    def compact(self, job: CompactionJob) -> CompactionResult:
        """Summarize the job's messages with the callable."""
        request = Message.from_text("user", self.instructions)
        summary = self.summarize(_dump(job.api, [*job.messages, request])).strip()
        if not summary:
            return CompactionResult("empty summary")
        message = Message.from_text("user", f"{self.label}\n\n{summary}")
        return CompactionResult("compacted", messages=(message,), summary=summary)


def add_beta(request: dict[str, Any], beta: str) -> None:
    """Turn on an Anthropic beta in a request.

    The SDKs send ``betas`` as the ``anthropic-beta`` header, and an
    ``anthropic-beta`` entry in ``extra_headers`` replaces it. So the beta
    goes into ``betas`` when the request has that list, for
    ``client.beta.messages.create``, and into the header entry when there is
    one or no list, which ``client.messages.create`` accepts too.
    """
    betas: object = request.get("betas")
    if isinstance(betas, list | tuple):
        listed = [*cast("Sequence[object]", betas)]
        if beta not in listed:
            listed.append(beta)
        request["betas"] = listed
    headers: object = request.get("extra_headers")
    fields = dict(cast("Mapping[str, Any]", headers)) if isinstance(headers, Mapping) else {}
    key = next((name for name in fields if name.lower() == "anthropic-beta"), None)
    if key is None and "betas" in request:
        return
    key = key or "anthropic-beta"
    current = fields.get(key)
    values = [part.strip() for part in current.split(",")] if isinstance(current, str) else []
    if beta not in values:
        values.append(beta)
    fields[key] = ",".join(value for value in values if value)
    request["extra_headers"] = fields


def _instructions(instructions: str | None) -> str:
    if instructions is None:
        return DEFAULT_INSTRUCTIONS
    if not instructions.strip() or len(instructions) > MAX_INSTRUCTIONS:
        raise ValueError(
            f"instructions must be a non-blank string of at most {MAX_INSTRUCTIONS} characters"
        )
    return instructions


def _shorter(boundaries: Sequence[int], summarized: int) -> int | None:
    """The boundary closest to half of the summarized part, if there's one before its end."""
    candidates = [boundary for boundary in boundaries if 0 < boundary < summarized]
    if not candidates:
        return None
    return min(candidates, key=lambda boundary: abs(boundary - summarized / 2))


def _status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _error_code(error: Exception) -> str | None:
    """The ``error.details.error_code`` of an Anthropic error, or the ``code`` of an OpenAI one."""
    code = getattr(error, "code", None)
    if isinstance(code, str):
        return code
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = cast("dict[str, object]", body).get("error")
        if isinstance(inner, dict):
            fields = cast("dict[str, object]", inner)
            details = fields.get("details")
            if isinstance(details, dict):
                value = cast("dict[str, object]", details).get("error_code")
                if isinstance(value, str):
                    return value
            value = fields.get("code")
            if isinstance(value, str):
                return value
    return None


def _api_failure(error: Exception, attempts: int) -> CompactionResult | None:
    """Turn an API error into a result when compaction can carry on without it."""
    status = _status(error)
    if status is None:
        return None
    code = _error_code(error)
    if code == "compaction_nothing_to_summarize":
        return CompactionResult("nothing to summarize", attempts=attempts)
    if code == "compaction_unavailable" or status in (429, 500, 502, 503, 504, 529):
        return CompactionResult("unavailable", attempts=attempts)
    return None


def _kind(item: object) -> object:
    """The ``type`` of a JSON object, or ``None`` for anything else."""
    return cast("Mapping[str, object]", item).get("type") if isinstance(item, Mapping) else None


def _dump(api: Format, messages: Sequence[Message]) -> list[JSONObject]:
    match api:
        case Format.ANTHROPIC_MESSAGES:
            return anthropic_messages.dump_messages(messages)
        case Format.OPENAI_RESPONSES:
            return openai_responses.dump_items(messages)
        case Format.OPENAI_CHAT:
            return openai_chat.dump_messages(messages)
