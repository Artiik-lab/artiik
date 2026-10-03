"""Tool-output clearing: old tool results make way for short stubs, and the originals are kept.

Tool results are often most of an agent's context, and the old ones are
rarely read again. Clearing replaces each with a stub such as::

    [cleared: tool "search", id 3f2a, 4,812 tokens. Call artiik_fetch("3f2a") to read it]

and keeps the original in the store's ``offload/`` directory. The model reads
it back with the ``artiik_fetch`` tool, which the context adds to every
request; :meth:`Context.fetch_result <artiik.Context.fetch_result>` answers
its calls.

Clearing changes the prompt, so it's built around the prompt cache:

- It starts at a ``trigger`` and clears in batches: every result it can at
  once, and only when that saves at least ``clear_at_least`` tokens. The
  prompt prefix changes once per batch, and batches are rare.
- The latest ``keep`` results, the results the model hasn't read yet, small
  results and the results of the tools in ``exclude`` stay as they are.
- A cleared result's stub never changes.

The strategies:

- :class:`Clearing`: artiik clears, for every API.
- :class:`AnthropicClearing`: Anthropic's server-side tool-result clearing
  (``clear_tool_uses_20250919``, beta ``context-management-2025-06-27``). The
  API clears each request on its way to the model; the history artiik sends
  doesn't change.

Reference: https://docs.anthropic.com/en/api/messages (``context_management``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from artiik.compaction import add_beta
from artiik.formats import anthropic_messages, openai_chat, openai_responses
from artiik.formats._json import clone
from artiik.messages import (
    Format,
    JSONObject,
    JSONValue,
    Message,
    RedactedThinking,
    Thinking,
    ToolResult,
    ToolUse,
)
from artiik.store import FileStore, Store

FETCH_TOOL = "artiik_fetch"
"""The name of the tool that reads a cleared tool output back."""

OFFLOAD = "offload"
"""The store directory that keeps the cleared tool outputs."""

ANTHROPIC_BETA = "context-management-2025-06-27"
"""The beta header of Anthropic's context editing."""

ANTHROPIC_EDIT = "clear_tool_uses_20250919"
"""The type of Anthropic's tool-result clearing edit."""

MIN_TOKENS = 100
"""The default ``min_tokens``: a stub costs about a quarter of this."""

ID_LENGTH = 4
"""The length of an offload id; a longer one is used when it's taken."""

_FETCH_DESCRIPTION = (
    "Read a tool output that was cleared from the conversation to save space. "
    "Pass the id from its [cleared: ...] note. The output comes back exactly as it was."
)
_FETCH_SCHEMA: JSONObject = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "The id in the [cleared: ...] note, such as 3f2a."}
    },
    "required": ["id"],
    "additionalProperties": False,
}


class _Settings:
    """What both strategies share: where clearing starts, what it keeps, and the least it clears."""

    def __init__(
        self,
        *,
        trigger: int | None,
        keep: int,
        clear_at_least: int | None,
        exclude: Iterable[str],
    ) -> None:
        if trigger is not None and trigger <= 0:
            raise ValueError(f"trigger must be a positive number of tokens, got {trigger}")
        if keep < 0:
            raise ValueError(f"keep must be zero or more, got {keep}")
        if clear_at_least is not None and clear_at_least < 0:
            raise ValueError(f"clear_at_least must be zero or more, got {clear_at_least}")
        self.trigger = trigger
        self.keep = keep
        self.clear_at_least = clear_at_least
        self.exclude = _names(exclude, "exclude")


class Clearing(_Settings):
    """artiik's tool-output clearing, for every API.

    - ``trigger``: a batch can run once a request would pass this many
      tokens. By default, half the budget, or two thirds of the compaction
      threshold when that's lower, so clearing comes before compaction.
    - ``keep``: the latest results, which always stay. The results the model
      hasn't read yet always stay too.
    - ``clear_at_least``: a batch only runs when it saves at least this many
      tokens, a quarter of the trigger by default. Until then, compaction or
      the guard makes room if the request needs it.
    - ``exclude``: tools whose results are never cleared.
    - ``min_tokens``: results smaller than this stay, as a stub would save little.
    - ``store``: where the originals go, under ``offload/``. By default a
      :class:`~artiik.store.FileStore` in ``.artiik``.
    - ``fetch``: whether the context offers the ``artiik_fetch`` tool. Without
      it, the stubs don't mention it, and the originals are kept for you only.

    Only the results of your own tools are cleared; results of the
    provider's server tools stay. With the Anthropic API, a result that comes
    before a thinking block stays as well: clearing it would change the
    history that block was made with, which models that check thinking
    blocks reject. :class:`AnthropicClearing` clears those on the server.
    """

    def __init__(
        self,
        *,
        trigger: int | None = None,
        keep: int = 3,
        clear_at_least: int | None = None,
        exclude: Iterable[str] = (),
        min_tokens: int = MIN_TOKENS,
        store: Store | None = None,
        fetch: bool = True,
    ) -> None:
        super().__init__(trigger=trigger, keep=keep, clear_at_least=clear_at_least, exclude=exclude)
        if min_tokens < 0:
            raise ValueError(f"min_tokens must be zero or more, got {min_tokens}")
        self.min_tokens = min_tokens
        self.store: Store = store if store is not None else FileStore()
        self.fetch = fetch


class AnthropicClearing(_Settings):
    """Anthropic's server-side tool-result clearing (``clear_tool_uses_20250919``).

    The context adds the edit to ``context_management`` on every request,
    with the beta header ``context-management-2025-06-27``. Only
    ``client.beta.messages.create`` takes ``context_management``, so send the
    requests there. Past the trigger, the API replaces the oldest tool
    results with a placeholder before the model reads them, and the response
    says what it cleared, which the context writes to the trace. The history
    artiik sends doesn't change, so the prompt cache and thinking blocks stay
    valid, but the cleared outputs can't be fetched back.

    - ``trigger``: the API clears once a request passes this many input
      tokens. The default is the same as :class:`Clearing`'s, or the API's
      own (100,000 tokens) when the context has no budget.
    - ``trigger_tool_uses``: clear once the conversation has more than this
      many tool uses, instead of a number of tokens.
    - ``keep``: the latest tool uses, which keep their results.
    - ``clear_at_least``: the API only clears when that saves at least this
      many tokens; by default a quarter of the trigger.
    - ``exclude``: tools whose uses are never cleared.
    - ``clear_inputs``: also clear the tool calls' inputs: ``True`` for every
      tool, or a list of tool names.
    """

    def __init__(
        self,
        *,
        trigger: int | None = None,
        trigger_tool_uses: int | None = None,
        keep: int = 3,
        clear_at_least: int | None = None,
        exclude: Iterable[str] = (),
        clear_inputs: bool | Iterable[str] = False,
    ) -> None:
        super().__init__(trigger=trigger, keep=keep, clear_at_least=clear_at_least, exclude=exclude)
        if trigger_tool_uses is not None and trigger is not None:
            raise ValueError("set trigger or trigger_tool_uses, not both")
        if trigger_tool_uses is not None and trigger_tool_uses <= 0:
            raise ValueError(f"trigger_tool_uses must be positive, got {trigger_tool_uses}")
        self.trigger_tool_uses = trigger_tool_uses
        self.clear_inputs: bool | tuple[str, ...] = (
            clear_inputs if isinstance(clear_inputs, bool) else _names(clear_inputs, "clear_inputs")
        )

    def edit(self, *, trigger: int | None = None, clear_at_least: int | None = None) -> JSONObject:
        """The ``clear_tool_uses_20250919`` edit, with the context's values for what isn't set."""
        edit: JSONObject = {"type": ANTHROPIC_EDIT}
        if self.trigger_tool_uses is not None:
            edit["trigger"] = {"type": "tool_uses", "value": self.trigger_tool_uses}
        elif trigger is not None:
            edit["trigger"] = {"type": "input_tokens", "value": trigger}
        edit["keep"] = {"type": "tool_uses", "value": self.keep}
        if clear_at_least is not None:
            edit["clear_at_least"] = {"type": "input_tokens", "value": clear_at_least}
        if self.exclude:
            edit["exclude_tools"] = list(self.exclude)
        if self.clear_inputs is True:
            edit["clear_tool_inputs"] = True
        elif self.clear_inputs:
            edit["clear_tool_inputs"] = list(self.clear_inputs)
        return edit

    def configure(
        self,
        api: Format,
        request: dict[str, Any],
        *,
        trigger: int | None = None,
        clear_at_least: int | None = None,
    ) -> None:
        """Add the edit to the request's ``context_management``, and its beta header.

        An edit of the same type already in the request is left as it is.
        """
        if api is not Format.ANTHROPIC_MESSAGES:
            raise ValueError("AnthropicClearing works with the anthropic-messages API")
        edit = self.edit(trigger=trigger, clear_at_least=clear_at_least)
        settings: object = request.get("context_management")
        edits = (
            cast("dict[str, object]", settings).get("edits") if isinstance(settings, dict) else None
        )
        if isinstance(edits, list):
            items = cast("list[object]", edits)
            if not any(_kind(item) == ANTHROPIC_EDIT for item in items):
                items.append(edit)
        else:
            request["context_management"] = {"edits": [edit]}
        add_beta(request, ANTHROPIC_BETA)


def default_trigger(budget: int | None, compact_at: int | None) -> int | None:
    """Where clearing starts by default: half the budget, or two thirds of ``compact_at``."""
    found = [budget // 2] if budget is not None else []
    if compact_at is not None:
        found.append(compact_at * 2 // 3)
    return max(min(found), 1) if found else None


@dataclass(frozen=True)
class Candidate:
    """A tool result a batch may clear: its place in the history, and the tool it comes from."""

    message: int
    block: int
    result: ToolResult
    tool: str


def candidates(
    api: Format,
    history: Sequence[Message],
    *,
    keep: int,
    exclude: Collection[str],
    cleared: Collection[str],
) -> tuple[list[Candidate], int]:
    """The results a batch may clear, oldest first, and how many stay for a thinking block.

    A result stays when it's among the latest ``keep``, when the model hasn't
    read it yet (no assistant message comes after it), when it's empty or
    already cleared (its call's id is in ``cleared``), or when it comes from
    a tool in ``exclude`` or from a call that isn't in the history. With the
    Anthropic API, a result before a thinking block stays too, and is counted.
    """
    names = {use.id: use.name for message in history for use in message.tool_uses}
    read_until = max(
        (index for index, message in enumerate(history) if message.role == "assistant"), default=-1
    )
    thinking = (
        max(
            (
                index
                for index, message in enumerate(history)
                if any(isinstance(block, Thinking | RedactedThinking) for block in message.blocks)
            ),
            default=-1,
        )
        if api is Format.ANTHROPIC_MESSAGES
        else -1
    )
    results = [
        (index, position, block)
        for index, message in enumerate(history)
        for position, block in enumerate(message.blocks)
        if isinstance(block, ToolResult)
    ]
    latest: set[tuple[int, int]] = (
        {(index, position) for index, position, _ in results[-keep:]} if keep else set()
    )
    found: list[Candidate] = []
    held = 0
    for index, position, result in results:
        tool = names.get(result.tool_use_id)
        if (
            (index, position) in latest
            or index > read_until
            or not result.content
            or result.tool_use_id in cleared
            or tool is None
            or tool in exclude
        ):
            continue
        if index < thinking:
            held += 1
            continue
        found.append(Candidate(message=index, block=position, result=result, tool=tool))
    return found, held


def stub(tool: str, id: str, tokens: int, *, fetch: bool) -> str:
    """The text that stands in for a cleared tool result."""
    text = f"[cleared: tool {json.dumps(tool, ensure_ascii=False)}, id {id}, {tokens:,} tokens"
    return f'{text}. Call {FETCH_TOOL}("{id}") to read it]' if fetch else f"{text}]"


def original(api: Format, result: ToolResult) -> JSONValue:
    """A tool result's content as the provider receives it: a string, or a list of parts."""
    match api:
        case Format.ANTHROPIC_MESSAGES:
            message = anthropic_messages.dump_message(Message(role="user", blocks=(result,)))
            block = cast("list[JSONObject]", message["content"])[0]
            return block.get("content")
        case Format.OPENAI_RESPONSES:
            item = openai_responses.dump_item(
                Message(role="tool", blocks=(result,), bare_item=True)
            )
            return item["output"]
        case Format.OPENAI_CHAT:
            return openai_chat.dump_message(Message(role="tool", blocks=(result,)))["content"]


def fetch_tool(api: Format) -> JSONObject:
    """The ``artiik_fetch`` tool definition, in the API's format."""
    schema = clone(_FETCH_SCHEMA)
    match api:
        case Format.ANTHROPIC_MESSAGES:
            return {"name": FETCH_TOOL, "description": _FETCH_DESCRIPTION, "input_schema": schema}
        case Format.OPENAI_RESPONSES:
            return {
                "type": "function",
                "name": FETCH_TOOL,
                "description": _FETCH_DESCRIPTION,
                "parameters": schema,
                "strict": True,
            }
        case Format.OPENAI_CHAT:
            function: JSONObject = {
                "name": FETCH_TOOL,
                "description": _FETCH_DESCRIPTION,
                "parameters": schema,
            }
            return {"type": "function", "function": function}


def fetch_id(call: ToolUse) -> str | None:
    """The id an ``artiik_fetch`` call asks for, or ``None`` when its arguments don't name one."""
    arguments = call.input
    value = arguments.get("id") if isinstance(arguments, dict) else None
    return value if isinstance(value, str) and value.strip() else None


def answer(api: Format, call_id: str, content: JSONValue, *, error: bool = False) -> JSONObject:
    """A tool result answering a call, in the API's format: a block, an item or a message."""
    match api:
        case Format.ANTHROPIC_MESSAGES:
            block: JSONObject = {"type": "tool_result", "tool_use_id": call_id, "content": content}
            if error:
                block["is_error"] = True
            return block
        case Format.OPENAI_RESPONSES:
            return {"type": "function_call_output", "call_id": call_id, "output": content}
        case Format.OPENAI_CHAT:
            return {"role": "tool", "tool_call_id": call_id, "content": content}


@dataclass(frozen=True)
class Offloaded:
    """A cleared tool output: its id, where the store keeps it, and what it was."""

    id: str
    path: str
    tool: str
    tool_use_id: str
    tokens: int


class Offload:
    """The tool outputs one conversation cleared, kept in a store.

    Each output gets a short id from a hash of its call's id and its content,
    made longer when it's taken, in this conversation or by another file in
    the store. Only the ids this conversation cleared can be read back.
    """

    def __init__(self, store: Store) -> None:
        self.store = store
        self._entries: dict[str, Offloaded] = {}

    @property
    def entries(self) -> tuple[Offloaded, ...]:
        """The cleared outputs, in the order they were cleared."""
        return tuple(self._entries.values())

    def put(
        self, api: Format, *, tool: str, tool_use_id: str, content: JSONValue, tokens: int
    ) -> Offloaded:
        """Keep a tool output, and return its entry."""
        digest = hashlib.sha256(_canonical([tool_use_id, content]).encode("ascii")).hexdigest()
        for length in range(ID_LENGTH, len(digest) + 1):
            id = digest[:length]
            if id in self._entries:
                continue
            path = f"{OFFLOAD}/{id}.json"
            record: JSONObject = {
                "id": id,
                "tool": tool,
                "tool_use_id": tool_use_id,
                "api": api.value,
                "tokens": tokens,
                "content": content,
            }
            data = (json.dumps(record, ensure_ascii=True, indent=2) + "\n").encode("ascii")
            existing = self.store.read(path)
            if existing is not None and existing != data:
                continue
            if existing is None:
                self.store.write(path, data)
            entry = Offloaded(id=id, path=path, tool=tool, tool_use_id=tool_use_id, tokens=tokens)
            self._entries[id] = entry
            return entry
        raise RuntimeError("every offload id for this output is taken")  # pragma: no cover

    def get(self, id: str) -> JSONValue:
        """The original output of a cleared result; ``KeyError`` for an id this one didn't clear."""
        entry = self._entries.get(id)
        if entry is None:
            raise KeyError(f"no cleared tool output has the id {id!r}")
        data = self.store.read(entry.path)
        record = cast(JSONValue, json.loads(data)) if data is not None else None
        if (
            not isinstance(record, dict)
            or record.get("tool_use_id") != entry.tool_use_id
            or "content" not in record
        ):
            raise KeyError(f"the store no longer has the tool output {id!r} ({entry.path})")
        return record["content"]


def _names(names: Iterable[str], field: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{field} takes a list of tool names, not a string")
    found = tuple(names)
    for name in found:
        if not isinstance(name, str) or not name:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"{field} takes tool names, got {name!r}")
    return found


def _canonical(value: JSONValue) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _kind(item: object) -> object:
    return cast("dict[str, object]", item).get("type") if isinstance(item, dict) else None
