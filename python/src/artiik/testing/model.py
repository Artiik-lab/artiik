"""What the fake models say: scripted replies, a tool-calling agent, and summarizers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from artiik.messages import Compaction, JSONObject, Message, ToolResult


def _empty_object() -> JSONObject:
    return {}


@dataclass(frozen=True, kw_only=True)
class ToolCall:
    """A tool call the fake model makes."""

    name: str
    arguments: JSONObject = field(default_factory=_empty_object)


@dataclass(frozen=True, kw_only=True)
class Reply:
    """A fake model's answer, independent of the wire format.

    - ``thinking`` becomes a signed ``thinking`` block (Anthropic) or a
      ``reasoning`` item with encrypted content (Responses), which the fake
      checks when it comes back. Chat Completions keeps reasoning hidden.
    - ``raw`` holds provider-format blocks (Anthropic) or output items
      (Responses) returned as given, such as server tool blocks. Chat
      Completions has no place for them and leaves them out.
    - ``stop_reason`` overrides the natural one (the fakes translate it to
      their format's field), and ``output_tokens`` overrides the counted value.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: str | None = None
    raw: tuple[JSONObject, ...] = ()
    stop_reason: str | None = None
    output_tokens: int | None = None


class Policy(Protocol):
    """Decides what the fake model answers to a conversation."""

    def reply(self, conversation: Sequence[Message]) -> Reply:
        """Answer the conversation as the model would."""
        ...


class Summarizer(Protocol):
    """Writes the summary a fake compaction returns."""

    def summarize(self, conversation: Sequence[Message], instructions: str | None) -> str:
        """Summarize the conversation."""
        ...


class ScriptedPolicy:
    """Answers with the given replies, in order, one per call."""

    def __init__(self, replies: Iterable[Reply]) -> None:
        self._replies = list(replies)
        self._next = 0

    def reply(self, conversation: Sequence[Message]) -> Reply:
        if self._next >= len(self._replies):
            raise AssertionError(f"the script has only {len(self._replies)} replies")
        reply = self._replies[self._next]
        self._next += 1
        return reply


class ToolLoopPolicy:
    """A tool-calling agent: ``steps`` rounds of tool calls after each user turn, then an answer.

    Each round calls ``tool_name`` ``calls_per_step`` times in parallel. If the
    history was compacted in the middle of a turn, the model answers right away,
    so a session always ends.
    """

    def __init__(
        self,
        *,
        tool_name: str = "read_log",
        calls_per_step: int = 1,
        steps: int = 2,
        answer: str = "Done.",
    ) -> None:
        self.tool_name = tool_name
        self.calls_per_step = calls_per_step
        self.steps = steps
        self.answer = answer

    def reply(self, conversation: Sequence[Message]) -> Reply:
        calls_made = 0
        for message in reversed(conversation):
            if any(isinstance(block, Compaction) for block in message.blocks):
                return Reply(text=self.answer)
            if is_user_turn(message):
                break
            calls_made += len(message.tool_uses)
        step = calls_made // max(self.calls_per_step, 1)
        if step >= self.steps:
            return Reply(text=self.answer)
        return Reply(
            tool_calls=tuple(
                ToolCall(name=self.tool_name, arguments={"step": step, "call": call})
                for call in range(self.calls_per_step)
            )
        )


class DigestSummarizer:
    """Keeps one line per message: its role and the start of its text.

    An earlier readable summary is carried over, so compacting twice keeps what
    the first compaction kept.
    """

    def __init__(self, *, chars_per_message: int = 80) -> None:
        self.chars_per_message = chars_per_message

    def summarize(self, conversation: Sequence[Message], instructions: str | None) -> str:
        lines = [f"Summary of {len(conversation)} messages."]
        for message in conversation:
            for block in message.blocks:
                if isinstance(block, Compaction) and block.summary:
                    lines.append(block.summary)
            text = " ".join(message.text.split())
            if text:
                lines.append(f"{message.role}: {text[: self.chars_per_message]}")
        return "\n".join(lines)


class ForgetfulSummarizer:
    """Wraps a summarizer and removes the given phrases, like a summary that drops constraints."""

    def __init__(self, forget: Iterable[str], base: Summarizer | None = None) -> None:
        self.forget = tuple(forget)
        self.base: Summarizer = base if base is not None else DigestSummarizer()

    def summarize(self, conversation: Sequence[Message], instructions: str | None) -> str:
        summary = self.base.summarize(conversation, instructions)
        for phrase in self.forget:
            summary = summary.replace(phrase, "")
        return summary


def is_user_turn(message: Message) -> bool:
    """Whether a message is a turn from the user, not a carrier of tool results."""
    return (
        message.role == "user"
        and bool(message.blocks)
        and not all(isinstance(block, ToolResult) for block in message.blocks)
    )
