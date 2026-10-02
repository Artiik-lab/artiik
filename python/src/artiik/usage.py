"""Token usage as providers report it, in one shape for every format.

References: https://docs.anthropic.com/en/api/messages,
https://platform.openai.com/docs/api-reference/responses and
https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

from dataclasses import dataclass, field

from artiik.formats._json import clone
from artiik.messages import Format, JSONObject


def _empty_object() -> JSONObject:
    return {}


@dataclass(frozen=True, kw_only=True)
class Usage:
    """The tokens one call used.

    ``input_tokens`` is the whole prompt, however the provider splits it.
    Anthropic reports uncached input, cache reads and cache writes separately,
    and they're added up here; OpenAI's counts already include cached tokens.
    An Anthropic compaction call reports zeros at the top level and its real
    numbers in ``usage.iterations``, which are used instead. ``raw`` keeps the
    provider's usage object.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    raw: JSONObject = field(default_factory=_empty_object)


def read_usage(api: Format, response: JSONObject) -> Usage:
    """Read the usage of a response in the given format; zeros when it has none."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    match api:
        case Format.ANTHROPIC_MESSAGES:
            return _anthropic(usage)
        case Format.OPENAI_RESPONSES:
            details = _object(usage, "input_tokens_details")
            output_details = _object(usage, "output_tokens_details")
            return Usage(
                input_tokens=_int(usage, "input_tokens"),
                output_tokens=_int(usage, "output_tokens"),
                cache_read_tokens=_int(details, "cached_tokens"),
                reasoning_tokens=_int(output_details, "reasoning_tokens"),
                raw=clone(usage),
            )
        case Format.OPENAI_CHAT:
            details = _object(usage, "prompt_tokens_details")
            output_details = _object(usage, "completion_tokens_details")
            return Usage(
                input_tokens=_int(usage, "prompt_tokens"),
                output_tokens=_int(usage, "completion_tokens"),
                cache_read_tokens=_int(details, "cached_tokens"),
                reasoning_tokens=_int(output_details, "reasoning_tokens"),
                raw=clone(usage),
            )


def _anthropic(usage: JSONObject) -> Usage:
    source = usage
    output = _int(usage, "output_tokens")
    raw_iterations = usage.get("iterations")
    iterations = (
        [item for item in raw_iterations if isinstance(item, dict)]
        if isinstance(raw_iterations, list)
        else []
    )
    if iterations and not any(_prompt_parts(usage)):
        source = iterations[0]
        output = sum(_int(item, "output_tokens") for item in iterations)
    uncached, cache_read, cache_write = _prompt_parts(source)
    return Usage(
        input_tokens=uncached + cache_read + cache_write,
        output_tokens=output,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        raw=clone(usage),
    )


def _prompt_parts(usage: JSONObject) -> tuple[int, int, int]:
    return (
        _int(usage, "input_tokens"),
        _int(usage, "cache_read_input_tokens"),
        _int(usage, "cache_creation_input_tokens"),
    )


def _int(data: JSONObject, key: str) -> int:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _object(data: JSONObject, key: str) -> JSONObject:
    value = data.get(key)
    return value if isinstance(value, dict) else {}
