"""Validated copies of provider data, and accessors that fail with a useful path."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from artiik.errors import FormatError
from artiik.messages import Block, Format, JSONObject, JSONValue


def to_json(value: object, path: str) -> JSONValue:
    """Return a deep copy of ``value``, checking that it only holds JSON types."""
    try:
        return _copy(value)
    except _NotJSON as error:
        location = path + "".join(reversed(error.steps))
        raise FormatError(f"{location}: {error.problem}") from None


class _NotJSON(Exception):
    """A value that isn't JSON, found deep inside :func:`to_json`'s input.

    The path is only built when this happens, from the steps each level adds
    on the way out, so copying valid data doesn't pay for it.
    """

    def __init__(self, problem: str) -> None:
        super().__init__(problem)
        self.problem = problem
        self.steps: list[str] = []


def _copy(value: object) -> JSONValue:
    kind = type(value)
    if kind is str or kind is int or kind is float or kind is bool or value is None:
        return cast(JSONValue, value)
    if kind is dict:
        return _copy_mapping(cast("dict[object, object]", value))
    if kind is list or kind is tuple:
        return _copy_items(cast("list[object] | tuple[object, ...]", value))
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return _copy_mapping(cast("Mapping[object, object]", value))
    if isinstance(value, list | tuple):
        return _copy_items(cast("list[object] | tuple[object, ...]", value))
    raise _NotJSON(
        f"expected JSON data, got {type(value).__name__}. "
        "Pass plain dicts and lists, for example the result of an SDK object's model_dump()."
    )


def _copy_mapping(mapping: Mapping[object, object]) -> JSONObject:
    result: JSONObject = {}
    for key, item in mapping.items():
        if type(key) is not str and not isinstance(key, str):
            raise _NotJSON(f"object keys must be strings, got {type(key).__name__}")
        try:
            result[key] = _copy(item)
        except _NotJSON as error:
            error.steps.append(f".{key}")
            raise
    return result


def _copy_items(items: list[object] | tuple[object, ...]) -> list[JSONValue]:
    copied: list[JSONValue] = []
    for index, item in enumerate(items):
        try:
            copied.append(_copy(item))
        except _NotJSON as error:
            error.steps.append(f"[{index}]")
            raise
    return copied


def to_object(value: object, path: str) -> JSONObject:
    """Return a validated deep copy of a JSON object."""
    return expect_object(to_json(value, path), path)


def plain(value: object) -> object:
    """Turn SDK objects into plain data, including ones nested in dicts and lists.

    The Anthropic and OpenAI SDK models convert with ``to_dict()``, which keeps
    only the fields the API returned. ``model_dump()`` is the fallback; it can
    add ``null`` fields that some APIs reject when a block is sent back. Both
    run in JSON mode, so dates come out as strings, and without the warnings
    pydantic gives for data the SDK types don't describe exactly, such as the
    user messages ``/responses/compact`` returns.
    """
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {key: plain(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        items = cast("list[object] | tuple[object, ...]", value)
        return [plain(item) for item in items]
    for name in ("to_dict", "model_dump"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                data = method(mode="json", warnings=False)
            except TypeError:
                data = method()
            return plain(data)
    return value


def expect_object(value: JSONValue, path: str) -> JSONObject:
    """Check that an already-validated value is an object."""
    if not isinstance(value, dict):
        raise FormatError(f"{path}: expected an object, got {describe(value)}")
    return value


def expect_list(value: JSONValue, path: str) -> list[JSONValue]:
    """Check that an already-validated value is an array."""
    if not isinstance(value, list):
        raise FormatError(f"{path}: expected an array, got {describe(value)}")
    return value


def get_str(data: JSONObject, key: str, path: str) -> str:
    """Return a required string field."""
    if key not in data:
        raise FormatError(f"{path}.{key}: missing")
    value = data[key]
    if not isinstance(value, str):
        raise FormatError(f"{path}.{key}: expected a string, got {describe(value)}")
    return value


def get_object(data: JSONObject, key: str, path: str) -> JSONObject:
    """Return a required object field."""
    if key not in data:
        raise FormatError(f"{path}.{key}: missing")
    return expect_object(data[key], f"{path}.{key}")


def without(data: JSONObject, *keys: str) -> JSONObject:
    """Return a shallow copy of ``data`` without ``keys``."""
    return {key: value for key, value in data.items() if key not in keys}


def clone(data: JSONObject) -> JSONObject:
    """Return a deep copy, so callers can't mutate artiik's state or the other way round."""
    return cast(JSONObject, _copy(data))


def parse_arguments(arguments: str) -> JSONValue:
    """Parse a JSON arguments string; ``None`` when the model produced invalid JSON."""
    try:
        parsed: object = json.loads(arguments)
    except ValueError:
        return None
    try:
        return to_json(parsed, "arguments")
    except FormatError:
        return None


def dump_arguments(raw_arguments: str | None, value: JSONValue) -> str:
    """Return the arguments string to send: the original one while it still matches."""
    if raw_arguments is not None and parse_arguments(raw_arguments) == value:
        return raw_arguments
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def start_block(block: Block, target: Format, default_type: str) -> JSONObject:
    """Start a block's output from its ``extra`` fields when it goes back to its own format."""
    result = clone(block.extra) if block.origin in (None, target) else {}
    result.setdefault("type", default_type)
    return result


def check_same_format(block: Block, target: Format, path: str) -> None:
    """Refuse to convert a block whose meaning is tied to the format it came from."""
    if block.origin is not None and block.origin is not target:
        raise FormatError(
            f"{path}: a {type(block).__name__} block from {block.origin.value} "
            f"can't be sent in {target.value}"
        )


def describe(value: JSONValue) -> str:
    """Name a JSON value's type for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    return "an object"
