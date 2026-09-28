"""Response objects with attribute access, like the ones provider SDKs return."""

from __future__ import annotations

import copy
from typing import cast

from artiik.messages import JSONObject, JSONValue


class FakeObject:
    """A read-only JSON object that also allows attribute access.

    ``response.content[0].text`` and ``response["content"]`` both work, and
    :meth:`model_dump` returns a copy of the underlying data, as SDK models do.
    """

    __slots__ = ("_data",)
    _data: JSONObject

    def __init__(self, data: JSONObject) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> object:
        data = cast(JSONObject, object.__getattribute__(self, "_data"))
        if name not in data:
            raise AttributeError(name)
        return wrap(data[name])

    def __getitem__(self, key: str) -> object:
        return wrap(self._data[key])

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("fake responses are read-only")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeObject) and self._data == other._data

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"FakeObject({self._data!r})"

    def model_dump(self, **_: object) -> JSONObject:
        """Return a copy of the data, like ``BaseModel.model_dump()``."""
        return copy.deepcopy(self._data)

    def to_dict(self) -> JSONObject:
        """Return a copy of the data."""
        return copy.deepcopy(self._data)


def wrap(value: JSONValue) -> object:
    """Wrap objects (and objects inside lists) in :class:`FakeObject`."""
    if isinstance(value, dict):
        return FakeObject(value)
    if isinstance(value, list):
        return [wrap(item) for item in value]
    return value
