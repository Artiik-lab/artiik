"""Where artiik keeps what leaves the prompt, such as the tool outputs it clears.

A store holds files by relative path, such as ``offload/3f2a.json``.
:class:`FileStore` keeps them on disk, under ``.artiik`` in the working
directory by default. :class:`MemoryStore` keeps them in memory, for tests
and short-lived agents. Any object with the same ``read`` and ``write``
methods works too.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Protocol

STORE_DIR = ".artiik"
"""The default directory of a :class:`FileStore`."""

_PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")


class Store(Protocol):
    """Files by relative path: ``/``-separated names of letters, digits, ``_``, ``.`` and ``-``."""

    def read(self, path: str) -> bytes | None:
        """The file's bytes, or ``None`` when there's no such file."""
        ...

    def write(self, path: str, data: bytes) -> None:
        """Create or replace a file."""
        ...


class MemoryStore:
    """A store in memory: its files go away with it."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    def read(self, path: str) -> bytes | None:
        """The file's bytes, or ``None`` when there's no such file."""
        return self._files.get(check_path(path))

    def write(self, path: str, data: bytes) -> None:
        """Create or replace a file."""
        self._files[check_path(path)] = bytes(data)


class FileStore:
    """A store on disk, under ``root``: ``.artiik`` in the working directory by default.

    Directories are created when a file is first written in them, and each
    write replaces the file in one step, so a reader never sees half a file.
    """

    def __init__(self, root: str | os.PathLike[str] = STORE_DIR) -> None:
        self.root = Path(root).absolute()

    def read(self, path: str) -> bytes | None:
        """The file's bytes, or ``None`` when there's no such file."""
        try:
            return self._file(path).read_bytes()
        except FileNotFoundError:
            return None

    def write(self, path: str, data: bytes) -> None:
        """Create or replace a file."""
        file = self._file(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        temporary = file.with_name(f".{file.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, file)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _file(self, path: str) -> Path:
        return self.root.joinpath(*check_path(path).split("/"))


def check_path(path: str) -> str:
    """Return ``path`` if it's a valid store path, and raise ``ValueError`` if it isn't.

    A valid path is relative and stays inside the store: no ``..``, no
    absolute paths, no backslashes or drive letters.
    """
    if (
        not isinstance(path, str)  # pyright: ignore[reportUnnecessaryIsInstance]
        or not _PATH.fullmatch(path)
        or any(part in (".", "..") for part in path.split("/"))
    ):
        raise ValueError(
            f"invalid store path {path!r}: use a relative path such as offload/a1b2.json"
        )
    return path
