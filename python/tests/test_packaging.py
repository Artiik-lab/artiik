import re
import tomllib
from pathlib import Path

import pytest

import artiik

PYTHON_DIR = Path(__file__).resolve().parents[1]


def test_version_is_exposed() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", artiik.__version__)


def test_core_has_no_required_dependencies() -> None:
    pyproject = tomllib.loads((PYTHON_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["dependencies"] == []


def test_license_copy_matches_the_repository_license() -> None:
    # python/LICENSE is a real copy (sdists can't carry a symlink to ../LICENSE).
    repository_license = PYTHON_DIR.parent / "LICENSE"
    if not repository_license.exists():
        pytest.skip("not running from a repository checkout")
    assert (PYTHON_DIR / "LICENSE").read_bytes() == repository_license.read_bytes()
