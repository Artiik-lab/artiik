"""
Artiik import alias for ContextManager public API.

Allows:
    from artiik import ContextManager, Config, ShortTermMemory, LongTermMemory

This module re-exports from the canonical `context_manager` package.
"""

from context_manager.core.orchestrator import ContextManager
from context_manager.core.config import Config
from context_manager.memory.short_term import ShortTermMemory
from context_manager.memory.long_term import LongTermMemory

try:
    from context_manager import __version__  # type: ignore
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "ContextManager",
    "Config",
    "ShortTermMemory",
    "LongTermMemory",
    "__version__",
]


