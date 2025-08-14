"""
ContextManager - A modular, plug-and-play memory and context management layer for AI agents.
"""

from .core.orchestrator import ContextManager
from .core.config import Config
from .memory.short_term import ShortTermMemory
from .memory.long_term import LongTermMemory

__version__ = "0.1.0"
__all__ = ["ContextManager", "Config", "ShortTermMemory", "LongTermMemory"] 