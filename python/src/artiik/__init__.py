"""artiik: context management for AI agents.

Pre-alpha. The 0.1 API (``ContextManager``) was removed in the 0.2 rewrite; it
stays available at the ``v0.1.1`` tag.
"""

from artiik.clearing import FETCH_TOOL, AnthropicClearing, Clearing
from artiik.compaction import (
    AnthropicCompaction,
    AnthropicThresholdCompaction,
    CompactionJob,
    CompactionResult,
    Compactor,
    OpenAICompaction,
    SummaryCompaction,
)
from artiik.context import Context
from artiik.errors import ArtiikError, BudgetError, FormatError, ValidationError
from artiik.messages import (
    Block,
    Compaction,
    Document,
    Format,
    Image,
    JSONObject,
    JSONValue,
    Message,
    Opaque,
    RedactedThinking,
    Role,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)
from artiik.pins import Ledger, Pin
from artiik.store import FileStore, MemoryStore, Store
from artiik.tokens import AnthropicTokenCounter, Estimator, TiktokenCounter, TokenCounter
from artiik.trace import Trace, TraceEvent
from artiik.usage import Usage

__version__ = "0.2.0.dev0"

__all__ = [
    "FETCH_TOOL",
    "AnthropicClearing",
    "AnthropicCompaction",
    "AnthropicThresholdCompaction",
    "AnthropicTokenCounter",
    "ArtiikError",
    "Block",
    "BudgetError",
    "Clearing",
    "Compaction",
    "CompactionJob",
    "CompactionResult",
    "Compactor",
    "Context",
    "Document",
    "Estimator",
    "FileStore",
    "Format",
    "FormatError",
    "Image",
    "JSONObject",
    "JSONValue",
    "Ledger",
    "MemoryStore",
    "Message",
    "Opaque",
    "OpenAICompaction",
    "Pin",
    "RedactedThinking",
    "Role",
    "Store",
    "SummaryCompaction",
    "Text",
    "Thinking",
    "TiktokenCounter",
    "TokenCounter",
    "ToolResult",
    "ToolUse",
    "Trace",
    "TraceEvent",
    "Usage",
    "ValidationError",
    "__version__",
]
