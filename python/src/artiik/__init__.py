"""artiik: context management for AI agents.

Pre-alpha. The 0.1 API (``ContextManager``) was removed in the 0.2 rewrite; it
stays available at the ``v0.1.1`` tag.
"""

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
from artiik.tokens import AnthropicTokenCounter, Estimator, TiktokenCounter, TokenCounter
from artiik.trace import Trace, TraceEvent
from artiik.usage import Usage

__version__ = "0.2.0.dev0"

__all__ = [
    "AnthropicTokenCounter",
    "ArtiikError",
    "Block",
    "BudgetError",
    "Compaction",
    "Context",
    "Document",
    "Estimator",
    "Format",
    "FormatError",
    "Image",
    "JSONObject",
    "JSONValue",
    "Message",
    "Opaque",
    "RedactedThinking",
    "Role",
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
