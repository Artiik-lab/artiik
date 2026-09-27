"""artiik: context management for AI agents.

Pre-alpha. The 0.1 API (``ContextManager``) was removed in the 0.2 rewrite; it
stays available at the ``v0.1.1`` tag.
"""

from artiik.errors import ArtiikError, FormatError
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

__version__ = "0.2.0.dev0"

__all__ = [
    "ArtiikError",
    "Block",
    "Compaction",
    "Document",
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
    "ToolResult",
    "ToolUse",
    "__version__",
]
