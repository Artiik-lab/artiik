"""Converters between provider wire formats and the artiik conversation model.

Each module parses its format into :class:`~artiik.messages.Message` objects and
dumps them back, losslessly: dumping what was parsed gives back equal JSON.
"""

from artiik.formats import anthropic_messages, openai_chat, openai_responses

__all__ = ["anthropic_messages", "openai_chat", "openai_responses"]
