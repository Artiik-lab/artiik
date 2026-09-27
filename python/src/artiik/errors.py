"""Exceptions raised by artiik."""


class ArtiikError(Exception):
    """Base class for every error artiik raises."""


class FormatError(ArtiikError, ValueError):
    """Provider data doesn't have the expected shape, or can't be sent in the target format.

    The message starts with the path to the offending value, for example
    ``messages[3].content[1].tool_use_id``.
    """
