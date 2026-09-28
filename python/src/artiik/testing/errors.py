"""Errors raised by the fake clients, shaped like the providers' API errors."""

from __future__ import annotations

from artiik.messages import JSONObject


class FakeAPIError(Exception):
    """An API error from a fake client.

    Like the SDKs' errors, it has ``status_code`` and ``body``, the error payload
    the provider would return. ``code`` is the machine-readable code when there
    is one, such as ``compaction_block_misplaced`` or ``context_length_exceeded``.
    """

    def __init__(
        self, status_code: int, message: str, *, body: JSONObject, code: str | None
    ) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.body = body
        self.code = code

    @classmethod
    def anthropic(
        cls,
        status_code: int,
        error_type: str,
        message: str,
        *,
        code: str | None = None,
    ) -> FakeAPIError:
        """An error shaped like Anthropic's: ``{"type": "error", "error": {...}}``."""
        error: JSONObject = {"type": error_type, "message": message}
        if code is not None:
            error["details"] = {"error_code": code}
        return cls(status_code, message, body={"type": "error", "error": error}, code=code)

    @classmethod
    def openai(
        cls,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> FakeAPIError:
        """An error shaped like OpenAI's: ``{"error": {"message", "type", "param", "code"}}``."""
        body: JSONObject = {
            "error": {"message": message, "type": error_type, "param": param, "code": code}
        }
        return cls(status_code, message, body=body, code=code)
