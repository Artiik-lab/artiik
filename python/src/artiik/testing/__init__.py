"""Test helpers: fake provider clients, a session driver and the Tier 0 invariants.

Nothing here needs a provider SDK or the network. The fake clients take the
same calls as the official SDKs, check each request the way the API does, and
record every call. A test can run a long agent session in milliseconds, then
check the invariants on what was actually sent::

    from artiik import Format
    from artiik.testing import FakeAnthropic, assert_holds, check_all, run_session

    fake = FakeAnthropic()
    run_session(fake.messages.create, Format.ANTHROPIC_MESSAGES, ["Check the logs."] * 50)
    assert_holds(check_all(fake.calls, budget=50_000))
"""

from artiik.testing.driver import (
    Create,
    Prepare,
    default_request,
    response_json,
    run_context,
    run_session,
)
from artiik.testing.environment import Environment
from artiik.testing.errors import FakeAPIError
from artiik.testing.fake_anthropic import COMPACTION_BETA, FakeAnthropic
from artiik.testing.fake_openai import FakeOpenAI
from artiik.testing.invariants import (
    Recall,
    Violation,
    assert_holds,
    check_all,
    check_append_only,
    check_budget,
    check_kept_whole,
    check_pins,
    check_scopes,
    check_tool_pairs,
)
from artiik.testing.model import (
    DigestSummarizer,
    ForgetfulSummarizer,
    Policy,
    Reply,
    ScriptedPolicy,
    Summarizer,
    ToolCall,
    ToolLoopPolicy,
)
from artiik.testing.objects import FakeObject
from artiik.testing.recording import RecordedCall
from artiik.testing.tokens import Tokenizer

__all__ = [
    "COMPACTION_BETA",
    "Create",
    "DigestSummarizer",
    "Environment",
    "FakeAPIError",
    "FakeAnthropic",
    "FakeObject",
    "FakeOpenAI",
    "ForgetfulSummarizer",
    "Policy",
    "Prepare",
    "Recall",
    "RecordedCall",
    "Reply",
    "ScriptedPolicy",
    "Summarizer",
    "Tokenizer",
    "ToolCall",
    "ToolLoopPolicy",
    "Violation",
    "assert_holds",
    "check_all",
    "check_append_only",
    "check_budget",
    "check_kept_whole",
    "check_pins",
    "check_scopes",
    "check_tool_pairs",
    "default_request",
    "response_json",
    "run_context",
    "run_session",
]
