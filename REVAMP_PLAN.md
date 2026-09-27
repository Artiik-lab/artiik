# artiik revamp plan

*Drafted 2026-09-27. Status: **approved 2026-09-27**. The decisions are in §11, and the work is tracked as GitHub issues under the [roadmap, #2](https://github.com/Artiik-lab/artiik/issues/2) (§12).*

## 0. Summary

**The mission stays the same.** artiik is a plug-and-play context management layer for people building agents. What changes is the delivery and the internals: the 2025 code solved a 2024 problem (small windows, string prompts, chat pairs), and its core path doesn't run.

**The target experience:**

```text
/plugin marketplace add artiik-lab/artiik
/plugin install artiik@artiik

> add context management to my agent
```

Or, without the plugin, add just the skill: `npx skills add artiik-lab/artiik`, or copy its folder into `.claude/skills/`. That works in any coding agent that supports Agent Skills.

Claude Code then:

1. detects the agent's stack;
2. installs artiik;
3. wires it in with a small diff that uses the provider's native features;
4. adds a regression test;
5. prints a before/after report.

The builder doesn't read docs or wire anything by hand.

**What we ship, all from this one repo:**

| # | Deliverable | What it is |
|---|---|---|
| 1 | **artiik runtime** | A small library with zero required dependencies (Python first, TypeScript in M6). It manages the context of the agent at run time. |
| 2 | **The artiik skill, packaged as a Claude Code plugin** | The main entry point. The skill teaches the coding agent to integrate, audit and tune artiik on any supported stack. **It stands alone:** add just the skill to Claude Code, Codex or any agent that supports Agent Skills, and everything works. The plugin adds one-command install and updates in Claude Code, plus the eval suite. |
| 3 | **artiik-bench** | Two benchmarks. **Integration:** does the plugin wire artiik in correctly? **Context quality:** does the wired agent remember what it must, and at what cost? |

**How:** a rewrite, not a patch. The existing `v0.1.0` and `v0.1.1` tags keep the old code reachable, and we replace the package. Roughly 11 weeks to 1.0 for one developer working with Claude Code, aiming at mid-December 2026 (§7).

---

## 1. What this plan takes from the analysis, and what it sets aside

**Taken as input** (the verified facts in §2–§3 of the strategic analysis dated 2026-09-27, which was shared separately and isn't in the repo):

- **The v0.1 code is broken.** Its defects include summarize-and-offload that can never run, `-1` FAISS hits, corrupted persistence after load→delete, and scope filtering that runs after top-k. Its README examples don't run either.
- **Its assumptions are outdated.** It assumes string prompts, `(user, assistant)` text pairs, GPT-4 tiktoken for every model, and 2024 model defaults. It also pulls in a ~60-package torch/CUDA install.
- **Providers and frameworks now ship the mechanics:**
  - compaction (Anthropic on demand and at a threshold; OpenAI Responses);
  - context editing (tool-result and thinking clearing);
  - Anthropic's memory tool;
  - prompt caching;
  - compaction hooks in the Claude Agent SDK and Claude Code.
- **What still breaks:** compaction silently drops constraints, decisions and the artifact trail. Fewer tokens don't mean lower cost once caching is counted.
- **How builders adopt tools now:** they work through coding agents. Skills raise integration success sharply, while agents rarely add a new dependency unprompted.

**Set aside, as you asked:** the "archive", "Pin & Prove", "Context CI", "memory firewall" and "flight recorder" proposals. Some of their mechanisms show up below as *features* of the layer: must-keep items in §2.2 and traces. None of them is the product.

---

## 2. Product definition

### 2.1 Who it's for, and the promise

- **Who:** developers building agents in Python or TypeScript who work in Claude Code (primary), or in any coding agent that supports Agent Skills (Codex, Cursor, Copilot, Gemini CLI).
- **The promise:** *"Install the plugin, ask for context management, and get a correct, tested, cache-friendly integration on your stack in minutes. It uses your provider's native features and adds only what they're missing."*

### 2.2 What "context management" means in artiik 2 (five jobs)

| Job | What artiik does | Native feature it builds on |
|---|---|---|
| **1. Budget** | Keeps every request under a token budget and the model's window. Never splits a `tool_use` from its `tool_result`, and never sends an invalid history. | Token counts from usage fields; Anthropic's `count_tokens` |
| **2. Compact** | When history grows, it first clears old tool outputs, the cheap step. It keeps the full output on disk behind a short stub and a fetch tool. Then it compacts. It changes history rarely and in batches, so the prompt cache survives. | Anthropic on-demand compaction (`compaction: {type: "summarize", instructions}`), threshold compaction and context editing (`clear_tool_uses_20250919`); OpenAI Responses `context_management`/`compact_threshold`. Where the provider has none, a summary made through the builder's own model call. |
| **3. Keep** | Keeps what must survive any compaction. **Pins** are rules, constraints, decisions and goals. The **artifact ledger** is derived automatically from tool calls (files, IDs and resources touched). Both live outside the compactable history, are re-stated right after every compaction, and are checked against the summary text when the provider returns readable text. | The "restate it after compaction in a `system` message" pattern that Anthropic's compaction docs recommend |
| **4. Remember** | Cross-session memory stored as plain files, scoped by user, agent and session. Scopes are enforced *before* retrieval on every read path. Recall is lexical (BM25, no dependencies), with optional embeddings. Memory is offered two ways: automatic recall injected into the current turn, and a memory tool the agent calls. | The Anthropic memory tool (`memory_20250818`), for which artiik becomes the storage backend; a function tool for other providers |
| **5. Show** | Writes a trace per request: tokens by segment, clears and compactions, pins restored, cache reads and writes, and estimated cost. Stored as JSONL under `.artiik/traces/`, with optional OpenTelemetry. `artiik inspect` reads it. | Usage fields, including `cache_read_input_tokens`, `cache_creation_input_tokens` and `usage.iterations` |

### 2.3 Non-goals

- **Not an LLM client.** The builder keeps their own client. The v0.1 `LLMAdapter` that called models itself goes away.
- **Not an agent framework, and not a RAG or ingestion framework.** `ingest_text`, `ingest_file` and `ingest_directory` are dropped.
- **Not a hosted memory service.** There's no vector database by default, and no torch or FAISS in the default install.
- **The MCP memory server (issue #1) is parked** until after 1.0.

### 2.4 Design principles

1. **Native first.** If the provider or framework has a feature, artiik configures it and only adds what's missing. The benchmark includes a "native only" arm to keep us honest.
2. **Zero required dependencies.** The core uses only the standard library. Embeddings, OpenTelemetry and tiktoken are optional extras.
3. **Structured messages, not strings.** A provider-neutral message model round-trips each provider's format without loss. Opaque blocks (a signed Anthropic compaction block, an OpenAI compaction item, encrypted reasoning) pass through untouched.
4. **Cache-safe by default:**
   - the prefix (system prompt and tools) stays stable;
   - history is append-only between compactions;
   - pins added mid-run are appended, never inserted;
   - recall rides on the newest user turn, which is uncached anyway.
5. **A minimal diff in the builder's code.** Integration takes one wrapper call, two calls in the loop, or one framework hook registration.
6. **Deterministic and testable.** Every strategy runs against a fake model, and the skill leaves a regression test in the builder's repo.
7. **Honest docs.** Every claim in the README links to a test or a benchmark result.

---

## 3. Target user experience

```text
> my support agent forgets the refund policy after long chats, can you fix its context handling?

● Skill(artiik)
● Detected: Python 3.12 · anthropic SDK · custom tool loop in agent/loop.py:41 · no compaction · no memory
  Plan: profile "long-running-agent"
        • Anthropic on-demand compaction, plus tool-result clearing at 60k tokens
        • pin: "Refunds above $500 need human approval" (found in prompts/system.md)
        • file memory scoped by customer_id
  Proceed? (y)
● uv add artiik
● Edit agent/loop.py (+9 −2)   Write artiik.toml   Write tests/test_artiik_context.py
● Bash(pytest -q)  → 23 passed
● Bash(artiik bench quick)
    scenario             before   after
    policy kept (%)        41      100
    tokens / turn        38.2k    14.9k
    cache hit rate        0.22     0.81
  Done. The changes and how to tune them are in artiik.toml.
```

*(The numbers above are illustrative. Real ones come from the benchmark in §6.)*

What the diff typically looks like, for the lightest integration level:

```python
import anthropic, artiik

ctx = artiik.Context.from_config("artiik.toml", scope={"user": customer_id})
client = artiik.wrap(anthropic.Anthropic(), ctx)   # the rest of the loop is unchanged
```

---

## 4. Runtime architecture

### 4.1 Three integration levels

| Level | Who uses it | Code |
|---|---|---|
| **L1 · wrap** | Raw SDK loops, and the default for simple agents | `client = artiik.wrap(anthropic.Anthropic(), ctx)` intercepts `messages.create` / `responses.create` / `chat.completions.create`, sync and async, streaming included. |
| **L2 · prepare/record** | Custom loops that want explicit control | `req = ctx.prepare(messages, system=..., tools=...)` → `resp = client.messages.create(**req, ...)` → `ctx.record(resp)` |
| **L3 · framework adapter** | Framework users | `artiik.integrations.<framework>.attach(agent, ctx)` registers the framework's own hooks or middleware. |

L2 is the v0.1 two-call contract (`build_context` → `prepare`, `observe` → `record`), rebuilt on structured messages.

**How L1 reconciles history.** The wrapper sees the caller's full `messages` list on each call. It hashes messages to find the new suffix, appends that suffix to its own managed history, and sends the managed (compacted) version. If the caller rewrites earlier messages, the wrapper logs a warning and rebuilds its managed history. L2 is the documented default until the L1 tests prove this path.

### 4.2 Stack support matrix

Hook names marked ✓ are verified in current docs. The others must be re-checked when the adapter is built.

| Stack | Hook point | Native features artiik configures | Milestone |
|---|---|---|---|
| Anthropic SDK: raw loop / tool runner | L1 or L2; the tool runner's `compact_before_next_turn()` ✓ | On-demand compaction (beta `compact-2026-09-04`, support checked via Models API `capabilities.compaction`) ✓, context editing ✓, memory tool ✓, `cache_control` ✓ | M1 |
| OpenAI Responses: raw loop | L1 or L2 | `context_management` + `compact_threshold`, or `/responses/compact` ✓. The compaction item is opaque, so pins and the ledger are kept outside it. | M1 |
| Chat Completions and OpenAI-compatible APIs (Ollama, vLLM, LiteLLM, local models) | L2 | None. artiik compacts through a summarize callable the builder supplies. This is the "small-window" profile, where v0.1's original use case lives on. | M1 |
| Claude Agent SDK (Python) | `UserPromptSubmit` (recall and pins), `PreCompact` (flush memory, snapshot pins) ✓; re-state pins on the next `UserPromptSubmit` | Harness auto-compaction | M3 |
| Claude Agent SDK (TypeScript) | The above plus `PostCompact` and `SessionStart` (TypeScript only) ✓ | Same | M6 |
| OpenAI Agents SDK | A custom `Session` plus a model-input filter (verify) | `OpenAIResponsesCompactionSession` | M3 |
| LangChain v1 / LangGraph | `AgentMiddleware` `before_model` / `after_model` (verify) | Configures `SummarizationMiddleware` rather than replacing it | M3 |
| Pydantic AI | `history_processors` / capabilities (verify) | Its compaction capabilities | M3 |
| Vercel AI SDK | `prepareStep` / middleware (verify) | Anthropic and OpenAI pass-through | M6 |
| AWS Strands | Conversation manager (verify) | Native message pinning (`metadata.custom.pinned`) | post-1.0 |

### 4.3 Request layout (cache-safe)

```text
[system + tools] ← cache breakpoint     stable for the whole run; startup pins live at the end of system
[compaction block or summary]           replaced only at a compaction event
[pins re-statement: system message]     inserted once, right after the first user turn that follows a compaction
[history …]                             append-only between compactions; old tool results become stubs in batches
[newest user turn + recall block]       uncached tail; recall and "pin added" notices ride here
```

### 4.4 Pipeline inside `prepare()`

The steps run in order, cheapest first. Each strategy has a native implementation per provider and a generic fallback.

1. **Validate.** Check tool pairs, role order and opaque-block placement.
2. **Clear** (above the soft threshold). Swap old tool results for stubs, keep the originals in `.artiik/offload/`, and expose an `artiik_fetch(id)` tool.
3. **Compact** (above the compaction threshold). Use native compaction where it exists, with artiik's default `instructions`: keep goals, constraints, decisions, open tasks and the artifact trail, and don't call tools. Then re-state the pins and the ledger, and write a receipt to the trace (for example, *"summary kept 5/7 pins; re-stated all 7"*). The Anthropic compaction request is sent separately and without `context_management`, because the API rejects the two together.
4. **Guard.** A hard budget check. If still over, drop the oldest whole turns while keeping tool pairs intact. This is a last resort, and it's logged.

### 4.5 Token accounting

- Ground truth comes from the last response's `usage`, including cache fields and `usage.iterations` for compaction calls.
- New messages are estimated with a heuristic calibrated per model family, which learns from observed usage. v0.1 assumed GPT-4 tiktoken for every model, which is wrong for newer Claude tokenizers.
- Exact counts are opt-in: `count_tokens` for Anthropic, tiktoken (as an extra) for OpenAI.

### 4.6 Storage

- A `Store` protocol with filesystem (the default) and in-memory implementations. Builders can implement it for Redis, S3 or a database.
- Layout on disk:

  ```text
  .artiik/
    memory/<scope>/*.md
    offload/
    traces/*.jsonl
  ```

- Plain, diffable files that fit the 2026 "memory as files" consensus.

### 4.7 Profiles and configuration

- **Profiles** set defaults: `assistant`, `long-running-agent`, `research`, `small-window`.
- **Configuration** is either in code or in `artiik.toml`, which the skill generates and comments.
- **Supported Python:** 3.11 and later. Python 3.10 reaches end of life in October 2026, and 3.11's built-in `tomllib` reads `artiik.toml` without adding a dependency.

---

## 5. The artiik skill and its Claude Code plugin

**Skill first.** The skill is the product, and the plugin is packaging. Adding the skill folder on its own, without the plugin, must be enough for everything to work. So:

- everything the skill uses lives inside its own folder: workflows, per-stack references, scripts and templates;
- the skill never depends on plugin-only parts such as hooks, subagents or MCP servers;
- integrate, audit and bench are workflows inside **one** skill, so there's a single thing to add.

### 5.1 Repo layout

Checked against Claude Code's `plugin validate`.

```text
.claude-plugin/marketplace.json        # the repo is its own marketplace
plugins/artiik/
  .claude-plugin/plugin.json
  skills/
    artiik/                            # the one skill: self-contained, works without the plugin
      SKILL.md                         # short router: which workflow, which stack reference
      workflows/
        integrate.md  audit.md  bench.md
      references/                      # loaded on demand, one file per stack
        anthropic.md  openai.md  compat-local.md  claude-agent-sdk.md
        openai-agents.md  langgraph.md  pydantic-ai.md  vercel-ai.md
        config.md  testing.md  troubleshooting.md
      scripts/
        detect_stack.py                # stdlib only → JSON (language, SDKs, framework, loop call sites)
        verify_integration.py          # drives the agent with a fake model; checks the invariants
      templates/                       # artiik.toml, test file
  evals/                               # `claude plugin eval` cases (§6.1)
python/                                # artiik runtime (PyPI: artiik)
typescript/                            # @artiik/core (M6)
spec/                                  # language-neutral conformance fixtures shared by py and ts
bench/
  integration/  context/  results/
docs/                                  # short: quickstart, concepts, one page per stack, benchmark
```

Notes from checking this against Claude Code:

- Skills must live under `skills/`, not at the plugin root. A root skill path (`"./"`) collides with `evals/`, and `plugin eval init` refuses to run.
- The plugin has **no hooks and no MCP server** in v1. That means no always-on cost, nothing to trust beyond small readable scripts, and no network calls.
- `claude plugin details` reports the always-on token cost. Target: under 150 tokens, since there's one skill description.

### 5.2 What the main skill does

1. **Detect.** Run `detect_stack.py` and show the findings, with file:line references.
2. **Ask, only if needed.** At most three questions: the profile, the memory scope key (such as `user_id`), and any must-keep rules. Otherwise use defaults. It also offers pins it found in the existing system prompt.
3. **Install** with the project's own tool (uv, poetry, pip, npm or pnpm). If the project forbids new dependencies, **vendor mode** copies the pure-Python core into the repo.
4. **Wire** it using the stack's reference recipe and the lightest level that works (L1, then L2, then L3).
5. **Write** a commented `artiik.toml` and `tests/test_artiik_context.py`. The test drives a long synthetic run through a fake client and asserts that:
   - the budget is respected;
   - tool pairs stay intact;
   - pins survive a forced compaction;
   - memory scoping holds.
6. **Run** the project's own tests and `verify_integration.py`.
7. **Report** the files changed, what's native versus what artiik adds, how to tune it, and the output of `artiik bench quick`.

The skill checks the installed artiik version and reads that version's docs instead of relying on training data. Coding agents writing context code from stale knowledge is the failure this avoids.

The **audit** workflow is the same detection plus a read-only review. It reports findings such as unbounded history, tool results that can be orphaned, cache-busting content in the system prompt and memory without scoping. It changes nothing. The **bench** workflow runs `artiik bench` on the builder's agent, quick or full.

### 5.3 Distribution

There are three ways in, and each one must work on its own:

1. **Claude Code plugin:** `/plugin marketplace add artiik-lab/artiik`, then `/plugin install artiik@artiik`. Releases are tagged with `claude plugin tag`. Once the benchmarks are published, the plugin is submitted to Anthropic's plugin directory.
2. **The skill alone:** `npx skills add artiik-lab/artiik`, or copy the `artiik` skill folder into `.claude/skills/` (project) or `~/.claude/skills/` (user). `SKILL.md` follows the Agent Skills standard. M2 confirms that `npx skills add` finds `plugins/artiik/skills/*`; if it doesn't, a root `skills/` entry points there.
3. **Codex and other agents:** the same skill folder. Codex is tested and packaged as a Codex plugin at the end, in M6.
- **CI:** runs `claude plugin validate --strict` and the eval suite in quick mode.

---

## 6. Benchmarks

Three tiers: the two you asked for, plus a fast one that guards every pull request.

### 6.0 Tier 0: invariant tests (every PR, under 1 minute, no API cost)

A fake model runs long scripted sessions (500+ turns, large tool outputs) and asserts that:

- the budget is never exceeded;
- no tool pair is ever orphaned;
- pins are present after every compaction;
- no bytes before the last cache breakpoint change between compactions;
- memory scopes never leak;
- opaque blocks round-trip byte for byte.

Each v0.1 defect gets a named regression test.

### 6.1 Track A: integration benchmark ("does the plugin integrate artiik correctly?")

- **Fixtures** (`bench/integration/fixtures/`): small, realistic agents, each with its own tests plus a **hidden verifier** that the coding agent never sees.
  - **M3, Python:** Anthropic raw loop · OpenAI Responses loop · OpenAI-compatible local model · Claude Agent SDK · OpenAI Agents SDK · LangGraph · Pydantic AI · an agent that already has naive homegrown memory, where the goal is to replace it without breaking it.
  - **M6, TypeScript:** Vercel AI SDK · Claude Agent SDK · Anthropic raw loop.
- **Prompts:** three phrasings per fixture:
  - vague: *"my agent forgets things in long chats"*;
  - specific: *"add compaction and cross-session memory"*;
  - explicit: `/artiik:artiik`.
- **Arms:**
  1. Claude Code *without* the plugin, same prompt. This is the baseline.
  2. Claude Code *with* the plugin.
  3. Later: Codex with the skill.

  Each arm runs with two coding-agent model tiers.
- **Metrics:**
  - **Primary:** the hidden-verifier pass rate.
  - **Secondary:**
    - the fixture's own tests still pass;
    - the skill triggered;
    - native features were used correctly;
    - diff size;
    - turns, cost and wall time;
    - an LLM-judged code-quality score.
- **Harness:**
  - `claude plugin eval` provides the with/without-plugin ablation and these graders: `tool_used: Skill` (did the skill fire), `regex` over changed files, `file_exists` for the generated test, and `llm` criteria.
  - It has no "run the tests" grader, so `bench/integration/run.py` adds hard verification:
    1. copy the fixture to a temporary directory;
    2. run `claude -p` with `--plugin-dir`;
    3. run the hidden verifier;
    4. write JSON.
  - N = 5 runs per cell.
- **1.0 targets:**
  - ≥ 90% hidden-verifier pass with the plugin on the Python fixtures;
  - the skill triggers ≥ 95% of the time on specific prompts and ≥ 80% on vague ones;
  - zero broken fixture tests;
  - the baseline number is published next to ours.

### 6.2 Track B: context-quality benchmark ("does the integrated agent manage context better, and at what cost?")

**Suites.** All are seeded and generated, with deterministic tools in a simulated environment, and are scored programmatically wherever possible.

| Suite | What it tests | Primary metric |
|---|---|---|
| B1 Constraint retention | Policies given early or mid-run; later tool calls that tempt a violation; at least 2 compactions | Violation rate; probe accuracy |
| B2 Cross-session memory | Facts spread over sessions, then asked about later (plus a LongMemEval-style subset if the license allows) | QA accuracy |
| B3 Artifact trail | Tool-heavy tasks in a sandbox filesystem | F1 on "which files did you change?"; rate of redone work |
| B4 Heavy tool output | Large logs and web pages | Task success; input tokens |
| B5 Plan continuity | A multi-step plan with steps already done when compaction hits | Steps revived or redone |
| B6 Small window | A local model with an 8–32k window | Task success; overflow errors |

**Arms:**

1. Naive full history (fails when the window is reached).
2. A naive sliding window.
3. **Provider-native only** (Anthropic compaction plus context editing; OpenAI compaction).
4. Framework-native (for example, LangChain `SummarizationMiddleware`).
5. **artiik default.**
6. artiik ablations: without pins, without memory, without clearing, and generic compaction instead of native.

**Agent models:** three Claude tiers (small, mid and frontier), one small OpenAI model and one local model. Exact model IDs are set in the benchmark config, not hard-coded. §6.3 shows which tier runs which part of the matrix.

**Metrics:** task success, retention and violations, recall accuracy, input tokens, cache hit rate, number of compactions and latency. Billed cost is computed from usage fields times a dated `pricing.toml`, cache reads and writes included. **The headline is cost per successful task, alongside retention.**

**Rigor:**

- ≥ 5 seeds per cell and bootstrap 95% confidence intervals.
- A record/replay cache of model responses, so anyone can rerun the numbers at no cost.
- Raw JSONL is published.
- An LLM judge is used only where a programmatic check can't be.
- Quick mode (1 seed, 2 suites) and `--max-cost-usd` keep costs under control.

**Honesty rule:** if "provider-native only" matches artiik on a suite, the report says so and artiik's defaults change to native-only for that case. The value then comes from correct wiring, pins, memory and visibility, not from reimplementing what the provider does.

**Output:** `bench/results/<date>/report.html` plus a summary table in the README.

### 6.3 Budget: €100 of API spend

**€100 (about $110) is enough for 1.0**, with the design below. The full matrix in §6.2 (every arm × every model × 5 seeds, all on API keys) would cost roughly €450, so we don't run it that way.

List prices on 2026-09-27, in USD per million tokens:

| Model tier | Input | Cache write (5 min) | Cache read | Output |
|---|---|---|---|---|
| Claude small (Haiku class) | $1 | $1.25 | $0.10 | $5 |
| Claude mid (Sonnet class) | $2 | $2.50 | $0.20 | $10 |
| Claude frontier (Opus class) | $4 | $5 | $0.20 | $20 |
| OpenAI small | $0.10 | n/a | $0.01 | $0.50 |

**What keeps it inside €100:**

1. **Track A runs on your Claude subscription, not the API.** `claude -p` and `claude plugin eval` run as the logged-in user, so the coding-agent sessions count against your plan's usage limits. On an API key, the ~240 sessions would cost about $170 (≈ $0.70 each at mid-tier prices).
2. **Scaled-down windows.** Track B compacts at about 32k tokens instead of 150k or more. What survives a compaction doesn't depend on the window's size, and small windows are 5–10× cheaper. One full-size spot check confirms it.
3. **A tiered matrix.** The cheap models run the whole matrix. The mid tier runs the suites where native compaction matters, and the frontier tier gets one spot check.
4. **Cache-hostile arms run small.** A naive sliding window changes the prompt prefix on every call, so it can't use the cache and costs 4–5× more per episode. It runs with one seed per suite, which is enough to show that cost.
5. **Replay and a hard cap.** Re-runs come from the record/replay cache at no cost, and `--max-cost-usd` stops a run before it overspends. The first three episodes of each cell measure the real cost, and the matrix is rescaled if the estimate is off.
6. **Optional:** the Batch API (50% off) for the final published run, by stepping all episodes in lockstep.

On-demand compaction isn't available on the current small tier. There, the "provider-native" arm uses context editing only, and native compaction is measured on the mid and frontier tiers.

The estimates assume an episode of about 60 model calls, about 20k tokens of context on average, at least 90% cache reads and 2–3 compactions.

| Block | Where it runs | Est. cost |
|---|---|---|
| Track A: 8 fixtures × 3 prompts × 2 arms × 5 runs | Claude subscription | $0 (plan limits) |
| Track A: LLM graders | Claude small | ~$3 |
| Track B: development and debugging | fake model, replay, OpenAI small | ~$5 |
| Track B: B1, B3, B5 × 4 arms (naive, native only, artiik, artiik without pins) × 5 seeds = 60 episodes at ~$0.80 | Claude mid | ~$48 |
| Track B: 6 suites × 3 arms × 3 seeds = 54 episodes at ~$0.30, plus the sliding window at 1 seed per suite | Claude small | ~$26 |
| Track B: 6 suites × 5 arms (framework-native included) × 5 seeds = 150 episodes | OpenAI small | ~$8 |
| Track B: frontier spot check, 1 suite × 2 arms × 3 seeds | Claude frontier | ~$7 |
| B6 small window | local model on your machine | $0 |
| Reserve for re-runs and surprises | | ~$13 |
| **Total** | | **≈ $110 ≈ €100** |

---

## 7. Roadmap

| Milestone | Scope | Est. | Dates |
|---|---|---|---|
| **M0 · Reset** | • Keep the `v0.1.0` / `v0.1.1` tags as the legacy reference.<br>• Delete `context_manager/`, `artiik/`, `Demos/`, `demo.py`, `serve_docs.py`, the old tests and the docsify site.<br>• `python/` with hatchling + uv, Python 3.11+.<br>• CI: ruff, pytest, pyright, `plugin validate`.<br>• New README, fixed LICENSE holder, CONTRIBUTING and CLAUDE.md.<br>• Land this plan on `main`; mark issue #1 as parked. | 2–3 days | Sep 28 – Sep 30 |
| **M1 · Core runtime (Python)** | • Message model and converters (Anthropic, OpenAI Responses, Chat Completions).<br>• Budget and guard.<br>• Clearing and offload.<br>• Compaction: native Anthropic, native OpenAI, and generic.<br>• Pins and artifact ledger.<br>• File memory, BM25, and the memory-tool backend.<br>• Traces and `artiik inspect`.<br>• Profiles and `artiik.toml`.<br>• `wrap()` for the Anthropic and OpenAI clients (sync and async).<br>• `artiik.testing` fake clients.<br>• Tier 0 invariant tests. | ~2 weeks | Oct 1 – Oct 14 |
| **M2 · Skill + plugin v1** | • The self-contained `artiik` skill with the Anthropic, OpenAI and local recipes, and its integrate, audit and bench workflows.<br>• `detect_stack`, `verify_integration` and vendor mode.<br>• Marketplace and plugin.<br>• All three install paths verified.<br>• First 3 fixtures and eval cases.<br>• Dogfood on 2 real agents. | ~1 week (overlaps the end of M1) | Oct 12 – Oct 21 |
| **M3 · Framework adapters** | • Claude Agent SDK (Python), OpenAI Agents SDK, LangChain v1/LangGraph and Pydantic AI.<br>• Each one ships an adapter, a skill reference, a fixture and an eval case. | ~2 weeks | Oct 22 – Nov 4 |
| **M4 · Integration benchmark** | • Runner, hidden verifiers and full runs.<br>• Iterate on the skill until the §6.1 targets hold. | ~1 week | Nov 5 – Nov 11 |
| **M5 · Context-quality benchmark** | • Suites B1–B6, the arms, runs and report.<br>• Tune the defaults from the results, within the §6.3 budget. | ~2 weeks | Nov 12 – Nov 25 |
| **M6 · TypeScript, Codex, launch** | • `@artiik/core`, built against the shared `spec/` fixtures.<br>• Vercel AI SDK and Claude Agent SDK (TypeScript) adapters, plus their fixtures.<br>• Codex: test the skill there, add a Codex arm to Track A, package a Codex plugin.<br>• Docs.<br>• Release 1.0 on PyPI, npm and as a plugin tag.<br>• Submit to the plugin directory.<br>• Benchmark write-up. | 2–3 weeks | Nov 26 – Dec 16 |

The versions run 0.2.x alphas from M1 to 1.0 at M6. Nobody depends on 0.1.0 (about 157 non-mirror downloads in six months), so we make a clean break and write a short migration note.

---

## 8. What happens to the current code

| v0.1 piece | Fate in v2 |
|---|---|
| `ContextManager.build_context` / `observe` | Become `Context.prepare` / `record` (L2), on structured messages. |
| `ShortTermMemory` (a deque of text pairs) | Replaced by the managed message history, which is aware of tool calls. |
| `HierarchicalSummarizer` + hard-coded `LLMAdapter`s | Replaced by native compaction, with generic compaction through the builder's own callable. |
| `LongTermMemory` (FAISS + sentence-transformers) | Replaced by file memory, BM25 and an optional `[embeddings]` extra, exposed as a memory tool. |
| Similarity + recency + importance scoring | **Kept** for memory recall ranking. |
| `session_id` / `task_id` scoping | **Kept as an idea**, now enforced as a pre-filter on every read path. |
| `_assemble_and_optimize_context` tiered pruning | **Kept as an idea**: clear → compact → guard, with a tier that's never pruned (pins). |
| `debug_context_building`, `get_stats` | Become traces and `artiik inspect`. |
| `ingest_*` | Dropped (non-goal). |
| `TokenCounter` (tiktoken for everything) | Replaced by usage-based accounting and a calibrated estimator. |
| Docs site (~3,800 lines) | Replaced by short docs; every claim is backed by a test or benchmark. |
| Tests (mocks that patch the wrong name) | Replaced by Tier 0 invariant tests and per-module unit tests. |

---

## 9. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Provider APIs move fast. On-demand compaction is a beta dated 2026-09-04. | Provider specifics live in `providers/`, behind capability detection and dated feature flags. A nightly live smoke test runs against each provider, and the skill reads the installed version's docs. |
| Transparent wrapping (L1) mis-reconciles history | L2 stays the documented default until L1 passes Tier 0 plus fuzzing on history edits. |
| Skill output varies from run to run | Deterministic scripts handle detection and verification, and the model does the wiring. The eval suite runs in CI, and the skill description is tuned against trigger-rate data from Track A. |
| Plugin fatigue and trust | One skill, no hooks, no MCP server and no network calls. The scripts are small and readable, and the skill works without the plugin. |
| Framework churn | Adapters stay thin, and each has a fixture that fails loudly when the framework upgrades. |
| Native features make parts of artiik redundant | The "native only" benchmark arm, plus the honesty rule in §6.2. |
| Benchmark spend | Capped at €100 (§6.3): Track A on the Claude subscription, record/replay, quick mode and `--max-cost-usd`. |
| Scope creep | The non-goals list. A new feature ships only with a benchmark suite showing it helps. |

---

## 10. Definition of done for 1.0

- [ ] A builder goes from plugin install to an integrated, tested agent in under 10 minutes on any Python fixture.
- [ ] Track A: ≥ 90% hidden-verifier pass with the plugin, with the no-plugin baseline published alongside.
- [ ] Track B: published report. artiik is at least as good as "provider-native only" on retention (B1, B3, B5) at a cost per successful task that is no worse, or the report says where it isn't.
- [ ] The core has zero required dependencies, and a wheel under 200 KB.
- [ ] `claude plugin validate --strict` passes, the eval suite is green in CI, and the always-on skill cost is under 150 tokens.
- [ ] The skill works when added on its own, without the plugin, in Claude Code and in Codex.
- [ ] Every README claim links to a test or a benchmark row.

---

## 11. Decisions (settled 2026-09-27)

| # | Decision | Outcome |
|---|---|---|
| 1 | Python first, or Python and TypeScript in parallel? | **Python first.** TypeScript comes in M6, and the `spec/` fixtures keep the port mechanical. |
| 2 | Install artiik as a package, or copy it into the agent's code? | **Package by default.** Vendor mode is the fallback when a project doesn't allow new dependencies. |
| 3 | A clean break from the 0.1 API? | **Yes**, with a short migration note. |
| 4 | Which frameworks in M3? | **Claude Agent SDK, OpenAI Agents SDK, LangChain v1/LangGraph and Pydantic AI.** |
| 5 | API budget for the benchmarks? | **€100**, designed in §6.3. |
| 6 | When to support Codex? | **At the end (M6).** The skill already works there unchanged; M6 tests it and packages a Codex plugin. |
| 7 | Skill or plugin as the product? | **The skill.** Adding it alone must be enough (§5). The plugin is packaging. |

---

## 12. How the work is tracked on GitHub

**Roadmap:** [#2](https://github.com/Artiik-lab/artiik/issues/2) · **Milestones:** #3 (M0) to #9 (M6) · **Tasks:** #10 to #51.

- **One roadmap issue** holds this plan's summary. Its sub-issues are the milestones.
- **One epic per milestone** (M0–M6): type *Feature*, label `epic`, with the org's *Start date*, *Target date*, *Priority* and *Effort* fields set, so a GitHub Project can draw them on a roadmap.
- **One task issue per deliverable**, as a sub-issue of its epic: type *Task*, plus an `area:*` label and a language label. Each task has acceptance criteria and names what blocks it.
- **Workflow:** a branch per task (or per milestone for small ones), and a pull request that says `Closes #n`, so merging closes the task and moves the epic's progress bar.
- **Parked ideas**, such as the MCP server (issue #1), carry the `parked` label and stay out of the roadmap until after 1.0.
