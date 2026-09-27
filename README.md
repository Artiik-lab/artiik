# artiik

**Context management for AI agents, added by your coding agent.**

> **Status: pre-alpha, being rewritten.** Nothing described below is usable yet. Follow the [roadmap](https://github.com/Artiik-lab/artiik/issues/2) and the [plan](REVAMP_PLAN.md).
> The previous 0.1 release is no longer maintained; its code stays at the [`v0.1.1` tag](https://github.com/Artiik-lab/artiik/tree/v0.1.1).

## What artiik will be

Long-running agents outgrow their context window. Providers now compact long conversations for you, but a summary can silently drop what the agent must not forget: a user's rule, a decision, the files it already changed.

artiik is a small layer between your agent and its model. It does five jobs:

| Job | What it does |
|---|---|
| **Budget** | Keeps every request under the budget, and never splits a tool call from its result. |
| **Compact** | Clears old tool output first, then compacts using your provider's native feature when there is one. |
| **Keep** | Restates pinned rules, decisions and the files the agent touched after every compaction. |
| **Remember** | Keeps memory across sessions as plain files, scoped per user, and exposes it as a memory tool. |
| **Show** | Records what each request contained: tokens per segment, cache hits, compactions and cost. |

It builds on what your provider already offers (Anthropic's compaction, context editing and memory tool; OpenAI's compaction) and adds only what's missing. The core has no required dependencies.

## How you'll add it

You won't wire it in by hand. Add the **artiik skill**, then ask your coding agent for context management:

```text
/plugin marketplace add artiik-lab/artiik      # Claude Code: the plugin
/plugin install artiik@artiik

npx skills add artiik-lab/artiik               # or the skill alone, in any agent that supports Agent Skills

> add context management to my agent
```

The skill detects your stack, installs artiik, wires it in with a small diff, adds a regression test and shows a before/after report.

Planned for 1.0: raw Anthropic and OpenAI SDK loops, OpenAI-compatible local models, the Claude Agent SDK, the OpenAI Agents SDK, LangChain v1/LangGraph and Pydantic AI in Python, then the Vercel AI SDK and the Claude Agent SDK in TypeScript.

## Proof, not promises

Every claim will link to a test or a benchmark result:

- **Integration benchmark:** does the skill wire artiik in correctly, compared with a coding agent working without it?
- **Context-quality benchmark:** does the agent keep what it must after compaction, and what does it cost per successful task, compared with provider-native features alone?

Results will be published in this repository, including the cases where native features were enough.

## Repository

| Path | What's there |
|---|---|
| [`python/`](python/) | The runtime (PyPI: `artiik`). |
| [`REVAMP_PLAN.md`](REVAMP_PLAN.md) | The plan for artiik 2. |
| `plugins/artiik/`, `spec/`, `bench/`, `typescript/` | Coming with the milestones in the [roadmap](https://github.com/Artiik-lab/artiik/issues/2). |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
