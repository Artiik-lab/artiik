# CLAUDE.md

Guidance for coding agents working in this repository.

## Context

artiik is being rebuilt as a context management layer for AI agents, delivered through a skill that coding agents apply for their users. [`REVAMP_PLAN.md`](REVAMP_PLAN.md) is the plan. The work is tracked in GitHub issues under the roadmap, issue #2. Before working on a task, read its issue: it has the scope, the acceptance criteria and what blocks it.

## Layout

- `python/`: the runtime. It uses a src layout (`python/src/artiik/`), with tests in `python/tests/`.
- `.github/workflows/`: `ci.yml` (lint, types, tests, packaging) and `release.yml` (PyPI via trusted publishing).
- Coming: `plugins/artiik/` (the skill and its Claude Code plugin), `spec/` (fixtures shared by Python and TypeScript), `bench/`, `typescript/`.

## Commands

Run them from `python/`:

- `uv sync`
- `uv run pytest`
- `uv run ruff check . && uv run ruff format --check .`
- `uv run pyright` (strict mode)
- `uv build`

## Rules

- **Never add to `[project].dependencies`.** The core has zero required dependencies, and `tests/test_packaging.py` and the CI packaging job enforce it. Optional features go behind extras.
- **No network in tests.** Use fake clients for anything model-shaped.
- **No hard-coded model IDs in library code.**
- **Keep provider specifics in one place.** Beta headers, block types and parameter shapes live in provider modules behind capability checks, each with a link to its documentation page.
- **Protect the prompt cache.** Never rewrite history before the last cache breakpoint between compactions; append instead.
- **Keep `python/LICENSE` identical to `LICENSE`.** A test checks it.
- **Reference the issue.** Commits say `Refs #n`, and pull requests say `Closes #n`.
