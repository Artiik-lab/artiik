# Contributing

## Workflow

1. **Start from an issue.** Each task issue has a scope, acceptance criteria and a **Blocked by** line.
2. **Branch from `main`**, named after the work: for example `feat/15-message-model` or `fix/42-scorer-rounding`.
3. **Commits** reference the issue with `Refs #15`.
4. **Open a pull request** whose description says `Closes #15`, so merging it closes the task.
5. **CI must be green** before merging.

## Development

The Python package lives in [`python/`](python/) and uses [uv](https://docs.astral.sh/uv/).

```sh
cd python
uv sync                                          # create the environment
uv run pytest                                    # tests
uv run ruff check . && uv run ruff format --check .
uv run pyright                                   # strict type checking
uv build                                         # wheel and sdist
```

### Live smoke tests

[`python/live/`](python/live/) has small tests that call the real provider APIs: one per compaction strategy, and one for pins per provider. CI never runs them. Run them by hand after changing provider code; a run costs a few cents.

```sh
cd python
export ARTIIK_LIVE=1
export ANTHROPIC_API_KEY=... ARTIIK_ANTHROPIC_MODEL=...   # a model with compaction on demand
export OPENAI_API_KEY=... ARTIIK_OPENAI_MODEL=...         # a model with Responses compaction
uv run --with anthropic --with openai pytest live -s
```

A test skips when its provider's key or model isn't set. With `ARTIIK_LIVE_HEAVY=1`, the Anthropic threshold test also sends about 60,000 input tokens, so the API really compacts.

## Rules

- **The core has zero required dependencies.** Provider SDKs and heavier features go behind optional extras. A test and a CI job enforce this.
- **Tests never use the network.** Anything that talks to a model uses the fake clients in `artiik.testing`, and runtime features check the Tier 0 invariants (`check_all`) on the calls those clients record. The hand-run smoke tests in `python/live/` are the only exception.
- **No hard-coded model IDs in library code.** Model choices come from the caller or from configuration.
- **Provider specifics are documented at the source.** Beta headers, block types and parameter shapes live in provider modules behind capability checks, each with a link to the provider page it comes from, because these APIs change often.
- **Protect the prompt cache.** Never rewrite history before the last cache breakpoint between compactions; append instead.
- **Every README claim links to a test or a benchmark result.**
- **Keep `python/LICENSE` identical to `LICENSE`.** A test checks it.
- **Supported Python:** 3.11 and later.
