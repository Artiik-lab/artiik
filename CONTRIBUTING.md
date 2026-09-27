# Contributing

## Workflow

1. **Start from an issue.** The [roadmap](https://github.com/Artiik-lab/artiik/issues/2) lists the milestones, and each milestone lists its tasks. Every task has a scope, acceptance criteria and a **Blocked by** line.
2. **Branch from `main`**, named after the issue: for example `feat/17-context-core` or `fix/42-scorer-rounding`.
3. **Commits** reference the issue with `Refs #17`.
4. **Open a pull request** whose description says `Closes #17`. Merging it closes the task and moves the milestone's progress bar.
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

## Rules

- **The core has zero required dependencies.** Provider SDKs and heavier features go behind optional extras. A test and a CI job enforce this.
- **Tests never use the network.** Anything that talks to a model uses fake clients.
- **No hard-coded model IDs in library code.** Model choices come from the caller or from configuration.
- **Provider specifics are documented at the source.** Beta headers, block types and parameter shapes carry a link to the provider page they come from, because these APIs change often.
- **Every README claim links to a test or a benchmark result.**
- **Supported Python:** 3.11 and later.
