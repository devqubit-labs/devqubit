# Contributing to devqubit

Thanks for your interest in devqubit! 🎉

Whether it's a typo fix, bug report, new adapter, or a wild feature idea — we appreciate it all. This guide will get you set up quickly.

- Follow our community standards in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- For security issues, **do not** open a public issue — see [Security](#security)
- Questions or support: use [GitHub Discussions](https://github.com/devqubit-labs/devqubit/discussions) (Issues are for actionable bugs/requests)

## Where to start?

- Browse [open issues](https://github.com/devqubit-labs/devqubit/issues) for bugs or feature requests
- Propose an idea or ask questions in [Discussions](https://github.com/devqubit-labs/devqubit/discussions)
- Pick anything that interests you — even small fixes matter

## Quickstart

```bash
git clone https://github.com/devqubit-labs/devqubit.git
cd devqubit

# Install all workspace packages
uv sync --locked --all-packages

# (Optional) Include all extras (adapters, UI, etc.)
uv sync --locked --all-packages --all-extras

# Install git hooks (required)
uv run pre-commit install

# Run checks and tests
uv run pre-commit run --all-files
uv run pytest
```

## Prerequisites

- Git
- Python (see `requires-python` in `pyproject.toml` for supported versions)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Project structure

```
devqubit/                    # Metapackage (re-exports from engine)
packages/
├── devqubit-engine/         # Core: tracking, storage, comparison, CLI
├── devqubit-ui/             # FastAPI web interface
├── devqubit-qiskit/         # Qiskit adapter
├── devqubit-qiskit-runtime/ # IBM Runtime adapter
├── devqubit-braket/         # Amazon Braket adapter
├── devqubit-cirq/           # Google Cirq adapter
├── devqubit-cudaq/          # NVIDIA CUDA-Q adapter
└── devqubit-pennylane/      # PennyLane adapter
```

## Project tooling

| Tool | Purpose |
|------|---------|
| **uv** | Dependency management, workspaces (monorepo) |
| **pre-commit** | Formatting, linting, repo hygiene |
| **pytest** | Testing |
| **towncrier** | Changelog generation |

CI runs `pre-commit` and `pytest` across multiple Python versions.

## Workspace (monorepo)

This repo uses **uv workspaces** — multiple packages managed together.

```bash
# Install all packages
uv sync --all-packages

# Run command in workspace environment
uv run <cmd>

# Run command for a specific package
uv run --package devqubit-engine pytest
uv run --package devqubit-engine python -m devqubit --help
```

## Pre-commit hooks

Pre-commit is **required**. Install once:

```bash
uv run pre-commit install
```

Run manually (matches CI):

```bash
uv run pre-commit run --all-files
```

Run a specific hook:

```bash
uv run pre-commit run <hook_id> --files path/to/file.py
```

## Running tests

```bash
# Full suite
uv run pytest

# Stop on first failure
uv run pytest -x

# Single file or test
uv run pytest path/to/test_file.py
uv run pytest -k test_name_substring
```

## Making changes

### Branching

Branch from `main`:

- `feat/<short-description>` — features
- `fix/<short-description>` — bug fixes
- `docs/<short-description>` — documentation

### Pull request checklist

Before requesting review:

- [ ] `uv run pre-commit run --all-files` passes
- [ ] `uv run pytest` passes
- [ ] New/changed behavior is covered by tests
- [ ] User-facing changes include a changelog fragment
- [ ] User-facing changes are documented (README/docs/examples)

### Code style

- Prefer clear, readable code over clever code
- Avoid breaking public APIs without discussion
- Keep changes compatible with supported Python versions

## Changelog (towncrier)

We generate `CHANGELOG.md` from fragments in `changelog.d/`.

### When to add a fragment

Add a fragment **only** for user-facing changes:

- New API/CLI features
- Behavior changes (including breaking)
- Bug fixes users would notice
- Deprecations, removals
- Security fixes

**Skip** for: internal refactors, tests, CI, formatting.

### Fragment format

```
changelog.d/<PR_NUMBER>.<type>.md
changelog.d/+<description>.<type>.md  # orphan (no PR yet)
```

Types: `added`, `changed`, `fixed`, `deprecated`, `removed`, `security`

Keep fragments short (1–3 lines), user-focused.

### Validate locally

```bash
uv run towncrier build --draft
```

## Dependencies

When changing dependencies:

```bash
# Update pyproject.toml (or use uv add / uv remove)
uv lock
uv sync --locked --all-packages
```

Upgrade all locked versions:

```bash
uv lock --upgrade
uv sync --locked --all-packages
```

> **Tip:** If `uv sync --locked` fails, run `uv lock` and commit the updated lockfile.

## Reporting bugs

Include:

- What you expected vs. what happened
- OS and Python version
- devqubit version or git commit
- Minimal reproduction (code snippet or command)

## Security

**Do not** report security vulnerabilities via public GitHub issues.

Use GitHub's [Private Vulnerability Reporting](https://github.com/devqubit-labs/devqubit/security/advisories/new).

## License

By contributing, you agree that your contributions are licensed under [Apache-2.0](LICENSE).
