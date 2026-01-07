# Releasing devqubit (monorepo)

This repository is a **uv workspace** (monorepo) that publishes multiple PyPI distributions:

- `devqubit` (workspace root)
- `devqubit-engine`
- `devqubit-ui`
- `devqubit-qiskit` (+ other adapters)

Releases are cut from git tags `vX.Y.Z`. GitHub Actions builds **all** distributions and publishes them to PyPI via **Trusted Publishing (OIDC)** — no long-lived PyPI tokens.

References (official):
- Publishing with GitHub Actions: https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
- Trusted Publishing (PyPI): https://docs.pypi.org/trusted-publishers/
- Publishing with a Trusted Publisher: https://docs.pypi.org/trusted-publishers/using-a-publisher/
- uv CLI reference: https://docs.astral.sh/uv/reference/cli/
- Towncrier docs: https://towncrier.readthedocs.io/

---

## Versioning policy

We use **lockstep versioning** across the monorepo:

- root `pyproject.toml` and every `packages/*/pyproject.toml` share the same `project.version`.
- release tags are `v{version}` (e.g. `v0.1.0`).

The release workflow should verify that the tag version matches all `pyproject.toml` versions.

---

## Changelog policy (towncrier)

This repo uses **towncrier**. Changes for the upcoming release live in `changelog.d/`.

### When a fragment is needed

Add a fragment **only for user-facing changes**, for example:

- New API/CLI features or options
- Behavior changes (including breaking changes)
- Bug fixes that users would notice
- Deprecations / removals
- Security fixes

Skip fragments for internal refactors, tests, CI, formatting-only changes, etc.

### Where it goes

Choose a category directory:

- `changelog.d/added/`
- `changelog.d/changed/`
- `changelog.d/fixed/`
- `changelog.d/deprecated/`
- `changelog.d/removed/`
- `changelog.d/security/`

Filename rules:

- Prefer: `<PR_NUMBER>.md` (e.g. `changelog.d/fixed/123.md`)
- If not tied to a PR/issue: use an **orphan fragment** that starts with `+`
  (e.g. `changelog.d/changed/+bundle-format.md`)

### Previewing the upcoming changelog (optional)

```bash
uv run towncrier build --draft
```

### Generating the release notes

Generate the release notes into `CHANGELOG.md`:

```bash
uv run towncrier build --version X.Y.Z
```

This updates `CHANGELOG.md` and removes the consumed fragments from `changelog.d/`.

---

## Release checklist (standard release)

> Do releases from a clean `main` branch with all checks green.

### 1) Ensure dependencies are in sync

Install the full workspace, including all optional dependencies:

```bash
uv sync --all-packages --all-extras
```

### 2) Run quality checks locally (recommended)

```bash
uv run pre-commit run --all-files
uv run pytest
```

### 3) Prepare changelog

```bash
uv run towncrier build --draft
uv run towncrier build --version X.Y.Z
```

### 4) Bump versions (lockstep)

Update `project.version` to `X.Y.Z` in:

- root `pyproject.toml`
- every `packages/*/pyproject.toml`

### 5) Final sanity check (recommended)

```bash
uv sync --all-packages --all-extras
uv run pytest
```

### 6) Commit, tag, push

```bash
git add -A
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

### 7) Verify publication

- GitHub Actions runs the Release workflow:
  - builds wheels/sdists for root + all `packages/*`
  - publishes all distributions found in `dist/` via Trusted Publishing
- Verify each project on PyPI:
  - `devqubit`
  - `devqubit-engine`
  - `devqubit-ui`
  - adapters (`devqubit-qiskit`, etc.)

---

## First release (one-time setup)

### A) Create/configure PyPI projects (names)

Ensure all distribution names you intend to publish are available on PyPI.

### B) Configure Trusted Publishing (OIDC) for each PyPI project

Trusted Publisher config is **per PyPI project**. For each project:

1. Go to the project’s “Trusted Publishers” settings on PyPI.
2. Add a GitHub Actions publisher pointing to:
   - GitHub owner + repository
   - workflow file: `.github/workflows/release.yml`
   - (recommended) GitHub Environment name: `pypi`

Notes:
- The publish job in GitHub Actions must have `permissions: id-token: write`.
- PyPI will mint a short-lived token at publish time (no secret tokens required).
