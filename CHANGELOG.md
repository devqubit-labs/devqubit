# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) and the changes for the upcoming release live in `changelog.d/`.

<!-- towncrier release notes start -->

## [0.1.11](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.11) - 2026-02-04

#### Added
- Added `RemoteStore` and `RemoteRegistry` for HTTP-based centralized tracking. ([#44](https://github.com/devqubit-labs/devqubit/pull/44))

#### Changed
- Migrated frontend from Jinja/HTMX to React. ([#41](https://github.com/devqubit-labs/devqubit/pull/41))

## [0.1.10](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.10) - 2026-01-30

#### Changed
- Migrated web UI from Jinja/HTMX to React + TypeScript for improved maintainability. ([#41](https://github.com/devqubit-labs/devqubit/pull/41))

## [0.1.9](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.9) - 2026-01-28

#### Added
- Add run deletion from UI via detail page modal or inline table buttons. ([#37](https://github.com/devqubit-labs/devqubit/pull/37))

#### Changed
- Optimized storage backends for scale (S3/GCS local index, SQLite connection pooling, GC pagination). ([#38](https://github.com/devqubit-labs/devqubit/pull/38))

## [0.1.8](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.8) - 2026-01-25

#### Added
- Added ``run_name`` display throughout the web UI. Runs now show human-readable names in tables, detail views, and comparison dropdowns. The ``Run`` class exposes a public ``run_name`` property.

## [0.1.7](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.7) - 2026-01-24

#### Fixed
- Fix `list_runs` failing with "no such column: name" when filtering by run name due to missing `run_name` column in queries.

## [0.1.6](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.6) - 2026-01-22

#### Added
- UEC-first architecture with strict adapter contract enforcement:
  - `parametric_hash` and `executed_parametric_hash` in `ProgramSnapshot` for VQE support
  - `canonicalize_bitstrings()` for consistent bit order normalization to `cbit0_right`
  - `ProgramMatchStatus` enum for detailed comparison reporting
  - `MissingEnvelopeError` when adapter run missing envelope (use `strict=False` for migration) ([#15](https://github.com/devqubit-labs/devqubit/pull/15))

- Support referencing runs by name via `--project` flag in CLI and `project` parameter in API functions (`diff`, `pack_run`, `verify_baseline`). ([#34](https://github.com/devqubit-labs/devqubit/pull/34))

#### Changed
- UEC ExecutionEnvelope is now the single source of truth for diffs/verification: adapter runs must emit a schema-valid envelope; “non-strict” fallback/synthesis for adapter runs was removed. Manual/replay runs still synthesize a best-effort envelope. Program comparison now distinguishes structural vs parametric hashes and results are canonicalized across bit orders. ([#16](https://github.com/devqubit-labs/devqubit/pull/16))

- Redesigned public API surface: added high-level `verify_baseline()`, new `devqubit.runs` module for run navigation, `devqubit.errors` for public exceptions, `devqubit.adapters` for extension API, moved low-level symbols to appropriate submodules. ([#17](https://github.com/devqubit-labs/devqubit/pull/17))

## [0.1.5](https://github.com/devqubit-labs/devqubit/releases/tag/vv0.1.5) - 2026-01-13

#### Changed
- Breaking: Align UEC Envelope 1.0 schemas/models and update all adapters to emit consistent producer/device/program/execution snapshots and unified artifact references. ([#14](https://github.com/devqubit-labs/devqubit/pull/14))

#### Fixed
- Fix `devqubit-ui` standalone installation by importing directly from `devqubit_engine` instead of the `devqubit` metapackage, and fixed `ArtifactRef` lazy import path. ([#10](https://github.com/devqubit-labs/devqubit/pull/10))

## [0.1.4](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.4) - 2026-01-11

#### Fixed
- Fix top-level imports from devqubit_ui by exposing run_server and create_app. ([#9](https://github.com/devqubit-labs/devqubit/pull/9))

## [0.1.3](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.3) - 2026-01-10

#### Fixed
- Workspace context preserved across navigation links in Teams mode.

## [0.1.2](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.2) - 2026-01-10

#### Added
- Add workspace selector to UI header for Teams integration. When current_workspace and workspaces are passed to templates, users can see and switch between workspaces.

## [0.1.1](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.1) - 2026-01-07

### Added

- User menu component in base template for Teams integration
- Support for `current_user` context variable in templates

## [0.1.0](https://github.com/devqubit-labs/devqubit/releases/tag/v0.1.0) - 2026-01-07

#### Added
- Initial public release of devqubit (core + engine + adapters + optional local UI).
