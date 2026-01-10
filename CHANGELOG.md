# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [towncrier](https://towncrier.readthedocs.io/) and the changes for the upcoming release live in `changelog.d/`.

<!-- towncrier release notes start -->

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
