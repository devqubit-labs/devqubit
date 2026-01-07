# devqubit

**devqubit** is a *local-first experiment tracker for quantum workloads*.

It helps you **capture the full execution context** (code + circuit/program artifacts + device/backend snapshot + results), so you can:

- reproduce runs reliably,
- compare two executions and understand what changed,
- detect regressions/drift (code, settings, hardware calibration),
- share portable “bundles” of results with your team.

## What you get

- **Tracking**: parameters, metrics, tags, and artifacts
- **Adapters**: automatic capture for popular quantum SDKs/backends
- **Comparison & verification**: diff runs, verify against baselines, CI-friendly output
- **Portable bundles**: pack/unpack runs into ZIPs for sharing and archiving

---

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/overview
concepts/workspace
concepts/uec
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/tracking
guides/adapters
guides/comparison
guides/configuration
guides/remote_storage
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/cli
reference/glossary
```
