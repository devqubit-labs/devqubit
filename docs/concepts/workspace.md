# Workspace and storage

devqubit is local-first: everything is stored in a **workspace directory** by default, but remote cloud storage is also supported.

## Default location

By default:

- macOS/Linux: `~/.devqubit`
- Windows: your home directory equivalent

Override with:

```bash
export DEVQUBIT_HOME=/path/to/workspace
```

## Layout

A typical local workspace:

```text
~/.devqubit/
├── objects/                 # content-addressed store (by digest)
│   └── sha256/
│       └── a1/
│           └── a1b2c3...    # artifacts stored by digest
├── registry.db              # run metadata index (SQLite)
└── baselines.json           # project baseline mappings
```

## Why content-addressing?

Artifacts are stored by digest, so devqubit can:

- deduplicate identical blobs,
- validate integrity,
- pack/unpack portable bundles reliably.

## Remote storage

For team collaboration or CI/CD pipelines, devqubit supports cloud storage:

| Backend | URL Scheme | Installation |
|---------|------------|--------------|
| Amazon S3 | `s3://bucket/prefix` | `pip install 'devqubit[s3]'` |
| Google Cloud Storage | `gs://bucket/prefix` | `pip install 'devqubit[gcs]'` |

Example:
```bash
export DEVQUBIT_STORAGE_URL="s3://my-bucket/devqubit/objects"
export DEVQUBIT_REGISTRY_URL="s3://my-bucket/devqubit"
```

See {doc}`remote-storage` for detailed configuration, authentication, and CI/CD integration.

## Configuration

See {doc}`../guides/configuration` for environment variables and programmatic configuration.
