# Workspace and Storage

devqubit is local-first: everything is stored in a **workspace directory** by default, with optional cloud storage for teams and CI/CD.

## Default Location

| Platform | Default path |
|----------|-------------|
| macOS/Linux | `~/.devqubit` |
| Windows | `%USERPROFILE%\.devqubit` |

Override with:

```bash
export DEVQUBIT_HOME=/path/to/workspace
```

## Workspace Layout

```
~/.devqubit/
├── objects/                 # Content-addressed artifact store
│   └── sha256/
│       └── a1/
│           └── a1b2c3...    # Artifacts stored by digest
├── registry.db              # Run metadata index (SQLite)
└── baselines.json           # Project baseline mappings
```

## Content-Addressing

Artifacts are stored by SHA-256 digest, enabling:
- Deduplication of identical blobs
- Integrity verification
- Portable bundles

## Remote Storage

For team collaboration or CI/CD, devqubit supports cloud storage:

| Backend | URL Scheme | Installation |
|---------|------------|--------------|
| Amazon S3 | `s3://bucket/prefix` | `pip install 'devqubit-engine[s3]'` |
| Google Cloud Storage | `gs://bucket/prefix` | `pip install 'devqubit-engine[gcs]'` |

Configure via environment variables:

```bash
export DEVQUBIT_STORAGE_URL="s3://my-bucket/devqubit/objects"
export DEVQUBIT_REGISTRY_URL="s3://my-bucket/devqubit"
```

Or programmatically:

```python
from devqubit.storage import create_store, create_registry

store = create_store("s3://my-bucket/devqubit/objects")
registry = create_registry("s3://my-bucket/devqubit")
```
