# Remote Storage

devqubit supports remote cloud storage backends for team collaboration and scalable infrastructure.

## Supported Backends

| Backend | URL Scheme | Installation |
|---------|------------|--------------|
| Local filesystem | `file://` | Built-in |
| Amazon S3 | `s3://` | `pip install 'devqubit[s3]'` |
| Google Cloud Storage | `gs://` | `pip install 'devqubit[gcs]'` |

---

## Amazon S3

### Installation

```bash
pip install 'devqubit[s3]'
```

This installs `boto3` and related AWS SDK components.

### Configuration

**URL format:**
```
s3://bucket-name/prefix?region=us-east-1
```

**Environment variables:**
```bash
export DEVQUBIT_STORAGE_URL="s3://my-bucket/devqubit/objects"
export DEVQUBIT_REGISTRY_URL="s3://my-bucket/devqubit"
```

**Programmatic:**
```python
from devqubit.storage import create_store, create_registry

store = create_store("s3://my-bucket/devqubit/objects")
registry = create_registry("s3://my-bucket/devqubit")
```

### URL Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `region` | AWS region | `us-east-1` |
| `endpoint_url` | Custom endpoint (for MinIO, LocalStack) | `http://localhost:9000` |

### AWS Credentials

devqubit uses standard boto3 credential resolution:

1. **Environment variables**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
2. **Shared credentials file**: `~/.aws/credentials`
3. **AWS config file**: `~/.aws/config`
4. **IAM role** (for EC2, ECS, Lambda)

Example with environment variables:
```bash
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_DEFAULT_REGION="us-east-1"
```

### S3-Compatible Storage (MinIO, LocalStack)

Use the `endpoint_url` parameter for S3-compatible services:

```python
from devqubit.storage import create_store

# MinIO
store = create_store(
    "s3://my-bucket/devqubit/objects",
    endpoint_url="http://localhost:9000",
)

# LocalStack
store = create_store(
    "s3://my-bucket/devqubit/objects",
    endpoint_url="http://localhost:4566",
)
```

Or via URL query parameter:
```bash
export DEVQUBIT_STORAGE_URL="s3://my-bucket/prefix?endpoint_url=http://localhost:9000"
```

---

## Google Cloud Storage

### Installation

```bash
pip install 'devqubit[gcs]'
```

This installs `google-cloud-storage` and related GCP SDK components.

### Configuration

**URL format:**
```
gs://bucket-name/prefix
```

**Environment variables:**
```bash
export DEVQUBIT_STORAGE_URL="gs://my-bucket/devqubit/objects"
export DEVQUBIT_REGISTRY_URL="gs://my-bucket/devqubit"
```

**Programmatic:**
```python
from devqubit.storage import create_store, create_registry

store = create_store("gs://my-bucket/devqubit/objects")
registry = create_registry("gs://my-bucket/devqubit")
```

### URL Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `project` | GCP project ID | `my-gcp-project` |

### GCP Authentication

devqubit uses standard Google Cloud authentication:

1. **Application Default Credentials (ADC)**: `gcloud auth application-default login`
2. **Service account key**: `GOOGLE_APPLICATION_CREDENTIALS` environment variable
3. **Compute Engine / GKE metadata** (automatic on GCP infrastructure)

Example with service account:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## Storage Layout

Both S3 and GCS use the same object layout:

```
<prefix>/
├── sha256/                      # Content-addressed object store
│   ├── a1/
│   │   └── a1b2c3...           # Objects sharded by first 2 hex chars
│   └── ff/
│       └── ff00ab...
├── runs/                        # Run records (JSON)
│   ├── 01ABCD123...json
│   └── 01EFGH456...json
└── baselines/                   # Project baselines (JSON)
    ├── my-project.json
    └── vqe-h2.json
```

---

## Combining Local and Remote Storage

You can use different backends for store and registry:

```python
from devqubit.storage import create_store, create_registry

# Objects in S3, metadata locally (faster queries)
store = create_store("s3://my-bucket/objects")
registry = create_registry("file:///home/user/.devqubit")

# Or vice versa
store = create_store("file:///home/user/.devqubit/objects")
registry = create_registry("gs://my-bucket/devqubit")
```

This is useful when:
- You want fast local queries but durable remote artifact storage
- You're migrating between storage backends
- You need to share artifacts but keep run metadata private
