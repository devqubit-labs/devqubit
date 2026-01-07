# Comparison and verification

Use comparison to answer: **did my result change?** and **why?**

devqubit compares two runs (or two portable bundles) across:
- metadata (project, adapter, backend)
- parameters, metrics
- program/circuit artifacts (exact + structural matching)
- results (counts/expectations) with noise-aware context
- device drift (when device snapshots are available)

Most users start with the CLI: {doc}`../reference/cli` → `devqubit diff` and `devqubit verify`.


## Comparing Runs

Compare two runs to detect differences in parameters, programs, metrics, and results:

```python
from devqubit import diff, create_registry

registry = create_registry()

result = diff("RUN_BASELINE", "RUN_CANDIDATE")

print(result)
```

Output:

```
======================================================================
RUN COMPARISON
======================================================================
Baseline:  01KDNZYSNPZFVPZG94DATP1DT6
Candidate: 01KDNZZ9KBYK1DDCW6KP38DA34

Overall: ✗ DIFFER

----------------------------------------------------------------------
Metadata
----------------------------------------------------------------------
  project: ✓
  backend: ✗
    fake_manila -> aer_simulator

----------------------------------------------------------------------
Program
----------------------------------------------------------------------
  ✓ Match (structural)

----------------------------------------------------------------------
Parameters
----------------------------------------------------------------------
  ✓ Match

----------------------------------------------------------------------
Results
----------------------------------------------------------------------
  TVD: 0.037598
  Noise threshold (p95): 0.068421
  p-value: 0.234
  Interpretation: Difference is consistent with sampling noise (p >= 0.10)

======================================================================
```

## Compare Anything

`diff` accepts run IDs or bundle files:

```python
from devqubit import diff

# Two run IDs
result = diff("RUN_A", "RUN_B")

# Bundle vs run
result = diff("baseline.zip", "RUN_B")

# Two bundles
result = diff("baseline.zip", "candidate.zip")
```

## Comparison Result

```python
from devqubit import diff

result = diff("RUN_A", "RUN_B")

# Overall
result.identical          # True if everything matches
result.run_id_a           # Baseline run ID
result.run_id_b           # Candidate run ID

# Metadata
result.metadata["project_match"]
result.metadata["adapter_match"]
result.metadata["backend_match"]

# Parameters
result.params["match"]    # True if all params match
result.params["changed"]  # {"shots": {"a": 1000, "b": 2000}}
result.params["added"]    # Params only in candidate
result.params["removed"]  # Params only in baseline

# Metrics
result.metrics["match"]   # True if all metrics match
result.metrics["changed"] # Changed metrics with values

# Program
result.program.exact_match       # True if artifact digests identical
result.program.structural_match  # True if circuit structure matches
result.program_match             # True if either matches

# Results
result.tvd                # Total variation distance (or None)
result.counts_a           # {"00": 500, "11": 500}
result.counts_b
result.noise_context      # Bootstrap-calibrated noise analysis

# Circuit
result.circuit_diff       # Semantic circuit comparison

# Device
result.device_drift       # Calibration drift analysis

# Serialize
result.to_dict()          # Dict representation
result.format_json()      # JSON string
result.format_summary()   # One-line summary
```

## Total Variation Distance (TVD)

TVD measures the difference between two probability distributions:

| TVD | Interpretation |
|-----|----------------|
| 0.0 | Identical distributions |
| 0.01–0.05 | Typical shot noise |
| 0.05–0.15 | Possible drift or change |
| > 0.15 | Significant difference |

## Bootstrap-Calibrated Noise Context

The `noise_context` uses parametric bootstrap to estimate shot noise thresholds:

```python
if result.noise_context:
    ctx = result.noise_context

    # Bootstrap-calibrated threshold (primary)
    print(f"Noise p95: {ctx.noise_p95:.4f}")     # 95th percentile threshold
    print(f"p-value: {ctx.p_value:.4f}")         # Empirical p-value
    print(f"Exceeds noise: {ctx.exceeds_noise}") # tvd > noise_p95?

    # Legacy fields (backward compatibility)
    print(f"Expected noise: {ctx.expected_noise:.4f}")
    print(f"Noise ratio: {ctx.noise_ratio:.2f}x")

    # Human-readable interpretation
    print(f"Interpretation: {ctx.interpretation()}")
```

### Interpretation Guidelines

| p-value | Interpretation |
|---------|----------------|
| ≥ 0.10 | Consistent with sampling noise |
| 0.05–0.10 | Borderline; consider increasing shots |
| 0.01–0.05 | Likely exceeds sampling noise |
| < 0.01 | Significantly exceeds sampling noise |

### Why Bootstrap?

The bootstrap approach:
1. Pools both distributions under H0 (null hypothesis: same distribution)
2. Simulates many measurement pairs from pooled distribution
3. Computes TVD for each simulated pair
4. Uses quantiles as calibrated thresholds

This is more robust than simple O(√k/n) heuristics, especially for non-uniform distributions.

---

## Baseline Verification

Verify a candidate run against the project's baseline:

```python
from devqubit import create_registry
from devqubit import verify_against_baseline
from devqubit.compare import VerifyPolicy

registry = create_registry()

policy = VerifyPolicy(
    params_must_match=True,
    program_must_match=True,
    noise_factor=1.0,  # Use bootstrap-calibrated threshold
)

result = verify_against_baseline(
    candidate=registry.load("RUN_CANDIDATE"),
    project="vqe-h2",
    policy=policy,
)

print(result)
print(f"Passed: {result.ok}")
```

## Verification Policy

```python
from devqubit.compare import VerifyPolicy

policy = VerifyPolicy(
    # Structural checks
    params_must_match=True,       # Parameters must be identical
    program_must_match=True,      # Program must match
    program_match_mode="either",  # exact, structural, or either
    fingerprint_must_match=False, # Full run fingerprint must match

    # Result checks
    tvd_max=0.1,                  # Fixed TVD threshold
    noise_factor=1.0,             # Dynamic: fail if TVD > N × noise_p95

    # Baseline handling
    allow_missing_baseline=False, # Pass when no baseline exists
)
```

### Program Match Modes

| Mode | Behavior | Best For |
|------|----------|----------|
| `EXACT` | Require identical artifact digests | Reproducibility checks |
| `STRUCTURAL` | Require same circuit structure | VQE/QAOA (different params OK) |
| `EITHER` | Pass if exact OR structural matches | General use (default) |

```python
from devqubit.compare import VerifyPolicy

# Strict reproducibility
policy = VerifyPolicy(program_match_mode="exact")

# VQE-friendly (ignore parameter values)
policy = VerifyPolicy(program_match_mode="structural")
```

### TVD Thresholds

```python
from devqubit.compare import VerifyPolicy

# Fixed threshold
policy = VerifyPolicy(tvd_max=0.1)

# Bootstrap-calibrated threshold (recommended for CI)
# Fails if TVD > noise_factor × noise_p95
policy = VerifyPolicy(noise_factor=1.0)  # Use raw p95 threshold

# More lenient (for noisy hardware)
policy = VerifyPolicy(noise_factor=1.5)  # 1.5× p95 threshold

# Combined (uses the larger threshold)
policy = VerifyPolicy(tvd_max=0.05, noise_factor=1.2)
```

**Recommended `noise_factor` values:**

| Value | Use Case |
|-------|----------|
| 1.0 | Strict CI gating (5% false positive rate under H0) |
| 1.2 | Standard CI (recommended default) |
| 1.5 | Lenient (noisy hardware, exploratory runs) |

## Setting Baselines

```python
from devqubit import create_registry

registry = create_registry()

# Set baseline
registry.set_baseline("vqe-h2", "RUN_PRODUCTION_V1")

# Get current baseline
baseline = registry.get_baseline("vqe-h2")
print(baseline["run_id"])
```

## Auto-Promote on Pass

```python
result = verify_against_baseline(
    candidate=candidate,
    project="vqe-h2",
    store=store,
    registry=registry,
    policy=policy,
    promote_on_pass=True,  # Update baseline if verification passes
)
```

---

## Device Drift Detection

Calibration drift is automatically detected during comparison:

```python
result = diff("RUN_A", "RUN_B", registry=registry, store=store)

if result.device_drift and result.device_drift.significant_drift:
    print("! Significant calibration drift detected")
    for metric in result.device_drift.top_drifts[:3]:
        print(f"  {metric.metric}: {metric.percent_change:+.1f}%")
```

---

## CI Integration

### GitHub Actions

```yaml
- name: Run experiment
  run: python run_experiment.py

- name: Verify against baseline
  run: |
    devqubit verify --project vqe-h2 $RUN_ID \
      --params-must-match \
      --program-match-mode either \
      --noise-factor 1.0 \
      --junit results.xml
```

### JUnit Output

```python
from devqubit import verify_against_baseline
from devqubit.ci import write_junit

result = verify_against_baseline(...)
write_junit(result, "results.xml")
```

### GitHub Annotations

```python
from devqubit.ci import result_to_github_annotations

print(result_to_github_annotations(result))
# ::notice title=Verification Passed::Candidate RUN_ID matches baseline
```

---

## CLI

```bash
# Compare two runs
devqubit diff RUN_A RUN_B

# Compare bundles
devqubit diff baseline.zip candidate.zip

# Output as JSON
devqubit diff RUN_A RUN_B --format json

# Verify against baseline
devqubit verify --project vqe-h2 RUN_CANDIDATE

# Verify with bootstrap-calibrated threshold
devqubit verify --project vqe-h2 RUN_CANDIDATE \
  --program-match-mode structural \
  --noise-factor 1.0 \
  --promote

# Replay experiment (experimental)
devqubit replay experiment.zip --experimental --seed 42
```
