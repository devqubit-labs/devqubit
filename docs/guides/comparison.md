# Comparison and Verification

Use comparison to answer: **did my result change?** and **why?**

devqubit compares two runs (or bundles) across: metadata, parameters, metrics, program artifacts (exact + structural matching), results with noise-aware context, and device calibration drift.

## Comparing Runs

```python
from devqubit.compare import diff

# By run name (recommended)
result = diff("baseline-v1", "candidate-v2", project="bell-state")

# Or by run ID
result = diff("01KDNZYSNPZFVPZG94DATP1DT6", "01KDNZZ9KBYK1DDCW6KP38DA34")

print(result)
```

Output:

```
══════════════════════════════════════════════════════════════════════
                            RUN COMPARISON
══════════════════════════════════════════════════════════════════════

  Baseline:      01KDNZYSNPZFVPZG94DATP1DT6  [vqe-h2]
  Candidate:     01KDNZZ9KBYK1DDCW6KP38DA34  [vqe-h2]

──────────────────────────────────────────────────────────────────────
  RESULT: ✗ DIFFER
──────────────────────────────────────────────────────────────────────

SUMMARY
───────
  ✓ Program:       structural match
  ✗ Parameters:    2 changed, 1 added
  ! Results:       TVD = 0.0870 (3.5x noise)
  ! Device:        calibration drift detected

METADATA DIFFERENCES
────────────────────
  backend:       ibm_brisbane => ibm_kyoto

PARAMETER CHANGES
─────────────────
  learning_rate         0.01 => 0.02 (+100.0%)
  num_layers            4 => 6 (+50.0%)

METRIC CHANGES
──────────────
  energy                -1.136 => -1.142 (-0.5%)
  fidelity              0.952 => 0.891 (-6.4%)

DISTRIBUTION ANALYSIS
─────────────────────
  TVD:           0.087000  !
  Expected noise: 0.025000
  Noise ratio:   3.48x
  Assessment:    Moderate divergence

DEVICE CALIBRATION
──────────────────
  Baseline cal:  2026-01-15T10:00:00Z
  Candidate cal: 2026-01-15T14:30:00Z

  ! Significant drift in 2 metric(s):
    median_t2_us        120.5 => 98.3 (-18.4%)
    median_t1_us        95.2 => 82.1 (-13.8%)

CIRCUIT DIFFERENCES
───────────────────
  depth          4 => 6 (+50.0%)
  2Q gates       8 => 12 (+50.0%)
  + gates        rz, rzz

WARNINGS
────────
  ! Backend changed between runs
  ! Calibration data may not be comparable across devices
```

`diff` accepts run names (with project), run IDs, or bundle files:

```python
result = diff("baseline", "candidate", project="myproj")  # Two run names
result = diff("01JD7X...", "01JD8Y...")                   # Two run IDs
result = diff("baseline.zip", "candidate.zip")            # Two bundles
```

## ComparisonResult

```python
result = diff("baseline", "candidate", project="myproj")

# Overall
result.identical          # True if everything matches
result.run_id_a           # Baseline run ID
result.run_id_b           # Candidate run ID

# Metadata
result.metadata["project_match"]
result.metadata["backend_match"]

# Parameters and metrics
result.params["match"]    # True if all params match
result.params["changed"]  # {"shots": {"a": 1000, "b": 2000}}
result.metrics["match"]

# Program comparison
result.program.exact_match       # Artifact digests identical
result.program.structural_match  # Circuit structure matches
result.program.parametric_match  # Structure + params match
result.program.matches("either") # Check with specific mode

# Results
result.tvd                # Total variation distance
result.counts_a           # {"00": 500, "11": 500}
result.noise_context      # Bootstrap noise analysis

# Device and circuit
result.device_drift       # Calibration drift analysis
result.circuit_diff       # Semantic circuit comparison

# Output
result.to_dict()
result.format_json()
result.format_summary()
```

## TVD and Noise Context

TVD measures distribution difference: 0.0 = identical, 0.01–0.05 = typical shot noise, >0.15 = significant difference.

The `noise_context` uses parametric bootstrap to estimate shot noise thresholds:

```python
if result.noise_context:
    ctx = result.noise_context
    print(f"Noise p95: {ctx.noise_p95:.4f}")     # 95th percentile threshold
    print(f"p-value: {ctx.p_value:.4f}")         # Empirical p-value
    print(f"Exceeds noise: {ctx.exceeds_noise}") # tvd > noise_p95?
    print(f"Interpretation: {ctx.interpretation()}")
```

| p-value | Interpretation |
|---------|----------------|
| ≥ 0.10 | Consistent with sampling noise |
| 0.05–0.10 | Borderline; consider increasing shots |
| < 0.05 | Likely exceeds sampling noise |

## Baseline Verification

Verify a candidate run against the project's baseline:

```python
from devqubit.compare import verify_baseline, VerifyPolicy

policy = VerifyPolicy(
    params_must_match=True,
    program_must_match=True,
    noise_factor=1.0,
)

result = verify_baseline(
    "nightly-run",  # run name or ID
    project="vqe-h2",
    policy=policy,
)

print(f"Passed: {result.ok}")
if not result.ok:
    print(result.failures)
```

## VerifyPolicy

```python
from devqubit.compare import VerifyPolicy, ProgramMatchMode

policy = VerifyPolicy(
    # Structural checks
    params_must_match=True,
    program_must_match=True,
    program_match_mode=ProgramMatchMode.EITHER,  # exact, structural, or either
    fingerprint_must_match=False,

    # TVD checks
    tvd_max=0.1,          # Hard limit
    noise_factor=1.0,     # Dynamic: fail if TVD > N × noise_p95

    # Bootstrap settings
    noise_alpha=0.95,
    noise_n_boot=1000,
    noise_seed=12345,

    # Behavior
    allow_missing_baseline=False,
)
```

When both `tvd_max` and `noise_factor` are set, the **stricter** (minimum) threshold is used.

**Program match modes:** `EXACT` (identical digests), `STRUCTURAL` (same circuit structure, VQE-friendly), `EITHER` (default).

**Recommended noise_factor:** 1.0 for strict CI, 1.2 for standard CI (recommended), 1.5 for noisy hardware.

## Setting Baselines

```python
from devqubit.runs import get_baseline, set_baseline, clear_baseline

set_baseline("vqe-h2", "production-v1")  # by name or ID
baseline = get_baseline("vqe-h2")
clear_baseline("vqe-h2")
```

Or via CLI:

```bash
devqubit baseline set vqe-h2 production-v1
devqubit baseline get vqe-h2
devqubit baseline clear vqe-h2
```

Auto-promote on pass:

```python
result = verify_baseline(
    "nightly-run",
    project="vqe-h2",
    policy=policy,
    promote_on_pass=True,
)
```

## Device Drift Detection

Calibration drift is automatically detected during comparison:

```python
if result.device_drift and result.device_drift.significant_drift:
    print("Significant calibration drift detected")
    for metric in result.device_drift.top_drifts[:3]:
        print(f"  {metric.metric}: {metric.percent_change:+.1f}%")
```

## CI Integration

```yaml
# GitHub Actions
- name: Verify against baseline
  run: |
    devqubit verify --project vqe-h2 nightly-run \
      --noise-factor 1.0 \
      --junit results.xml
```

```python
from devqubit.compare import verify_baseline
from devqubit.ci import write_junit

result = verify_baseline("nightly-run", project="vqe-h2")
write_junit(result, "results.xml")
```

## CLI

```bash
devqubit diff baseline-v1 candidate-v2 --project myproj
devqubit diff baseline-v1 candidate-v2 --project myproj --format json
devqubit verify --project vqe-h2 nightly-run
devqubit verify --project vqe-h2 nightly-run --noise-factor 1.0 --promote
```
