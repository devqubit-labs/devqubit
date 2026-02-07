# Glossary

| Term | Meaning |
|------|---------|
| **Adapter** | SDK-specific wrapper that auto-captures circuits, results, and snapshots. |
| **Artifact** | Any stored blob/file (circuit, result JSON, device snapshot, …). |
| **Baseline** | Reference run for a project, used by `verify`. |
| **Bundle** | Portable ZIP created by `devqubit pack`, containing a run and all artifacts. |
| **Calibration** | Device-specific error rates, T1/T2 times, and gate fidelities. |
| **DeviceSnapshot** | Captured backend state: topology, calibration, SDK versions. |
| **Digest** | SHA-256 hash identifying an artifact in the content-addressed store. |
| **Doctor** | CLI command (`devqubit doctor`) that detects and recovers stale RUNNING runs. |
| **ExecutionEnvelope** | Standardized record of an execution (UEC): device + program + execution + result snapshots. |
| **ExecutionSnapshot** | Submission metadata: shots, job IDs, transpilation info. |
| **Fingerprint** | Hash computed from run contents for reproducibility tracking. |
| **Frontend** | In multi-layer stacks (e.g., PennyLane), the user-facing API layer. |
| **GC** | Garbage collection — `devqubit gc` prunes old runs and cleans orphaned objects. |
| **Group** | Collection of related runs (e.g., parameter sweep) identified by `group_id`. |
| **JUnit** | XML test report format used for CI integration (`write_junit`). |
| **Lineage** | Parent-child relationship between runs (`parent_run_id`). |
| **Metric** | Numeric value logged during a run (`log_metric`). |
| **Metric series** | Time series of metrics logged with `step` parameter. |
| **Noise context** | Bootstrap-calibrated thresholds and p-value for TVD under shot noise. |
| **Parameter** | Configuration value logged at run start (`log_param`). |
| **ProgramSnapshot** | Captured circuit artifacts: logical, physical, hashes. |
| **Project** | A named collection of runs (e.g., `bell-state`, `vqe-h2`). |
| **Provenance** | Git state (commit, branch, dirty) captured at run time. |
| **Registry** | Database storing run metadata (SQLite by default). |
| **ResultSnapshot** | Normalized measurement counts or expectation values. |
| **Role** | Semantic category of an artifact (`program`, `results`, `device_raw`, `envelope`, …). |
| **Run** | A single tracked execution with metadata, params, metrics, artifacts, and fingerprints. |
| **Store** | Content-addressed object store for artifacts (file-based by default). |
| **Tag** | String key-value pair for categorization (`set_tag`). |
| **TVD** | Total Variation Distance — measures difference between two probability distributions. |
| **UEC** | Uniform Execution Contract — standardized format for execution records. |
| **Verification** | Checking a candidate run against baseline using a policy. |
| **VerifyPolicy** | Configuration for verification: thresholds, match requirements. |
| **Workspace** | Local directory containing the object store and metadata registry. |
