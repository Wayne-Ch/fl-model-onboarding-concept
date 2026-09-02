# SmolLM2 Round 6 JSON regression follow-up

## Scope

- Target blocker: optimized JSON-format regression for `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Frozen revision: `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Scope guard held: diagnostics-only artifacts; no production/config/profile/contract changes.

## A) Audit-gap fixes completed

1. **Runtime toolchain probe now uses explicit runtime interpreter**
   - `run_smollm_json_regression_diagnostics.py` now executes a runtime-subprocess probe with `--runtime-python`.
   - It records requested executable vs reported executable and package versions.
   - It fails when harness/runtime are the same unless explicitly acknowledged with `--allow-interpreter-conflation`.

2. **Selected-input identity, sibling inventory, and hash stability**
   - Selection is now strict by exact artifact id short-hash directory (`artifact_prefix-<artifact_id[:12]>`), not best-effort fallback.
   - Report records preexisting sibling inventory for matching snapshot/artifact prefixes.
   - Report records full selected snapshot/package fingerprints before and after diagnostics and confirms they remain unchanged.

3. **Contained cleanup of exact stray root**
   - Cleanup target is hard-capped to exactly the configured stray diagnostics root (`%SystemDrive%\\fmo-r6-smollm-diagnostics`).
   - Report records entries observed, bytes freed, and selected-input fingerprints remain unchanged after cleanup.

4. **Failed variant auditability**
   - Variant failures now record longer sanitized stderr/stdout tails, failure classification, and last exception line.

## B) Full 4-prompt suite evidence (blocking ask)

Using the fixed `textgen-basic-quality-v1` prompt order through the same runtime batch-worker path:

- **default INT4 (selected Round 6 artifact):** 3/3 complete batches, structural JSON regression present in 3/3, not promotion-eligible.
- **block_size=64 candidate:** 3/3 complete batches, 0/3 structural JSON regressions, promotion-eligible 3/3.

Each full-suite trial persists bounded per-prompt baseline/candidate outputs, batch timings, and full `evaluate_quality_validation` evidence (promotion, recipe verification, model capability, baseline comparison, functional checks).

## C) Block-size paradox evidence

Report now includes a direct cost/perf matrix for default/16/32/64:

- package bytes,
- Olive optimize wall time (when rebuilt),
- Foundry load + generation timing,
- best-effort peak RSS (psutil sampling when available).

Numeric fidelity probe was attempted via bounded OGA next-token trace divergence on a fixed prompt.  
If the probe is unavailable in a runtime/API context, the report marks fidelity unknown with exact error.  
Regardless of probe availability, full-suite strict quality evidence remains the gating guard.

## D) Required conclusion and next gate

- `block_size=64` remains a **SmolLM-targeted candidate** under this follow-up evidence.
- Cross-model generalization is still **unproven** until an unchanged full frozen five-model rerun completes.
- If production direction proceeds, this evidence supports a deterministic hard-capped retry ladder concept for Round 7:
  default build first, retry block64 only on baseline-pass/optimized-structural-regression, max two builds, no model-id logic.

## Artifacts

- Harness: `evaluation/recipe-agent-v1/diagnostics/smollm-json-regression/run_smollm_json_regression_diagnostics.py`
- Report: `evaluation/recipe-agent-v1/diagnostics/smollm-json-regression/diagnostic-report.json`
- Artifact tests: `evaluation/recipe-agent-v1/diagnostics/smollm-json-regression/test_smollm_json_regression_artifacts.py`
