# Recipe Agent v1 Round 6 Report

- **Run ID:** `r6-20260902T172246Z`
- **Branch:** `wayne-ch-linus-running-round-6`
- **Commit:** `4e07f09f4963db50978e3683a0b796659e84d35a`
- **Window (UTC):** `2026-09-02T17:22:46.199587+00:00` -> `2026-09-02T17:46:17.199596+00:00`
- **valid_baseline:** `True`
- **recipe_verified_count/5:** `4/5`
- **model_capability_all_pass_count/5:** `0/5`
- **Recipe Verification counts:** VERIFIED=4, BLOCKED=1, INCONCLUSIVE=0, UNKNOWN=0

## Recipe Verification (blocking / promotion)

| Model | Status | Gate | Can promote | Runtime functional | Baseline available | Regression free |
| --- | --- | --- | --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | verified | passed | True | True | True | True |
| HuggingFaceTB/SmolLM2-360M-Instruct | blocked | failed | False | True | True | False |
| Qwen/Qwen2-1.5B-Instruct | verified | passed | True | True | True | True |
| Qwen/Qwen2-0.5B-Instruct | verified | passed | True | True | True | True |
| ibm-granite/granite-3.2-2b-instruct | verified | passed | True | True | True | True |

## Model Capability (non-blocking advisory)

| Model | Checks passed | All pass | Confidence |
| --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0/4 | False | low |
| HuggingFaceTB/SmolLM2-360M-Instruct | 1/4 | False | low |
| Qwen/Qwen2-1.5B-Instruct | 3/4 | False | low |
| Qwen/Qwen2-0.5B-Instruct | 2/4 | False | low |
| ibm-granite/granite-3.2-2b-instruct | 3/4 | False | low |

## Reuse verification (post-promotion)

| Model | Reuse identity match | Reuse attempt id | build_invocation_delta |
| --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | True | 5cb1efa6-a052-4e69-a178-e5460ce21502 | 0 |
| Qwen/Qwen2-1.5B-Instruct | True | 50ffeed4-0930-4776-8876-c4e3d5d04fa4 | 0 |
| Qwen/Qwen2-0.5B-Instruct | True | 2229def2-7768-4d3a-81b1-97aad9a81b50 | 0 |
| ibm-granite/granite-3.2-2b-instruct | True | e60c48c2-b976-4e16-95d1-d3705b6e925f | 0 |

## Setup, failures, and cleanup

- Toolchain ready: `True`; missing_required: `[]`
- Failure category counts: `{"validation_failed": 1}`
- Round 5 delta (semantics changed):
  - Round 5 success_rate=0/5 (legacy quality-gate semantics)
  - Round 6 recipe_verified_rate=4/5 and model_capability_all_pass_rate=0/5
  - Note: Round 6 reports split outcomes by product decision: Recipe Verification is blocking/promotion integrity, while Model Capability is non-blocking absolute task quality.
- Cleanup bytes: current_run_freed=3698966337, runtime_bytes=72762007216, cache_bytes=17131789589, state_bytes=241664, workspace_bytes=55629975963
- Lingering process total after per-model cleanup checks: `0`

## Remaining path to 5/5 Recipe Verification

- `HuggingFaceTB/SmolLM2-360M-Instruct` => status `blocked`; next action: Resolve quality prompt failures before attempting promotion.
