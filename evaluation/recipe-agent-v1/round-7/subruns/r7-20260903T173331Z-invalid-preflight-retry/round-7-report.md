# Recipe Agent v1 Round 7 Report

- **Run ID:** `r7-20260903T173331Z`
- **Branch:** `wayne-ch-linus-running-round-7`
- **Commit:** `c107d5a8e130b2d574438990e17dac2b4e38dc61`
- **Window (UTC):** `2026-09-03T17:33:31.511996+00:00` -> `2026-09-03T17:50:04.727215+00:00`
- **valid_baseline:** `True`
- **Recipe Verification (winner-selected):** `1/5`
- **Model Capability (all checks passed):** `0/5`
- **Selected-candidate reuse zero-build evidence:** `1/5`

## Recipe Verification (winner-selected)

| Model | Workflow outcome | Winner candidate | Winner recipe status | First request Mobius/Olive | Reuse 0-build |
| --- | --- | --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | selected | 0:default-int4 | verified | 1/1 | True |
| HuggingFaceTB/SmolLM2-360M-Instruct | exhausted | - | blocked | 1/1 | False |
| Qwen/Qwen2-1.5B-Instruct | exhausted | - | None | None/None | False |
| Qwen/Qwen2-0.5B-Instruct | exhausted | - | None | None/None | False |
| ibm-granite/granite-3.2-2b-instruct | exhausted | - | None | None/None | False |

## Model Capability (non-blocking advisory)

| Model | Checks passed | Confidence | Warnings |
| --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0/4 | low | arithmetic-addition-17-plus-28:shared_capability_failure, factual-red-planet:divergent_capability_failure, instruction-two-words-blue-river:divergent_capability_failure, format-json-answer-unit:shared_capability_failure |
| HuggingFaceTB/SmolLM2-360M-Instruct | - | - | - |
| Qwen/Qwen2-1.5B-Instruct | - | - | - |
| Qwen/Qwen2-0.5B-Instruct | - | - | - |
| ibm-granite/granite-3.2-2b-instruct | - | - | - |

## Retry and dispatch evidence

- Retry expected only for `HuggingFaceTB/SmolLM2-360M-Instruct`: expected `1`, actual `0`.
- First-request aggregate invocation totals: Mobius=2, Olive=2.
- Selected-candidate reuse with measured zero dispatch (5-model denominator): `1/5`.

## Cleanup and process evidence

- Workspace cleanup after durable evidence extraction: retained=1, deleted=0, blocked=0, freed_bytes=0.
- Final lingering process count under runtime root: `0`.
- Branch source identity: module_under_repo_root=True, distribution_under_repo_root=False, expected_commit=c107d5a8e130b2d574438990e17dac2b4e38dc61.

## Delta from Round 6

- Round 6 Recipe Verification: `4/5` -> Round 7: `1/5`.
- Explanation: Round 7 counts recipe verification by selected winner candidate (including quality-retry fallback), while preserving parent-attempt-only rate in parent_attempt_recipe_verified_rate.
- Remaining path to 5/5:
  - `HuggingFaceTB/SmolLM2-360M-Instruct` -> status `blocked`; next action: Resolve quality prompt failures before attempting promotion.
  - `Qwen/Qwen2-1.5B-Instruct` -> status `unknown`; next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
  - `Qwen/Qwen2-0.5B-Instruct` -> status `unknown`; next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
  - `ibm-granite/granite-3.2-2b-instruct` -> status `unknown`; next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
