# Recipe Agent v1 Round 7 Report

- **Run ID:** `r7-20260903T203830Z`
- **Branch:** `wayne-ch-linus-running-round-7`
- **Commit:** `ba056fce7fcd91fd4284ea0a5f630bdd70944a0f`
- **Window (UTC):** `2026-09-03T20:38:31.598548+00:00` -> `2026-09-03T20:57:51.143983+00:00`
- **valid_baseline:** `True`
- **Recipe Verification (winner-selected):** `5/5`
- **Model Capability (all checks passed):** `0/5`
- **Selected-candidate reuse zero-build evidence:** `5/5`

## Recipe Verification (winner-selected)

| Model | Workflow outcome | Winner candidate | Winner recipe status | First request Mobius/Olive | Reuse 0-build |
| --- | --- | --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | selected | 0:default-int4 | verified | 1/1 | True |
| HuggingFaceTB/SmolLM2-360M-Instruct | selected | 1:int4-block-size-64 | verified | 1/2 | True |
| Qwen/Qwen2-1.5B-Instruct | selected | 0:default-int4 | verified | 1/1 | True |
| Qwen/Qwen2-0.5B-Instruct | selected | 0:default-int4 | verified | 1/1 | True |
| ibm-granite/granite-3.2-2b-instruct | selected | 0:default-int4 | verified | 1/1 | True |

## Model Capability (non-blocking advisory)

| Model | Checks passed | Confidence | Warnings |
| --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0/4 | low | arithmetic-addition-17-plus-28:shared_capability_failure, factual-red-planet:divergent_capability_failure, instruction-two-words-blue-river:divergent_capability_failure, format-json-answer-unit:shared_capability_failure |
| HuggingFaceTB/SmolLM2-360M-Instruct | 2/4 | low | arithmetic-addition-17-plus-28:shared_capability_failure, instruction-two-words-blue-river:divergent_capability_failure |
| Qwen/Qwen2-1.5B-Instruct | 3/4 | low | arithmetic-addition-17-plus-28:shared_capability_failure |
| Qwen/Qwen2-0.5B-Instruct | 2/4 | low | arithmetic-addition-17-plus-28:divergent_capability_failure, instruction-two-words-blue-river:shared_capability_failure |
| ibm-granite/granite-3.2-2b-instruct | 3/4 | low | arithmetic-addition-17-plus-28:shared_capability_failure |

## Retry and dispatch evidence

- Retry expected only for `HuggingFaceTB/SmolLM2-360M-Instruct`: expected `1`, actual `1`.
- First-request aggregate invocation totals: Mobius=5, Olive=6.
- Selected-candidate reuse with measured zero dispatch (5-model denominator): `5/5`.

## Cleanup and process evidence

- Workspace cleanup after durable evidence extraction: retained=5, deleted=0, blocked=0, freed_bytes=0.
- Final lingering process count under runtime root: `0`.
- Branch source identity: module_under_repo_root=True, distribution_under_repo_root=False, expected_commit=ba056fce7fcd91fd4284ea0a5f630bdd70944a0f.

## Delta from Round 6

- Round 6 Recipe Verification: `4/5` -> Round 7: `5/5`.
- Explanation: Round 7 counts recipe verification by selected winner candidate (including quality-retry fallback), while preserving parent-attempt-only rate in parent_attempt_recipe_verified_rate.
- Remaining path to 5/5: none.
