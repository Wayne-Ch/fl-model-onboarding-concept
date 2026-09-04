# Recipe Agent v1 Round 4 Report

- **Run ID:** `r4-20260902T083935Z`
- **Branch:** `wayne-ch-linus-fixing-generated-execution`
- **Commit:** `72a87d0abaeed8ac8b5027cc0f7978e321edb94c`
- **Window (UTC):** `2026-09-02T08:39:35.324524+00:00` -> `2026-09-02T08:55:46.059283+00:00`
- **Round classification:** `valid_baseline`
- **Baseline valid evidence:** `True`
- **Model success rate:** **0/5**
- **Retained external evidence root:** `scratch://round-4/r4-20260902T083935Z`

## Frozen manifest and deterministic checks

- Manifest invariants pass: **True**
- `recipe-agent frozen-validate` exit code: **0**
- `recipe-agent frozen-dry-run` exit code: **0**
- Prior pinned snapshot cache seed status:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea1ffedaf415f4da2f062534de366a451e6` => `copied`
- `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951` => `copied`
- `Qwen/Qwen2-1.5B-Instruct` @ `ba1cf1846d7df0a0591d6c00649f57e798519da8` => `copied`
- `Qwen/Qwen2-0.5B-Instruct` @ `c540970f9e29518b1d8f06ab8b24cba66ad77b6d` => `copied`
- `ibm-granite/granite-3.2-2b-instruct` @ `641593c3b25bec0b1efe9f0f7d7a67f7243f86a3` => `copied`

## Deterministic quality profile snapshot

- Profile: `textgen-basic-quality-v1` v`1.0.0` for task `text-generation`
- Deterministic inference config: `{"temperature": 0.0, "seed": 17, "max_tokens": 64}`
- Runtime-reported unsupported deterministic fields: `temperature, seed`

## Per-model outcomes

| Model | Attempt state | First failed stage | Classification | Prior passed gates |
| --- | --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |
| HuggingFaceTB/SmolLM2-360M-Instruct | failed | succeeded | validation_failed | mobius_build, olive_optimize, onnx_validation, ort_validation, oga_validation, fl_sdk_inference |
| Qwen/Qwen2-1.5B-Instruct | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |
| Qwen/Qwen2-0.5B-Instruct | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |
| ibm-granite/granite-3.2-2b-instruct | failed | succeeded | validation_failed | mobius_build, olive_optimize, onnx_validation, ort_validation, oga_validation, fl_sdk_inference |

## Failure analysis

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: {"ok": false, "error": "Model output was not found: logits"}`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://588416f7-7a8f-429f-afe9-08b7b7ec8d30`
   - Generated recipe provenance: fingerprint `fde0420b626832d11a9138f3b69ae9f43713cb8c8c67446f350d8c7a51b98577`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `3ba0f69d071bb52253c9dcd2d62030ebd4871fa2e686b5e162fd18597a1d74d9`
2. **HuggingFaceTB/SmolLM2-360M-Instruct**
   - First failed stage/classification: `succeeded` / `validation_failed`
   - Error signature: `Quality validation failed: arithmetic-addition-17-plus-28:exact_match_failed|max_words_exceeded; instruction-two-words-blue-river:max_words_exceeded; format-json-answer-unit:forbidden_token_present:```|json_format_invalid; baseline:baseline_failed_functional_checks|optimized_failed_prompt:format-json-answer-unit`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `quality_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Resolve quality prompt failures before attempting promotion.
   - Evidence refs: `job://17e1e928-4eff-42ac-a908-dd23328c5a60`
   - Generated recipe provenance: fingerprint `2c7616bcb5c377210ee24ce9a0a199dfaab3a587b5fb529be63ee5579f464e46`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `32ebdc1fd1a4f5a646626403301350b9b3934035cf66a4e2ea77ab554a2f6986`
3. **Qwen/Qwen2-1.5B-Instruct**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: {"ok": false, "error": "[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for GatherBlockQuantized(1) node with name 'model/embed_tokens/Gather_node_0_Q4'"}`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://40064d6c-b183-4e3a-b231-42ae92467c1b`
   - Generated recipe provenance: fingerprint `39bae0ace88df5c288b46133af71b37207255f0cb46c96d23fb3cfbfeaae8229`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `264dff405d4ad4493de5306103d659511ede8df54b41023314f7b4dc86143192`
4. **Qwen/Qwen2-0.5B-Instruct**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: {"ok": false, "error": "[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for GatherBlockQuantized(1) node with name 'model/embed_tokens/Gather_node_0_Q4'"}`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://a01afb5b-6977-4977-afb7-0582b18aa9ee`
   - Generated recipe provenance: fingerprint `d08ade5c87402496fef273b32f47843dff2e01722def5df45ededc91202561db`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `83132f6e8d047a396fb1f5a3d6e63fedba4e2781a1404d8c6629b01bf4c64edd`
5. **ibm-granite/granite-3.2-2b-instruct**
   - First failed stage/classification: `succeeded` / `validation_failed`
   - Error signature: `Quality validation failed: arithmetic-addition-17-plus-28:exact_match_failed|relevance_keyword_missing; baseline:baseline_failed_functional_checks`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `quality_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Resolve quality prompt failures before attempting promotion.
   - Evidence refs: `job://67e5821f-a563-4bb6-a677-7c64ae6e6b54`
   - Generated recipe provenance: fingerprint `ff3f3668f8b2c4f06afcd00ebb2d470e04978813411b7e2f0d13eacce2e1540d`, capability `d8556c39d05d7ef454c9a729be58eb03261c8873ffcfcb4e081ea5309698c2c2`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `a96ea0e7d7095996594001bd398b38e5fd47d77ed8118093218dfcdba424c14d`

## Verified recipe reuse re-check (no rebuild)

- No successful models to re-check for verified reuse.

## Scratch retention

- Runtime root retained size: **12.977 GiB**
- Cache retained size: **12.977 GiB**
- Workspace retained size: **0.000 GiB**
- State retained size: **0.000 GiB**
- Current-run failed workspace cleanup: deleted=5, retained-debug-opt-in=0, blocked=0, freed=46.164 GiB
- Obsolete failed workspace cleanup:
- r2-20260902t1950z: status=completed, deleted=5, blocked=0, freed=30.374 GiB
- r3-20260902T031122Z: status=completed, deleted=5, blocked=0, freed=46.164 GiB
- Total obsolete workspace bytes freed: **76.538 GiB**
- Paths represented in committed artifacts by `scratch://round-4/<run_id>/...` placeholders.
