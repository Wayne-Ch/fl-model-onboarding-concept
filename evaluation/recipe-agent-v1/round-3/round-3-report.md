# Recipe Agent v1 Round 3 Report

- **Run ID:** `r3-20260902T031122Z`
- **Branch:** `wayne-ch-linus-fixing-generated-execution`
- **Commit:** `5a9f2c1cd2b778c4a4a6c92caeedc979744f92a0`
- **Window (UTC):** `2026-09-02T03:11:22.332953+00:00` -> `2026-09-02T03:23:16.729118+00:00`
- **Round classification:** `valid_baseline`
- **Baseline valid evidence:** `True`
- **Model success rate:** **0/5**
- **Retained external evidence root:** `scratch://round-3/r3-20260902T031122Z`

## Frozen manifest and deterministic checks

- Manifest invariants pass: **True**
- `recipe-agent frozen-validate` exit code: **0**
- `recipe-agent frozen-dry-run` exit code: **0**
- Round 2 snapshot reuse seed status:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea1ffedaf415f4da2f062534de366a451e6` => `junction`
- `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951` => `junction`
- `Qwen/Qwen2-1.5B-Instruct` @ `ba1cf1846d7df0a0591d6c00649f57e798519da8` => `junction`
- `Qwen/Qwen2-0.5B-Instruct` @ `c540970f9e29518b1d8f06ab8b24cba66ad77b6d` => `junction`
- `ibm-granite/granite-3.2-2b-instruct` @ `641593c3b25bec0b1efe9f0f7d7a67f7243f86a3` => `junction`

## Deterministic quality profile snapshot

- Profile: `textgen-basic-quality-v1` v`1.0.0` for task `text-generation`
- Deterministic inference config: `{"temperature": 0.0, "seed": 17, "max_tokens": 64}`
- Runtime-reported unsupported deterministic fields: `temperature, seed`

## Per-model outcomes

| Model | Attempt state | First failed stage | Classification | Prior passed gates |
| --- | --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |
| HuggingFaceTB/SmolLM2-360M-Instruct | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |
| Qwen/Qwen2-1.5B-Instruct | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |
| Qwen/Qwen2-0.5B-Instruct | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |
| ibm-granite/granite-3.2-2b-instruct | failed | runtime_validating | gate_failed | mobius_build, olive_optimize |

## Failure analysis

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: <python-exe>: Error while finding module specification for 'fl_model_onboarding.runtime_worker' (ModuleNotFoundError: No module named 'fl_model_onboarding')`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://7048867b-8807-4196-8337-2d0d113f36a0`
   - Generated recipe provenance: fingerprint `fde0420b626832d11a9138f3b69ae9f43713cb8c8c67446f350d8c7a51b98577`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `3ba0f69d071bb52253c9dcd2d62030ebd4871fa2e686b5e162fd18597a1d74d9`
2. **HuggingFaceTB/SmolLM2-360M-Instruct**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: <python-exe>: Error while finding module specification for 'fl_model_onboarding.runtime_worker' (ModuleNotFoundError: No module named 'fl_model_onboarding')`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://6f602ac4-830c-4bf8-af56-14b95da00bc9`
   - Generated recipe provenance: fingerprint `2c7616bcb5c377210ee24ce9a0a199dfaab3a587b5fb529be63ee5579f464e46`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `32ebdc1fd1a4f5a646626403301350b9b3934035cf66a4e2ea77ab554a2f6986`
3. **Qwen/Qwen2-1.5B-Instruct**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: <python-exe>: Error while finding module specification for 'fl_model_onboarding.runtime_worker' (ModuleNotFoundError: No module named 'fl_model_onboarding')`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://fd43a3a7-5614-4c97-8682-f018a49a763d`
   - Generated recipe provenance: fingerprint `39bae0ace88df5c288b46133af71b37207255f0cb46c96d23fb3cfbfeaae8229`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `264dff405d4ad4493de5306103d659511ede8df54b41023314f7b4dc86143192`
4. **Qwen/Qwen2-0.5B-Instruct**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: <python-exe>: Error while finding module specification for 'fl_model_onboarding.runtime_worker' (ModuleNotFoundError: No module named 'fl_model_onboarding')`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://2d52b7bd-89ed-48ac-af47-d903cb335d3d`
   - Generated recipe provenance: fingerprint `d08ade5c87402496fef273b32f47843dff2e01722def5df45ededc91202561db`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `83132f6e8d047a396fb1f5a3d6e63fedba4e2781a1404d8c6629b01bf4c64edd`
5. **ibm-granite/granite-3.2-2b-instruct**
   - First failed stage/classification: `runtime_validating` / `gate_failed`
   - Error signature: `Runtime validation failed: <python-exe>: Error while finding module specification for 'fl_model_onboarding.runtime_worker' (ModuleNotFoundError: No module named 'fl_model_onboarding')`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `onnx_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://c79323d3-ac76-4241-b20b-d8e7ae6be84e`
   - Generated recipe provenance: fingerprint `ff3f3668f8b2c4f06afcd00ebb2d470e04978813411b7e2f0d13eacce2e1540d`, capability `d8556c39d05d7ef454c9a729be58eb03261c8873ffcfcb4e081ea5309698c2c2`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `a96ea0e7d7095996594001bd398b38e5fd47d77ed8118093218dfcdba424c14d`

## Verified recipe reuse re-check (no rebuild)

- No successful models to re-check for verified reuse.

## Scratch retention

- Runtime root retained size: **57.433 GiB**
- Cache retained size: **11.269 GiB**
- Workspace retained size: **46.164 GiB**
- State retained size: **0.000 GiB**
- Paths represented in committed artifacts by `scratch://round-3/<run_id>/...` placeholders.
