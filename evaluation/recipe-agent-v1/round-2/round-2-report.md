# Recipe Agent v1 Round 2 Report

- **Run ID:** `r2-20260902t1950z`
- **Branch:** `wayne-ch-linus-fixing-generated-execution`
- **Commit:** `dc14760f72273bba5abb7cdbcc22b98d602b054a`
- **Window (UTC):** `2026-09-02T02:50:24.392816+00:00` -> `2026-09-02T02:59:21.550459+00:00`
- **Round classification:** `valid_baseline`
- **Baseline valid evidence:** `True`
- **Model success rate:** **0/5**
- **Retained external evidence root:** `scratch://round-2/r2-20260902t1950z`

## Frozen manifest and deterministic checks

- Manifest invariants pass: **True**
- `recipe-agent frozen-validate` exit code: **0**
- `recipe-agent frozen-dry-run` exit code: **0**

## Deterministic quality profile snapshot

- Profile: `textgen-basic-quality-v1` v`1.0.0` for task `text-generation`
- Deterministic inference config: `{"temperature": 0.0, "seed": 17, "max_tokens": 64}`
- Runtime-reported unsupported deterministic fields: `temperature, seed`

## Per-model outcomes

| Model | Attempt state | First failed stage | Classification | Prior passed gates |
| --- | --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | failed | mobius_building | gate_failed | - |
| HuggingFaceTB/SmolLM2-360M-Instruct | failed | mobius_building | gate_failed | - |
| Qwen/Qwen2-1.5B-Instruct | failed | mobius_building | gate_failed | - |
| Qwen/Qwen2-0.5B-Instruct | failed | mobius_building | gate_failed | - |
| ibm-granite/granite-3.2-2b-instruct | failed | mobius_building | gate_failed | - |

## Failure analysis

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0**
   - First failed stage/classification: `mobius_building` / `gate_failed`
   - Error signature: `Mobius build failed: ...uild _save_package(pkg, output_dir, args, optimize, component_filter) File "<redacted-absolute-path>", line 330, in _save_package artifacts = write_ort_genai_config( ^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 1119, in write_ort_genai_config tokenizer_files = _copy_tokenizer_files(hf_model_id, directory) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 224, in _copy_tokenizer_files src = hf_hub_download(model_id, filename) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 84, in _inner_fn validate_repo_id(arg_value) File "<redacted-absolute-path>", line 138, in validate_repo_id raise HFValidationError( huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: 'scratch://round-2/r2-20260902t1950z\workspace\1b0e2955-ab0f-4079-a3fa-88ce740ba43e\snapshot'.`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `mobius_build` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://1b0e2955-ab0f-4079-a3fa-88ce740ba43e`
   - Generated recipe provenance: fingerprint `fde0420b626832d11a9138f3b69ae9f43713cb8c8c67446f350d8c7a51b98577`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `3ba0f69d071bb52253c9dcd2d62030ebd4871fa2e686b5e162fd18597a1d74d9`
2. **HuggingFaceTB/SmolLM2-360M-Instruct**
   - First failed stage/classification: `mobius_building` / `gate_failed`
   - Error signature: `Mobius build failed: ...uild _save_package(pkg, output_dir, args, optimize, component_filter) File "<redacted-absolute-path>", line 330, in _save_package artifacts = write_ort_genai_config( ^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 1119, in write_ort_genai_config tokenizer_files = _copy_tokenizer_files(hf_model_id, directory) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 224, in _copy_tokenizer_files src = hf_hub_download(model_id, filename) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 84, in _inner_fn validate_repo_id(arg_value) File "<redacted-absolute-path>", line 138, in validate_repo_id raise HFValidationError( huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: 'scratch://round-2/r2-20260902t1950z\workspace\f04412df-6216-4d2b-8101-9d90f7cc600f\snapshot'.`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `mobius_build` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://f04412df-6216-4d2b-8101-9d90f7cc600f`
   - Generated recipe provenance: fingerprint `2c7616bcb5c377210ee24ce9a0a199dfaab3a587b5fb529be63ee5579f464e46`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `32ebdc1fd1a4f5a646626403301350b9b3934035cf66a4e2ea77ab554a2f6986`
3. **Qwen/Qwen2-1.5B-Instruct**
   - First failed stage/classification: `mobius_building` / `gate_failed`
   - Error signature: `Mobius build failed: ...uild _save_package(pkg, output_dir, args, optimize, component_filter) File "<redacted-absolute-path>", line 330, in _save_package artifacts = write_ort_genai_config( ^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 1119, in write_ort_genai_config tokenizer_files = _copy_tokenizer_files(hf_model_id, directory) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 224, in _copy_tokenizer_files src = hf_hub_download(model_id, filename) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 84, in _inner_fn validate_repo_id(arg_value) File "<redacted-absolute-path>", line 138, in validate_repo_id raise HFValidationError( huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: 'scratch://round-2/r2-20260902t1950z\workspace\5c1201dd-5abd-4c91-9f7e-76265a37b159\snapshot'.`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `mobius_build` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://5c1201dd-5abd-4c91-9f7e-76265a37b159`
   - Generated recipe provenance: fingerprint `39bae0ace88df5c288b46133af71b37207255f0cb46c96d23fb3cfbfeaae8229`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `264dff405d4ad4493de5306103d659511ede8df54b41023314f7b4dc86143192`
4. **Qwen/Qwen2-0.5B-Instruct**
   - First failed stage/classification: `mobius_building` / `gate_failed`
   - Error signature: `Mobius build failed: ...uild _save_package(pkg, output_dir, args, optimize, component_filter) File "<redacted-absolute-path>", line 330, in _save_package artifacts = write_ort_genai_config( ^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 1119, in write_ort_genai_config tokenizer_files = _copy_tokenizer_files(hf_model_id, directory) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 224, in _copy_tokenizer_files src = hf_hub_download(model_id, filename) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 84, in _inner_fn validate_repo_id(arg_value) File "<redacted-absolute-path>", line 138, in validate_repo_id raise HFValidationError( huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: 'scratch://round-2/r2-20260902t1950z\workspace\ba0348a0-18ad-4626-a732-c2c189dfdc20\snapshot'.`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `mobius_build` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://ba0348a0-18ad-4626-a732-c2c189dfdc20`
   - Generated recipe provenance: fingerprint `d08ade5c87402496fef273b32f47843dff2e01722def5df45ededc91202561db`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `83132f6e8d047a396fb1f5a3d6e63fedba4e2781a1404d8c6629b01bf4c64edd`
5. **ibm-granite/granite-3.2-2b-instruct**
   - First failed stage/classification: `mobius_building` / `gate_failed`
   - Error signature: `Mobius build failed: ...uild _save_package(pkg, output_dir, args, optimize, component_filter) File "<redacted-absolute-path>", line 330, in _save_package artifacts = write_ort_genai_config( ^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 1119, in write_ort_genai_config tokenizer_files = _copy_tokenizer_files(hf_model_id, directory) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 224, in _copy_tokenizer_files src = hf_hub_download(model_id, filename) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "<redacted-absolute-path>", line 84, in _inner_fn validate_repo_id(arg_value) File "<redacted-absolute-path>", line 138, in validate_repo_id raise HFValidationError( huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: 'scratch://round-2/r2-20260902t1950z\workspace\faf90eff-86c2-4099-9558-71fe9143bb13\snapshot'.`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `mobius_build` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key.
   - Evidence refs: `job://faf90eff-86c2-4099-9558-71fe9143bb13`
   - Generated recipe provenance: fingerprint `ff3f3668f8b2c4f06afcd00ebb2d470e04978813411b7e2f0d13eacce2e1540d`, capability `d8556c39d05d7ef454c9a729be58eb03261c8873ffcfcb4e081ea5309698c2c2`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `a96ea0e7d7095996594001bd398b38e5fd47d77ed8118093218dfcdba424c14d`

## Verified recipe reuse re-check (no rebuild)

- No successful models to re-check for verified reuse.

## Scratch retention

- Runtime root retained size: **30.374 GiB**
- Cache retained size: **0.000 GiB**
- Workspace retained size: **30.374 GiB**
- State retained size: **0.000 GiB**
- Paths represented in committed artifacts by `scratch://round-2/<run_id>/...` placeholders.
