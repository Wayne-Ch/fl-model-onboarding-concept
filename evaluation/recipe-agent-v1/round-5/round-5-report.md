# Recipe Agent v1 Round 5 Report

- **Run ID:** `r5-0902d`
- **Branch:** `wayne-ch-linus-applying-round-5-fixes`
- **Commit:** `ab9a9a41f24a427cb072cb04e3b85090a65eba29`
- **Window (UTC):** `2026-09-02T11:26:19.105195+00:00` -> `2026-09-02T13:48:28.297061+00:00`
- **Round classification:** `valid_baseline`
- **Baseline valid evidence:** `True`
- **Model success rate:** **0/5**
- **Retained external evidence root:** `scratch://round-5/r5-0902d`

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

| Model | Attempt state | First failed stage | Classification | Prior passed gates | Quality evidence |
| --- | --- | --- | --- | --- | --- |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | failed | succeeded | validation_failed | mobius_build, olive_optimize, onnx_validation, ort_validation, oga_validation, fl_sdk_inference | metrics-ref-missing |
| HuggingFaceTB/SmolLM2-360M-Instruct | failed | succeeded | validation_failed | mobius_build, olive_optimize, onnx_validation, ort_validation, oga_validation, fl_sdk_inference | loaded (3 prompt check failures) |
| Qwen/Qwen2-1.5B-Instruct | failed | succeeded | validation_failed | mobius_build, olive_optimize, onnx_validation, ort_validation, oga_validation, fl_sdk_inference | loaded (1 prompt check failures) |
| Qwen/Qwen2-0.5B-Instruct | failed | succeeded | validation_failed | mobius_build, olive_optimize, onnx_validation, ort_validation, oga_validation, fl_sdk_inference | loaded (2 prompt check failures) |
| ibm-granite/granite-3.2-2b-instruct | failed | succeeded | validation_failed | mobius_build, olive_optimize, onnx_validation, ort_validation, oga_validation, fl_sdk_inference | loaded (1 prompt check failures) |

## Failure analysis

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0**
   - First failed stage/classification: `succeeded` / `validation_failed`
   - Error signature: `Baseline quality prompt 'factual-red-planet' failed: Command timed out after 900s: ('<redacted-absolute-path> Files\\Python311\\python.exe', '-m', 'fl_model_onboarding.runtime_worker', 'foundry-infer', '--model-dir', '<redacted-absolute-path>', '--model-name', 'tinyllama-1-1b-chat-v1-0-onboarding-30c0d2e5-338-mobius-baseline:1', '--request-file', '<redacted-absolute-path>')`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `quality_validation` with status `unavailable`
   - Source owner: `fl-onboarding`
   - Next action: Ensure a pre-Olive Mobius baseline package can run deterministic prompt validation before retrying promotion.
   - Evidence refs: `job://30c0d2e5-3386-4f96-bfe7-cd2e10886afd`
   - Generated recipe provenance: fingerprint `fde0420b626832d11a9138f3b69ae9f43713cb8c8c67446f350d8c7a51b98577`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `3ba0f69d071bb52253c9dcd2d62030ebd4871fa2e686b5e162fd18597a1d74d9`
   - Quality evidence status: `metrics-ref-missing`
2. **HuggingFaceTB/SmolLM2-360M-Instruct**
   - First failed stage/classification: `succeeded` / `validation_failed`
   - Error signature: `Quality validation failed: arithmetic-addition-17-plus-28:exact_match_failed|max_words_exceeded; instruction-two-words-blue-river:max_words_exceeded; format-json-answer-unit:forbidden_token_present:```|json_format_invalid; baseline:baseline_failed_functional_checks|optimized_failed_prompt:format-json-answer-unit`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `quality_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Resolve quality prompt failures before attempting promotion.
   - Evidence refs: `job://7256f663-6d12-4e92-882d-dd7454066fdb`
   - Generated recipe provenance: fingerprint `2c7616bcb5c377210ee24ce9a0a199dfaab3a587b5fb529be63ee5579f464e46`, capability `286e622f6ffa4db8a93d3231939f8b0b12a25355e72c43f121e39c961c02b3a7`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `32ebdc1fd1a4f5a646626403301350b9b3934035cf66a4e2ea77ab554a2f6986`
   - Quality evidence status: `loaded`
   - Quality metrics ref: `quality-metrics://7256f663-6d12-4e92-882d-dd7454066fdb/quality-validation-evidence.json`
   - Bounded quality prompt evidence:
     - `arithmetic-addition-17-plus-28` baseline_passed=False optimized_passed=False
       baseline: `17 + 28 = 45`
       optimized: `17 + 28 = 45`
     - `instruction-two-words-blue-river` baseline_passed=False optimized_passed=False
       baseline: `River`
       optimized: `The river is blue.`
     - `format-json-answer-unit` baseline_passed=True optimized_passed=False
       baseline: `{\n  "answer": 12,\n  "unit": "cm"\n}`
       optimized: `Yes, I can assist with that. Here is the valid JSON object with the specified keys and unit:\n\n```json\n{\n  "answer": 12,\n  "unit": "cm"\n}\n````
3. **Qwen/Qwen2-1.5B-Instruct**
   - First failed stage/classification: `succeeded` / `validation_failed`
   - Error signature: `Quality validation failed: arithmetic-addition-17-plus-28:exact_match_failed|relevance_keyword_missing; baseline:baseline_failed_functional_checks`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `quality_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Resolve quality prompt failures before attempting promotion.
   - Evidence refs: `job://0c84301c-a7ea-4c8d-a920-c180707a75b9`
   - Generated recipe provenance: fingerprint `e1abc6ce982ec5617dd8d0c58d2e074406d324acba8de975c421162e9b2007ae`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `4f80aa828f34f0dbf110443195a7322331b1f88eae67b04431d69dea6178374c`
   - Quality evidence status: `loaded`
   - Quality metrics ref: `quality-metrics://0c84301c-a7ea-4c8d-a920-c180707a75b9/quality-validation-evidence.json`
   - Bounded quality prompt evidence:
     - `arithmetic-addition-17-plus-28` baseline_passed=False optimized_passed=False
       baseline: `1728`
       optimized: `1728`
4. **Qwen/Qwen2-0.5B-Instruct**
   - First failed stage/classification: `succeeded` / `validation_failed`
   - Error signature: `Quality validation failed: arithmetic-addition-17-plus-28:exact_match_failed|max_words_exceeded; instruction-two-words-blue-river:max_words_exceeded; baseline:baseline_failed_functional_checks`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `quality_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Resolve quality prompt failures before attempting promotion.
   - Evidence refs: `job://ee7b3072-0706-4120-803d-e50ad04988c2`
   - Generated recipe provenance: fingerprint `d089580184b636ec9847ee55d60ec31afcbb63f8bad2e6bf559cfcb224da8895`, capability `9dfacd17b15f8427b0a5f8c319bdd1225a5179b818b084188373cff55dc1e9f4`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `40148bef0c7133a7aa0441d52f8ce03ecc17c15d85cd292a5834c22432646fb1`
   - Quality evidence status: `loaded`
   - Quality metrics ref: `quality-metrics://ee7b3072-0706-4120-803d-e50ad04988c2/quality-validation-evidence.json`
   - Bounded quality prompt evidence:
     - `arithmetic-addition-17-plus-28` baseline_passed=False optimized_passed=False
       baseline: `35`
       optimized: `The sum of 17 and 28 is 45.`
     - `instruction-two-words-blue-river` baseline_passed=False optimized_passed=False
       baseline: `Blue River, River of the Blue Mountains`
       optimized: `Blue River, River of the Blue.`
5. **ibm-granite/granite-3.2-2b-instruct**
   - First failed stage/classification: `succeeded` / `validation_failed`
   - Error signature: `Quality validation failed: arithmetic-addition-17-plus-28:exact_match_failed|relevance_keyword_missing; baseline:baseline_failed_functional_checks`
   - Preflight recipe status: `unregistered` (expected in generated-recipe preflight; not an environment blocker).
   - First failed gate: `quality_validation` with status `failed`
   - Source owner: `fl-onboarding`
   - Next action: Resolve quality prompt failures before attempting promotion.
   - Evidence refs: `job://70b5003d-4d83-40a6-a81a-a0b79399b5fd`
   - Generated recipe provenance: fingerprint `ff3f3668f8b2c4f06afcd00ebb2d470e04978813411b7e2f0d13eacce2e1540d`, capability `d8556c39d05d7ef454c9a729be58eb03261c8873ffcfcb4e081ea5309698c2c2`, toolchain `fa0e873479662bc7af42469c6a64b8981dc9e696eda2e201229beab3107f235d`, profile `a96ea0e7d7095996594001bd398b38e5fd47d77ed8118093218dfcdba424c14d`
   - Quality evidence status: `loaded`
   - Quality metrics ref: `quality-metrics://70b5003d-4d83-40a6-a81a-a0b79399b5fd/quality-validation-evidence.json`
   - Bounded quality prompt evidence:
     - `arithmetic-addition-17-plus-28` baseline_passed=False optimized_passed=False
       baseline: `35`
       optimized: `25`

## Verified recipe reuse re-check (no rebuild)

- No successful models to re-check for verified reuse.

## Scratch retention

- Runtime root retained size: **15.955 GiB**
- Cache retained size: **15.955 GiB**
- Workspace retained size: **0.000 GiB**
- State retained size: **0.000 GiB**
- Current-run failed workspace cleanup: deleted=5, retained-debug-opt-in=0, blocked=0, freed=55.254 GiB
- Obsolete failed workspace cleanup: not requested.
- Paths represented in committed artifacts by `scratch://round-5/<run_id>/...` placeholders.
