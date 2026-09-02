# Recipe Agent v1 runtime diagnostics (Round 4)

## Outcome

Round 4 runtime failures split into two independent issues:

1. **TinyLlama is a config/graph output mismatch** after Olive (`logits` renamed to `logits_Q4`).
2. **Qwen2 INT4 failures are a runtime capability boundary** in current ORT/OGA when BF16-typed contrib-op scales are produced.

No product source or frozen model-set content was modified in this session.

## Scope and evidence

- Round 4 evidence: `evaluation/recipe-agent-v1/round-4/**`
- Added deterministic diagnostics: `evaluation/recipe-agent-v1/diagnostics/runtime-round4/**`
- Added machine-readable evidence: `evaluation/recipe-agent-v1/diagnostics/runtime-round4/runtime_round4_evidence.json`
- Upstream references inspected:
  - Olive RTN quantization implementation (`olive-ai==0.13.0`):
    - `olive/passes/onnx/rtn_quantization.py`
    - `olive/passes/onnx/model_builder.py`
  - ORT contrib CPU kernels (`onnxruntime==1.29.0`):
    - `onnxruntime/contrib_ops/cpu/quantization/gather_block_quantized.cc`
    - `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc`
    - `onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc`

## TinyLlama diagnosis

### What failed

- Round 4 signature: `Model output was not found: logits`

### What changed across stages

- Mobius ONNX output includes: `logits`
- Olive ONNX output includes: `logits_Q4`
- `genai_config.json` still mapped `model.decoder.outputs.logits` to `logits`

### Generic reconciliation test (no model-ID logic)

- A generic output-reconciliation helper matched missing decoder outputs to a unique quantized suffix variant (`<name>_Q<bits>`).
- Applying only this mapping update (`logits -> logits_Q4`) made both:
  - runtime gate pass (ONNX checker + ORT CPU load + OGA generation), and
  - Foundry Local inference pass.

### Root-cause class

- **generic config/artifact reconciliation**

## Qwen2 diagnosis (rebuilt variant: Qwen2-0.5B)

### What failed

- Round 4 signatures for both Qwen2-1.5B and Qwen2-0.5B:
  `GatherBlockQuantized(1) NOT_IMPLEMENTED` at `model/embed_tokens/Gather_node_0_Q4`

### Which Olive path creates the failing op

- Olive ONNX RTN quantization (`OnnxBlockWiseRtnQuantization`) quantizes:
  - `MatMul -> MatMulNBits`
  - `Gather -> GatherBlockQuantized`

### Runtime capability boundary observed

- Default Qwen path (Mobius dtype unset, Olive INT4):
  - graph contains `GatherBlockQuantized` and `MatMulNBits`
  - sampled scale dtype is `BFLOAT16`
  - runtime fails on `GatherBlockQuantized(1) NOT_IMPLEMENTED`

- Generic rule test: exclude all Gather nodes (pass-level `nodes_to_exclude`)
  - removes `GatherBlockQuantized`
  - `MatMulNBits` remains with BF16 scales
  - runtime still fails (`MatMulNBits(1) NOT_IMPLEMENTED`)

- Generic rule test: convert remaining `MatMulNBits` to QDQ
  - removes `MatMulNBits`, introduces `DequantizeLinear`
  - runtime still fails (`DequantizeLinear(24) NOT_IMPLEMENTED`)

- Mobius `--dtype f32` + Olive INT4:
  - quantized contrib-op scales become `FLOAT`
  - runtime gate passes with INT4 preserved

### ORT 1.29 source evidence

- `GatherBlockQuantized` CPU kernel constrains scale type to float/fp16 and has explicit BF16 not-implemented behavior.
- `MatMulNBits` kernels are registered for `T1=float` and `T1=MLFloat16` (not BF16).
- Kernel registry shows typed contrib registrations for float/fp16 variants, not BF16.

### Olive config-surface evidence

- `olive optimize` ONNX flow generated only `OnnxPeepholeOptimizer` + `OnnxBlockWiseRtnQuantization` passes in saved config.
- `--extra_mb_options int4_op_types_to_quantize=MatMul` did not change this ONNX RTN pass behavior in the tested flow.
- Pass-level node include/exclude control is available via `olive run-pass` for `onnxblockwisertnquantization`.

### Root-cause classes

- **missing ORT/OGA source/runtime capability** (BF16-path contrib execution)
- **architecture-level Olive rule** (default RTN quantization behavior + current optimize surface)

## Classification summary

| Issue | Category |
| --- | --- |
| TinyLlama decoder output mapping mismatch after Olive | generic config/artifact reconciliation |
| Qwen INT4 default path emits BF16-typed contrib-op scales not executable in current ORT/OGA runtime | missing ORT/OGA source/runtime capability |
| Olive ONNX RTN default quantizes Gather + MatMul; production optimize path does not expose pass-level include/exclude controls | architecture-level Olive rule |
| System Python lacked required packages while Round 4 used dedicated venv (repro setup risk only) | environment/package issue |

## Smallest generic fix and owner

1. **TinyLlama-class fix (owner: fl-onboarding, source-level)**
   - Add a generic post-Olive, pre-packaging reconciliation step:
     - compare final ONNX graph outputs to `genai_config.json` decoder output mapping,
     - auto-remap only when missing output has exactly one deterministic quantized-suffix match,
     - fail closed on ambiguity.
   - Most direct integration point:
     - `src/fl_model_onboarding/production_runner.py` immediately after Olive output selection (`source_dir = olive_dir`) and before packaging/runtime validation.

2. **Qwen-class fix without precision fallback (owner: fl-onboarding, source-level)**
   - Keep INT4 as requested (no silent fallback to fp16/fp32 precision).
   - Ensure build dtype is explicit `f32` for current tool-supported-unverified INT4 families until ORT/OGA BF16 contrib support exists.
   - Concrete control points:
     - capability data: `config/architecture-capabilities.json` (`qwen`/`phi` currently have `mobius_rules.dtype: null`)
     - compiler propagation: `src/fl_model_onboarding/recipe_compiler.py`
     - Mobius argv emission: `src/fl_model_onboarding/adapters/mobius_cli.py` and `src/fl_model_onboarding/production_runner.py`

3. **Optional Olive-surface enhancement (owner: fl-onboarding + Olive integration)**
   - If Gather exclusion is needed as a tunable policy, expose pass-level RTN controls in production flow (not model-ID branching).
   - Current adapter surface uses `olive optimize` only:
     - `src/fl_model_onboarding/adapters/olive_cli.py`

4. **Upstream runtime capability path (owner: ORT/OGA upstream)**
   - Add BF16-compatible implementations/registrations for affected contrib/dequantization paths, or
   - provide deterministic preflight rejection for unsupported BF16 INT4 graphs to avoid late runtime failure.

## INT4 requirement vs AUTO semantics

- Keep explicit INT4 behavior as the contract for these candidates.
- Do **not** silently downgrade precision to pass runtime.
- If INT4 cannot be satisfied under current runtime/toolchain for an architecture family, mark capability status accordingly instead of implicit precision fallback.

## Deterministic diagnostic tests added

- `evaluation/recipe-agent-v1/diagnostics/runtime-round4/test_runtime_round4_diagnostics.py`
  - validates Round 4 scope signatures for TinyLlama and Qwen2,
  - validates TinyLlama reconciliation evidence,
  - validates Qwen experiment matrix and capability boundary,
  - validates generic reconciliation logic.

- Helper scripts:
  - `evaluation/recipe-agent-v1/diagnostics/runtime-round4/reconcile_genai_outputs.py`
  - `evaluation/recipe-agent-v1/diagnostics/runtime-round4/inspect_quantized_onnx.py`

## Full-set rerun requirements

1. **After TinyLlama reconciliation fix:** rerun full frozen Round 4 set (all 5) because packaging/runtime validation contract is shared.
2. **After Qwen dtype-policy/source fix:** rerun at minimum both Qwen frozen revisions, then full frozen Round 4 set for baseline comparability.
3. **After any ORT/OGA runtime upgrade:** rerun full frozen Round 4 set end-to-end because runtime kernel surface changed.
4. Promotion remains gated by all existing checks (Mobius, Olive, ONNX, ORT, OGA, Foundry inference, quality). No gate weakening.

## External workspace cleanup and retained size

- Deleted external diagnostic workspace root: `C:\fmo-r4-diag`
  - freed: **13,535,209,283 bytes** (~12.607 GiB)
- Deleted local Olive cache created during diagnostics: `.olive-cache`
  - freed: **10,282,937,120 bytes** (~9.577 GiB)
- Total freed by this diagnosis session: **23,818,146,403 bytes** (~22.184 GiB)
- Retained pinned Round 4 cache root: `C:\fmo-r4`
  - retained: **13,934,208,209 bytes** (~12.980 GiB)
