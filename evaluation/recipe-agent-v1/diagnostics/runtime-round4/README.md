# Runtime Round 4 diagnostics (TinyLlama + Qwen2)

This folder contains deterministic diagnostics for the Round 4 runtime failures:

- TinyLlama runtime validation failing with `Model output was not found: logits`.
- Qwen2 runtime validation failing with `GatherBlockQuantized(1) NOT_IMPLEMENTED`.

## Artifacts

- `runtime_round4_evidence.json`  
  Sanitized machine-readable evidence from the scoped reproduction runs.
- `reconcile_genai_outputs.py`  
  Generic decoder output reconciliation helper (`genai_config.json` vs ONNX graph outputs), with no model-ID branching.
- `inspect_quantized_onnx.py`  
  Generic ONNX quantized-op inspector (counts `GatherBlockQuantized`, `MatMulNBits`, `DequantizeLinear`, and reports scale dtypes).
- `test_runtime_round4_diagnostics.py`  
  Deterministic tests asserting scope signatures and validated root-cause evidence.

## Local execution

```powershell
python -m pytest evaluation/recipe-agent-v1/diagnostics/runtime-round4/test_runtime_round4_diagnostics.py
```
