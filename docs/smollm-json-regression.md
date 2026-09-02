# SmolLM2 Round 6 JSON regression diagnosis

## Scope

- Target blocker: `HuggingFaceTB/SmolLM2-360M-Instruct` optimized JSON-format regression in Recipe Agent v1 Round 6.
- Frozen identity: `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512eabd3dde888204e902eca88bddb4951`.
- Strict contract preserved: no fence stripping, parser-only acceptance, validator relaxation, or output rewriting.

## Answers to the five blocker questions

1. **Deterministic or variance?**  
   Bounded repeats were deterministic in this toolchain path: baseline JSON prompt stayed valid/unfenced (`6/6`), optimized stayed fenced/invalid (`6/6`), and recipe integrity regression signature reproduced (`6/6`).

2. **Exact outputs/structure/timing/prompt-template inputs?**  
   The fixed quality prompt `format-json-answer-unit` produced:
   - baseline: `{"answer":12,"unit":"cm"}` shape (valid JSON, unfenced),
   - optimized: markdown-prefixed fenced JSON (invalid strict JSON contract due ` ``` ` + parse failure).  
   Bounded timings and per-prompt outputs are captured in `evaluation/recipe-agent-v1/diagnostics/smollm-json-regression/diagnostic-report.json` under `reproduction_single_batch` and `determinism_repeated_trials`.

3. **Which layer introduces the difference?**  
   Evidence points to the **quantized graph layer**:
   - `genai_config.json` and tokenizer/chat-template fingerprints matched baseline vs optimized.
   - Hybrid swap tests showed:
     - baseline package + optimized graph reproduces fenced regression,
     - optimized package + baseline graph returns clean JSON and passes.

4. **Smallest generic fix that keeps CPU INT4 + strict JSON contract?**  
   The strongest tested generic candidate is capability-level Olive INT4 policy `--block_size 64`.  
   On SmolLM2 this removed the fence regression and passed quality integrity in bounded repeats (`can_promote 5/5`, JSON parse ok `5/5`, fenced `0/5`) without changing prompts/contracts/gates.

5. **Generalization status and required rerun?**  
   Candidate is **proven on target model only**, not yet proven across the frozen set.  
   Mandatory next step is an unchanged full five-model Round 6 rerun with strict gates intact.

## Decoding-control probe notes

- Foundry SDK settings accepted in probe: `max_tokens`, `temperature`, `top_p`, `seed`, `do_sample` (settable surface).
- OGA search options accepted: `max_length`, `temperature`, `top_p`, `do_sample`; rejected: `max_tokens`, `seed`.
- Round 6 quality evidence still records partial determinism support in gate metadata; diagnostics preserved that contract.

## Rejected or non-viable variants

- `int4_default`: regression persists (`can_promote 0/3`, fenced `3/3`).
- `int4_block_size_16`: non-JSON failure mode, still blocked (`can_promote 0/3`).
- `int4_block_size_32`: non-JSON failure and additional degradation (`can_promote 0/3`).
- `int4_block_size_-1`: unsupported/failing optimize path in this toolchain.
- `int4_act_precision_uint8`: optimize path fails in this toolchain environment.

## Artifacts

- Harness: `evaluation/recipe-agent-v1/diagnostics/smollm-json-regression/run_smollm_json_regression_diagnostics.py`
- Report: `evaluation/recipe-agent-v1/diagnostics/smollm-json-regression/diagnostic-report.json`
- Artifact test: `evaluation/recipe-agent-v1/diagnostics/smollm-json-regression/test_smollm_json_regression_artifacts.py`
