# Recipe Agent v1 – Round 4 Quality Diagnostics

**Diagnostic ID:** `quality-round4-diagnostic-001`
**Round:** `r4-20260902T083935Z` on branch `wayne-ch-linus-fixing-generated-execution` @ `72a87d0`
**Profile:** `textgen-basic-quality-v1` v1.0.0 (4 prompts, max_tokens=64, temp/seed unsupported by runtime)

## Evidence gap

Round 4 ran with `retain_failed_workspaces=false`. The actual baseline and optimized output
text for each quality prompt was **not persisted** in committed artifacts. Only failure-code
error signatures survive in the model result JSON files. A bounded inference rerun using the
retained 12.977 GiB pinned cache is needed to recover verbatim outputs; no full rebuild
required.

## SmolLM2-360M-Instruct

| Prompt | Category | Optimized failures | Baseline also failed? | Classification |
|---|---|---|---|---|
| arithmetic-addition-17-plus-28 | arithmetic | exact_match_failed, max_words_exceeded | Yes | **Baseline functional failure** |
| instruction-two-words-blue-river | instruction | max_words_exceeded | Yes | **Baseline functional failure** |
| format-json-answer-unit | output-format | forbidden_token_present:\`\`\`, json_format_invalid | **No — baseline passed** | **Quantization regression** |
| factual-red-planet | factual-recall | _(none)_ | N/A (both passed) | Passed |

**Diagnosis:** 1 genuine INT4 quantization regression (JSON format — optimized introduced
markdown code fences; the forbidden_token contract independently rejects \`\`\` even if the
JSON parser were fence-aware). 2 baseline functional failures where both FP32 and INT4 failed
(arithmetic, instruction word count) — not quantization regressions. Without retained raw
output text, the detailed root cause for the baseline failures (model capability vs
prompt/template vs inference plumbing) is unproven.

**Path forward:** SmolLM2-360M fails 3 of 4 prompts. It remains in the model set. A bounded
rerun with output capture is the prerequisite for further root-cause analysis.

## Granite-3.2-2B-Instruct

| Prompt | Category | Optimized failures | Baseline also failed? | Classification |
|---|---|---|---|---|
| arithmetic-addition-17-plus-28 | arithmetic | exact_match_failed, relevance_keyword_missing | Yes | **Baseline functional failure** |
| factual-red-planet | factual-recall | _(none)_ | N/A (both passed) | Passed |
| instruction-two-words-blue-river | instruction | _(none)_ | N/A (both passed) | Passed |
| format-json-answer-unit | output-format | _(none)_ | N/A (both passed) | Passed |

**Diagnosis:** Zero quantization regressions. The single failure is a baseline functional
failure — both FP32 baseline and INT4 optimized failed identically.
`relevance_keyword_missing` specifically establishes that the answer token "45" is absent from
the output (wrong answer). Without retained raw output text, root cause beyond wrong-answer is
unproven.

**Path forward:** Granite passes 3/4 with zero optimization-caused degradation. A bounded
rerun with output capture is needed to establish root cause for the arithmetic failure.

## Cross-cutting findings

### Chat template verification
Same `_execute_quality_prompts` code path for baseline and optimized. Same tokenizer source.
No template divergence.

### Determinism
Only `max_tokens=64` enforced. `temperature` and `seed` are unsupported by the runtime. This
affects cross-run reproducibility but not within-run baseline-vs-optimized fairness.

### Prompt fairness (0.36B–3B range)
- **Factual recall (Red Planet):** Fair — all models expected to pass.
- **Instruction following (two words):** Strict but fair — prompt explicitly says "exactly two words."
- **JSON format:** Requires clean JSON without markdown wrapping; legitimate contract.
- **Arithmetic (17+28):** Legitimate functional bar.

## Recommendations

| Action | Scope | Detail |
|---|---|---|
| **Keep all failures** | Both models | All current failures are legitimate under the existing quality contract. |
| **Bounded rerun** | Both models | Capture verbatim baseline + optimized outputs to establish root cause for baseline functional failures. |

## Next round harness requirement

The next round harness must retain **bounded sanitized baseline and optimized output text** per
quality prompt, plus determinism enforcement metadata, in committed artifacts. This enables
root-cause analysis without requiring a rerun. Required fields per prompt:
`prompt_id`, `baseline_output_text`, `optimized_output_text`, `applied_determinism`,
`unsupported_determinism_fields`. Truncate outputs to max_tokens limit; strip absolute paths.
No model-specific prompt changes.
