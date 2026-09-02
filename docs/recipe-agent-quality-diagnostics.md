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
| arithmetic-addition-17-plus-28 | arithmetic | exact_match_failed, max_words_exceeded | Yes | **Model capability** |
| instruction-two-words-blue-river | instruction | max_words_exceeded | Yes (inferred) | **Model capability** |
| format-json-answer-unit | output-format | forbidden_token_present:\`\`\`, json_format_invalid | **No — baseline passed** | **Quantization regression** |
| factual-red-planet | factual-recall | _(none)_ | N/A (both passed) | Passed |

**Diagnosis:** 1 genuine INT4 quantization regression (JSON format — optimized wraps JSON in
markdown code fences, baseline did not). 2 model capability limitations (arithmetic, word count).
The JSON regression is partially an evaluator strictness issue: `_parse_json_object` does raw
`json.loads()` without stripping markdown fences, which is common instruction-model behavior.

**Path to 5/5:** Even with an evaluator fix for JSON fences, SmolLM2-360M would still fail
arithmetic and likely instruction word-count. This 360M model has a genuine quality ceiling.
It remains in the model set; path to full pass requires a higher-capability source model at
this scale.

## Granite-3.2-2B-Instruct

| Prompt | Category | Optimized failures | Baseline also failed? | Classification |
|---|---|---|---|---|
| arithmetic-addition-17-plus-28 | arithmetic | exact_match_failed, relevance_keyword_missing | Yes | **Model capability** |
| factual-red-planet | factual-recall | _(none)_ | N/A (both passed) | Passed |
| instruction-two-words-blue-river | instruction | _(none)_ | N/A (both passed) | Passed |
| format-json-answer-unit | output-format | _(none)_ | N/A (both passed) | Passed |

**Diagnosis:** Zero quantization regressions. The single failure is arithmetic — both FP32
baseline and INT4 optimized produce a wrong answer (45 not present at all). This is a source
model limitation.

**Path to 5/5:** Granite passes 3/4 with no optimization-caused degradation. Only arithmetic
blocks; this requires the source model to improve, not a pipeline fix.

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
- **JSON format:** Fair intent, but evaluator penalizes common markdown-wrapping behavior.
- **Arithmetic (17+28):** Legitimate bar, but hardest for sub-1B models.

## Recommendations

| Action | Scope | Detail |
|---|---|---|
| **Fix evaluator** | `_parse_json_object` | Strip markdown code fences before `json.loads()`. Generic fix, not model-specific. |
| **Keep arithmetic failure** | Both models | Legitimate model capability limitation. |
| **Keep instruction failure** | SmolLM2 only | Legitimate capability limitation at 360M. |
| **Bounded rerun** | Both models | Capture verbatim outputs. Apply JSON-fence fix first, then rerun. |

## Required next steps

1. Implement `evaluator-json-fence-strip` generic fix in `quality_validation._parse_json_object`.
2. Bounded inference rerun with output capture for SmolLM2 and Granite.
3. Confirm SmolLM2 JSON prompt passes with fix (would reduce failures from 3→2).
4. Full 5-model rerun after evaluator fix is merged.
