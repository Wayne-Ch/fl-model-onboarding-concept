# Recipe selection policy (Slice 1)

`recipe_selection_policy.py` is a typed, versioned, **planning-only** module. It declares
which CPU INT4 quantization candidates a recipe *may* attempt and the single, declarative
condition under which a non-default candidate becomes eligible. It does not execute
recipes, does not call Olive/Mobius/onnxruntime, and does not select or reference any
model (no model ID, name, org, or hidden-size selectors appear anywhere in the policy).

## Data

- `config/recipe-selection-policies.json` — the policy data, validated against
  `contracts/recipe-selection-policy.schema.json`.
- Each policy declares `target_device` (`cpu`), `quantization` (`int4`), `max_candidates`,
  and an ordered `candidates` array.

## Candidates (current policy `cpu-int4-recipe-selection-v1`)

- **Candidate 0 (`default-int4`)** — the existing default quantization arguments, with no
  override. Always eligible.
- **Candidate 1 (`int4-block-size-64`)** — the same recipe with the supported Olive INT4
  `block_size=64` override, exactly as proven in the SmolLM JSON-regression diagnostics
  (see `docs/smollm-json-regression.md`). It is declared with
  `eligibility_trigger: "retryable_optimized_structural_regression"` — the only allowlisted
  trigger — and is **not** a static architecture/model default. It only becomes eligible
  once a recipe's quality validation produces a matching retry disposition (see below).

`RecipeSelectionPolicy.plan(trigger=...)` returns the deterministic, ordered set of
candidates eligible for a given trigger value. Passing no trigger (or an unrecognized one)
returns only the default candidate.

## Quality retry disposition

`quality_validation.py` additively exposes `QualityRetryDisposition` and
`QualityValidationResult.quality_retry_evaluation`. The disposition is
`retryable_optimized_structural_regression` only when, from existing typed gate evidence
(never free-text report prose):

1. a baseline exists and passes the affected structural constraint;
2. the optimized run is runtime-functional (no pathological output);
3. optimized alone fails an allowlisted structural output-format requirement (currently:
   invalid JSON output, or a forbidden formatting token in an output-format prompt); and
4. the recipe is otherwise blocked solely because of that optimized-only regression.

Every other case (baseline unavailable, both baseline and optimized failing, capability-only
wrong answers, optimized improvement, matched failure, pathological/runtime failure, or
unknown/malformed evidence) resolves to `not_retryable`.

## What this slice does *not* do

This module is metadata and disposition-derivation only. Wiring the retry disposition into
an actual recipe re-attempt (choosing candidate 1's override and re-running the recipe) is
explicit follow-up work and is out of scope here.
