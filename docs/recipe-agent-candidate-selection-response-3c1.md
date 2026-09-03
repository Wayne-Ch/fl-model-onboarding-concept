# Recipe Agent Slice 3C1: public candidate plan/timeline/selection/reuse response contract

Slice 3C1 is **serialization only**: it exposes the already-durable candidate-plan,
candidate-timeline/selection, and candidate-selection-reuse evidence that Slice 2/3A/3B
persist, as two small, purely additive fields on two existing public responses. No
retry execution, state machine, or store schema changed. No web UI changed. No real
tools/models run anywhere in this slice or its tests.

Owned files: `src/fl_model_onboarding/local_service.py` (response/serialization
helpers only), `contracts/openapi.yaml`, and the new
`tests/test_recipe_candidate_selection_response.py` (plus one additive assertion
block in `tests/test_contract_files.py`).

## 1. Preview: `generated_recipe.candidate_plan`

Added to both `GET /api/recipes/generated/preview` and `GET /api/models/detail`
(both call the same internal `_generated_recipe_payload` helper), alongside the
existing `verified_reuse` / `candidate_selection_reuse` fields.

`candidate_plan` is a pure, static projection of the approved CPU INT4 selection
policy (`RecipeSelectionPolicy`/`recipe_selection_policy.py`) — it never depends on
any particular attempt's execution history. It is **non-null exactly when** the
compiled generated recipe's actual `olive.device`/`olive.precision` fall inside the
approved policy's scope (currently `cpu`/`int4`), and **null** for every other
generated recipe (wrong device/precision) — this is the same eligibility check
`_generated_record_is_cpu_int4_eligible` already uses to decide whether to register
a candidate-0 lineage at attempt-creation time, so this field can never silently
drift from what actually happens when an attempt is created. It is also null
whenever recipe compilation itself failed (`generated_recipe == null`).

### Example: CPU INT4-eligible model

```json
{
  "generated_recipe": {
    "eligible_for_automatic_recipe_attempt": true,
    "...": "...",
    "candidate_plan": {
      "policy_id": "cpu-int4-recipe-selection-v1",
      "policy_version": "1.0.0",
      "policy_fingerprint": "b6b2...e91a",
      "max_candidates": 2,
      "candidates": [
        {
          "candidate_index": 0,
          "candidate_id": "default-int4",
          "role": "default",
          "quantization_override": null,
          "eligibility_trigger": null
        },
        {
          "candidate_index": 1,
          "candidate_id": "int4-block-size-64",
          "role": "quality_retry",
          "quantization_override": { "block_size": 64 },
          "eligibility_trigger": "retryable_optimized_structural_regression"
        }
      ]
    }
  }
}
```

### Example: not CPU INT4-eligible / compile failed / static recipe

```json
{ "generated_recipe": { "candidate_plan": null, "...": "..." } }
```

### Frontend (3C2) label guidance

- `role: "default"` → user-facing label **"First recipe"**.
- `role: "quality_retry"` → user-facing label **"Automatic quality retry"**.
- Never surface `candidate_id`, `eligibility_trigger`, `policy_fingerprint`, or
  `quantization_override.block_size` directly in primary UI copy — these are stable,
  machine-readable identifiers meant for support/debugging, expandable/"details"
  panels, or telemetry, not primary prose. A short explanatory line such as
  *"If the first recipe doesn't pass quality checks, Foundry Local automatically
  tries one alternative before giving up."* covers the conditional-fallback nuance
  without exposing internal jargon.
- The fallback is **only ever attempted after** the default candidate fails a
  narrow, specific quality gate (`eligibility_trigger:
  "retryable_optimized_structural_regression"`); it is not a general retry-on-any-
  failure mechanism. `max_candidates` (currently always `2`) is the hard ceiling —
  never assume more candidates can appear.

## 2. Attempt/timeline: `workflow_outcome` + `candidate_selection`

Added to `GET /api/recipes/generated/attempts/{attempt_id}` (`get_recipe_attempt`)
and to the attempt embedded in `POST /api/recipes/generated/attempts`
(`create_generated_recipe_attempt`'s response `attempt` field) — both go through the
same `_serialize_recipe_attempt` helper, so the shape is always identical whether an
attempt was just created or is being polled.

### `workflow_outcome` (always present, never omitted)

One of exactly five stable, machine-readable codes:

| Code | Meaning |
|---|---|
| `not_applicable` | This attempt is not part of any candidate-selection lineage (legacy attempt predating candidate tracking, or a generated recipe outside the CPU INT4 policy scope) **and** is not itself a candidate-selection-reuse materialization. `candidate_selection` is always `null` in this case. |
| `pending` | This attempt's lineage exists and is still deciding (candidate(s) still running, or terminal but the fallback-trigger/exhaustion decision has not yet been recorded). |
| `selected` | This attempt's lineage has a single verified winner. **This can be true even when the attempt you queried is itself the failed default** — see the worked example below. |
| `exhausted` | Every candidate in this attempt's lineage is terminal and none was verified; no winner was ever selected. |
| `reused` | This specific attempt is a candidate-selection-reuse materialization: it never dispatched a new build, and instead durably aliases a previously selected/verified winner's own artifact/package. |

`workflow_outcome` intentionally distinguishes the **lineage's overall outcome**
from **this attempt's own, never-rewritten `state`**. A default candidate that
regressed and lost stays `state: "failed"` forever, even once its fallback sibling
is verified and selected — `workflow_outcome` is the field that tells you "but the
overall workflow still succeeded, via a different candidate."

### `candidate_selection` (`null` iff `workflow_outcome == "not_applicable"`)

```json
{
  "policy_id": "cpu-int4-recipe-selection-v1",
  "policy_version": "1.0.0",
  "policy_fingerprint": "b6b2...e91a",
  "max_candidates": 2,
  "lineage_selection_state": "pending | selected | exhausted | null",
  "selected_candidate": {
    "candidate_attempt_id": "...",
    "attempt_id": "...",
    "candidate_index": 0,
    "candidate_id": "default-int4",
    "selected_by": "validation",
    "selection_reason": "Candidate 0 ('default-int4') verified: ...",
    "selected_utc": "2026-01-01T00:00:00+00:00"
  },
  "candidates": [
    {
      "candidate_attempt_id": "...",
      "attempt_id": "...",
      "candidate_index": 0,
      "candidate_id": "default-int4",
      "role": "default",
      "attempt_state": "succeeded | running | generated | failed | cancelled",
      "recipe_fingerprint": "...",
      "quantization_override": null,
      "eligibility_trigger": null,
      "disposition": null,
      "disposition_reasons": [],
      "selection_status": "not_selected | selected",
      "artifact_ref": "job://<job_id>/artifact/<artifact_id>",
      "package_ref": "job://<job_id>/package",
      "invocation_counters": {
        "mobius_build_invocation_count": 1,
        "olive_optimize_invocation_count": 1,
        "total_invocation_count": 2,
        "wall_clock_seconds": null,
        "estimated_cost_usd": null
      },
      "validated_scope": {
        "target_device": "cpu",
        "target_ep": "...",
        "toolchain_fingerprint": "...",
        "environment_scope": "foundry-local-onboarding:..."
      }
    }
  ],
  "aggregate_invocation_counters": {
    "mobius_build_invocation_count": 1,
    "olive_optimize_invocation_count": 1,
    "total_invocation_count": 2,
    "wall_clock_seconds": null,
    "estimated_cost_usd": null
  },
  "reuse": null
}
```

Every count/duration/cost field preserves `null` ("never measured") distinctly from
`0` ("measured zero"). `candidates` is always ordered by `candidate_index` ascending
(store-guaranteed, deterministic).

#### Worked example: fallback verified, default remains failed

Querying **either** the default's own `attempt_id` **or** the fallback's own
`attempt_id` for the same lineage returns the identical `workflow_outcome` and
`candidate_selection.candidates` set:

```json
{
  "attempt_id": "<default-attempt-id>",
  "state": "failed",
  "workflow_outcome": "selected",
  "candidate_selection": {
    "lineage_selection_state": "selected",
    "selected_candidate": { "candidate_index": 1, "candidate_id": "int4-block-size-64", "...": "..." },
    "candidates": [
      { "candidate_index": 0, "attempt_state": "failed", "selection_status": "not_selected", "...": "..." },
      { "candidate_index": 1, "attempt_state": "succeeded", "selection_status": "selected", "...": "..." }
    ]
  }
}
```

The default candidate's `attempt_state` is **never** rewritten to `succeeded` or
omitted; `state`/`attempt_state: "failed"` for candidate 0 and
`workflow_outcome: "selected"` for the overall lineage are both simultaneously true
and both must be surfaced together.

#### Aggregate invocation counters are derived, never constants

`aggregate_invocation_counters` sums each metric across every candidate **currently
registered** in the lineage, from real persisted per-candidate evidence:

- Single-candidate lineage (default verified on first try): `mobius_build_invocation_count: 1`,
  `olive_optimize_invocation_count: 1` (one real Mobius build + one real Olive
  optimize for the one candidate that ran).
- Two-candidate lineage (default regressed, block64 fallback verified):
  `mobius_build_invocation_count: 1` (only the default candidate ever invokes
  Mobius; the fallback reuses the captured pre-Olive artifact and never invokes
  Mobius again), `olive_optimize_invocation_count: 2` (one Olive optimize per
  candidate).
- A metric is `null` exactly when **no** candidate in the lineage has recorded a
  real value for it yet (e.g. mid-flight, before any candidate reached a terminal
  state) — it is never coerced to `0`. Once at least one candidate has a real
  value, only the known values are summed; an as-yet-unmeasured sibling is excluded
  from the sum rather than forcing the whole aggregate back to `null`.

### `candidate_selection.reuse` (non-null iff `workflow_outcome == "reused"`)

```json
{
  "reused_without_build": true,
  "source_attempt_id": "<winner-attempt-id>",
  "source_candidate_attempt_id": "<winner-candidate-attempt-id>",
  "source_parent_attempt_id": "<winner-lineage-parent-attempt-id>",
  "policy_id": "cpu-int4-recipe-selection-v1",
  "policy_version": "1.0.0",
  "policy_fingerprint": "b6b2...e91a",
  "quality_profile_fingerprint": "...",
  "runner_dispatch_count": 0,
  "mobius_invocation_count": 0,
  "olive_invocation_count": 0,
  "recorded_utc": "2026-01-01T00:00:00+00:00"
}
```

A `reused` attempt has **no lineage/candidates of its own**
(`candidate_selection.lineage_selection_state`, `max_candidates`, and
`selected_candidate` are all `null`; `candidates` is `[]`): it is a durable alias of
a previously selected winner. The returned build `job_id`/artifact/package **belong
to the original winner's build** — frontend must never imply a new build ran for a
reused attempt; `reused_without_build: true` and the zeroed invocation counts are
the explicit, durable signal for this. Missing/incomplete dispatch evidence is never
synthesized: if evidence genuinely cannot be found, `reuse` (and thus the whole
`candidate_selection` object for that path) stays consistent with `workflow_outcome:
"not_applicable"` rather than fabricating zero counts from nothing.

## 3. Backward compatibility

Both new fields are purely additive:

- `candidate_plan` (preview) and `candidate_selection` (attempt) are only ever
  populated once real candidate-plan/lineage/reuse evidence exists; every existing
  field on both responses is completely unchanged in name, type, or value.
- Legacy/static/non-CPU-INT4-eligible attempts get `workflow_outcome:
  "not_applicable"` and `candidate_selection: null` — old clients that only read
  pre-3C1 fields (`state`, `gates`, `quality_validation`, `verified_reuse`, ...)
  observe byte-for-byte the same values as before this slice.
- `contracts/openapi.yaml` is updated in place (`GeneratedRecipePreview`,
  `RecipeAttempt`, and eight new component schemas:
  `CandidateRole`, `CandidateQuantizationOverride`, `CandidatePlanEntry`,
  `CandidateSelectionPlan`, `CandidateSelectionReusePreview`,
  `CandidateInvocationCounters`, `CandidateValidatedScope`,
  `CandidateTimelineEntry`, `CandidateSelectedSummary`, `CandidateReuseEvidence`,
  `RecipeAttemptCandidateSelection`); neither new field is added to any existing
  `required` array, so old payloads still validate.

## 4. Integrity behavior (fail-closed, never swallowed)

`recipe_candidate_lineages` and `candidate_attempts` rows must always be created and
read together for one lineage — the store never allows a candidate row to exist
without its parent lineage row. If the serializer ever resolves a known candidate
row whose parent lineage is missing (on-disk corruption, or a row written through an
unsupported path), it raises a `ServiceError` (`RECIPE_ATTEMPT_STORE_ERROR`,
HTTP 500) instead of silently reporting `not_applicable` or any other normal-looking
summary — corruption is always surfaced as an error, never presented as ordinary
"no candidate plan" behavior. The same applies if a lineage's own
`selected_candidate_attempt_id` does not resolve to one of its registered
candidates.

## 5. Tests

`tests/test_recipe_candidate_selection_response.py` covers, using the same fully
faked `ProcessRunner`/`TextInferenceBackend` fixtures as
`test_local_service_candidate_orchestration.py` (no real tools/models):

- Preview `candidate_plan` shape/content for the approved policy, wired into both
  `generated_recipe_preview()` and `model_detail()`, and `null` when the compiled
  recipe is not CPU INT4-eligible.
- Attempt `workflow_outcome`/`candidate_selection` for: single-candidate `selected`
  (default verified first try), two-candidate `selected` (fallback verified,
  default still `failed`, queried from both attempt ids), `exhausted` (both
  candidates regress), `pending` (mid-flight, blocked mid-Mobius-build, invocation
  counters/aggregate all `null`), `reused` for both a reused default winner and a
  reused block64 winner (measured-zero evidence, both from `create_...` and a
  follow-up poll), and `not_applicable` for a legacy/non-eligible generated
  attempt (with an explicit backward-compatibility assertion on the surrounding
  legacy fields).
- No path leakage: every `artifact_ref`/`package_ref` matches the sanitized
  `job://<job_id>/(artifact/<id>|package)` shape.
- A corrupted-store integrity case (candidate row present, parent lineage row
  deleted) surfaces a `ServiceError` instead of a normal summary.
- `tests/test_contract_files.py` gained an additive assertion block verifying the
  new OpenAPI schemas/enums/required-field sets exist with the exact names this
  slice's serializers emit.
