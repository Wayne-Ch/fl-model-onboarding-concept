# Recipe Attempt Store (v1 + Slice 2 candidate persistence)

`src/fl_model_onboarding/recipe_attempt_store.py` persists deterministic recipe-agent candidate generation, build/validation attempts, and verified promotions with exact identity matching. Slice 2 additively layers durable, immutable retry-candidate persistence, migration, fingerprints, and policy-aware reuse identity on top, without changing any Slice 1 table, column, or behavior.

## Scope and guarantees

- Stores immutable typed records for:
  - `GeneratedRecipeRecord`
  - `RecipeAttempt`
  - `AttemptGateResult`
  - `AttemptFailure`
  - `VerifiedRecipeRecord`
  - `CandidateAttemptRecord` (Slice 2)
  - `RecipeCandidateLineage` (Slice 2)
- Uses SQLite with:
  - schema versioning (`PRAGMA user_version`)
  - forward migration (`v1 -> v2 -> v3`)
  - WAL mode for one writer + multiple readers
  - transactional writes (`BEGIN IMMEDIATE`) for idempotency and promotion atomicity
- Enforces attempt state contract:
  - `generated -> running -> succeeded|failed|cancelled`
  - terminal states are immutable
- Enforces ordered gate sequence:
  - `mobius_build`
  - `olive_optimize`
  - `onnx_validation`
  - `ort_validation`
  - `oga_validation`
  - `fl_sdk_inference`
  - `quality_validation`

## Identity and idempotency

- Generated recipe upsert is keyed by immutable recipe fingerprint.
- Attempt creation requires both:
  - idempotency key
  - canonical request fingerprint (`build_attempt_request_fingerprint`)
- Same key + same request fingerprint replays the same attempt.
- Same key + different request fingerprint conflicts.
- Full identity is preserved (no truncation):
  - HF pinned SHA (40 hex)
  - compiler/capability fingerprint
  - toolchain fingerprint
  - profile fingerprint

## Promotion and reuse

- Verified promotion requires:
  - compiler-provided complete promotion evidence
  - succeeded attempt
  - source recipe fingerprint match
  - exact revision/toolchain/capability/profile identity match
  - matching per-gate evidence references
- Reuse query returns verified recipes only when **all** identity keys match exactly:
  - model id
  - revision SHA
  - requested device
  - requested precision
  - compiler/capability fingerprint
  - toolchain fingerprint
  - profile fingerprint
- Stale toolchain/profile identities miss by design.

## Security and bounded persistence

- Evidence references and failure payloads are validated before persistence.
- Rejects credential-like tokens and absolute private paths.
- Rejects multiline/unbounded evidence payloads and oversized failure text.
- Stores structured references only (not raw logs/model artifacts).

## Contract and fail-closed loading

- Contract: `contracts/recipe-attempt.schema.json`
- Record serialization/deserialization is schema-validated and fail-closed:
  - required keys enforced
  - unknown keys rejected
  - invalid enum/type/pattern rejected

## Restart behavior

- On startup, any attempt left in `running` is deterministically recovered to terminal `failed`
  with `interrupted` failure classification and explicit recovery evidence.
- Verified records remain queryable after restart.

## Integration boundary with existing BuildJob SQLite state

- **Build execution state source of truth stays in `local_service.SQLiteStateStore`** (`jobs/events/artifacts/validations`).
- **Recipe attempt store source of truth is recipe-agent lifecycle state** (generated candidate identity, attempt gates/failures, verified promotion/reuse index).
- To avoid duplicated execution state:
  - keep BuildJob stage transitions and runtime artifacts in BuildJob store
  - keep recipe promotion/reuse eligibility in recipe-attempt store
  - link across stores via structured evidence references (for example BuildJob IDs / event IDs), not copied logs or artifact blobs.

## Slice 2: candidate plan / selection

Slice 2 adds two new tables only (`recipe_candidate_lineages`, `candidate_attempts`); no
existing table is altered, recreated, or dropped, and legacy attempts created before
Slice 2 remain fully readable with no lineage/candidates and unchanged behavior.

### Design: candidates reuse the existing attempt state machine

A "candidate" under a policy's plan is **not** a duplicate state machine. Candidate
index 0 (the default) *is* the parent's own existing `RecipeAttempt` (`attempt_id ==
parent_attempt_id`). Any further candidate (index >= 1, e.g. the `block_size=64`
fallback) is registered against its own, separately created `RecipeAttempt` (its own
idempotency key, its own compiled `GeneratedRecipe`/recipe fingerprint, its own gate
sequence). The new `candidate_attempts` table only adds candidate-plan and selection
metadata on top, linking back to the attempt that actually carries gate/state/failure
data. A candidate counts as **verified** exactly when its linked attempt reaches
`AttemptState.SUCCEEDED` (every required gate, including `quality_validation`, passed).

### APIs

- `register_candidate_attempt(...)` — create/validate one candidate-plan slot from an
  approved `RecipeSelectionPolicy`. Candidate index 0 must be registered first
  (`attempt_id == parent_attempt_id`, no trigger/disposition). Any further index
  requires the exact `eligibility_trigger` the policy declares for that index, and a
  `QualityRetryEvaluation` whose disposition is retryable and whose
  `.disposition.value` matches the supplied trigger byte-for-byte (this is the runtime
  half of the cross-module trigger-constant enforcement described below). Enforces
  `(parent_attempt_id, candidate_index)`/`candidate_id` uniqueness, in-order
  registration, a fixed policy/quality-profile identity per lineage, and
  `policy.max_candidates`. Also fails closed (`CandidatePlanValidationError`) unless
  the child attempt's generation identity (`model_id`, `revision_sha`,
  `requested_device`, `requested_precision`, `compiler_version`,
  `capability_fingerprint`, `toolchain_fingerprint`, `profile_fingerprint`) is
  byte-for-byte identical to the parent's -- only `recipe_fingerprint` and the
  candidate's own quantization override may differ. Parent and child are loaded and
  compared inside the same transaction, before any candidate row is inserted, so a
  fully separate (e.g. different model) successful attempt can never be registered as
  a fallback candidate.
- `finalize_candidate_attempt_evidence(...)` — atomically attach nullable
  artifact/package refs and invocation counters once the linked attempt is terminal.
  Write-once: identical values are a no-op, differing values raise.
- `select_verified_candidate_attempt(...)` — atomically choose the single winner.
  Fails closed unless the lineage is `PENDING` and the candidate's linked attempt is
  `SUCCEEDED`. Always persists `selected_by="validation"`. Never overwrites a failed
  default with a fallback: the failed default row is untouched and stays queryable.
  Defense in depth: re-validates the candidate's linked attempt against the parent's
  generation identity (the same check as registration) before committing selection,
  raising `CandidateSelectionConflictError` if it was ever violated (e.g. a
  preexisting or tampered row).
- `finalize_exhausted_candidate_lineage(...)` — atomically marks a lineage
  `EXHAUSTED` (no winner). Fails closed if any candidate is not yet terminal, or if any
  candidate is verified-but-unselected (that candidate must be selected instead).
- `list_candidate_attempts(parent_attempt_id)` / `get_candidate_attempt(...)` /
  `get_candidate_lineage(...)` — deterministic, ordered reads. The negative (failed)
  default candidate remains discoverable alongside the winner.
- `find_reusable_candidate_selection(query: CandidateSelectionReuseQuery)` — read-only
  lookup of a previously *selected* verified candidate by complete provenance
  identity. Never executes anything and never infers invocation counts. Defense in
  depth: also re-validates the winner's linked attempt against its own parent's
  generation identity before returning it, raising `CandidateReuseIntegrityError`
  (fail closed) rather than ever silently serving a cross-identity or corrupt winner.

### Fingerprints

- `build_candidate_recipe_fingerprint(recipe_fingerprint, quantization_override,
  policy_fingerprint)` is the child's identity: the actual fully-resolved recipe
  fingerprint + actual quantization override (`None` for the default) + the selection
  policy's fingerprint. Deliberately excludes timestamps, filesystem paths, and
  candidate-attempt UUIDs.
- `CandidateSelectionReuseQuery` extends the existing generated-recipe identity
  (model/revision/device/precision/compiler/capability/toolchain/profile) with two
  more identities Slice 2 introduces: the quality-validation profile fingerprint (a
  quality-profile version bump invalidates reuse) and the selection policy fingerprint
  (a policy id/version/candidate-set change invalidates reuse). This is a separate
  type from the Slice 1 `RecipeReuseQuery`, so existing `local_service.py` callers are
  unaffected.

### Nullable counters/cost placeholders

- `CandidateInvocationCounters` fields (Mobius/Olive invocation counts, wall-clock
  seconds, estimated cost) are all `None` until a later slice actually instruments a
  real run. `None` is never coerced to or serialized as `0`.

### Cross-module trigger-constant enforcement

`recipe_selection_policy.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER` is now
imported directly from `QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION.value`
(single source), `recipe_attempt_store.py` asserts their equality at import time, and
`register_candidate_attempt` re-verifies the exact value used to justify each
fallback-candidate registration. A future rename on either side fails fast instead of
silently disabling the fallback candidate.

