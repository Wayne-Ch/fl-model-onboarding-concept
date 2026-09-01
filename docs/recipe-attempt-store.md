# Recipe Attempt Store (v1)

`src/fl_model_onboarding/recipe_attempt_store.py` persists deterministic recipe-agent candidate generation, build/validation attempts, and verified promotions with exact identity matching.

## Scope and guarantees

- Stores immutable typed records for:
  - `GeneratedRecipeRecord`
  - `RecipeAttempt`
  - `AttemptGateResult`
  - `AttemptFailure`
  - `VerifiedRecipeRecord`
- Uses SQLite with:
  - schema versioning (`PRAGMA user_version`)
  - forward migration (`v1 -> v2`)
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
