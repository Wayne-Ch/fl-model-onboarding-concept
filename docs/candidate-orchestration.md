# Recipe Agent Slice 3B1: default candidate + trusted block64 fallback orchestration

This slice wires the previously-unused durable candidate lineage/evidence/selection/
exhaustion API (`RecipeAttemptStore.register_candidate_attempt` /
`finalize_candidate_attempt_evidence` / `select_verified_candidate_attempt` /
`finalize_exhausted_candidate_lineage`, all added in Slice 2), Slice 3A1's trusted
block64 compilation (`compile_trusted_candidate_recipe`), and Slice 3A2's pre-Olive
Mobius reuse (`ProductionBuildStageRunner.run_fallback_with_pre_olive_reuse`) into
`LocalOnboardingService`'s generated-recipe-attempt lifecycle.

Everything below lives in `local_service.py` (plus two small additive helpers: a
`RecipeAttemptStore.find_candidate_attempt_by_attempt_id` reverse lookup, and a
`ProductionBuildStageRunner.on_mobius_ready` result hook). No API/OpenAPI route,
`web`/`web_dist`, frozen models/profile/prompts, or root index changed. No real
model/tool runs anywhere in this slice's tests.

## State model: no new tables, one additive orchestration seam

The existing `RecipeAttempt`/`BuildJob` state machines are reused completely
unchanged. A "candidate" is simply an existing `RecipeAttempt` (its own gates,
`GENERATED → RUNNING → {SUCCEEDED,FAILED,CANCELLED}`) linked into a
`RecipeCandidateLineage`/`CandidateAttemptRecord` row pair (Slice 2 schema,
`recipe_candidate_lineages` / `candidate_attempts` tables — no migration needed for
3B1). Candidate 0 (the default) *is* the parent attempt (`attempt_id ==
parent_attempt_id`); candidate 1 (the trusted block64 fallback) is a **separate**
`RecipeAttempt` + `BuildJob` + workspace with the same generation identity and an
immutable, deterministically fingerprinted recipe.

Orchestration hooks entirely off of two existing extension points:

- `LocalOnboardingService.create_generated_recipe_attempt` — after `start_attempt`,
  idempotently registers candidate 0 when the record is CPU INT4-eligible.
- `LocalOnboardingService._sync_generated_attempt_with_job` — the single place every
  job's terminal state (success, gate failure, quality-validation failure,
  cancellation) already flows through. Candidate evidence/selection/fallback-trigger/
  exhaustion all hook in here, gated behind one reverse lookup:
  `RecipeAttemptStore.find_candidate_attempt_by_attempt_id(attempt_id)`. When it
  returns `None` (a static recipe, or a generated attempt whose actual compiled
  device/precision do not match the approved policy), **every** Slice 3B1 code path
  is skipped and behavior is byte-for-byte identical to pre-3B1.

No terminal state is ever rewound: the default candidate's own `RecipeAttempt`
transitions to `FAILED` exactly once (via the existing `finish_attempt_failed`) and
is never revisited, even when the fallback candidate later succeeds and the overall
user-facing operation is reported as succeeded through the *fallback's own*
attempt/recipe.

## Eligibility: CPU INT4 only

```python
LocalOnboardingService._generated_record_is_cpu_int4_eligible(record) -> bool
```

reads the record's own *compiled* `recipe.olive.device`/`recipe.olive.precision` and
compares them against `DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY.target_device` /
`.quantization` — deliberately mirroring exactly the same check
`compile_trusted_candidate_recipe` itself enforces, so eligibility here can never
silently drift from what a fallback compile would actually accept. A generated
attempt that resolves to any other device/precision (none exist yet in this
codebase, but the check is device/precision-driven, not model-driven, so it is
future-proof) and every static-recipe build fall through untouched — this is the
"legacy generated attempts without policy continue existing path" requirement.

## Default candidate flow

1. `create_generated_recipe_attempt` creates+starts the parent `RecipeAttempt`
   (unchanged), then calls `_register_default_candidate_lineage_if_eligible`, which
   idempotently registers candidate 0 (a no-op if a lineage already exists for this
   `attempt_id`, or if it is not CPU INT4-eligible).
2. The job runs through the *unchanged* `ProductionBuildStageRunner.run()`. Right
   after a real Mobius build succeeds — while `mobius_dir` is guaranteed to still
   exist and before any future retention cleanup could remove it — the new
   `on_mobius_ready` hook (`_capture_pre_olive_descriptor_if_eligible`) captures an
   immutable `PreOliveArtifactDescriptor` into `self._pre_olive_descriptors[attempt_id]`
   (process-memory only) if and only if this is a still-`PENDING`, CPU INT4-eligible
   default candidate. Any failure to capture is swallowed — it only ever means a
   later fallback trigger finds no usable descriptor and correctly refuses to retry.
3. `_sync_generated_attempt_with_job` runs quality validation exactly as before. If
   it passes: `_on_candidate_attempt_terminal` persists real invocation counters
   (`production_invocation_evidence_to_candidate_counters`) and compact
   `job://.../artifact/...` / `job://.../package` references, then
   `_select_verified_candidate` calls `select_verified_candidate_attempt` with real
   `validated_target_device`/`validated_target_ep`/`validated_toolchain_fingerprint`/
   `validated_environment_scope` derived from the candidate's own compiled recipe and
   persisted generation identity (never left implicitly "verified for every scope"
   by omission). Promotion then proceeds exactly as before (unchanged code, now
   simply also reached for the fallback candidate when applicable).
4. If quality validation fails, the attempt is finalized failed exactly as before
   (unchanged message/classification/next_action), and *then*
   `_maybe_launch_fallback_candidate` evaluates every required trigger precondition.

## Fallback trigger — every precondition is independently checked

`_maybe_launch_fallback_candidate` returns `True` (fallback launched) only when
**all** of the following hold; any single failure returns `False` and the caller
finalizes the lineage exhausted instead:

1. `candidate.candidate_index == 0` — only the default may ever trigger a fallback;
   the fallback candidate's own failure never triggers a third candidate.
2. `lineage.policy_max_candidates > 1` and the lineage is still `PENDING`.
3. `quality_outcome.quality_retry_evaluation.is_retryable` is `True` — the sole
   allowlisted disposition, derived deterministically by
   `quality_validation._derive_quality_retry_disposition` from existing gate
   evidence only. Integrity evidence identifies the blocking prompt IDs and their
   structural correspondence; eligibility then requires the *full optimized
   failure tuple* for each blocking prompt to be entirely allowlisted structural
   codes (`json_format_invalid` / output-format `forbidden_token_present:*`) with
   exact structural-entry correspondence. Unknown/malformed/partial/duplicate or
   mismatched integrity evidence, missing prompt coverage, baseline unavailable,
   pathological/runtime failures, baseline+optimized failures on the same prompt,
   and any mixed same-prompt non-structural failure all resolve to
   `NOT_RETRYABLE` (fail closed), never by free-text matching. Model-capability
   advisories on unrelated prompts remain advisory-only and do not suppress this
   retry trigger.
4. `job.state == JobState.SUCCEEDED` — defense in depth: Mobius, Olive, and every
   ONNX/ORT/OGA/FL runtime gate must have actually succeeded for *this specific
   job*, independent of the disposition check above.
5. A captured pre-Olive descriptor exists in `self._pre_olive_descriptors` for this
   parent attempt, **and** `revalidate_pre_olive_source(descriptor)` still succeeds
   (re-walks and re-hashes the captured source; fails closed on drift/tamper/a
   symlink or reparse point now present).
6. The default candidate's recipe recompiles successfully
   (`_recompile_generated_recipe_record`).

Only once every check passes is the descriptor consumed (popped from the cache) and
`_launch_fallback_candidate_attempt` called.

## Fallback candidate flow

`_launch_fallback_candidate_attempt`:

1. `compile_trusted_candidate_recipe(default_recipe, policy=..., candidate=policy.candidates[1])`
   — the *only* boundary permitted to layer the approved `block_size=64` override.
2. `upsert_generated_recipe` persists the resulting `GeneratedRecipe` (a new,
   immutable, deterministically fingerprinted `GeneratedRecipeRecord` — distinct
   from the default's).
3. `create_attempt` with a **deterministic idempotency key**
   (`f"fallback-candidate-1::{parent_attempt_id}"`), so repeated worker delivery for
   the same parent always resolves to the exact same fallback attempt row.
4. `start_attempt` (skipped if already started by a prior partial attempt) and
   `register_candidate_attempt(candidate_index=1, trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
   retry_evaluation=...)` — both skipped if candidate 1 is already registered.
5. A fresh `BuildRequest`/`BuildJob`/workspace via the existing
   `_build_request_for_generated_attempt` (reused unchanged), then enqueued —
   skipped entirely if a live job already exists for this attempt (idempotent
   against duplicate worker delivery).
6. The descriptor + fallback generation identity are stashed in
   `self._fallback_launch_context[job_id]`, consumed exactly once by `_run_job`,
   which dispatches that one job through
   `ProductionBuildStageRunner.run_fallback_with_pre_olive_reuse` instead of the
   ordinary `run()` — Mobius is never invoked for this job; Olive runs exactly once
   with `--block_size 64`.

The fallback candidate then flows through the exact same
`_sync_generated_attempt_with_job` success/failure paths as any other candidate
(evidence, selection-or-exhaustion), because that logic is candidate-index-agnostic.

## Exhaustion, cancellation, and the restart contract

- **Non-retryable / fallback-not-triggered**: `_finalize_lineage_exhausted_if_applicable`
  calls `finalize_exhausted_candidate_lineage` (a no-op if the lineage is already
  finalized, or if the store's own "every candidate terminal, none
  verified-but-unselected" invariant is not yet satisfied).
- **Cancellation** (candidate 0 or candidate 1) always finalizes the lineage
  exhausted — cancellation stops the whole lineage; no further candidate is ever
  attempted afterward.
- **Fallback process failure** (Olive itself fails for candidate 1) finalizes
  candidate 1 failed and the lineage exhausted; never a third candidate.
- **Restart recovery**: the pre-Olive descriptor cache is process-memory only and
  never survives a restart. `LocalOnboardingService._recover_orphaned_candidate_lineages`
  runs once at startup (after the existing interrupted-job recovery pass) and, for
  every still-`PENDING` lineage whose registered candidates are *all* terminal:
  - if none is verified, finalizes the lineage exhausted -- with an explicit
    `restart_fail_closed` reason when a fallback candidate was never registered
    (never rebuilding Mobius to resume it), or a `restart_recovery` reason when
    every candidate slot the policy allows (`policy_max_candidates`) already ran
    and failed/cancelled. Candidate count being exactly at the policy max is
    never treated differently from being below it -- a crash between committing
    the last candidate's terminal state and finalizing the lineage must not
    orphan it forever, since the parent attempt is itself already terminal and
    the ordinary lazy-sync path never revisits it again.
  - if exactly one candidate is verified but was never selected (a crash between
    the candidate's own attempt reaching `succeeded` and
    `select_verified_candidate_attempt` committing), selects it using the same
    trusted `_select_verified_candidate` logic the live sync path uses, instead
    of ever exhausting the lineage around it. If the generated recipe backing it
    can no longer be located, the lineage is left `PENDING` (explicit and still
    actionable) rather than risk discarding a verified winner.

  A lineage that is still genuinely in flight (any candidate not yet terminal) is
  left completely untouched; the ordinary lazy-sync path (a `get_recipe_attempt`
  poll, or the same interrupted-job recovery pass) finalizes it normally once
  safe, including launching the fallback candidate if that candidate's own gates
  were simply never reached before the restart. An already-finalized
  (`selected`/`exhausted`) lineage is always a no-op, making repeated restart
  recovery idempotent.

## Locking: a per-attempt guard, never the global lock, across unbounded I/O

`_sync_generated_attempt_with_job` (the method that runs quality validation,
evaluates the fallback trigger, and -- when triggered -- revalidates the
captured pre-Olive artifact and compiles/registers/launches the fallback
candidate) always executes through `_safe_sync_generated_attempt`, which no
longer serializes it under `LocalOnboardingService._lock` (the service-wide
lock guarding `self._jobs`/`self._attempt_to_build_job`/`self._cancel_events`/
`self._queue`/etc.). Instead:

1. A brief `self._lock` critical section looks up or creates a per-attempt
   `_AttemptSyncGuard` (a plain `threading.Lock`, refcounted so the guard map
   never grows without bound across the service's lifetime), then releases
   `self._lock`.
2. The (potentially slow, unbounded) sync body itself runs holding only that
   per-attempt guard -- serializing a duplicate sync of the *same* attempt
   (e.g. the worker finishing a job racing a `get_recipe_attempt` poll's own
   lazy re-sync) without ever holding `self._lock` across manifest hashing
   (`revalidate_pre_olive_source`), process execution, or quality validation.
3. `_sync_generated_attempt_with_job`'s callees still take `self._lock`
   themselves, but only briefly, around actual shared in-memory map/queue
   mutations (see `_launch_fallback_candidate_attempt`) -- always nested
   *inside* an already-held per-attempt guard, never the reverse. No call
   site ever holds `self._lock` while blocking to acquire a per-attempt guard
   (`cancel_build` and every `_run_job` call site release `self._lock` before
   calling `_safe_sync_generated_attempt`), so there is no lock-order
   inversion between the two.
4. Immediately before committing to launch the fallback candidate --
   after `revalidate_pre_olive_source` returns, which can take an unbounded
   amount of wall-clock time -- `_maybe_launch_fallback_candidate` re-checks
   the job's cancellation signal, the candidate's own terminal/non-verified
   state, and the lineage's still-`PENDING` selection state, so a
   cancellation or any other concurrent mutation that landed during the
   revalidation can never be silently raced past.

This keeps unrelated `get_build`/`get_recipe_attempt`/`cancel_build`/new-
submission calls -- which only ever need brief `self._lock` sections --
responsive while one attempt's sync revalidates a potentially many-GB
pre-Olive artifact. See `tests/test_local_service_lock_narrowing.py` for the
concurrency regression coverage.

## Aggregate counters: always real, never inferred

`CandidateAttemptRecord.invocation_counters` is populated once per candidate from
that candidate's own `BuildJob.production_invocation_evidence` via
`production_invocation_evidence_to_candidate_counters` (Slice 3A1, unchanged): a
tool that never launched stays `None`, never `0`. A normal default success is
Mobius 1 / Olive 1 (single candidate). A fallback success is the parent's Mobius 1 /
Olive 1 plus the child's Mobius `None` / Olive 1 — summing the two real,
independently-recorded rows gives Mobius 1 / Olive 2, exactly as specified, with
nothing hardcoded or inferred.

## Two small 3A2 hardening follow-ups (`production_runner.py`)

1. `materialize_pre_olive_copy`'s per-entry copy loop now calls
   `_assert_no_link_in_relative_path_chain(source_dir, entry.relative_path)` instead
   of checking only the fully-composed leaf path. A Windows junction swapped into
   an *intermediate* ancestor directory between the initial revalidation walk and
   this specific file's copy would previously go undetected (the leaf-only check
   only inspects the final path component's own attributes; `lstat` still
   transparently follows an intermediate junction while resolving the rest of the
   path). See `test_materialize_pre_olive_copy_rejects_mocked_windows_junction_race_on_intermediate_ancestor`.
2. `_build_directory_manifest` now calls
   `_assert_no_case_insensitive_relative_path_collisions` before hashing: two
   distinct relative paths differing only by case (e.g. `Model.onnx` vs
   `model.onnx`) are rejected before any copy is attempted, since such a manifest
   is faithful to a case-sensitive source but would silently collapse one file into
   the other on a case-insensitive destination filesystem (the default on
   Windows/NTFS). Post-copy hash verification remains mandatory and unchanged.

## Slice 3B2a delivered

- **Selected-candidate reuse is now wired into preview + generated-attempt dispatch.**
  `local_service.py` now builds an exact `CandidateSelectionReuseQuery` from the
  current request identity (model/revision/device/precision/compiler/capability/
  toolchain/profile) plus current quality-profile and policy fingerprints, calls
  `find_reusable_candidate_selection`, and short-circuits to the already-selected
  winner when every identity/scope/integrity check passes.
- **No new build dispatch on reuse hit.** A reuse hit now returns without queuing a
  new runner job; the original selected winner's artifact/package lineage remains the
  source of truth. Candidate-selection reuse metadata is carried in preview and the
  returned attempt payload (`source_parent_attempt_id`, winner candidate/attempt ids,
  winner recipe fingerprint, and selection reason).
- **Fail-closed boundaries.** `CandidateReuseIntegrityError` and winner-provenance
  mismatches (lineage state drift, missing winner recipe/job refs, policy candidate
  re-derivation mismatch) surface explicitly as integrity errors; ordinary identity or
  validated-scope mismatches remain safe misses and proceed down the normal build path.
- **No API/OpenAPI/route/UI changes.** Candidate lineage/evidence/selection state is
  only ever visible through direct `RecipeAttemptStore` access (as these tests do);
  `_serialize_recipe_attempt`'s response shape is unchanged.
- **No real model/tool runs.** Every test in
  `tests/test_local_service_candidate_orchestration.py` uses a real
  `ProductionBuildStageRunner` wired with a fully faked `ProcessRunner` and a fake,
  deterministic `TextInferenceBackend` — no network access, no real Mobius/Olive/
  onnxruntime/Foundry Local tooling.

## Slice 3B2b delivered: full invalidation matrix, evidence, abandoned-recovery

Scope for this slice, entirely inside `local_service.py` /
`recipe_attempt_store.py` and their test files. No API/OpenAPI response
contract, `web`/`web_dist`, runner build semantics, selection trigger/policy,
quality profile/prompts, or root index changed anywhere in this slice. No
real tools/models.

### A. Complete exact invalidation matrix

Selected-candidate reuse (`RecipeAttemptStore.find_reusable_candidate_selection`)
already compared every field of `CandidateSelectionReuseQuery` by strict SQL
equality against the winner's own recorded provenance; this slice adds
exhaustive, parameterized proof of that fact for every one of the ten
reuse-identity fields, plus a full sanity check that the exact, unmodified
query is a genuine hit before any single field is mismatched:

- model_id, revision_sha, requested_device, requested_precision,
  compiler_version, capability_fingerprint, toolchain_fingerprint,
  profile_fingerprint (the eight generation-identity fields shared with
  Slice 1's verified-recipe reuse and Slice 2's candidate registration), plus
- quality_profile_fingerprint and policy_fingerprint (the two Slice 2
  additions specific to candidate-selection reuse).

See `tests/test_recipe_candidate_attempts.py::
test_find_reusable_candidate_selection_rejects_every_reuse_identity_field_mismatch`
(parameterized over all ten fields). Null/missing/malformed fields are never
wildcarded: `LocalOnboardingService._build_candidate_selection_reuse_query`
already returns `None` (safe miss, no reuse offered at all) the moment any
single identity field on the current request is missing/blank, and
`_normalize_candidate_selection_reuse_query` rejects malformed hex/empty
values outright rather than coercing them into a match. Only the fields the
existing contracts already explicitly normalize (device/precision case,
hex-fingerprint case) are ever normalized; there is still no cross-host or
cross-toolchain reuse path anywhere.

The selection's *validated* scope (`validated_target_device`,
`validated_target_ep`, `validated_toolchain_fingerprint`,
`validated_environment_scope`) is a second, independent invalidation layer
enforced in `LocalOnboardingService._resolve_reusable_candidate_selection`:
a winner missing any one of those four fields
(`CandidateAttemptRecord.has_fully_validated_selection_scope`) is never
treated as reusable regardless of identity match, and a mismatch between the
current request's expected device/EP/toolchain/environment (derived from the
*current* generated recipe's compiled `olive`/`mobius` payload) and the
winner's recorded validated fields is likewise a safe miss, not an error.
`tests/test_local_service_candidate_orchestration.py::
test_policy_fingerprint_mismatch_forces_normal_build_dispatch_not_reuse`
proves the full-stack, service-level consequence for the policy-identity
dimension specifically: once the service's active policy fingerprint no
longer matches the winner's recorded lineage policy fingerprint, a request
against the exact same recipe fingerprint safely misses reuse and dispatches
a genuine new `BuildJob` (`build_job_id` is set, a new job id distinct from
the winner's own job is created, and the build actually runs and succeeds
end to end) — never the stale-policy winner's artifact.

### B. Stale/corrupt safety

Newly covered, service-level, in `tests/test_local_service_candidate_orchestration.py`:

- **Missing winner generated recipe.** If the winner candidate's own
  `recipe_fingerprint` no longer resolves to a `generated_recipes` row (a
  stale cache eviction, or direct corruption), reuse fails closed with
  `CANDIDATE_SELECTION_REUSE_INTEGRITY_ERROR` (500) before ever creating an
  attempt row for the requesting Idempotency-Key
  (`test_reuse_resolution_fails_closed_when_winner_generated_recipe_row_is_missing`).
- **Tampered winner recipe fingerprint.** If the winner's recorded
  `recipe_fingerprint` is repointed at a different, *genuinely existing*
  generated recipe (not merely a dangling reference), the trusted-recompile
  cross-check in `_resolve_reusable_candidate_selection` (re-deriving the
  expected trusted candidate recipe from the parent's own generated record
  and the current policy, then comparing fingerprints) still fails closed
  rather than ever serving the wrong artifact
  (`test_reuse_resolution_fails_closed_when_winner_recipe_fingerprint_is_tampered`).
- **Non-`SUCCEEDED` source.** If the winner's own linked attempt is no
  longer `SUCCEEDED` (corrupted/tampered after selection), reuse fails closed
  rather than ever reusing a non-verified source
  (`test_reuse_resolution_fails_closed_when_winner_attempt_state_is_tampered_non_succeeded`).
- **Cross-model link / unselected-lineage corruption** were already covered
  at the store level (`test_select_verified_candidate_attempt_refuses_corrupt_cross_identity_row`,
  `test_find_reusable_candidate_selection_refuses_corrupt_selected_row` in
  `tests/test_recipe_candidate_attempts.py`) and remain unchanged.
- `CandidateReuseIntegrityError` is never swallowed anywhere in this path: it
  always surfaces as a 500 `ServiceError`, and no test in this slice ever
  observes a mismatched/corrupt reuse attempt reach a `succeeded` state or a
  wrong-artifact attempt row. No private filesystem paths or raw internal
  data ever appear in these error messages — every message is built from
  already-public attempt/candidate ids and enum values, exactly like the
  existing 3B2a integrity errors.

### C. Persisted measured-zero dispatch evidence

New additive migration `_migrate_v4_to_v5` (schema version 4 → 5) creates
`candidate_reuse_dispatch_evidence` — one row per successfully materialized
reuse attempt:

| column | meaning |
| --- | --- |
| `reused_attempt_id` (PK) | the consumer's own attempt that was reused into |
| `source_attempt_id` / `source_candidate_attempt_id` / `parent_attempt_id` | the winner's own attempt/candidate/lineage identity |
| `policy_id` / `policy_version` / `policy_fingerprint` / `quality_profile_fingerprint` | the winner's recorded selection scope identity |
| `reused_without_build` | always `1` (`True`) |
| `runner_dispatch_count` / `mobius_invocation_count` / `olive_invocation_count` | always `0` for this reuse operation |
| `recorded_utc` | bookkeeping only — never part of any fingerprint/identity comparison |

`RecipeAttemptStore.finish_reused_attempt_succeeded_with_dispatch_evidence`
(see the atomicity revision below) is called from exactly three
service-layer branches, all of which are the same branches that return a
completed reuse result *without* ever creating, enqueuing, or invoking the
build stage runner: `_materialize_reused_generated_attempt`'s fresh
(`GENERATED`) and RUNNING-resume branches, and the abandoned-reuse recovery
path (below). It is never derived merely from a generation-identity
match — `tests/test_local_service_candidate_orchestration.py::
test_candidate0_default_reuse_persists_measured_zero_dispatch_evidence` and
`test_candidate1_block64_reuse_persists_measured_zero_dispatch_evidence`
install the existing runner-dispatch spy (`_install_runner_dispatch_spy`)
around the reuse call and assert both that the spy saw zero calls *and* that
the persisted evidence row shows zero counts, tying the durable record to the
actually-observed absence of dispatch. The write is idempotent (a resumed
materialization calling it twice for the same `reused_attempt_id` is a safe
no-op — the first row is never overwritten, proven in
`tests/test_recipe_candidate_attempts.py::
test_record_reuse_dispatch_evidence_persists_measured_zero_and_is_idempotent`),
and it never rewrites the source winner's own `CandidateInvocationCounters`
row — both tests assert the winner's real (non-zero, e.g. Mobius 1/Olive 1
for candidate 0 or Mobius `None`/Olive 1 for the block64 fallback) counters
are exactly value-equal before and after. Legacy (pre-migration) attempts
remain fully readable; `get_reuse_dispatch_evidence` returns `None` for any
attempt with no recorded evidence, and the migration is additive/idempotent
on reopen (`test_migration_v4_to_v5_is_additive_and_idempotent_on_reopen`).

### D. Abandoned RUNNING reuse recovery

Reviewer-identified gap: a marked `RUNNING` candidate-selection-reuse attempt
never gets a `BuildJob` mapping (reuse never dispatches one), so the
pre-existing `get_recipe_attempt` job-sync path could never reach it — if the
client only ever polls `get_recipe_attempt` and never resubmits the original
Idempotency-Key (the only thing that previously drove
`_materialize_reused_generated_attempt`'s RUNNING-resume branch), the attempt
stayed `RUNNING` forever.

`LocalOnboardingService._recover_abandoned_reuse_attempt(attempt_id)` closes
this gap:

1. Reads the durable `reuse_source_attempt_id` marker directly (never trusts
   identity match alone). Returns `None` (a strict no-op) if the attempt
   doesn't exist, isn't `RUNNING`, or carries no reuse marker at all —
   **ordinary non-reuse `RUNNING` attempts are never auto-resumed here**
   (`test_get_recipe_attempt_poll_never_auto_resumes_ordinary_non_reuse_running_attempt`).
2. Revalidates: the source is still a registered, `SELECTED` candidate
   winner (`RecipeAttemptStore.find_candidate_attempt_by_attempt_id`) with a
   fully validated selection scope, its own linked attempt actually reached
   `SUCCEEDED`, and this attempt's own recorded gate history is a strict
   prefix of the source's gate history — the same defense-in-depth check
   `_materialize_reused_generated_attempt`'s own RUNNING-resume branch
   already performs.
3. If every check passes: copies only the remaining genuine source gate
   evidence, finishes the attempt `SUCCEEDED`, and records measured-zero
   dispatch evidence — never dispatching to the build stage runner.
4. If the source is missing/corrupt/untrusted at any point:
   `_finalize_abandoned_reuse_attempt_failed` fails the attempt terminally
   (classification `INTERNAL_ERROR`, stage
   `candidate_selection_reuse_recovery`) with an explicit, sanitized reason —
   never left `RUNNING` forever
   (`test_get_recipe_attempt_poll_fails_abandoned_reuse_attempt_terminally_when_source_untrusted`).

Wired into two call sites, both idempotent and safe across restarts/polls:

- **`get_recipe_attempt`** — when an attempt is `RUNNING` with no mapped
  `BuildJob`, a single poll now completes (or fails) an abandoned reuse
  attempt with no resubmission at all
  (`test_get_recipe_attempt_poll_recovers_abandoned_running_reuse_attempt_without_resubmission`).
- **`LocalOnboardingService.__init__`** — a new
  `_recover_abandoned_reuse_attempts_at_startup` pass (run once, after
  `_recover_orphaned_candidate_lineages`) proactively resolves every
  abandoned `RUNNING` reuse attempt on a fresh service startup alone, with no
  client poll or resubmission at all
  (`test_fresh_service_startup_recovers_abandoned_running_reuse_attempt_with_no_poll_or_resubmission`).

Both call sites — and a concurrent Idempotency-Key resubmission through
`_materialize_reused_generated_attempt` — share the exact same per-attempt
`_AttemptSyncGuard` keyed by `attempt_id` that 3B2a already introduced, so a
poll racing a resubmission (or two concurrent polls) can never double-finish,
double-fail, or deadlock: every caller serializes on the same lock, re-reads
the attempt's current state once it acquires it, and only one caller ever
performs the terminal transition
(`test_abandoned_reuse_poll_and_resubmit_race_is_safe`). The guard map is
always empty again once every concurrent caller has returned, exactly as in
3B2a's own concurrency tests.

## Reviewer-REJECTED fix: reuse success/evidence atomicity (Linus revision)

Scope for this revision, entirely inside `local_service.py` /
`recipe_attempt_store.py` and their test files, same constraints as Slice
3B2b above. No API/OpenAPI/route/UI, runner, selection trigger/policy,
quality profile/prompts, or root index changed. No real tools/models.

### Issue 1 — non-atomic success + evidence

The original 3B2b code called `finish_attempt_succeeded()` (its own
committed transaction) and then separately `record_reuse_dispatch_evidence()`
in a second transaction. A crash or error between the two left a terminal
`succeeded` reuse attempt with no evidence row — invisible forever to every
recovery/poll/resubmit path, all of which only ever looked for a
*non-terminal* (`RUNNING`) reuse attempt to resume.

`RecipeAttemptStore.finish_reused_attempt_succeeded_with_dispatch_evidence`
closes this gap: one `BEGIN IMMEDIATE` transaction that (1) loads/validates
the reused attempt is `RUNNING` (or idempotently already `succeeded` with
identical evidence), (2) verifies the durable `reuse_source_attempt_id`
marker matches the supplied source, (3) revalidates the source using the
same trusted store invariants 3B2a/3B2b already established (registered,
`SELECTED` candidate winner with a fully validated selection scope, its own
attempt `SUCCEEDED`, full generation-identity match via
`_require_matching_candidate_generation_identity`), (4) inserts the durable
measured-zero evidence row — deriving every identity-bearing field from the
source's own trusted candidate-plan row, never from caller-supplied values —
and (5) transitions the attempt to `succeeded`. Any failure at any step rolls
back the whole transaction (the store's `_connect()` context manager already
rolls back on any raised exception): the attempt is left exactly as it was
(`RUNNING`, no evidence row), fully recoverable by the existing
startup/poll/resubmit resume logic, never a terminal `succeeded` attempt with
no evidence.

`_materialize_reused_generated_attempt` (both its fresh-`GENERATED` and
`RUNNING`-resume branches) and `_recover_abandoned_reuse_attempt` now call
this single atomic method directly; every split
"`finish_attempt_succeeded` then separately record evidence" call site is
gone. Ordinary (non-reuse) `finish_attempt_succeeded` is untouched.

Already-`succeeded` with **no** evidence (the exact legacy/crash shape the
old code could produce) is deliberately never silently re-completed by this
method — see
`tests/test_recipe_candidate_attempts.py::test_finish_reused_attempt_succeeded_rejects_already_succeeded_missing_evidence`.
That case can only be healed by the explicit legacy-backfill path below.

Crash-injection coverage (`tests/test_recipe_candidate_attempts.py`):
`test_finish_reused_attempt_succeeded_rolls_back_both_on_injected_evidence_insert_failure`
and
`test_finish_reused_attempt_succeeded_rolls_back_both_on_injected_failure_after_update_before_commit`
each inject a failure at a different point inside the transaction (before
the evidence insert, and after the evidence insert *and* the `succeeded`
`UPDATE` but before the final commit) and assert full rollback (`RUNNING`,
no evidence) followed by a single successful completion on retry. At the
service level,
`tests/test_local_service_candidate_orchestration.py::test_reuse_materialization_atomic_finish_evidence_injected_failure_rolls_back_and_resumes_once_on_fresh_service`
proves the same thing through the real `create_generated_recipe_attempt`
request path and a fresh service instance, mirroring the existing 3B2a/3B2b
crash-recovery tests.

### Issue 2 — conflicting evidence silently ignored

`record_reuse_dispatch_evidence` previously returned the existing row on a
repeat call for the same `reused_attempt_id` without ever comparing the new
call's identity fields against what was already persisted. It now routes
through a shared `_insert_or_verify_reuse_dispatch_evidence_locked`: every
identity-bearing field (`source_attempt_id`, `source_candidate_attempt_id`,
`parent_attempt_id`, `policy_id`, `policy_version`, `policy_fingerprint`,
`quality_profile_fingerprint`) is compared against an already-persisted row.
An identical replay stays a safe no-op; any single mismatched field raises
`CandidateReuseIntegrityError` naming only the mismatched field(s), never
overwriting the existing row and never echoing raw values. The four
store-fixed measured fields (`reused_without_build`,
`runner_dispatch_count`, `mobius_invocation_count`,
`olive_invocation_count`) are never compared — they are always the same
constants for every row, never caller-supplied. See
`tests/test_recipe_candidate_attempts.py::test_record_reuse_dispatch_evidence_rejects_every_conflicting_identity_field_mismatch`
(parameterized over all seven fields).

### Legacy/crash backfill

For attempts a pre-fix service could already have left `succeeded` with a
reuse marker but no evidence:

- `RecipeAttemptStore.find_legacy_succeeded_reuse_attempts_missing_evidence`
  is a bounded, indexed query (`idx_attempts_reuse_source_by_state`) finding
  every such row without a full table scan.
- `RecipeAttemptStore.backfill_reused_attempt_dispatch_evidence` atomically
  revalidates the source with the exact same trusted invariants as above and
  inserts the missing evidence — it never reruns any tool, never touches the
  already-immutable attempt row, and never rewrites any gate/source
  invocation counter.
- If the source is corrupt/untrusted, no evidence is ever fabricated;
  instead `LocalOnboardingService._backfill_legacy_reuse_dispatch_evidence`
  records an explicit, sanitized audit row via
  `RecipeAttemptStore.record_reuse_evidence_backfill_integrity_failure` (new
  `candidate_reuse_evidence_backfill_failures` table) so the gap stays
  detectable instead of silently persisting forever — never a hard failure
  for an unrelated poll/resubmit caller.
- Wired opportunistically into `get_recipe_attempt` (poll), the
  `_materialize_reused_generated_attempt` resubmission early-return branch,
  and a new startup sweep `_recover_legacy_reuse_dispatch_evidence_at_startup`
  (run once, after `_recover_abandoned_reuse_attempts_at_startup`).

See `tests/test_recipe_candidate_attempts.py::
test_find_legacy_succeeded_reuse_attempts_missing_evidence_and_backfill_is_idempotent`,
`test_backfill_reused_attempt_dispatch_evidence_fails_closed_for_untrusted_source`,
and `test_record_reuse_evidence_backfill_integrity_failure_is_durable_sanitized_and_idempotent`
at the store level, and
`tests/test_local_service_candidate_orchestration.py::
test_startup_backfills_legacy_succeeded_reuse_attempt_missing_evidence`,
`test_poll_backfills_legacy_succeeded_reuse_attempt_missing_evidence`,
`test_resubmit_backfills_legacy_succeeded_reuse_attempt_missing_evidence`, and
`test_backfill_records_audit_failure_and_never_fabricates_evidence_for_untrusted_legacy_source`
at the service level.

### Schema: additive `_migrate_v5_to_v6` (schema version 5 → 6)

Migrations are treated as immutable/append-only once shipped: the 3B2b
`_migrate_v4_to_v5` step already produced real `user_version=5` databases,
so this revision's new `candidate_reuse_evidence_backfill_failures` table and
`idx_attempts_reuse_source_by_state` index are added in a brand-new
`_migrate_v5_to_v6` step instead of being folded back into the already-shipped
v4→v5 migration, which a real v5 database would never re-run. See
`tests/test_recipe_candidate_attempts.py::test_migration_v5_to_v6_is_additive_and_idempotent_on_reopen`,
which constructs an exact f57ba7e-shape v5 database (containing
`candidate_reuse_dispatch_evidence` but neither of the new v6 objects) and
proves the upgrade is additive, data-preserving, and idempotent on reopen.

### Locking

Every new/changed store method uses the store's own transaction
(`BEGIN IMMEDIATE`) plus the existing per-attempt `_AttemptSyncGuard`; no
call site holds the service-wide `self._lock` across any of this work,
preserving the existing 3B1 lock-ordering design.

## Slice 3C handoff

Everything above is internal service/store behavior with **no** change to
any public FastAPI/OpenAPI response contract, `web`/`web_dist`, runner build
semantics, selection trigger/policy, quality profile/prompts, or root index.
The exact fields Slice 3C's public API/UI work will need to surface, all
already present on internal records today:

- **Candidate-selection reuse** (`_candidate_selection_reuse_payload`,
  already returned in preview/attempt payloads): `source_parent_attempt_id`,
  `winner_candidate_attempt_id`, `winner_attempt_id`,
  `winner_candidate_index`, `winner_candidate_id`,
  `winner_recipe_fingerprint`, `selection_reason`, `policy_fingerprint`,
  `quality_profile_fingerprint`, `winner_job_id`.
- **Measured-zero dispatch evidence** (`CandidateReuseDispatchEvidence`, new
  in this slice, read via `RecipeAttemptStore.get_reuse_dispatch_evidence`):
  `reused_attempt_id`, `source_attempt_id`, `source_candidate_attempt_id`,
  `parent_attempt_id`, `policy_id`, `policy_version`, `policy_fingerprint`,
  `quality_profile_fingerprint`, `reused_without_build`,
  `runner_dispatch_count`, `mobius_invocation_count`,
  `olive_invocation_count`, `recorded_utc` (bookkeeping only — should not be
  surfaced as if it were a fingerprint/identity field).
- **Abandoned-reuse recovery outcome**: an attempt recovered this way is
  indistinguishable in its serialized shape from any other reuse
  materialization (`state`, `build_job_id: null`, full `gates`) — Slice 3C
  does not need a separate "recovered" flag; the durable
  `CandidateReuseDispatchEvidence` row (if surfaced) already lets a caller
  tell a reuse attempt apart from a real build.
- **No product claims beyond tests**: everything documented above is proven
  by the specific tests named inline; nothing here describes behavior beyond
  what those tests exercise.
