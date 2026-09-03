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
   evidence only (baseline passes the affected structural constraint, optimized is
   runtime-functional, the *only* failures are the narrow structural allowlist
   (`json_format_invalid` / an output-format `forbidden_token_present`), and the
   recipe is blocked *solely* because of that regression). Unknown/malformed/
   partial evidence, baseline unavailable, both baseline and optimized failing,
   capability-only mismatches, pathological/runtime failures, and plain
   non-structural regressions are all `NOT_RETRYABLE` by construction — never by
   free-text matching.
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

## What this slice explicitly does *not* do (exact Slice 3B2 handoff)

- **No selected-candidate reuse short-circuit for a *new* user request.** Slice 2's
  `find_reusable_candidate_selection(CandidateSelectionReuseQuery)` — a read-only
  lookup of a previously-selected verified candidate winner by full
  policy-aware/quality-profile-aware identity — is never called anywhere in
  `local_service.py`. Every generated-attempt request in this slice still always
  executes candidate 0 (and, when triggered, candidate 1) for real; there is no
  zero-build path. (Note: the pre-existing, policy-unaware `find_reusable_verified_recipe`
  / `generated_recipe_preview` "verified recipe" reuse cache is untouched and
  continues to work exactly as it did before 3B1 — including for a recipe promoted
  via the fallback candidate, since `build_profile_fingerprint` does not encode
  `block_size`. That is pre-existing Slice 1 behavior, not new "candidate selection
  reuse", and is unrelated to the scope excluded here.)
- **Exact 3B2 handoff**: wire `find_reusable_candidate_selection` into
  `create_generated_recipe_attempt` (or an earlier preview/routing step), keyed by
  `CandidateSelectionReuseQuery` (model/revision/device/precision/compiler/
  capability/toolchain/profile fingerprints **plus** `quality_profile_fingerprint`
  and `policy_fingerprint`). On a hit, serve the already-selected winner's
  artifact/package directly — no `BuildJob`, no candidate 0/1 execution at all —
  while still respecting `CandidateReuseIntegrityError`'s fail-closed
  cross-identity guard. This slice deliberately stops one call short of that wiring.
- **No API/OpenAPI/route/UI changes.** Candidate lineage/evidence/selection state is
  only ever visible through direct `RecipeAttemptStore` access (as these tests do);
  `_serialize_recipe_attempt`'s response shape is unchanged.
- **No real model/tool runs.** Every test in
  `tests/test_local_service_candidate_orchestration.py` uses a real
  `ProductionBuildStageRunner` wired with a fully faked `ProcessRunner` and a fake,
  deterministic `TextInferenceBackend` — no network access, no real Mobius/Olive/
  onnxruntime/Foundry Local tooling.
