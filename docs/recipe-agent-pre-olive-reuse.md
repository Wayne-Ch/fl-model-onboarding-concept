# Recipe Agent Slice 3A2: safe immutable reuse of a successful pre-Olive Mobius artifact

This slice adds a runner-level, runtime-only primitive to `production_runner.py`: an
approved fallback candidate can run Olive against an independent copy of an already-built,
successful default candidate's pre-Olive Mobius output, instead of re-running Mobius itself.

Nothing here wires into `local_service.py`, the recipe-attempt store's orchestration, the
API/OpenAPI routes, `web`/`web_dist`, `runtime_worker` quality, frozen models/profile/prompts,
or the root index. `capture_pre_olive_artifact`, `validate_pre_olive_reuse`,
`revalidate_pre_olive_source`, and `materialize_pre_olive_copy` are standalone functions; the
only new method on `ProductionBuildStageRunner` is `run_fallback_with_pre_olive_reuse`, an
additive entry point that never runs unless a caller (Slice 3B) explicitly invokes it. The
existing `run()`/`_run()` one-shot path is behaviorally unchanged.

## Why this is safe: the trust chain

Reuse is only ever safe because Slice 3A1's `compile_trusted_candidate_recipe` guarantees a
fallback (block_size-override) candidate's recipe is identical to the default candidate's
recipe in every respect *except* `OliveRecipeArgs.block_size` — same model/revision, same
Mobius `ep`/`runtime`/`dtype`/`task`, same Olive `task`/`precision`/`device`/`provider`. Slice
3A2 does not re-derive that guarantee; it re-*checks* the two pieces of it that matter for
Mobius reuse (generation identity, Mobius arguments) at reuse time, independently of whatever
compiled the fallback recipe, so a bug or a hand-built (non-compiler) fallback request can never
silently bypass the check.

## Runtime-only source path vs. persistable/sanitized evidence

`PreOliveArtifactDescriptor.mobius_source_dir` is a real absolute filesystem path on the machine
that ran the default candidate's Mobius build. It is **never** included in
`PreOliveArtifactDescriptor.sanitized_payload()`. Callers that need to log, persist, or return
this descriptor over an API must use `sanitized_payload()` (or `descriptor.logical_ref`
directly), which only ever exposes:

- `generation_identity` (the 8 identity fields, all fingerprints/ids — no paths),
- `mobius_args_fingerprint`,
- `manifest_hash` / `manifest_total_bytes` / `manifest_file_count`,
- `logical_ref` (== `manifest_hash`; the stable, path-free reference to reuse),
- `captured_utc`, `source_attempt_id`, `source_candidate_id`.

`test_sanitized_payload_excludes_raw_source_path` asserts the raw source directory string never
appears anywhere in the serialized sanitized payload.

## Capturing a descriptor

```python
descriptor = capture_pre_olive_artifact(
    mobius_source_dir=default_job.request.workspace_root / "mobius",
    authorized_root=default_job.request.workspace_root,
    generation_identity=pre_olive_generation_identity_from_generated_record(default_generated_record),
    mobius_args=default_recipe.mobius,
    source_attempt_id=default_attempt_id,      # optional
    source_candidate_id="0",                    # optional
)
```

`capture_pre_olive_artifact`:

1. Asserts `mobius_source_dir` resolves inside `authorized_root` (the owning job's workspace
   root) — an escape fails closed.
2. Walks `mobius_source_dir` and builds a `PreOliveManifest`: every regular file's
   relative path, size, and streaming SHA-256 (bounded-memory reads, never a full file load), a
   deterministic `manifest_hash` over the sorted entry list, `total_bytes`, and `file_count`.
   Any symlink or Windows reparse point (junction) encountered anywhere in the tree — including
   the root itself — is rejected, never followed.
3. Refuses to capture an empty directory (zero files).
4. Returns an immutable `PreOliveArtifactDescriptor` combining the generation identity, a
   `compute_mobius_args_fingerprint(mobius_args)` fingerprint, the manifest, a capture
   timestamp, and the optional source attempt/candidate identifiers.

`PreOliveGenerationIdentity` mirrors `recipe_attempt_store.RecipeReuseQuery`'s eight fields
(`model_id`, `revision_sha`, `requested_device`, `requested_precision`, `compiler_version`,
`capability_fingerprint`, `toolchain_fingerprint`, `profile_fingerprint`) but is a separate,
narrower type — this runner-level primitive has no coupling to recipe-attempt store
persistence. `pre_olive_generation_identity_from_generated_record` is a convenience constructor
for the common case where both candidates are persisted `GeneratedRecipeRecord` rows.
`compute_mobius_args_fingerprint` hashes exactly `ep`/`runtime`/`dtype`/`task` — the fields
`ProductionBuildStageRunner._run` actually threads into the real Mobius `build` command line —
independent of (and in addition to) the broader generation-identity match, satisfying the
explicit "Mobius args are identical" requirement.

## Validating and reusing

`ProductionBuildStageRunner.run_fallback_with_pre_olive_reuse(job, *, descriptor,
fallback_generation_identity, persist, cancellation_event)`:

1. Resolves `job.request` through the same `RecipeExecutionResolver` every other execution path
   uses, so every existing generated-recipe/attempt-identity/experimental-status check still
   applies unchanged.
2. Re-checks the resolved recipe's model id/revision against `descriptor.generation_identity`
   directly, then calls `validate_pre_olive_reuse(descriptor, candidate_identity=...,
   candidate_mobius_args=recipe.mobius)`, which fails closed
   (`PreOliveReuseError`, classification `COMPATIBILITY`) on **any** difference in the 8-field
   generation identity or in the Mobius-args fingerprint. Only an approved Olive `block_size`
   difference is tolerated — because `block_size` is not part of either check.
3. Transitions `DOWNLOADING` → `MOBIUS_BUILDING` (no Hugging Face download or Mobius launch
   happens on this path — the transitions exist only because the job state machine requires
   passing through them) and calls `materialize_pre_olive_copy` to revalidate the source and
   copy it into the fallback job's own `workspace_root / "mobius"`.
4. Transitions to `MOBIUS_VALIDATING`, then delegates to the same `_run_from_olive` helper the
   legacy path uses — Olive optimize, packaging, decoder-output reconciliation, runtime
   validation, Foundry Local SDK inference, and artifact registration all run completely
   unchanged.

`materialize_pre_olive_copy(descriptor, *, destination_dir, authorized_roots,
cancellation_event=None)`:

1. Calls `revalidate_pre_olive_source(descriptor)` first — re-walks and re-hashes the captured
   source directory and fails closed on drift: missing directory, missing/added/renamed files,
   changed content (tamper), or a symlink/reparse point now present anywhere in the tree.
2. Asserts `destination_dir` resolves inside one of `authorized_roots`, does not already exist,
   and does not equal or overlap the source directory (neither contains the other).
3. Copies every manifested file with a bounded-memory streaming read/write — never a hardlink,
   symlink, junction, or `shutil.copytree` shortcut — checking the cancellation event and each
   source file's link/reparse status before every copy.
4. Re-manifests the destination and compares its hash against the descriptor's manifest hash;
   any mismatch (a partial or corrupted copy) fails closed.
5. On **any** failure — including an observed cancellation — removes exactly the
   `destination_dir` this call created (never a broader or unresolved path) before re-raising,
   so the source is always left untouched and the failure is auditable.

## Counter semantics for a fallback child

Mobius is never invoked for the fallback path, so `job.production_invocation_evidence.mobius`
stays at its ordinary not-run default (`invocation_count=0`, `terminal_stage=NOT_RUN`).
`production_invocation_evidence_to_candidate_counters` (unchanged from Slice 3A1) already maps a
0-count, not-run tool to `None`, not `0` — so a fallback child's evidence surfaces as Mobius
`None`/Olive `1` today. **Aggregating** a parent (Mobius 1) and its fallback child (Olive 1) into
the combined "Mobius1/Olive2" view described in the Slice 3A2 spec is Slice 3B's job (it owns
`candidate_attempt`/lineage aggregation in `recipe_attempt_store.py`); this slice only guarantees
each individual job's own counters are correct in isolation.

## Slice 3B call sequence

```python
# 1. Run the default candidate exactly as today (no change).
production.run(default_job, persist=persist, cancellation_event=event)
assert default_job.state == JobState.SUCCEEDED

# 2. Capture its pre-Olive artifact while its workspace is still on disk.
descriptor = capture_pre_olive_artifact(
    mobius_source_dir=default_job.request.workspace_root / "mobius",
    authorized_root=default_job.request.workspace_root,
    generation_identity=pre_olive_generation_identity_from_generated_record(default_record),
    mobius_args=default_recipe.mobius,
    source_attempt_id=default_attempt_id,
    source_candidate_id="0",
)

# 3. Compile the trusted fallback recipe (Slice 3A1, unchanged) and build its own
#    BuildRequest/BuildJob in a *separate* workspace_root.
fallback_recipe = compile_trusted_candidate_recipe(default_candidate, policy=policy, candidate=policy.candidates[1])
fallback_identity = pre_olive_generation_identity_from_generated_record(fallback_record)

# 4. Run the fallback candidate against the captured descriptor -- Mobius never runs.
production.run_fallback_with_pre_olive_reuse(
    fallback_job,
    descriptor=descriptor,
    fallback_generation_identity=fallback_identity,
    persist=persist,
    cancellation_event=event,
)
assert fallback_job.state == JobState.SUCCEEDED
# fallback_job.production_invocation_evidence.mobius.invocation_count == 0
# fallback_job.production_invocation_evidence.olive.invocation_count == 1
```

`tests/test_pre_olive_reuse.py::test_run_fallback_with_trusted_block64_candidate_end_to_end`
exercises this exact sequence with real `compile_generated_recipe`/`compile_trusted_candidate_recipe`
output (fake process runners only) and asserts the real Olive command line carries
`--block_size 64`.

## Safety invariants

- **Fail closed before any Olive launch.** Missing/tampered content, an out-of-authorized-root
  path, a symlink/reparse point, a manifest mismatch, source/destination overlap, or a
  generation-identity/Mobius-args mismatch all raise `PreOliveReuseError` before
  `materialize_pre_olive_copy` or `run_fallback_with_pre_olive_reuse` ever reach the Olive
  command line.
- **Reject links, never follow them.** `_path_is_link_or_reparse_point` fails closed (treats an
  un-`lstat`-able path as unsafe) and is checked for every directory/file encountered during a
  manifest walk or copy, on the manifest root itself, and on each source file immediately before
  it is copied.
- **Cleanup is narrow.** A failed/cancelled copy removes exactly the `destination_dir` it
  created — never a broader or unresolved path — leaving the source directory and any
  already-existing destination untouched.
- **Default path is unchanged.** `run()`/`_run()` behave identically whether or not this slice
  exists; `_run` now delegates its post-Mobius half to the shared `_run_from_olive` helper, but
  that is a pure refactor with no observable behavioral difference
  (`test_default_legacy_path_without_descriptor_remains_mobius1_olive1`).
- **Concurrency-safe.** Each `run_fallback_with_pre_olive_reuse` call operates on its own
  `BuildJob`/`BuildRequest`/workspace, so concurrent reuse calls against the same descriptor
  never share or race on destinations or invocation evidence
  (`test_concurrent_fallback_reuse_calls_have_isolated_destinations_and_evidence`).

## What this slice does *not* do

- No service orchestration: nothing here calls into `local_service.py`, registers/finalizes a
  real `candidate_attempt_id` in `recipe_attempt_store.py`, or exposes any of this over an
  API/route. `run_fallback_with_pre_olive_reuse` must be called directly by Slice 3B.
- No Mobius1/Olive2 lineage aggregation — that view is computed by Slice 3B from the parent's
  and fallback child's individually-correct evidence.
- No real model/tool runs; every test in `tests/test_pre_olive_reuse.py` uses fake process
  runners.
- No changes to `web`/`web_dist`, `runtime_worker` quality validation, frozen models/profile/
  prompts, or the root index.
