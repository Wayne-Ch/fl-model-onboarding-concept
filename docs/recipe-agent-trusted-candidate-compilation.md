# Recipe Agent Slice 3A1: trusted candidate compilation + real invocation instrumentation

This slice adds two things, both scoped to `recipe_compiler.py` and `production_runner.py`:

1. A trusted internal boundary that compiles Slice 1's approved candidate 1 (Olive CPU INT4
   `block_size=64`) into a *real*, executable `GeneratedRecipe` — the actual field/command
   `ProductionBuildStageRunner`/Olive consume, not just planning metadata.
2. Real, per-job Mobius/Olive invocation instrumentation in `ProductionBuildStageRunner`,
   replacing the nullable placeholder counters Slice 2 already defined
   (`CandidateInvocationCounters` in `recipe_attempt_store.py`) with actual measured values.

Neither of these wires anything into `local_service.py`, the API/OpenAPI routes, or actual
baseline/pre-Olive reuse. See "What this slice does not do" below.

## A. Trusted candidate compilation boundary

`compile_trusted_candidate_recipe(default_recipe, *, policy, candidate, schema_path=None)` in
`recipe_compiler.py` is the **only** place in this codebase permitted to layer a quantization
override onto a compiled recipe's actual Olive arguments.

Inputs:

- `default_recipe`: an already-compiled default `GeneratedRecipe` (from
  `compile_generated_recipe`).
- `policy`: a `RecipeSelectionPolicy` (Slice 1).
- `candidate`: one `RecipeSelectionCandidate` from that policy.

There is no raw/free-form argument parameter — the function cannot be asked to apply an
arbitrary Olive flag; it only knows how to apply the one override type
(`RecipeQuantizationOverride.block_size`) that Slice 1's schema allows.

### Identity validation (fail-closed)

Both `policy` and `candidate` must **exactly** identity-match (full dataclass equality) an
entry already present in the trusted, schema-validated
`DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY`:

- `policy.target_device` must be `cpu` and `policy.quantization` must be `int4`.
- `policy.policy_id` must resolve in the trusted registry, and the *entire* resolved policy
  object (version, fingerprint, full candidate set) must equal the object passed in.
- `candidate.candidate_index` must be in range, and `policy.candidates[candidate_index]` must
  equal the `candidate` object passed in — byte-for-byte, including its `quantization_override`.

Any drift — a different `block_size`, a renamed `candidate_id`, an unknown `policy_id`, an
out-of-range/mismatched `candidate_index`, an altered `eligibility_trigger` — raises
`TrustedCandidateCompilationError` before any recipe field is touched. Nothing here branches on
model id/name/org/shape: the same identity check and override application applies uniformly to
whatever `default_recipe` is passed in.

### Candidate 0 (default)

Returned **unchanged**: same object, same fingerprint, same payload as `default_recipe`.
Behavior/payload compatibility with the existing compiler is exact, not approximate. A candidate
0 that declares a (tampered) `quantization_override` is rejected by the identity check above
before it ever reaches the "candidate 0 must not override" check.

### Candidate 1 (block_size=64)

1. Asserts the default recipe's Olive `device`/`precision` match the policy's
   `target_device`/`quantization` and that no `block_size` is already set (refusing to layer an
   override on top of another).
2. Applies `OliveRecipeArgs.block_size = override.block_size` via `dataclasses.replace` — the
   *only* field touched.
3. Asserts `device`/`precision` are unchanged after the `replace` (defense in depth against a
   future edit to this function smuggling a device/precision change through).
4. Attaches a `TrustedCandidateProvenance` record (`policy_id`, `policy_version`,
   `policy_fingerprint`, `candidate_index`, `candidate_id`, `resolved_block_size`) to the
   recipe's provenance.

`OliveRecipeArgs.block_size` defaults to `None` for every statically-registered recipe in
`recipes.py` — it is never a static architecture/model default, only ever set by this trusted
boundary.

### Fingerprint identity

The resulting recipe/fingerprint is deterministic and reflects:

- the actual resolved override (`block_size` is part of the canonical Olive payload the
  fingerprint hashes over), **and**
- the exact policy/candidate identity that approved it (`TrustedCandidateProvenance` is part of
  the canonical provenance payload too — so two different policy versions that happen to resolve
  the same `block_size` still produce different fingerprints),

while the pinned revision, toolchain versions, capability id/version, and task profile are
carried over from `default_recipe` unchanged (verified by
`tests/test_recipe_compiler.py::test_trusted_candidate_block64_applies_override_and_changes_fingerprint`).

### Schema

`contracts/generated-recipe.schema.json` adds:

- `$defs.olive_args.properties.block_size`: `integer | null`, optional.
- `$defs.provenance.properties.trusted_candidate` → `$defs.trusted_candidate`: `object | null`,
  required (always present, `null` when this recipe was not produced by the trusted boundary).

### Execution wiring already in place

`production_runner.py`'s existing generated-recipe execution path
(`RecipeExecutionResolver._resolve_generated_recipe` → `_load_generated_recipe_execution_plan` →
`_recipe_from_payload`) now also parses `olive.block_size` back out of a persisted
`GeneratedRecipeRecord.canonical_json`, and `ProductionBuildStageRunner._run` adds
`--block_size <n>` to the real Olive `optimize` command line whenever
`recipe.olive.block_size is not None`. No other part of that execution path changed.

## B. Real Mobius/Olive invocation instrumentation

`contracts.py` adds `ToolInvocationEvidence` (one tool) and `ProductionInvocationEvidence`
(both tools), and `BuildJob` gets a new `production_invocation_evidence` field, defaulting to
`None`.

### Count semantics

- `invocation_count` increments **only** when the external process launch is actually attempted
  — immediately before `ProcessRunner.run(...)` is called for that tool — and it increments
  exactly once per attempt, regardless of what happens next (success, non-zero exit, timeout,
  cancellation, or any other exception).
- If upstream validation (recipe/task-profile/skip_olive checks, missing Olive settings, etc.)
  prevents the launch from ever happening, the count stays `0` and `terminal_stage` stays
  `not_run` — it is never inferred as "zero invocations occurred" for a real run.
- `terminal_stage` records one of: `not_run`, `completed` (exit 0), `failed` (non-zero exit or
  any other exception after launch), `timed_out` (`TimeoutError`), `cancelled` (any exception
  raised while the caller's `cancellation_event` is set).
- `wall_seconds`/`started_utc`/`finished_utc` are recorded around the actual launch attempt;
  `success` is `True` only for `completed`.

### Per-job isolation, not global state

A fresh `ProductionInvocationEvidence()` is created at the start of `ProductionBuildStageRunner._run`
and stored on that job's `BuildJob.production_invocation_evidence` — never as mutable state on the
runner instance. Since `ProductionBuildStageRunner` is a single long-lived object that may run many
jobs (including concurrently, from different threads), storing evidence per-`BuildJob` instead of
per-runner is what makes concurrent jobs safe: `test_concurrent_runs_have_isolated_invocation_counters`
runs several jobs against the same runner instance from separate threads and asserts each job's
counters land at exactly 1/1, never accumulated or raced.

### Persistable/sanitized shape

`ProductionInvocationEvidence.sanitized_payload()` returns only counts, terminal stage/success,
and wall-clock timing — never raw commands/argv, secrets, or absolute filesystem paths.

### Bridging to the Slice 2 store shape

`production_invocation_evidence_to_candidate_counters(evidence)` in `production_runner.py`
converts real per-job evidence into the store's existing nullable
`CandidateInvocationCounters` shape (`recipe_attempt_store.py`, unmodified by this slice): a
tool's count is only ever reported once that tool was actually attempted; an untouched tool
stays `None`, matching that dataclass's documented null semantics. This function only builds the
value — actually calling `RecipeAttemptStore.finalize_candidate_attempt_evidence(...)` to persist
it against a specific `candidate_attempt_id` is orchestration wiring for a later slice.

## What this slice does *not* do

- No baseline/pre-Olive reuse across candidates. Safe reuse of already-built Mobius output when
  only the Olive step differs between the default and block_size=64 candidates is **Slice 3A2**,
  not implemented here.
- No service orchestration: nothing here calls `register_candidate_attempt` /
  `finalize_candidate_attempt_evidence` on a real `candidate_attempt_id`, and `local_service.py`
  is unchanged.
- No API/UI wiring, no OpenAPI changes, no real model/tool runs (all tests use fake process
  runners).
- No model ID/name/org/shape branches anywhere in the trusted compilation path.
