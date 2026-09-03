"""Slice 3A2: safe immutable reuse of a successful pre-Olive Mobius artifact.

These tests exercise the runner-level primitives added to
`production_runner.py` -- `capture_pre_olive_artifact`,
`validate_pre_olive_reuse`, `revalidate_pre_olive_source`,
`materialize_pre_olive_copy`, and
`ProductionBuildStageRunner.run_fallback_with_pre_olive_reuse` -- entirely
with fake process runners (no real Mobius/Olive/model tooling). Nothing here
wires into `local_service.py`, the recipe-attempt store's orchestration, or
any API/route.
"""

from __future__ import annotations

import json
import sys
import threading

from pathlib import Path
from threading import Event

import pytest

import fl_model_onboarding.production_runner as production_runner_module
from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec
from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import (
    BuildJob,
    BuildRequest,
    FailureClassification,
    JobState,
    ToolInvocationTerminalStage,
)
from fl_model_onboarding.production_runner import (
    ProductionBuildStageRunner,
    PreOliveGenerationIdentity,
    PreOliveReuseError,
    SMOLLM2_REVISION,
    capture_pre_olive_artifact,
    materialize_pre_olive_copy,
    pre_olive_generation_identity_from_generated_record,
    revalidate_pre_olive_source,
    validate_pre_olive_reuse,
)
from fl_model_onboarding.recipe_compiler import compile_trusted_candidate_recipe
from fl_model_onboarding.recipe_selection_policy import DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
from fl_model_onboarding.recipes import MobiusRecipeArgs
from fl_model_onboarding.state_machine import transition

# Reuse existing generated-attempt compilation/fixture helpers from the
# Slice 3A1 test module instead of duplicating them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_production_runner as tpr  # noqa: E402


class ContractProcessRunner:
    def __init__(self) -> None:
        self.specs: list[CommandSpec] = []
        self.cancel_events: list[object] = []

    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        self.specs.append(spec)
        self.cancel_events.append(cancel_event)
        argv = spec.argv
        stdout = ""
        if argv[:2] == ("mobius", "build"):
            output = Path(argv[-1])
            (output / "model.onnx").write_bytes(b"onnx")
            (output / "genai_config.json").write_text("{}", encoding="utf-8")
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")
        elif argv[:2] == ("olive", "optimize"):
            output = Path(argv[argv.index("--output_path") + 1])
            (output / "model.onnx").write_bytes(b"optimized")
            (output / "genai_config.json").write_text("{}", encoding="utf-8")
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")
        elif "validate-runtime" in argv:
            stdout = json.dumps(
                {
                    "ok": True,
                    "checks": ["onnx_checker=1", "ort_cpu_load=passed", "oga_generation=passed"],
                }
            )
        elif "foundry-infer" in argv:
            stdout = json.dumps({"ok": True, "output": "OK"})
        return CommandResult(spec=spec, exit_code=0, stdout=stdout, stderr="")


class PinnedSnapshot:
    def acquire_snapshot(
        self,
        model_id: str,  # noqa: ARG002
        local_dir: Path,
        revision: str | None = None,  # noqa: ARG002
        allow_patterns=None,  # noqa: ANN001, ARG002
    ) -> Path:
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return local_dir


def _request(tmp_path: Path, *, name: str) -> BuildRequest:
    return BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=tmp_path / name,
        model_cache_dir=tmp_path / f"{name}-cache",
        output_dir=tmp_path / name / "output",
        task_profile="llm-cpu-int4",
        hf_revision=SMOLLM2_REVISION,
    )


def _identity(**overrides: object) -> PreOliveGenerationIdentity:
    base: dict[str, object] = dict(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision_sha=SMOLLM2_REVISION,
        requested_device="cpu",
        requested_precision="int4",
        compiler_version="1.0.0",
        capability_fingerprint="cap" + "0" * 61,
        toolchain_fingerprint="tool" + "0" * 60,
        profile_fingerprint="prof" + "0" * 60,
    )
    base.update(overrides)
    return PreOliveGenerationIdentity(**base)  # type: ignore[arg-type]


_SMOLLM2_MOBIUS_ARGS = MobiusRecipeArgs(ep="cpu", runtime="ort-genai", dtype="f32")


def _run_default_job_to_success(tmp_path: Path, *, name: str) -> tuple[BuildJob, BuildRequest, Path]:
    """Run the existing legacy one-shot path (static verified SmolLM2 recipe)
    to a real SUCCEEDED job using fakes, and return the job/request plus the
    still-on-disk pre-Olive Mobius directory it produced."""
    request = _request(tmp_path, name=name)
    request.workspace_root.mkdir(parents=True)
    request.model_cache_dir.mkdir(parents=True)
    job = BuildJob(job_id=f"{name}-job", request=request)
    transition(job, JobState.PREFLIGHT, "Preflight passed.")
    runner = ContractProcessRunner()
    production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]
    production.run(job, persist=lambda: None, cancellation_event=Event())
    assert job.state == JobState.SUCCEEDED
    return job, request, request.workspace_root / "mobius"


def _capture(request: BuildRequest, mobius_dir: Path, *, identity: PreOliveGenerationIdentity | None = None):
    return capture_pre_olive_artifact(
        mobius_source_dir=mobius_dir,
        authorized_root=request.workspace_root,
        generation_identity=identity or _identity(),
        mobius_args=_SMOLLM2_MOBIUS_ARGS,
        source_attempt_id="default-attempt-id",
        source_candidate_id="default-candidate-0",
    )


# --- 1. Capture -------------------------------------------------------------


def test_capture_pre_olive_artifact_from_successful_default_job(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-capture")
    descriptor = _capture(request, mobius_dir)

    assert descriptor.manifest.file_count == 4  # model.onnx, genai_config.json, tokenizer.json, inference_model.json
    assert descriptor.manifest.total_bytes > 0
    assert descriptor.mobius_source_dir == mobius_dir
    assert descriptor.logical_ref == descriptor.manifest.manifest_hash
    assert descriptor.source_attempt_id == "default-attempt-id"
    assert descriptor.source_candidate_id == "default-candidate-0"
    relative_paths = {entry.relative_path for entry in descriptor.manifest.entries}
    assert relative_paths == {"model.onnx", "genai_config.json", "tokenizer.json", "inference_model.json"}
    for entry in descriptor.manifest.entries:
        assert entry.size_bytes >= 0
        assert len(entry.sha256) == 64


def test_capture_rejects_empty_directory(tmp_path: Path) -> None:
    request = _request(tmp_path, name="empty-source")
    request.workspace_root.mkdir(parents=True)
    empty_mobius_dir = request.workspace_root / "mobius"
    empty_mobius_dir.mkdir()

    with pytest.raises(PreOliveReuseError):
        _capture(request, empty_mobius_dir)


def test_capture_rejects_path_outside_authorized_root(tmp_path: Path) -> None:
    request = _request(tmp_path, name="outside-root")
    request.workspace_root.mkdir(parents=True)
    outside_dir = tmp_path / "elsewhere" / "mobius"
    outside_dir.mkdir(parents=True)
    (outside_dir / "model.onnx").write_bytes(b"onnx")

    with pytest.raises(PreOliveReuseError):
        capture_pre_olive_artifact(
            mobius_source_dir=outside_dir,
            authorized_root=request.workspace_root,
            generation_identity=_identity(),
            mobius_args=_SMOLLM2_MOBIUS_ARGS,
        )


# --- 2. Independent copy / mutation isolation --------------------------------


def test_materialize_pre_olive_copy_is_independent_of_source(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-copy")
    descriptor = _capture(request, mobius_dir)

    fallback_request = _request(tmp_path, name="fallback-copy")
    fallback_request.workspace_root.mkdir(parents=True)
    destination = fallback_request.workspace_root / "mobius"

    result = materialize_pre_olive_copy(
        descriptor,
        destination_dir=destination,
        authorized_roots=(fallback_request.workspace_root,),
    )
    assert result == destination

    original_bytes = (mobius_dir / "model.onnx").read_bytes()
    # Simulate Olive mutating the destination copy in place.
    (destination / "model.onnx").write_bytes(b"olive-mutated-content")

    assert (mobius_dir / "model.onnx").read_bytes() == original_bytes
    assert (destination / "model.onnx").read_bytes() == b"olive-mutated-content"


def test_manifest_is_deterministic_across_copy(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-manifest")
    descriptor = _capture(request, mobius_dir)

    fallback_request = _request(tmp_path, name="fallback-manifest")
    fallback_request.workspace_root.mkdir(parents=True)
    destination = fallback_request.workspace_root / "mobius"
    materialize_pre_olive_copy(
        descriptor,
        destination_dir=destination,
        authorized_roots=(fallback_request.workspace_root,),
    )

    destination_manifest = production_runner_module._build_directory_manifest(destination)
    assert destination_manifest.manifest_hash == descriptor.manifest.manifest_hash
    assert destination_manifest.entries == descriptor.manifest.entries
    assert destination_manifest.total_bytes == descriptor.manifest.total_bytes
    assert destination_manifest.file_count == descriptor.manifest.file_count


# --- 3. Fallback execution: skip Mobius, run Olive exactly once -------------


def test_run_fallback_with_pre_olive_reuse_skips_mobius_runs_olive_once(tmp_path: Path) -> None:
    _default_job, default_request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-fallback")
    identity = _identity()
    descriptor = _capture(default_request, mobius_dir, identity=identity)

    fallback_request = _request(tmp_path, name="fallback-run")
    fallback_request.workspace_root.mkdir(parents=True)
    fallback_request.model_cache_dir.mkdir(parents=True)
    fallback_job = BuildJob(job_id="fallback-run-job", request=fallback_request)
    transition(fallback_job, JobState.PREFLIGHT, "Preflight passed.")
    fallback_runner = ContractProcessRunner()
    production = ProductionBuildStageRunner(fallback_runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run_fallback_with_pre_olive_reuse(
        fallback_job,
        descriptor=descriptor,
        fallback_generation_identity=identity,
        persist=lambda: None,
        cancellation_event=Event(),
    )

    assert fallback_job.state == JobState.SUCCEEDED
    assert not any(spec.argv[:2] == ("mobius", "build") for spec in fallback_runner.specs)
    olive_specs = [spec for spec in fallback_runner.specs if spec.argv[:2] == ("olive", "optimize")]
    assert len(olive_specs) == 1

    evidence = fallback_job.production_invocation_evidence
    assert evidence is not None
    assert evidence.mobius.invocation_count == 0
    assert evidence.mobius.terminal_stage == ToolInvocationTerminalStage.NOT_RUN
    assert evidence.olive.invocation_count == 1
    assert evidence.olive.terminal_stage == ToolInvocationTerminalStage.COMPLETED

    counters = production_runner_module.production_invocation_evidence_to_candidate_counters(evidence)
    assert counters.mobius_build_invocation_count is None
    assert counters.olive_optimize_invocation_count == 1
    assert counters.total_invocation_count == 1

    # The fallback's own Mobius destination is a real independent copy, not
    # the default job's original directory.
    assert (fallback_request.workspace_root / "mobius").exists()
    assert (fallback_request.workspace_root / "mobius") != mobius_dir
    assert mobius_dir.exists()


def test_default_legacy_path_without_descriptor_remains_mobius1_olive1(tmp_path: Path) -> None:
    """Explicit Slice 3A2 regression: the default/legacy one-shot `run()`
    path (no `pre_olive_reuse`/descriptor involved at all) must remain
    behavior-identical -- exactly one real Mobius launch and one real Olive
    launch."""
    job, _request_obj, _mobius_dir = _run_default_job_to_success(tmp_path, name="legacy-unchanged")
    evidence = job.production_invocation_evidence
    assert evidence is not None
    assert evidence.mobius.invocation_count == 1
    assert evidence.olive.invocation_count == 1
    counters = production_runner_module.production_invocation_evidence_to_candidate_counters(evidence)
    assert counters.mobius_build_invocation_count == 1
    assert counters.olive_optimize_invocation_count == 1
    assert counters.total_invocation_count == 2


def test_run_fallback_with_trusted_block64_candidate_end_to_end(tmp_path: Path) -> None:
    """Real Slice 3B call shape: default candidate compiled/executed through
    the generated-recipe attempt path, its pre-Olive artifact captured, and a
    trusted block64 fallback candidate (only Olive `block_size` differs)
    reuses it -- Mobius never runs for the fallback, Olive runs once with
    `--block_size 64`."""
    default_candidate = tpr._compile_generated_candidate(
        "owner/fallback-3a2-block64-model",
        "3234567890abcdef1234567890abcdef12345678",
    )
    default_record = tpr._generated_record_for(default_candidate)
    default_attempt_id = "44444444-4444-4444-4444-444444444444"
    default_attempt = tpr._attempt_for_generated(attempt_id=default_attempt_id, record=default_record)
    default_request = tpr._generated_request(
        tmp_path / "default-3a2-scope", default_candidate, attempt_id=default_attempt_id
    )
    default_request.workspace_root.mkdir(parents=True)
    default_request.model_cache_dir.mkdir(parents=True)
    default_job = BuildJob(job_id="default-3a2-job", request=default_request)
    transition(default_job, JobState.PREFLIGHT, "Preflight passed.")
    default_runner = ContractProcessRunner()
    default_store = tpr.InMemoryAttemptStore(attempt=default_attempt, generated=default_record)
    ProductionBuildStageRunner(
        default_runner,
        model_acquisition=tpr.GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=default_store,  # type: ignore[arg-type]
    ).run(default_job, persist=lambda: None, cancellation_event=Event())
    assert default_job.state == JobState.SUCCEEDED

    default_mobius_dir = default_request.workspace_root / "mobius"
    default_identity = pre_olive_generation_identity_from_generated_record(default_record)
    descriptor = capture_pre_olive_artifact(
        mobius_source_dir=default_mobius_dir,
        authorized_root=default_request.workspace_root,
        generation_identity=default_identity,
        mobius_args=default_candidate.recipe.mobius,
        source_attempt_id=default_attempt_id,
        source_candidate_id="0",
    )

    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    fallback_recipe = compile_trusted_candidate_recipe(
        default_candidate,
        policy=policy,
        candidate=policy.candidates[1],
    )
    assert fallback_recipe.recipe.olive is not None
    assert fallback_recipe.recipe.olive.block_size == 64
    fallback_record = tpr._generated_record_for(fallback_recipe)
    fallback_identity = pre_olive_generation_identity_from_generated_record(fallback_record)
    # Block-size-only trusted override must not perturb the generation
    # identity used for pre-Olive reuse.
    assert fallback_identity == default_identity

    fallback_attempt_id = "55555555-5555-5555-5555-555555555555"
    fallback_attempt = tpr._attempt_for_generated(attempt_id=fallback_attempt_id, record=fallback_record)
    fallback_request = tpr._generated_request(
        tmp_path / "fallback-3a2-scope", fallback_recipe, attempt_id=fallback_attempt_id
    )
    fallback_request.workspace_root.mkdir(parents=True)
    fallback_request.model_cache_dir.mkdir(parents=True)
    fallback_job = BuildJob(job_id="fallback-3a2-job", request=fallback_request)
    transition(fallback_job, JobState.PREFLIGHT, "Preflight passed.")
    fallback_runner = ContractProcessRunner()
    fallback_store = tpr.InMemoryAttemptStore(attempt=fallback_attempt, generated=fallback_record)
    ProductionBuildStageRunner(
        fallback_runner,
        model_acquisition=tpr.GenericSnapshot(),  # type: ignore[arg-type]
        recipe_attempt_store=fallback_store,  # type: ignore[arg-type]
    ).run_fallback_with_pre_olive_reuse(
        fallback_job,
        descriptor=descriptor,
        fallback_generation_identity=fallback_identity,
        persist=lambda: None,
        cancellation_event=Event(),
    )

    assert fallback_job.state == JobState.SUCCEEDED
    assert not any(spec.argv[:2] == ("mobius", "build") for spec in fallback_runner.specs)
    olive = next(spec for spec in fallback_runner.specs if spec.argv[:2] == ("olive", "optimize"))
    assert "--block_size" in olive.argv
    assert olive.argv[olive.argv.index("--block_size") + 1] == "64"

    evidence = fallback_job.production_invocation_evidence
    assert evidence is not None
    assert evidence.mobius.invocation_count == 0
    assert evidence.olive.invocation_count == 1
    assert default_mobius_dir.exists()  # default candidate's own Mobius output untouched


# --- 5. Fail-closed rejection cases ------------------------------------------


def test_validate_pre_olive_reuse_allows_matching_identity_and_mobius_args(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-validate-ok")
    identity = _identity()
    descriptor = _capture(request, mobius_dir, identity=identity)

    validate_pre_olive_reuse(
        descriptor,
        candidate_identity=identity,
        candidate_mobius_args=_SMOLLM2_MOBIUS_ARGS,
    )  # must not raise


@pytest.mark.parametrize(
    "override_field,override_value",
    [
        ("model_id", "owner/some-other-model"),
        ("revision_sha", "f" * 40),
        ("requested_device", "gpu"),
        ("requested_precision", "int8"),
        ("compiler_version", "9.9.9"),
        ("capability_fingerprint", "z" * 64),
        ("toolchain_fingerprint", "z" * 64),
        ("profile_fingerprint", "z" * 64),
    ],
)
def test_validate_pre_olive_reuse_rejects_each_identity_field_mismatch(
    tmp_path: Path,
    override_field: str,
    override_value: str,
) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name=f"default-mismatch-{override_field}")
    descriptor = _capture(request, mobius_dir)
    mismatched_identity = _identity(**{override_field: override_value})

    with pytest.raises(PreOliveReuseError) as excinfo:
        validate_pre_olive_reuse(
            descriptor,
            candidate_identity=mismatched_identity,
            candidate_mobius_args=_SMOLLM2_MOBIUS_ARGS,
        )
    assert excinfo.value.classification == FailureClassification.COMPATIBILITY


def test_validate_pre_olive_reuse_rejects_mobius_args_mismatch(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-mobius-mismatch")
    identity = _identity()
    descriptor = _capture(request, mobius_dir, identity=identity)

    different_mobius_args = MobiusRecipeArgs(ep="cpu", runtime="ort-genai", dtype="f16")
    with pytest.raises(PreOliveReuseError) as excinfo:
        validate_pre_olive_reuse(
            descriptor,
            candidate_identity=identity,
            candidate_mobius_args=different_mobius_args,
        )
    assert excinfo.value.classification == FailureClassification.COMPATIBILITY


def test_revalidate_pre_olive_source_rejects_tampered_content(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-tamper")
    descriptor = _capture(request, mobius_dir)

    (mobius_dir / "model.onnx").write_bytes(b"tampered-bytes")

    with pytest.raises(PreOliveReuseError):
        revalidate_pre_olive_source(descriptor)


def test_revalidate_pre_olive_source_rejects_missing_file(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-missing-file")
    descriptor = _capture(request, mobius_dir)

    (mobius_dir / "tokenizer.json").unlink()

    with pytest.raises(PreOliveReuseError):
        revalidate_pre_olive_source(descriptor)


def test_revalidate_pre_olive_source_rejects_missing_directory(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-missing-dir")
    descriptor = _capture(request, mobius_dir)

    import shutil as _shutil

    _shutil.rmtree(mobius_dir)

    with pytest.raises(PreOliveReuseError):
        revalidate_pre_olive_source(descriptor)


def test_materialize_pre_olive_copy_rejects_destination_outside_authorized_roots(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-outside-dest")
    descriptor = _capture(request, mobius_dir)

    unauthorized_root = tmp_path / "unauthorized"
    unauthorized_root.mkdir()
    destination = unauthorized_root / "mobius"

    with pytest.raises(PreOliveReuseError):
        materialize_pre_olive_copy(
            descriptor,
            destination_dir=destination,
            authorized_roots=(tmp_path / "some-other-authorized-root",),
        )
    assert not destination.exists()


def test_materialize_pre_olive_copy_rejects_destination_equal_to_source(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-same-dest")
    descriptor = _capture(request, mobius_dir)

    with pytest.raises(PreOliveReuseError):
        materialize_pre_olive_copy(
            descriptor,
            destination_dir=mobius_dir,
            authorized_roots=(request.workspace_root,),
        )


def test_materialize_pre_olive_copy_rejects_overlapping_destination(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-overlap-dest")
    descriptor = _capture(request, mobius_dir)

    nested_destination = mobius_dir / "nested-inside-source"
    with pytest.raises(PreOliveReuseError):
        materialize_pre_olive_copy(
            descriptor,
            destination_dir=nested_destination,
            authorized_roots=(request.workspace_root,),
        )
    assert not nested_destination.exists()


def test_materialize_pre_olive_copy_rejects_preexisting_destination(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-preexisting-dest")
    descriptor = _capture(request, mobius_dir)

    fallback_request = _request(tmp_path, name="fallback-preexisting-dest")
    fallback_request.workspace_root.mkdir(parents=True)
    destination = fallback_request.workspace_root / "mobius"
    destination.mkdir()

    with pytest.raises(PreOliveReuseError):
        materialize_pre_olive_copy(
            descriptor,
            destination_dir=destination,
            authorized_roots=(fallback_request.workspace_root,),
        )


def test_build_directory_manifest_rejects_mocked_symlink_or_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates both a POSIX symlink and a Windows junction/reparse point
    uniformly: both are detected by `_path_is_link_or_reparse_point`, so a
    monkeypatched True return for one path exercises the exact same
    fail-closed branch a real reparse point would hit -- without requiring
    elevated privileges to create a real one in CI."""
    root = tmp_path / "manifest-root"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"onnx")
    suspicious_path = root / "sneaky_reparse_point"
    suspicious_path.write_bytes(b"stand-in for a junction/symlink target")

    real_check = production_runner_module._path_is_link_or_reparse_point

    def fake_check(path: Path) -> bool:
        if path.name == "sneaky_reparse_point":
            return True
        return real_check(path)

    monkeypatch.setattr(production_runner_module, "_path_is_link_or_reparse_point", fake_check)

    with pytest.raises(PreOliveReuseError):
        production_runner_module._build_directory_manifest(root)


def test_build_directory_manifest_rejects_real_symlink_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "manifest-root-real-symlink"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"onnx")
    target = tmp_path / "outside-target.onnx"
    target.write_bytes(b"outside")
    link_path = root / "linked.onnx"
    try:
        link_path.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks is not permitted in this environment.")

    with pytest.raises(PreOliveReuseError):
        production_runner_module._build_directory_manifest(root)


# --- 6. Partial copy / cancellation / failure cleanup ------------------------


def test_materialize_pre_olive_copy_cleans_up_destination_on_cancellation(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-cancel")
    descriptor = _capture(request, mobius_dir)

    fallback_request = _request(tmp_path, name="fallback-cancel")
    fallback_request.workspace_root.mkdir(parents=True)
    destination = fallback_request.workspace_root / "mobius"
    cancel_event = Event()
    cancel_event.set()

    with pytest.raises(PreOliveReuseError) as excinfo:
        materialize_pre_olive_copy(
            descriptor,
            destination_dir=destination,
            authorized_roots=(fallback_request.workspace_root,),
            cancellation_event=cancel_event,
        )
    assert excinfo.value.classification == FailureClassification.CANCELLED
    assert not destination.exists()
    assert mobius_dir.exists()


def test_materialize_pre_olive_copy_cleans_up_destination_on_mid_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-midfail")
    descriptor = _capture(request, mobius_dir)
    assert descriptor.manifest.file_count > 1

    fallback_request = _request(tmp_path, name="fallback-midfail")
    fallback_request.workspace_root.mkdir(parents=True)
    destination = fallback_request.workspace_root / "mobius"

    call_count = {"n": 0}
    real_copy = production_runner_module._stream_copy_file

    def flaky_copy(source: Path, dest: Path) -> None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated disk failure mid-copy")
        real_copy(source, dest)

    monkeypatch.setattr(production_runner_module, "_stream_copy_file", flaky_copy)

    with pytest.raises(OSError):
        materialize_pre_olive_copy(
            descriptor,
            destination_dir=destination,
            authorized_roots=(fallback_request.workspace_root,),
        )
    assert not destination.exists()
    assert mobius_dir.exists()


def test_materialize_pre_olive_copy_rejects_silently_corrupted_destination_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct from a copy that raises mid-flight: every file copies without
    error, but the destination bytes silently diverge from the source (e.g. a
    corrupted write). The post-copy manifest-hash comparison must still catch
    this and fail closed, cleaning up the exact destination it created."""
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-corrupt")
    descriptor = _capture(request, mobius_dir)

    fallback_request = _request(tmp_path, name="fallback-corrupt")
    fallback_request.workspace_root.mkdir(parents=True)
    destination = fallback_request.workspace_root / "mobius"

    def corrupting_copy(source: Path, dest: Path) -> None:  # noqa: ARG001
        # Write plausible-looking but wrong content; no exception raised.
        dest.write_bytes(b"silently-corrupted-during-copy")

    monkeypatch.setattr(production_runner_module, "_stream_copy_file", corrupting_copy)

    with pytest.raises(PreOliveReuseError, match="manifest hash mismatch"):
        materialize_pre_olive_copy(
            descriptor,
            destination_dir=destination,
            authorized_roots=(fallback_request.workspace_root,),
        )
    assert not destination.exists()
    assert mobius_dir.exists()
    # Source remains byte-identical to what was captured.
    revalidate_pre_olive_source(descriptor)
    # Source content must remain byte-identical to what was captured.
    revalidate_pre_olive_source(descriptor)


def test_run_fallback_with_pre_olive_reuse_cleans_up_only_its_own_package_paths_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure inside the fallback run (after the copy already succeeded)
    must fail the job closed without deleting the shared default candidate's
    source Mobius output."""
    _default_job, default_request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-fail-cleanup")
    identity = _identity()
    descriptor = _capture(default_request, mobius_dir, identity=identity)

    fallback_request = _request(tmp_path, name="fallback-fail-cleanup")
    fallback_request.workspace_root.mkdir(parents=True)
    fallback_request.model_cache_dir.mkdir(parents=True)
    fallback_job = BuildJob(job_id="fallback-fail-cleanup-job", request=fallback_request)
    transition(fallback_job, JobState.PREFLIGHT, "Preflight passed.")

    class OliveFailsProcessRunner(ContractProcessRunner):
        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            if spec.argv[:2] == ("olive", "optimize"):
                self.specs.append(spec)
                self.cancel_events.append(cancel_event)
                return CommandResult(spec=spec, exit_code=1, stdout="", stderr="simulated Olive failure")
            return super().run(spec, cancel_event)

    fallback_runner = OliveFailsProcessRunner()
    production = ProductionBuildStageRunner(fallback_runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]

    production.run_fallback_with_pre_olive_reuse(
        fallback_job,
        descriptor=descriptor,
        fallback_generation_identity=identity,
        persist=lambda: None,
        cancellation_event=Event(),
    )

    assert fallback_job.state == JobState.FAILED
    assert mobius_dir.exists()  # default candidate's source untouched
    # The fallback's own materialized copy is left in place (matches the
    # legacy path's existing behavior of not deleting mobius_dir/olive_dir
    # on a later-stage failure); only staging/package paths are cleaned.
    assert (fallback_request.workspace_root / "mobius").exists()


# --- 7. Concurrency isolation -------------------------------------------------


def test_concurrent_fallback_reuse_calls_have_isolated_destinations_and_evidence(tmp_path: Path) -> None:
    _default_job, default_request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-concurrent")
    identity = _identity()
    descriptor = _capture(default_request, mobius_dir, identity=identity)

    results: dict[int, BuildJob] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run_one(index: int) -> None:
        try:
            fallback_request = _request(tmp_path, name=f"fallback-concurrent-{index}")
            fallback_request.workspace_root.mkdir(parents=True)
            fallback_request.model_cache_dir.mkdir(parents=True)
            job = BuildJob(job_id=f"fallback-concurrent-{index}-job", request=fallback_request)
            transition(job, JobState.PREFLIGHT, "Preflight passed.")
            runner = ContractProcessRunner()
            production = ProductionBuildStageRunner(runner, model_acquisition=PinnedSnapshot())  # type: ignore[arg-type]
            production.run_fallback_with_pre_olive_reuse(
                job,
                descriptor=descriptor,
                fallback_generation_identity=identity,
                persist=lambda: None,
                cancellation_event=Event(),
            )
            with lock:
                results[index] = job
        except BaseException as exc:  # noqa: BLE001 - surfaced via assertion below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=run_one, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors
    assert len(results) == 3
    destinations = set()
    for index in range(3):
        job = results[index]
        assert job.state == JobState.SUCCEEDED
        evidence = job.production_invocation_evidence
        assert evidence is not None
        assert evidence.mobius.invocation_count == 0
        assert evidence.olive.invocation_count == 1
        destinations.add(job.artifacts[0].path)
    assert len(destinations) == 3
    assert mobius_dir.exists()  # shared source untouched throughout


# --- 8. Sanitized evidence has no private path -------------------------------


def test_sanitized_payload_excludes_raw_source_path(tmp_path: Path) -> None:
    _job, request, mobius_dir = _run_default_job_to_success(tmp_path, name="default-sanitized")
    descriptor = _capture(request, mobius_dir)

    sanitized = descriptor.sanitized_payload()
    assert "mobius_source_dir" not in sanitized
    serialized = json.dumps(sanitized)
    assert str(mobius_dir) not in serialized
    assert str(request.workspace_root) not in serialized
    assert sanitized["logical_ref"] == descriptor.manifest.manifest_hash
    assert sanitized["manifest_file_count"] == descriptor.manifest.file_count
    assert sanitized["manifest_total_bytes"] == descriptor.manifest.total_bytes
