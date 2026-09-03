"""Slice 3B1: backend service orchestration for the default CPU INT4 generated
candidate, the narrowly-triggered trusted block64 fallback candidate, and the
durable candidate lifecycle/evidence/selection/exhaustion this wires into
`LocalOnboardingService`.

These tests exercise the *real* `ProductionBuildStageRunner` (so Mobius/Olive
skip semantics and pre-Olive reuse are genuinely exercised end to end) wired
with a fully faked `ProcessRunner` (no real Mobius/Olive/onnxruntime/Foundry
Local tooling) and a fake, deterministic `TextInferenceBackend` for quality
prompt validation. Nothing here runs a real model or touches the network.
"""

from __future__ import annotations

import json
import sys
import time

from pathlib import Path
from threading import Barrier, Event, Thread

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_local_service as tls  # noqa: E402

from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec  # noqa: E402
from fl_model_onboarding.contracts import CandidateModality, JobState  # noqa: E402
from fl_model_onboarding.local_service import BuildSubmission, LocalOnboardingService, ServiceError  # noqa: E402
from fl_model_onboarding.production_runner import ProductionBuildStageRunner, SMOLLM2_REVISION  # noqa: E402
from fl_model_onboarding.recipe_attempt_store import (  # noqa: E402
    AttemptState,
    CandidateLineageSelectionState,
    CandidateWinnerStatus,
    RecipeAttemptStoreError,
    build_attempt_request_fingerprint,
    build_attempt_request_from_generated,
)

_JSON_PROMPT = "Return valid JSON object with keys answer and unit, where answer is 12 and unit is cm."
_QUALITY_PASS_RESPONSES = {
    "What is 17 + 28? Reply using only digits.": "45",
    "Which planet is known as the Red Planet? Reply with one word.": "Mars",
    "Output exactly two words: blue river": "blue river",
    _JSON_PROMPT: '{"answer":12,"unit":"cm"}',
}


class ContractProcessRunner:
    """Fakes Mobius/Olive/runtime-validation/Foundry Local inference process
    launches with no real tooling, mirroring the fixture already used by
    `test_pre_olive_reuse.py` / `test_production_runner.py`."""

    def __init__(self) -> None:
        self.specs: list[CommandSpec] = []

    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        self.specs.append(spec)
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


class FailingOliveProcessRunner(ContractProcessRunner):
    """Like `ContractProcessRunner`, but every `olive optimize` invocation
    fails -- used to exercise "the fallback process itself fails"."""

    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        if spec.argv[:2] == ("olive", "optimize"):
            self.specs.append(spec)
            return CommandResult(spec=spec, exit_code=1, stdout="", stderr="simulated Olive failure")
        return super().run(spec, cancel_event)


class BlockingNthOccurrenceProcessRunner(ContractProcessRunner):
    """Blocks the Nth (1-indexed) command matching `match` until the test
    calls `.release.set()`, then -- if cancellation was requested meanwhile --
    returns a failure result, simulating a real subprocess actually having
    been terminated by `ProcessOwnershipRegistry`. Used to deterministically
    land a cancellation request while a specific tool invocation is "in
    flight" without any real timing races."""

    def __init__(self, *, match, occurrence: int = 1) -> None:  # noqa: ANN001
        super().__init__()
        self._match = match
        self._occurrence = occurrence
        self._count = 0
        self.started = Event()
        self.release = Event()

    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        if self._match(spec.argv):
            self._count += 1
            if self._count == self._occurrence:
                self.started.set()
                released = self.release.wait(timeout=10)
                if not released:
                    raise TimeoutError("test never released the blocked command")
                if cancel_event is not None and cancel_event.is_set():
                    self.specs.append(spec)
                    return CommandResult(spec=spec, exit_code=1, stdout="", stderr="terminated: cancelled")
        return super().run(spec, cancel_event)


class DeterministicQualityTextBackend:
    """Deterministic, fake `TextInferenceBackend` for quality-prompt
    validation. `regress_recipe_fingerprints` selects, by *recipe*
    fingerprint (learned from a generated-recipe preview call before any
    build job exists, so it is always known ahead of time), which
    optimized-only runs against that recipe must fail the JSON
    output-format prompt with invalid JSON -- the sole allowlisted
    retryable structural regression. A plain mutable `set` so a test can
    arm/disarm the regression after constructing the backend and service."""

    def __init__(self) -> None:
        self.regress_recipe_fingerprints: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        is_baseline = str(artifact.artifact_id).startswith("baseline-")
        self.calls.append((artifact.artifact_id, prompt))
        generated_attempt = job.request.generated_recipe_attempt
        is_regressed_recipe = (
            generated_attempt is not None
            and generated_attempt.recipe_fingerprint in self.regress_recipe_fingerprints
        )
        if is_regressed_recipe and not is_baseline and prompt == _JSON_PROMPT:
            return "answer is 12 cm"
        return _QUALITY_PASS_RESPONSES.get(prompt, f"{artifact.artifact_id}:{prompt}:{max_tokens}")


class AlwaysRegressedTextBackend(DeterministicQualityTextBackend):
    """Every optimized run (default and fallback alike) fails the JSON
    output-format prompt -- used for "default and fallback both regress"."""

    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        is_baseline = str(artifact.artifact_id).startswith("baseline-")
        self.calls.append((artifact.artifact_id, prompt))
        if not is_baseline and prompt == _JSON_PROMPT:
            return "answer is 12 cm"
        return _QUALITY_PASS_RESPONSES.get(prompt, f"{artifact.artifact_id}:{prompt}:{max_tokens}")


class BaselineUnavailableTextBackend(DeterministicQualityTextBackend):
    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        if str(artifact.artifact_id).startswith("baseline-"):
            raise RuntimeError("baseline runtime unavailable")
        return super().infer(artifact=artifact, job=job, prompt=prompt, max_tokens=max_tokens)


class NonStructuralRegressionTextBackend(DeterministicQualityTextBackend):
    """Optimized fails the *arithmetic* prompt only -- a non-allowlisted,
    non-structural regression that must never be retryable."""

    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        is_baseline = str(artifact.artifact_id).startswith("baseline-")
        if not is_baseline and prompt == "What is 17 + 28? Reply using only digits.":
            return "forty five"
        return super().infer(artifact=artifact, job=job, prompt=prompt, max_tokens=max_tokens)


class BothBaselineAndOptimizedFailTextBackend(DeterministicQualityTextBackend):
    """Both the baseline and the optimized run fail the JSON output-format
    prompt, but with *different* structural failure codes (baseline: missing
    required JSON key; optimized: invalid JSON entirely) -- genuinely blocked,
    but never retryable, since the baseline itself already fails."""

    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        if prompt == _JSON_PROMPT:
            is_baseline = str(artifact.artifact_id).startswith("baseline-")
            return '{"answer":12}' if is_baseline else "answer is 12 cm"
        return super().infer(artifact=artifact, job=job, prompt=prompt, max_tokens=max_tokens)


class GenericSnapshot:
    """Fake `HuggingFaceAcquisitionClient`: no network access, just a local
    directory with the minimal files the recipe compiler/preflight expect."""

    def acquire_snapshot(
        self,
        model_id: str,  # noqa: ARG002
        local_dir: Path,
        revision: str | None = None,  # noqa: ARG002
        allow_patterns=None,  # noqa: ANN001, ARG002
    ) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        return local_dir


def _model() -> str:
    return "owner/unregistered-model"


def _service(
    tmp_path: Path,
    *,
    process_runner=None,
    text_backend=None,
) -> LocalOnboardingService:
    return LocalOnboardingService(
        db_path=tmp_path / "state.sqlite3",
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        hf_metadata=tls.FakeHFMetadata(),  # type: ignore[arg-type]
        foundry_catalog=tls.FakeFoundryCatalog(),  # type: ignore[arg-type]
        preflight_inspector=tls.PassingPreflightInspector(),  # type: ignore[arg-type]
        process_runner=process_runner or ContractProcessRunner(),  # type: ignore[arg-type]
        text_inference_backend=text_backend or DeterministicQualityTextBackend(),  # type: ignore[arg-type]
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        enable_production_runner=True,
    )


def _create_default_attempt(service: LocalOnboardingService, *, idempotency_key: str):
    preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
    generated = preview["generated_recipe"]
    assert generated["eligible_for_automatic_recipe_attempt"] is True
    fingerprint = str(generated["fingerprint"])
    job, replay, attempt = service.create_generated_recipe_attempt(
        recipe_fingerprint=fingerprint,
        idempotency_key=idempotency_key,
        model_id=_model(),
    )
    return job, attempt, fingerprint


def _wait_for_job_terminal(service: LocalOnboardingService, job_id: str, timeout_seconds: float = 10.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        job = service.get_build(job_id)
        if job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for terminal state for job {job_id}")


def _wait_for_lineage_finalized(service: LocalOnboardingService, parent_attempt_id: str, timeout_seconds: float = 10.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        lineage = service._recipe_attempt_store.get_candidate_lineage(parent_attempt_id)
        if lineage is not None and lineage.selection_state != CandidateLineageSelectionState.PENDING:
            return lineage
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for lineage '{parent_attempt_id}' to finalize")


def _install_runner_dispatch_spy(
    service: LocalOnboardingService,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    runner = service._build_stage_runner
    counters = {"run": 0, "run_fallback_with_pre_olive_reuse": 0}

    def _unexpected_run(*args, **kwargs):  # noqa: ANN002, ANN003
        counters["run"] += 1
        raise AssertionError("build stage runner run() must not be called on selected-candidate reuse")

    monkeypatch.setattr(runner, "run", _unexpected_run)
    if isinstance(runner, ProductionBuildStageRunner):
        def _unexpected_fallback(*args, **kwargs):  # noqa: ANN002, ANN003
            counters["run_fallback_with_pre_olive_reuse"] += 1
            raise AssertionError(
                "run_fallback_with_pre_olive_reuse() must not be called on selected-candidate reuse"
            )

        monkeypatch.setattr(runner, "run_fallback_with_pre_olive_reuse", _unexpected_fallback)
    return counters


# --- 1. Default candidate verifies -------------------------------------------


def test_default_candidate_verifies_selects_and_promotes_once(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        job, attempt, fingerprint = _create_default_attempt(service, idempotency_key="default-verify-1")
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        attempt_id = attempt["attempt_id"]
        # Force the lazy sync so the store reflects the final candidate state.
        service.get_recipe_attempt(attempt_id=attempt_id)
        lineage = _wait_for_lineage_finalized(service, attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.SELECTED
        assert lineage.selected_candidate_attempt_id is not None

        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
        winner = candidates[0]
        assert winner.candidate_index == 0
        assert winner.selection_status == CandidateWinnerStatus.SELECTED
        assert winner.selected_by == "validation"
        assert winner.has_fully_validated_selection_scope is True
        assert winner.invocation_counters.mobius_build_invocation_count == 1
        assert winner.invocation_counters.olive_optimize_invocation_count == 1
        assert winner.invocation_counters.total_invocation_count == 2

        verified = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)[
            "generated_recipe"
        ]
        assert verified["eligible_for_automatic_recipe_attempt"] is False
        assert verified["verified_reuse"]["available"] is True
        assert verified["verified_reuse"]["source_recipe_fingerprint"] == fingerprint
    finally:
        service.close()


# --- 2. Generic structural regression triggers a verified block64 fallback --


def test_structural_regression_triggers_verified_block64_fallback(tmp_path: Path) -> None:
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        default_fingerprint = str(preview["generated_recipe"]["fingerprint"])
        backend.regress_recipe_fingerprints.add(default_fingerprint)

        job, attempt, fingerprint = _create_default_attempt(service, idempotency_key="fallback-verify-1")
        assert fingerprint == default_fingerprint
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED  # Mobius/Olive/runtime all succeeded.

        default_status = service.get_recipe_attempt(attempt_id=default_attempt_id)
        assert default_status["state"] == "failed"
        quality_gate = next(row for row in default_status["gates"] if row["gate"] == "quality_validation")
        assert quality_gate["status"] == "failed"

        lineage = _wait_for_lineage_finalized(service, default_attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.SELECTED

        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 2
        default_candidate = next(row for row in candidates if row.candidate_index == 0)
        fallback_candidate = next(row for row in candidates if row.candidate_index == 1)

        # Default candidate: terminal, non-verified, unselected, and its own
        # history is never rewritten by the fallback's later success.
        assert default_candidate.attempt_state == AttemptState.FAILED
        assert default_candidate.selection_status == CandidateWinnerStatus.NOT_SELECTED
        assert default_candidate.invocation_counters.mobius_build_invocation_count == 1
        assert default_candidate.invocation_counters.olive_optimize_invocation_count == 1

        # Fallback candidate: verified and selected; Mobius was never invoked
        # (reused pre-Olive artifact), Olive ran exactly once with block_size=64.
        assert fallback_candidate.attempt_state == AttemptState.SUCCEEDED
        assert fallback_candidate.selection_status == CandidateWinnerStatus.SELECTED
        assert fallback_candidate.selected_by == "validation"
        assert fallback_candidate.quantization_override_block_size == 64
        assert fallback_candidate.eligibility_trigger == "retryable_optimized_structural_regression"
        assert fallback_candidate.disposition == "retryable_optimized_structural_regression"
        assert fallback_candidate.invocation_counters.mobius_build_invocation_count is None
        assert fallback_candidate.invocation_counters.olive_optimize_invocation_count == 1
        assert fallback_candidate.invocation_counters.total_invocation_count == 1
        assert fallback_candidate.has_fully_validated_selection_scope is True

        assert lineage.selected_candidate_attempt_id == fallback_candidate.candidate_attempt_id

        # Overall aggregate evidence across the two real candidates: Mobius1 (default
        # only) / Olive2 (one per candidate) -- computed from real counters, never
        # hardcoded.
        aggregate_mobius = (default_candidate.invocation_counters.mobius_build_invocation_count or 0) + (
            fallback_candidate.invocation_counters.mobius_build_invocation_count or 0
        )
        aggregate_olive = (default_candidate.invocation_counters.olive_optimize_invocation_count or 0) + (
            fallback_candidate.invocation_counters.olive_optimize_invocation_count or 0
        )
        assert aggregate_mobius == 1
        assert aggregate_olive == 2

        # The overall user-facing operation succeeded via the fallback candidate's
        # own attempt/recipe -- promoted independently, without mutating the
        # default candidate's failed history.
        fallback_status = service.get_recipe_attempt(attempt_id=fallback_candidate.attempt_id)
        assert fallback_status["state"] == "succeeded"
        default_status_again = service.get_recipe_attempt(attempt_id=default_attempt_id)
        assert default_status_again["state"] == "failed"

        reused = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)[
            "generated_recipe"
        ]
        assert reused["eligible_for_automatic_recipe_attempt"] is False
        assert reused["verified_reuse"]["available"] is True
        assert reused["verified_reuse"]["source_recipe_fingerprint"] == fallback_candidate.recipe_fingerprint

        # No Mobius launch for the fallback candidate's own job.
        fallback_job_id = None
        for job_id, mapped_attempt_id in service._build_job_to_attempt.items():
            if mapped_attempt_id == fallback_candidate.attempt_id:
                fallback_job_id = job_id
        assert fallback_job_id is not None
        fallback_job = service.get_build(fallback_job_id)
        assert not any(
            spec.argv[:2] == ("mobius", "build")
            for spec in service._process_runner.specs  # type: ignore[attr-defined]
            if spec.cwd == fallback_job.request.workspace_root
        )
    finally:
        service.close()


def test_selected_default_candidate_reuse_short_circuits_without_new_build_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    try:
        source_job, source_attempt, source_fingerprint = _create_default_attempt(
            service, idempotency_key="reuse-selected-default-source-1"
        )
        assert _wait_for_job_terminal(service, source_job.job_id).state == JobState.SUCCEEDED
        source_attempt_id = source_attempt["attempt_id"]
        assert service.get_recipe_attempt(attempt_id=source_attempt_id)["state"] == "succeeded"
        _wait_for_lineage_finalized(service, source_attempt_id)

        source_candidate = service._recipe_attempt_store.list_candidate_attempts(source_attempt_id)[0]
        jobs_before = len(service._jobs)
        source_mobius = source_candidate.invocation_counters.mobius_build_invocation_count
        source_olive = source_candidate.invocation_counters.olive_optimize_invocation_count

        dispatch_counts = _install_runner_dispatch_spy(service, monkeypatch)
        reuse_preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)["generated_recipe"]
        assert reuse_preview["candidate_selection_reuse"]["available"] is True
        assert reuse_preview["candidate_selection_reuse"]["winner_candidate_index"] == 0

        reused_job, replay, reused_attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=source_fingerprint,
            idempotency_key="reuse-selected-default-consumer-1",
            model_id=_model(),
        )
        assert replay is False
        assert reused_job.job_id == source_job.job_id
        assert reused_attempt["state"] == "succeeded"
        assert reused_attempt["build_job_id"] is None
        assert reused_attempt["recipe_fingerprint"] == source_candidate.recipe_fingerprint
        assert reused_attempt["candidate_selection_reuse"]["winner_candidate_attempt_id"] == source_candidate.candidate_attempt_id
        assert len(service._jobs) == jobs_before
        assert dispatch_counts["run"] == 0
        assert dispatch_counts["run_fallback_with_pre_olive_reuse"] == 0

        source_attempt_after = service.get_recipe_attempt(attempt_id=source_attempt_id)
        assert source_attempt_after["state"] == "succeeded"
        source_candidate_after = service._recipe_attempt_store.get_candidate_attempt(source_candidate.candidate_attempt_id)
        assert source_candidate_after.invocation_counters.mobius_build_invocation_count == source_mobius
        assert source_candidate_after.invocation_counters.olive_optimize_invocation_count == source_olive
    finally:
        service.close()


def test_selected_block64_candidate_reuse_short_circuits_to_winner_without_runner_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        default_preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        default_fingerprint = str(default_preview["generated_recipe"]["fingerprint"])
        backend.regress_recipe_fingerprints.add(default_fingerprint)

        source_job, source_attempt, _fingerprint = _create_default_attempt(
            service, idempotency_key="reuse-selected-block64-source-1"
        )
        assert _wait_for_job_terminal(service, source_job.job_id).state == JobState.SUCCEEDED
        default_attempt_id = source_attempt["attempt_id"]
        _wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)

        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        default_candidate = next(row for row in candidates if row.candidate_index == 0)
        fallback_candidate = next(row for row in candidates if row.candidate_index == 1)
        fallback_job_id = _fallback_job_id_for(service, fallback_candidate.attempt_id)
        assert fallback_job_id is not None
        default_attempt_before = service.get_recipe_attempt(attempt_id=default_attempt_id)
        assert default_attempt_before["state"] == "failed"
        default_mobius = default_candidate.invocation_counters.mobius_build_invocation_count
        default_olive = default_candidate.invocation_counters.olive_optimize_invocation_count
        jobs_before = len(service._jobs)

        dispatch_counts = _install_runner_dispatch_spy(service, monkeypatch)
        reuse_preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)["generated_recipe"]
        assert reuse_preview["candidate_selection_reuse"]["available"] is True
        assert reuse_preview["candidate_selection_reuse"]["winner_candidate_index"] == 1
        assert (
            reuse_preview["candidate_selection_reuse"]["winner_recipe_fingerprint"]
            == fallback_candidate.recipe_fingerprint
        )

        reused_job, replay, reused_attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=default_fingerprint,
            idempotency_key="reuse-selected-block64-consumer-1",
            model_id=_model(),
        )
        assert replay is False
        assert reused_job.job_id == fallback_job_id
        assert reused_attempt["state"] == "succeeded"
        assert reused_attempt["build_job_id"] is None
        assert reused_attempt["candidate_selection_reuse"]["winner_candidate_id"] == "int4-block-size-64"
        assert reused_attempt["candidate_selection_reuse"]["winner_candidate_index"] == 1
        assert reused_attempt["recipe_fingerprint"] == fallback_candidate.recipe_fingerprint
        assert reused_attempt["recipe_fingerprint"] != default_candidate.recipe_fingerprint
        assert len(service._jobs) == jobs_before
        assert dispatch_counts["run"] == 0
        assert dispatch_counts["run_fallback_with_pre_olive_reuse"] == 0

        default_attempt_after = service.get_recipe_attempt(attempt_id=default_attempt_id)
        assert default_attempt_after["state"] == "failed"
        default_candidate_after = service._recipe_attempt_store.get_candidate_attempt(default_candidate.candidate_attempt_id)
        assert default_candidate_after.invocation_counters.mobius_build_invocation_count == default_mobius
        assert default_candidate_after.invocation_counters.olive_optimize_invocation_count == default_olive
    finally:
        service.close()


def _fallback_job_id_for(service: LocalOnboardingService, attempt_id: str) -> str | None:
    for job_id, mapped_attempt_id in service._build_job_to_attempt.items():
        if mapped_attempt_id == attempt_id:
            return job_id
    return None


# --- 3. Both default and fallback regress: exhausted, no third candidate ----


def test_default_and_fallback_both_regress_exhausted_no_third_candidate(tmp_path: Path) -> None:
    backend = AlwaysRegressedTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        job, attempt, fingerprint = _create_default_attempt(service, idempotency_key="both-regress-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        lineage = _wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        assert lineage.selected_candidate_attempt_id is None

        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 2
        assert all(row.attempt_state == AttemptState.FAILED for row in candidates)
        assert all(row.selection_status == CandidateWinnerStatus.NOT_SELECTED for row in candidates)
        assert not any(row.is_verified for row in candidates)

        # No third candidate: the fallback's own quality failure never triggers
        # another fallback, regardless of its own disposition.
        assert not any(row.candidate_index == 2 for row in candidates)
        assert len(service._recipe_attempt_store.list_verified_recipes()) == 0

        default_status = service.get_recipe_attempt(attempt_id=default_attempt_id)
        assert default_status["state"] == "failed"
        fallback_row = next(row for row in candidates if row.candidate_index == 1)
        fallback_status = service.get_recipe_attempt(attempt_id=fallback_row.attempt_id)
        assert fallback_status["state"] == "failed"
    finally:
        service.close()


# --- 4. Every non-retryable disposition class yields exactly one candidate --


def test_non_retryable_arithmetic_regression_yields_one_candidate_no_fallback(tmp_path: Path) -> None:
    service = _service(tmp_path, text_backend=NonStructuralRegressionTextBackend())
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="non-structural-1")
        attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        lineage = _wait_for_lineage_finalized(service, attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
        assert candidates[0].candidate_index == 0
        assert candidates[0].disposition is None or candidates[0].disposition == "not_retryable"
    finally:
        service.close()


def test_non_retryable_baseline_unavailable_yields_one_candidate_no_fallback(tmp_path: Path) -> None:
    service = _service(tmp_path, text_backend=BaselineUnavailableTextBackend())
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="baseline-unavailable-1")
        attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        lineage = _wait_for_lineage_finalized(service, attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
    finally:
        service.close()


def test_non_retryable_both_baseline_and_optimized_fail_yields_one_candidate(tmp_path: Path) -> None:
    service = _service(tmp_path, text_backend=BothBaselineAndOptimizedFailTextBackend())
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="both-fail-1")
        attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        lineage = _wait_for_lineage_finalized(service, attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
    finally:
        service.close()


# --- 5. Missing/tampered pre-Olive descriptor never allows an unsafe fallback


def test_missing_pre_olive_descriptor_prevents_unsafe_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fl_model_onboarding.local_service as local_service_module

    # Patch before the service (and therefore the runner's bound
    # `on_mobius_ready` hook) is constructed: `ProductionBuildStageRunner`
    # captures a bound method reference once at construction time, so patching
    # the class afterward would never take effect.
    monkeypatch.setattr(
        local_service_module.LocalOnboardingService,
        "_capture_pre_olive_descriptor_if_eligible",
        lambda self, job, mobius_dir: None,
    )

    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="missing-descriptor-1")
        attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        lineage = _wait_for_lineage_finalized(service, attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
    finally:
        service.close()


def test_tampered_pre_olive_source_prevents_unsafe_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fl_model_onboarding.local_service as local_service_module

    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        def _always_tampered(descriptor):  # noqa: ANN001
            from fl_model_onboarding.production_runner import PreOliveReuseError

            raise PreOliveReuseError("simulated tamper: manifest hash mismatch")

        monkeypatch.setattr(local_service_module, "revalidate_pre_olive_source", _always_tampered)

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="tampered-descriptor-1")
        attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        lineage = _wait_for_lineage_finalized(service, attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
    finally:
        service.close()


# --- 6. Cancellation ----------------------------------------------------------


def test_cancel_before_fallback_trigger_registers_no_candidate_one(tmp_path: Path) -> None:
    runner = BlockingNthOccurrenceProcessRunner(match=lambda argv: argv[:2] == ("mobius", "build"), occurrence=1)
    service = _service(tmp_path, process_runner=runner)
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="cancel-before-trigger-1")
        attempt_id = attempt["attempt_id"]
        assert runner.started.wait(timeout=5.0)
        service.cancel_build(job.job_id)
        runner.release.set()

        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.CANCELLED

        lineage = _wait_for_lineage_finalized(service, attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(attempt_id)
        assert len(candidates) == 1
        assert candidates[0].attempt_state == AttemptState.CANCELLED
        assert len(service._recipe_attempt_store.list_verified_recipes()) == 0
    finally:
        service.close()


def test_cancel_during_fallback_execution_no_promotion_terminal_lineage(tmp_path: Path) -> None:
    runner = BlockingNthOccurrenceProcessRunner(match=lambda argv: argv[:2] == ("olive", "optimize"), occurrence=2)
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, process_runner=runner, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="cancel-during-fallback-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED  # default candidate's own build succeeded

        # Wait until the fallback candidate is registered and its Olive stage is
        # blocked (the second real `olive optimize` invocation overall).
        assert runner.started.wait(timeout=10.0)
        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        fallback_candidate = next(row for row in candidates if row.candidate_index == 1)
        fallback_job_id = _fallback_job_id_for(service, fallback_candidate.attempt_id)
        assert fallback_job_id is not None

        service.cancel_build(fallback_job_id)
        runner.release.set()

        fallback_job = _wait_for_job_terminal(service, fallback_job_id)
        assert fallback_job.state == JobState.CANCELLED

        lineage = _wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        assert lineage.selected_candidate_attempt_id is None

        refreshed_candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(refreshed_candidates) == 2
        assert not any(row.candidate_index == 2 for row in refreshed_candidates)
        refreshed_fallback = next(row for row in refreshed_candidates if row.candidate_index == 1)
        assert refreshed_fallback.attempt_state == AttemptState.CANCELLED
        assert refreshed_fallback.selection_status == CandidateWinnerStatus.NOT_SELECTED
        default_row = next(row for row in refreshed_candidates if row.candidate_index == 0)
        assert default_row.attempt_state == AttemptState.FAILED  # unchanged/not rewound
        assert len(service._recipe_attempt_store.list_verified_recipes()) == 0
    finally:
        service.close()


def test_fallback_process_failure_no_promotion_exhausted(tmp_path: Path) -> None:
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, process_runner=FailingOliveProcessRunner(), text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="fallback-process-fails-1")
        default_attempt_id = attempt["attempt_id"]

        # The default candidate itself never reaches Olive successfully (Olive
        # always fails in this fixture), so it fails for a plain gate reason and
        # is never retry-eligible; assert the lineage still finalizes cleanly.
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.FAILED

        lineage = _wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 1
        assert len(service._recipe_attempt_store.list_verified_recipes()) == 0
    finally:
        service.close()


# --- 7. Idempotency / duplicate delivery -------------------------------------


def test_duplicate_sync_delivery_does_not_duplicate_candidate_or_job(tmp_path: Path) -> None:
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="duplicate-delivery-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        _wait_for_lineage_finalized(service, default_attempt_id)

        jobs_before = len(service._jobs)
        candidates_before = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)

        # Simulate a repeated worker delivery / redundant sync trigger for the
        # already-terminal default attempt's job.
        service._safe_sync_generated_attempt(job=completed)
        service._safe_sync_generated_attempt(job=completed)

        assert len(service._jobs) == jobs_before
        candidates_after = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates_after) == len(candidates_before)

        # And directly re-entering the fallback launch helper for an already-
        # registered/launched candidate 1 must never create a second row/job.
        fallback_candidate = next(row for row in candidates_after if row.candidate_index == 1)
        generated_record = service._recipe_attempt_store.get_generated_recipe(fallback_candidate.recipe_fingerprint)
        default_candidate = next(row for row in candidates_after if row.candidate_index == 0)
        default_generated_record = service._recipe_attempt_store.get_generated_recipe(
            default_candidate.recipe_fingerprint
        )
        default_recompiled = service._recompile_generated_recipe_record(record=default_generated_record)
        service._launch_fallback_candidate_attempt(
            parent_attempt_id=default_attempt_id,
            default_record=default_generated_record,
            default_candidate_recipe=default_recompiled,
            descriptor=None,  # type: ignore[arg-type]
            retry_evaluation=None,  # type: ignore[arg-type]
        )
        assert len(service._jobs) == jobs_before
        candidates_final = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates_final) == 2
        assert generated_record is not None
    finally:
        service.close()


# --- 8. Eligibility gating: only CPU INT4 registers a lineage ----------------


class _StubGeneratedRecord:
    def __init__(self, payload: dict) -> None:  # noqa: ANN001
        self._payload = payload

    def payload(self) -> dict:
        return self._payload


def test_non_cpu_int4_generated_record_is_not_eligible_and_registers_no_lineage(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        record = _StubGeneratedRecord({"recipe": {"olive": {"device": "gpu", "precision": "int4"}}})
        assert service._generated_record_is_cpu_int4_eligible(record) is False
        service._register_default_candidate_lineage_if_eligible(attempt_id="stub-attempt-id", record=record)
        assert service._recipe_attempt_store.get_candidate_lineage("stub-attempt-id") is None

        record_fp32 = _StubGeneratedRecord({"recipe": {"olive": {"device": "cpu", "precision": "fp32"}}})
        assert service._generated_record_is_cpu_int4_eligible(record_fp32) is False

        record_ok = _StubGeneratedRecord({"recipe": {"olive": {"device": "cpu", "precision": "int4"}}})
        assert service._generated_record_is_cpu_int4_eligible(record_ok) is True
    finally:
        service.close()


def test_static_recipe_build_never_touches_candidate_orchestration(tmp_path: Path) -> None:
    class PinnedSmolLM2Preflight(tls.PassingPreflightInspector):
        def inspect(self, request):  # noqa: ANN001
            from dataclasses import replace as _replace

            base = super().inspect(request)
            return _replace(base, huggingface_revision=SMOLLM2_REVISION, huggingface_sha=SMOLLM2_REVISION)

    service = LocalOnboardingService(
        db_path=tmp_path / "state.sqlite3",
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        hf_metadata=tls.FakeHFMetadata(),  # type: ignore[arg-type]
        foundry_catalog=tls.FakeFoundryCatalog(),  # type: ignore[arg-type]
        preflight_inspector=PinnedSmolLM2Preflight(),  # type: ignore[arg-type]
        process_runner=ContractProcessRunner(),  # type: ignore[arg-type]
        model_acquisition=GenericSnapshot(),  # type: ignore[arg-type]
        enable_production_runner=True,
    )
    try:
        job, _replay = service.create_build(
            BuildSubmission(
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                task=CandidateModality.LLM,
                task_profile="llm-cpu-int4",
            ),
            idempotency_key="static-recipe-1",
        )
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        # Static-recipe builds never bind a generated-recipe attempt at all, so
        # every Slice 3B1 hook (gated on `attempt_id`) is a pure no-op for them.
        assert service._attempt_id_for_job(job.job_id) is None
    finally:
        service.close()


# --- 9. Restart fail-closed recovery -----------------------------------------


def test_restart_fail_closed_when_fallback_never_registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    default_attempt_id: str
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        backend.regress_recipe_fingerprints.add(str(preview["generated_recipe"]["fingerprint"]))

        # Simulate a crash exactly between "default candidate finalized failed"
        # and "fallback candidate registered/launched": the trigger evaluation
        # itself is short-circuited to pretend it launched something, without
        # actually registering candidate 1 or creating its job.
        monkeypatch.setattr(
            service,
            "_maybe_launch_fallback_candidate",
            lambda **kwargs: True,  # noqa: ARG005
        )

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="restart-orphan-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        lineage = service._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage is not None and lineage.selection_state == CandidateLineageSelectionState.PENDING
        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 1
    finally:
        service.close()

    # "Restart": construct a fresh service instance pointed at the same
    # on-disk store/workspace, with no in-memory pre-Olive descriptor cache.
    restarted = _service(tmp_path, text_backend=DeterministicQualityTextBackend())
    try:
        lineage_after_restart = restarted._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_after_restart is not None
        assert lineage_after_restart.selection_state == CandidateLineageSelectionState.EXHAUSTED
        assert lineage_after_restart.selection_reason is not None
        assert "restart_fail_closed" in lineage_after_restart.selection_reason
        candidates_after_restart = restarted._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates_after_restart) == 1
    finally:
        restarted.close()


def test_restart_exhausts_lineage_when_both_candidates_terminal_at_policy_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact reviewer crash reproduction for Issue 2: both candidates (the
    default and its trusted block64 fallback) reach a terminal, non-verified
    state -- candidate count == `policy_max_candidates` -- and each
    candidate's own terminal evidence/state is durably committed, but the
    process "dies" before `finalize_exhausted_candidate_lineage` itself ever
    commits (simulated by patching that exact store call to a no-op after the
    child terminal write, rather than by actually crashing). A fresh service
    instance pointed at the same on-disk store/workspace must finalize the
    lineage exhausted on restart, and this must remain stable (idempotent)
    across further restarts/polls.
    """
    backend = AlwaysRegressedTextBackend()
    service = _service(tmp_path, text_backend=backend)
    default_attempt_id: str
    try:
        # Patch only the final lineage-finalization store call to a no-op:
        # every other write (candidate terminal evidence, attempt state
        # transitions) commits normally, exactly matching "candidate 1
        # terminal is committed, then the process dies before
        # `finalize_exhausted_candidate_lineage`".
        monkeypatch.setattr(
            service._recipe_attempt_store,
            "finalize_exhausted_candidate_lineage",
            lambda *args, **kwargs: None,
        )

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="restart-max-candidates-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        # Give the fallback candidate's own job time to reach its terminal
        # state too (it is launched synchronously by the same sync call, but
        # runs on the worker thread).
        deadline = time.time() + 10.0
        candidates: tuple = ()
        while time.time() < deadline:
            candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
            if len(candidates) == 2 and all(row.attempt_state in {AttemptState.FAILED} for row in candidates):
                break
            time.sleep(0.02)
        assert len(candidates) == 2
        assert all(row.attempt_state == AttemptState.FAILED for row in candidates)
        assert not any(row.is_verified for row in candidates)

        # The crash window: candidate count == policy max, both terminal, but
        # the lineage itself was never finalized because the store call was
        # patched to a no-op.
        lineage = service._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage is not None
        assert lineage.selection_state == CandidateLineageSelectionState.PENDING
        assert len(candidates) == lineage.policy_max_candidates
    finally:
        service.close()

    # "Restart" #1: a fresh service (no patched store method) must heal this
    # via `_recover_orphaned_candidate_lineages` during `__init__`.
    restarted = _service(tmp_path, text_backend=DeterministicQualityTextBackend())
    try:
        lineage_after_restart = restarted._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_after_restart is not None
        assert lineage_after_restart.selection_state == CandidateLineageSelectionState.EXHAUSTED
        assert lineage_after_restart.selected_candidate_attempt_id is None
        candidates_after_restart = restarted._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates_after_restart) == 2

        # A poll of either candidate's own attempt status must not resurrect
        # or mutate the now-exhausted lineage.
        for row in candidates_after_restart:
            restarted.get_recipe_attempt(attempt_id=row.attempt_id)
        lineage_after_poll = restarted._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_after_poll is not None
        assert lineage_after_poll.selection_state == CandidateLineageSelectionState.EXHAUSTED
    finally:
        restarted.close()

    # "Restart" #2: repeated restart recovery must be idempotent -- no error,
    # same terminal outcome, unchanged candidate rows.
    restarted_again = _service(tmp_path, text_backend=DeterministicQualityTextBackend())
    try:
        lineage_final = restarted_again._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_final is not None
        assert lineage_final.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates_final = restarted_again._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates_final) == 2
    finally:
        restarted_again.close()


def test_recovery_leaves_pending_lineage_untouched_when_candidate_still_running(tmp_path: Path) -> None:
    """A candidate that is still genuinely in flight (its linked attempt has
    not reached a terminal state) must be left completely untouched by
    restart recovery, regardless of candidate count vs. policy max."""
    runner = BlockingNthOccurrenceProcessRunner(match=lambda argv: argv[:2] == ("mobius", "build"), occurrence=1)
    service = _service(tmp_path, process_runner=runner)
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="recover-running-1")
        default_attempt_id = attempt["attempt_id"]
        assert runner.started.wait(timeout=5.0), "Mobius build never reached the blocking point"

        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 1
        assert candidates[0].attempt_state == AttemptState.RUNNING

        # Directly re-run the exact same restart-recovery pass `__init__`
        # performs, while the candidate is still genuinely in flight.
        service._recover_orphaned_candidate_lineages()

        lineage = service._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage is not None
        assert lineage.selection_state == CandidateLineageSelectionState.PENDING
    finally:
        runner.release.set()
        service.close()


def test_recovery_selects_verified_unselected_candidate_instead_of_exhausting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash window: the default candidate's own attempt reaches
    `AttemptState.SUCCEEDED` (verified) and its terminal evidence is
    committed, but the process dies before `select_verified_candidate_attempt`
    itself commits. Restart recovery must select the verified candidate --
    never exhaust the lineage around it -- using the same trusted selection
    logic the live sync path uses."""
    backend = DeterministicQualityTextBackend()
    service = _service(tmp_path, text_backend=backend)
    default_attempt_id: str
    try:
        monkeypatch.setattr(service, "_select_verified_candidate", lambda **kwargs: None)  # noqa: ARG005

        job, attempt, _fp = _create_default_attempt(service, idempotency_key="restart-verified-unselected-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        # Force the lazy sync so the store reflects the attempt's final
        # (verified) state, even though selection itself was short-circuited.
        service.get_recipe_attempt(attempt_id=default_attempt_id)

        candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates) == 1
        assert candidates[0].is_verified is True
        assert candidates[0].selection_status == CandidateWinnerStatus.NOT_SELECTED

        lineage = service._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage is not None
        assert lineage.selection_state == CandidateLineageSelectionState.PENDING
    finally:
        service.close()

    restarted = _service(tmp_path, text_backend=DeterministicQualityTextBackend())
    try:
        lineage_after_restart = restarted._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_after_restart is not None
        assert lineage_after_restart.selection_state == CandidateLineageSelectionState.SELECTED
        candidates_after_restart = restarted._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates_after_restart) == 1
        winner = candidates_after_restart[0]
        assert winner.selection_status == CandidateWinnerStatus.SELECTED
        assert lineage_after_restart.selected_candidate_attempt_id == winner.candidate_attempt_id
    finally:
        restarted.close()

    # Idempotent across a second restart.
    restarted_again = _service(tmp_path, text_backend=DeterministicQualityTextBackend())
    try:
        lineage_final = restarted_again._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_final is not None
        assert lineage_final.selection_state == CandidateLineageSelectionState.SELECTED
    finally:
        restarted_again.close()


def test_recovery_noop_for_already_selected_lineage(tmp_path: Path) -> None:
    service = _service(tmp_path)
    default_attempt_id: str
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="recovery-noop-selected-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        service.get_recipe_attempt(attempt_id=default_attempt_id)
        lineage = _wait_for_lineage_finalized(service, default_attempt_id)
        assert lineage.selection_state == CandidateLineageSelectionState.SELECTED
        selected_candidate_attempt_id = lineage.selected_candidate_attempt_id

        service._recover_orphaned_candidate_lineages()

        lineage_after = service._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_after is not None
        assert lineage_after.selection_state == CandidateLineageSelectionState.SELECTED
        assert lineage_after.selected_candidate_attempt_id == selected_candidate_attempt_id
    finally:
        service.close()


def test_recovery_noop_for_already_exhausted_lineage(tmp_path: Path) -> None:
    backend = AlwaysRegressedTextBackend()
    service = _service(tmp_path, text_backend=backend)
    default_attempt_id: str
    try:
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="recovery-noop-exhausted-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED
        lineage = _wait_for_lineage_finalized(service, default_attempt_id, timeout_seconds=15.0)
        assert lineage.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates_before = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)

        service._recover_orphaned_candidate_lineages()

        lineage_after = service._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_after is not None
        assert lineage_after.selection_state == CandidateLineageSelectionState.EXHAUSTED
        candidates_after = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
        assert len(candidates_after) == len(candidates_before)
    finally:
        service.close()


def test_recovery_transaction_failure_leaves_lineage_pending_and_is_retried_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store-level failure while exhausting a lineage during restart
    recovery must leave the lineage exactly as it was (still `PENDING`,
    candidate rows unchanged) rather than partially committing, and a later
    recovery pass (once the transient failure clears) must still be able to
    finalize it."""
    backend = AlwaysRegressedTextBackend()
    service = _service(tmp_path, text_backend=backend)
    default_attempt_id: str
    try:
        monkeypatch.setattr(
            service._recipe_attempt_store,
            "finalize_exhausted_candidate_lineage",
            lambda *args, **kwargs: None,
        )
        job, attempt, _fp = _create_default_attempt(service, idempotency_key="recovery-txn-failure-1")
        default_attempt_id = attempt["attempt_id"]
        completed = _wait_for_job_terminal(service, job.job_id)
        assert completed.state == JobState.SUCCEEDED

        deadline = time.time() + 10.0
        candidates: tuple = ()
        while time.time() < deadline:
            candidates = service._recipe_attempt_store.list_candidate_attempts(default_attempt_id)
            if len(candidates) == 2 and all(row.attempt_state == AttemptState.FAILED for row in candidates):
                break
            time.sleep(0.02)
        assert len(candidates) == 2
        candidates_before = candidates
    finally:
        service.close()

    def _raise_transaction_failure(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RecipeAttemptStoreError("simulated transaction failure")

    import fl_model_onboarding.recipe_attempt_store as recipe_attempt_store_module

    # Patch the *class* method before construction so the service's own
    # `__init__`-triggered recovery pass (not a later manual call) is the one
    # that hits the simulated failure -- otherwise `__init__`'s own recovery
    # would already have healed the lineage before we get a chance to patch
    # anything on the instance.
    with monkeypatch.context() as failure_context:
        failure_context.setattr(
            recipe_attempt_store_module.RecipeAttemptStore,
            "finalize_exhausted_candidate_lineage",
            _raise_transaction_failure,
        )
        failing_restart = _service(tmp_path, text_backend=DeterministicQualityTextBackend())
        try:
            lineage_after_failure = failing_restart._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
            assert lineage_after_failure is not None
            assert lineage_after_failure.selection_state == CandidateLineageSelectionState.PENDING
            candidates_after_failure = failing_restart._recipe_attempt_store.list_candidate_attempts(
                default_attempt_id
            )
            assert len(candidates_after_failure) == len(candidates_before)
            assert all(row.attempt_state == AttemptState.FAILED for row in candidates_after_failure)

            # Re-running recovery again, still under the simulated failure,
            # must remain a safe, consistent no-op (never partially commit).
            failing_restart._recover_orphaned_candidate_lineages()
            lineage_still_pending = failing_restart._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
            assert lineage_still_pending is not None
            assert lineage_still_pending.selection_state == CandidateLineageSelectionState.PENDING
        finally:
            failing_restart.close()

    # A later recovery pass (no store failure this time) must still heal it.
    healthy_restart = _service(tmp_path, text_backend=DeterministicQualityTextBackend())
    try:
        lineage_final = healthy_restart._recipe_attempt_store.get_candidate_lineage(default_attempt_id)
        assert lineage_final is not None
        assert lineage_final.selection_state == CandidateLineageSelectionState.EXHAUSTED
    finally:
        healthy_restart.close()


# --------------------------------------------------------------------------
# Reviewer-REJECTED 3B2a fix: concurrent same-Idempotency-Key reuse
# materialization race (Linus revision).
#
# `_materialize_reused_generated_attempt` drives a multi-transaction sequence
# (get GENERATED -> start -> copy winner gates -> finish SUCCEEDED). Before
# this revision, two concurrent `create_generated_recipe_attempt` calls
# resolving the SAME Idempotency-Key -- and therefore the same attempt_id --
# could both observe a stale GENERATED read and both call `start_attempt`,
# so the loser surfaced a 409/500 instead of transparently joining the
# winner's materialization. The tests below exercise the real
# `create_generated_recipe_attempt` request path (not the store in
# isolation) against the fix: a per-attempt guard
# (`_acquire_attempt_sync_guard`/`_release_attempt_sync_guard`, the same
# primitive `_safe_sync_generated_attempt` already uses) that serializes
# concurrent callers for the same attempt_id without ever holding the
# service-wide `self._lock` across the guarded work, plus a durable
# `reuse_source_attempt_id` marker (written atomically with the `RUNNING`
# transition) that lets a fresh process safely resume an interrupted
# materialization instead of guessing.
# --------------------------------------------------------------------------


def _run_concurrently(fns) -> tuple[list[object], list[BaseException | None]]:
    """Run each zero-arg callable in `fns` on its own thread, releasing every
    thread simultaneously via a shared `Barrier` once all of them have
    reached the starting line, and return `(results, errors)` in the same
    order as `fns`: `results[i]` is `fns[i]`'s return value (or `None` if it
    raised) and `errors[i]` is the exception it raised (or `None` on
    success). This maximizes the chance of exercising the exact race window
    the reviewer's repro describes, rather than a merely-sequential retry."""
    barrier = Barrier(len(fns))
    results: list[object] = [None] * len(fns)
    errors: list[BaseException | None] = [None] * len(fns)

    def _runner(index: int, fn) -> None:  # noqa: ANN001
        barrier.wait()
        try:
            results[index] = fn()
        except BaseException as exc:  # noqa: BLE001 - captured for the caller to inspect/raise
            errors[index] = exc

    threads = [Thread(target=_runner, args=(index, fn)) for index, fn in enumerate(fns)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
        assert not thread.is_alive(), "Concurrent reuse-materialization thread did not complete: possible deadlock."
    return results, errors


def _assert_no_thread_errors(errors: list[BaseException | None], *, context: str) -> None:
    for index, error in enumerate(errors):
        if error is not None:
            raise AssertionError(f"{context}: thread {index} raised {error!r}") from error


def test_concurrent_same_idempotency_key_reuse_materialization_is_race_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact reviewer repro: multiple threads calling
    `create_generated_recipe_attempt` with the SAME Idempotency-Key for a
    candidate-selection-reuse fingerprint must all succeed with the same
    materialized attempt/reuse result -- never a 409/500 from a losing
    thread -- and the reuse path must never dispatch to the build stage
    runner. Repeated across many cycles (fresh Idempotency-Key per cycle) to
    catch an intermittent race rather than a merely-lucky single trial."""
    service = _service(tmp_path)
    try:
        source_job, source_attempt, default_fingerprint = _create_default_attempt(
            service, idempotency_key="race-same-key-source-1"
        )
        assert _wait_for_job_terminal(service, source_job.job_id).state == JobState.SUCCEEDED
        source_attempt_id = source_attempt["attempt_id"]
        _wait_for_lineage_finalized(service, source_attempt_id)
        source_candidate = service._recipe_attempt_store.list_candidate_attempts(source_attempt_id)[0]
        source_mobius = source_candidate.invocation_counters.mobius_build_invocation_count
        source_olive = source_candidate.invocation_counters.olive_optimize_invocation_count

        dispatch_counts = _install_runner_dispatch_spy(service, monkeypatch)
        thread_count = 6
        cycles = 12
        for cycle in range(cycles):
            key = f"race-same-key-consumer-{cycle}"

            def _call(key: str = key):
                return service.create_generated_recipe_attempt(
                    recipe_fingerprint=default_fingerprint,
                    idempotency_key=key,
                    model_id=_model(),
                )

            results, errors = _run_concurrently([_call for _ in range(thread_count)])
            _assert_no_thread_errors(errors, context=f"cycle {cycle}")

            attempt_ids = {result[2]["attempt_id"] for result in results}  # type: ignore[index]
            states = {result[2]["state"] for result in results}  # type: ignore[index]
            job_ids = {result[0].job_id for result in results}  # type: ignore[index]
            assert len(attempt_ids) == 1, f"cycle {cycle}: expected exactly one materialized attempt, got {attempt_ids}"
            assert states == {"succeeded"}, f"cycle {cycle}: every caller must observe the succeeded reuse result"
            assert job_ids == {source_job.job_id}, f"cycle {cycle}: job_id must alias the winner's existing build"
            for result in results:
                assert result[2]["candidate_selection_reuse"]["winner_candidate_attempt_id"] == (  # type: ignore[index]
                    source_candidate.candidate_attempt_id
                )
            assert not service._attempt_sync_guards, f"cycle {cycle}: attempt sync guard leaked"

        assert dispatch_counts["run"] == 0
        assert dispatch_counts["run_fallback_with_pre_olive_reuse"] == 0

        # The winner's own recorded history/counters are never mutated by any
        # of the concurrent reuse materializations.
        source_candidate_after = service._recipe_attempt_store.get_candidate_attempt(
            source_candidate.candidate_attempt_id
        )
        assert source_candidate_after.invocation_counters.mobius_build_invocation_count == source_mobius
        assert source_candidate_after.invocation_counters.olive_optimize_invocation_count == source_olive
    finally:
        service.close()


def test_concurrent_different_idempotency_keys_reuse_same_winner_create_distinct_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent callers using DIFFERENT Idempotency-Keys against the same
    reusable winner must each safely materialize their OWN distinct attempt
    (the per-attempt guard is keyed by attempt_id, so unrelated attempt_ids
    never contend on the same lock) -- all succeed, none dispatch to the
    runner, and the shared winner's own counters/history stay untouched."""
    service = _service(tmp_path)
    try:
        source_job, source_attempt, default_fingerprint = _create_default_attempt(
            service, idempotency_key="race-distinct-keys-source-1"
        )
        assert _wait_for_job_terminal(service, source_job.job_id).state == JobState.SUCCEEDED
        source_attempt_id = source_attempt["attempt_id"]
        _wait_for_lineage_finalized(service, source_attempt_id)
        source_candidate = service._recipe_attempt_store.list_candidate_attempts(source_attempt_id)[0]
        source_mobius = source_candidate.invocation_counters.mobius_build_invocation_count
        source_olive = source_candidate.invocation_counters.olive_optimize_invocation_count

        dispatch_counts = _install_runner_dispatch_spy(service, monkeypatch)
        thread_count = 6
        keys = [f"race-distinct-keys-consumer-{index}" for index in range(thread_count)]

        def _make_call(key: str):
            def _call():
                return service.create_generated_recipe_attempt(
                    recipe_fingerprint=default_fingerprint,
                    idempotency_key=key,
                    model_id=_model(),
                )

            return _call

        results, errors = _run_concurrently([_make_call(key) for key in keys])
        _assert_no_thread_errors(errors, context="distinct-key reuse")

        attempt_ids = [result[2]["attempt_id"] for result in results]  # type: ignore[index]
        assert len(set(attempt_ids)) == thread_count, "every distinct Idempotency-Key must materialize its own attempt"
        for result in results:
            assert result[2]["state"] == "succeeded"  # type: ignore[index]
            assert result[0].job_id == source_job.job_id  # type: ignore[index]
            assert result[2]["candidate_selection_reuse"]["winner_candidate_attempt_id"] == (  # type: ignore[index]
                source_candidate.candidate_attempt_id
            )

        assert dispatch_counts["run"] == 0
        assert dispatch_counts["run_fallback_with_pre_olive_reuse"] == 0
        assert not service._attempt_sync_guards, "attempt sync guards leaked"

        source_candidate_after = service._recipe_attempt_store.get_candidate_attempt(
            source_candidate.candidate_attempt_id
        )
        assert source_candidate_after.invocation_counters.mobius_build_invocation_count == source_mobius
        assert source_candidate_after.invocation_counters.olive_optimize_invocation_count == source_olive
    finally:
        service.close()


def test_reuse_materialization_resumes_after_interrupted_gates_on_fresh_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash/partial safety: interrupt the store-level gate-copy loop partway
    through (simulating a process crash after `start_attempt` committed and
    3 of 7 winner gates were durably recorded, but before
    `finish_attempt_succeeded`), then retry the SAME Idempotency-Key against
    a brand-new `LocalOnboardingService` instance sharing only the
    persisted, on-disk state stores (an in-memory-only per-attempt guard
    from the crashed instance can never help here). The durable
    `reuse_source_attempt_id` marker written atomically with the `RUNNING`
    transition lets the fresh instance recognize this as a safe reuse resume
    and complete it deterministically -- copying only the remaining gates,
    never re-dispatching to the build stage runner -- exactly once."""
    idempotency_key = "crash-recovery-consumer-1"
    service_a = _service(tmp_path)
    try:
        source_job, source_attempt, default_fingerprint = _create_default_attempt(
            service_a, idempotency_key="crash-recovery-source-1"
        )
        assert _wait_for_job_terminal(service_a, source_job.job_id).state == JobState.SUCCEEDED
        source_attempt_id = source_attempt["attempt_id"]
        _wait_for_lineage_finalized(service_a, source_attempt_id)
        winner_candidate = service_a._recipe_attempt_store.list_candidate_attempts(source_attempt_id)[0]

        store = service_a._recipe_attempt_store
        original_record_gate = store.record_attempt_gate
        call_count = {"n": 0}
        raise_after = 3  # 3 of 7 gates are durably recorded before the simulated crash.

        def _interrupting_record_gate(**kwargs):  # noqa: ANN003
            call_count["n"] += 1
            if call_count["n"] > raise_after:
                raise RuntimeError("simulated crash mid gate materialization")
            return original_record_gate(**kwargs)

        store.record_attempt_gate = _interrupting_record_gate  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="simulated crash"):
            service_a.create_generated_recipe_attempt(
                recipe_fingerprint=default_fingerprint,
                idempotency_key=idempotency_key,
                model_id=_model(),
            )
        assert not service_a._attempt_sync_guards, "guard leaked after simulated crash"

        interrupted = next(
            candidate_attempt
            for candidate_attempt in store.list_attempts()
            if candidate_attempt.idempotency_key == idempotency_key
        )
        assert interrupted.state == AttemptState.RUNNING
        assert len(interrupted.gate_results) == raise_after
        assert store.get_attempt_reuse_source(interrupted.attempt_id) == winner_candidate.attempt_id
    finally:
        service_a.close()

    # Fresh service instance -- a stand-in for a fresh process after a crash --
    # sharing only the persisted, on-disk service/job and recipe-attempt
    # stores (no in-memory state, including per-attempt guards, survives).
    service_b = _service(tmp_path)
    try:
        dispatch_counts = _install_runner_dispatch_spy(service_b, monkeypatch)
        jobs_before = len(service_b._jobs)

        resumed_job, _replay, resumed_attempt = service_b.create_generated_recipe_attempt(
            recipe_fingerprint=default_fingerprint,
            idempotency_key=idempotency_key,
            model_id=_model(),
        )
        assert resumed_attempt["state"] == "succeeded"
        assert resumed_attempt["attempt_id"] == interrupted.attempt_id
        assert resumed_job.job_id == source_job.job_id
        assert len(service_b._jobs) == jobs_before, "resume must never create a new BuildJob"
        assert dispatch_counts["run"] == 0
        assert dispatch_counts["run_fallback_with_pre_olive_reuse"] == 0
        assert not service_b._attempt_sync_guards

        final_attempt = service_b._recipe_attempt_store.get_attempt(interrupted.attempt_id)
        assert final_attempt.state == AttemptState.SUCCEEDED
        assert len(final_attempt.gate_results) == 7
        winner_full = service_b._recipe_attempt_store.get_attempt(winner_candidate.attempt_id)
        assert [g.gate for g in final_attempt.gate_results] == [g.gate for g in winner_full.gate_results]
        assert [g.status for g in final_attempt.gate_results] == [g.status for g in winner_full.gate_results]
        assert [g.evidence_ref for g in final_attempt.gate_results] == [
            g.evidence_ref for g in winner_full.gate_results
        ]

        # Retrying yet again (already SUCCEEDED, matching marker) stays a safe,
        # idempotent no-op: same result, no re-finish, no runner dispatch.
        _resumed_job_again, _replay_again, resumed_attempt_again = service_b.create_generated_recipe_attempt(
            recipe_fingerprint=default_fingerprint,
            idempotency_key=idempotency_key,
            model_id=_model(),
        )
        assert resumed_attempt_again["state"] == "succeeded"
        assert resumed_attempt_again["attempt_id"] == interrupted.attempt_id
        assert dispatch_counts["run"] == 0
        assert dispatch_counts["run_fallback_with_pre_olive_reuse"] == 0
        assert not service_b._attempt_sync_guards
    finally:
        service_b.close()


def test_reuse_materialization_resumes_after_interrupted_start_with_no_gates_on_fresh_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same crash/partial-safety scenario as
    `test_reuse_materialization_resumes_after_interrupted_gates_on_fresh_service`,
    but interrupting immediately after `start_attempt` -- before even the
    first gate is recorded -- to cover the narrower recoverable window."""
    idempotency_key = "crash-recovery-no-gates-consumer-1"
    service_a = _service(tmp_path)
    try:
        source_job, source_attempt, default_fingerprint = _create_default_attempt(
            service_a, idempotency_key="crash-recovery-no-gates-source-1"
        )
        assert _wait_for_job_terminal(service_a, source_job.job_id).state == JobState.SUCCEEDED
        source_attempt_id = source_attempt["attempt_id"]
        _wait_for_lineage_finalized(service_a, source_attempt_id)
        winner_candidate = service_a._recipe_attempt_store.list_candidate_attempts(source_attempt_id)[0]

        store = service_a._recipe_attempt_store

        def _always_interrupt(**kwargs):  # noqa: ANN003, ARG001
            raise RuntimeError("simulated crash immediately after start_attempt")

        store.record_attempt_gate = _always_interrupt  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="simulated crash"):
            service_a.create_generated_recipe_attempt(
                recipe_fingerprint=default_fingerprint,
                idempotency_key=idempotency_key,
                model_id=_model(),
            )
        assert not service_a._attempt_sync_guards

        interrupted = next(
            candidate_attempt
            for candidate_attempt in store.list_attempts()
            if candidate_attempt.idempotency_key == idempotency_key
        )
        assert interrupted.state == AttemptState.RUNNING
        assert len(interrupted.gate_results) == 0
        assert store.get_attempt_reuse_source(interrupted.attempt_id) == winner_candidate.attempt_id
    finally:
        service_a.close()

    service_b = _service(tmp_path)
    try:
        dispatch_counts = _install_runner_dispatch_spy(service_b, monkeypatch)

        resumed_job, _replay, resumed_attempt = service_b.create_generated_recipe_attempt(
            recipe_fingerprint=default_fingerprint,
            idempotency_key=idempotency_key,
            model_id=_model(),
        )
        assert resumed_attempt["state"] == "succeeded"
        assert resumed_attempt["attempt_id"] == interrupted.attempt_id
        assert resumed_job.job_id == source_job.job_id
        assert dispatch_counts["run"] == 0
        assert dispatch_counts["run_fallback_with_pre_olive_reuse"] == 0
        assert not service_b._attempt_sync_guards

        final_attempt = service_b._recipe_attempt_store.get_attempt(interrupted.attempt_id)
        assert final_attempt.state == AttemptState.SUCCEEDED
        assert len(final_attempt.gate_results) == 7
    finally:
        service_b.close()


def test_running_attempt_without_reuse_marker_fails_closed_instead_of_resuming(
    tmp_path: Path,
) -> None:
    """A `RUNNING` attempt with no durable reuse marker (or a marker naming a
    different winner) must never be treated as a safe resume: it is always
    either a real, in-flight non-reuse build or an untrusted/corrupt state,
    and `_materialize_reused_generated_attempt` must fail closed with the
    typed 409 exactly like any other unrecoverable state, never silently
    finishing it via copied winner gates."""
    service = _service(tmp_path)
    try:
        source_job, source_attempt, default_fingerprint = _create_default_attempt(
            service, idempotency_key="fail-closed-source-1"
        )
        assert _wait_for_job_terminal(service, source_job.job_id).state == JobState.SUCCEEDED
        source_attempt_id = source_attempt["attempt_id"]
        _wait_for_lineage_finalized(service, source_attempt_id)
        winner_candidate = service._recipe_attempt_store.list_candidate_attempts(source_attempt_id)[0]
        winner_attempt = service._recipe_attempt_store.get_attempt(winner_candidate.attempt_id)

        store = service._recipe_attempt_store
        reusable = service._resolve_reusable_candidate_selection(
            record=store.get_generated_recipe(default_fingerprint)
        )
        assert reusable is not None

        # Fabricate a RUNNING attempt for a fresh Idempotency-Key with no reuse
        # marker at all -- as if a real (non-reuse) build were genuinely
        # in-flight for it.
        attempt_request = build_attempt_request_from_generated(reusable.winner_generated_record)
        request_fingerprint = build_attempt_request_fingerprint(attempt_request)
        untrusted_attempt, _replay = store.create_attempt(
            idempotency_key="fail-closed-untrusted-1",
            request=attempt_request,
            request_fingerprint=request_fingerprint,
        )
        store.start_attempt(untrusted_attempt.attempt_id)  # no reuse_source_attempt_id
        assert store.get_attempt_reuse_source(untrusted_attempt.attempt_id) is None

        with pytest.raises(ServiceError) as excinfo:
            service._materialize_reused_generated_attempt(
                attempt_id=untrusted_attempt.attempt_id,
                winner_attempt=winner_attempt,
            )
        assert excinfo.value.code == "RECIPE_ATTEMPT_ALREADY_STARTED"
        assert excinfo.value.status_code == 409
        assert not service._attempt_sync_guards

        # Still RUNNING, untouched -- never silently finished.
        assert store.get_attempt(untrusted_attempt.attempt_id).state == AttemptState.RUNNING
    finally:
        service.close()


def test_non_reuse_generated_attempt_idempotent_replay_is_unaffected(tmp_path: Path) -> None:
    """Sanity check that the per-attempt guard added for the reuse path does
    not change ordinary (non-reuse) `create_generated_recipe_attempt`
    idempotent replay: a second call with the same Idempotency-Key -- before
    any candidate-selection winner exists -- must still short-circuit to the
    exact same job/attempt via the pre-existing `_attempt_to_build_job`
    mapping, with no new job created."""
    service = _service(tmp_path)
    try:
        preview = service.generated_recipe_preview(model_id=_model(), task=CandidateModality.LLM)
        generated = preview["generated_recipe"]
        assert generated["candidate_selection_reuse"] is None
        fingerprint = str(generated["fingerprint"])

        first_job, first_replay, first_attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="non-reuse-replay-1",
            model_id=_model(),
        )
        assert first_replay is False
        jobs_before = len(service._jobs)

        second_job, second_replay, second_attempt = service.create_generated_recipe_attempt(
            recipe_fingerprint=fingerprint,
            idempotency_key="non-reuse-replay-1",
            model_id=_model(),
        )
        assert second_replay is True
        assert second_job.job_id == first_job.job_id
        assert second_attempt["attempt_id"] == first_attempt["attempt_id"]
        assert len(service._jobs) == jobs_before
    finally:
        service.close()
