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
from threading import Event

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_local_service as tls  # noqa: E402

from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec  # noqa: E402
from fl_model_onboarding.contracts import CandidateModality, JobState  # noqa: E402
from fl_model_onboarding.local_service import BuildSubmission, LocalOnboardingService  # noqa: E402
from fl_model_onboarding.production_runner import ProductionBuildStageRunner, SMOLLM2_REVISION  # noqa: E402
from fl_model_onboarding.recipe_attempt_store import (  # noqa: E402
    AttemptState,
    CandidateLineageSelectionState,
    CandidateWinnerStatus,
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

