from __future__ import annotations

import hashlib
import json
import uuid

from dataclasses import asdict, dataclass
from pathlib import Path

from .candidates import resolve_candidate
from .cancellation import ProcessOwnershipRegistry
from .contracts import BuildArtifact, BuildJob, BuildRequest, CandidateModality, JobEvent, JobState
from .job_runner import LocalJobRunner
from .workspace_layout import workspace_root_for_job


class IdempotencyConflictError(ValueError):
    pass


class BuildNotFoundError(KeyError):
    pass


class ArtifactNotFoundError(KeyError):
    pass


class ArtifactNotReadyError(RuntimeError):
    pass


class ArtifactCapabilityError(ValueError):
    pass


@dataclass(frozen=True)
class BuildCreateBody:
    candidate_key: str
    task_profile: str
    hf_revision: str | None = None
    dry_run: bool = True
    skip_olive: bool = False


@dataclass(frozen=True)
class IdempotencyRecord:
    body_sha256: str
    job_id: str


class BuildService:
    def __init__(
        self,
        job_runner: LocalJobRunner,
        model_cache_dir: Path,
        workspace_base: Path | None = None,
        process_registry: ProcessOwnershipRegistry | None = None,
    ) -> None:
        self._job_runner = job_runner
        self._model_cache_dir = model_cache_dir.resolve()
        self._workspace_base = workspace_base
        self._process_registry = process_registry or ProcessOwnershipRegistry()
        self._jobs: dict[str, BuildJob] = {}
        self._idempotency: dict[str, IdempotencyRecord] = {}
        self._artifact_to_job: dict[str, str] = {}

    def create_build(self, body: BuildCreateBody, idempotency_key: str) -> tuple[BuildJob, bool]:
        if not idempotency_key.strip():
            raise ValueError("Idempotency-Key is required.")

        body_digest = _sha256_json(asdict(body))
        existing = self._idempotency.get(idempotency_key)
        if existing is not None:
            if existing.body_sha256 != body_digest:
                raise IdempotencyConflictError(
                    "Idempotency-Key was reused with a different request body."
                )
            return self.get_build(existing.job_id), True

        job_id = str(uuid.uuid4())
        candidate = resolve_candidate(body.candidate_key)
        workspace_root = workspace_root_for_job(job_id=job_id, base_dir=self._workspace_base)
        workspace_root.mkdir(parents=True, exist_ok=True)
        request = BuildRequest(
            candidate=candidate,
            workspace_root=workspace_root,
            model_cache_dir=self._model_cache_dir,
            output_dir=workspace_root / "output",
            task_profile=body.task_profile,
            hf_revision=body.hf_revision,
            skip_olive=body.skip_olive,
            dry_run=body.dry_run,
        )

        if not body.dry_run:
            raise NotImplementedError("Non-dry-run build execution is not wired in BuildService yet.")

        plan = self._job_runner.run_dry(request=request, job_id_override=job_id)
        job = plan.job
        self._jobs[job.job_id] = job
        self._idempotency[idempotency_key] = IdempotencyRecord(body_sha256=body_digest, job_id=job.job_id)
        if job.result_artifact_id:
            self._artifact_to_job[job.result_artifact_id] = job.job_id
        return job, False

    def get_build(self, job_id: str) -> BuildJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise BuildNotFoundError(job_id)
        return job

    def get_events(self, job_id: str, after: int = 0) -> tuple[JobEvent, ...]:
        job = self.get_build(job_id)
        return job.events_after(after)

    def cancel_build(self, job_id: str, reason: str = "Cancelled by client request.") -> tuple[BuildJob, Path | None]:
        job = self.get_build(job_id)
        quarantine = self._process_registry.cancel(job, reason=reason)
        return job, quarantine

    def record_artifact(self, job_id: str, artifact: BuildArtifact) -> None:
        job = self.get_build(job_id)
        job.register_artifact(artifact)
        self._artifact_to_job[artifact.artifact_id] = job_id

    def ensure_inference_target(self, artifact_id: str, infer_kind: str) -> BuildJob:
        job_id = self._artifact_to_job.get(artifact_id)
        if job_id is None:
            raise ArtifactNotFoundError(artifact_id)
        job = self.get_build(job_id)
        if job.state != JobState.SUCCEEDED:
            raise ArtifactNotReadyError(
                f"Artifact '{artifact_id}' belongs to job '{job_id}' in state '{job.state.value}'."
            )
        if infer_kind == "text" and job.request.candidate.modality != CandidateModality.LLM:
            raise ArtifactCapabilityError("Artifact does not support text inference.")
        if infer_kind == "asr" and job.request.candidate.modality != CandidateModality.ASR:
            raise ArtifactCapabilityError("Artifact does not support ASR inference.")
        return job


def _sha256_json(payload: dict[str, object]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
