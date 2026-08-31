from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import threading
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Callable, Protocol

from .adapters.foundry_cli import FoundryCliCatalogAdapter
from .adapters.huggingface_metadata import HuggingFaceMetadataAdapter
from .adapters.interfaces import FoundryCatalogClient, HuggingFaceMetadataClient, ProcessRunner
from .cancellation import ProcessOwnershipRegistry
from .contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    BuildRequest,
    CandidateModality,
    FailureClassification,
    FailureInfo,
    JobEvent,
    JobState,
    MatchConfidence,
    ModelCandidate,
    PreflightResult,
    ToolAvailability,
    ValidationResult,
    ValidationStatus,
)
from .failures import failure
from .hf_policy import config_requires_remote_code
from .paths import ensure_dir
from .preflight import PreflightInspector
from .serialization import to_jsonable
from .state_machine import CANCELLABLE_STATES, fail_job, transition
from .subprocess_runner import SafeSubprocessRunner
from .workspace_layout import default_workspace_base, workspace_root_for_job

_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{8,})")
_API_KEY_RE = re.compile(r"(?i)\b(api[-_ ]?key\s*[=:]\s*)(\S+)")

_DEFAULT_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def default_data_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise EnvironmentError("LOCALAPPDATA is required for local onboarding service data.")
    return Path(local_app_data) / "fl-onboard"


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in {"localhost", "::1", "127.0.0.1"}:
        return True
    octets = normalized.split(".")
    if len(octets) != 4:
        return False
    if not all(part.isdigit() for part in octets):
        return False
    return octets[0] == "127"


def enforce_loopback_host(host: str, allow_non_loopback: bool) -> str | None:
    if is_loopback_host(host):
        return None
    if not allow_non_loopback:
        raise ValueError(
            "Non-loopback host binding is blocked by default. Re-run with --allow-non-loopback to opt in."
        )
    return (
        "WARNING: non-loopback host binding is enabled. "
        "This can expose local build APIs to your network."
    )


class ServiceError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        detail: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.detail = detail or {}


@dataclass(frozen=True)
class BuildSubmission:
    model_id: str
    task: CandidateModality
    task_profile: str = "default"
    hf_revision: str | None = None
    skip_olive: bool = False

    def normalized(self) -> "BuildSubmission":
        return BuildSubmission(
            model_id=self.model_id.strip(),
            task=self.task,
            task_profile=self.task_profile.strip() or "default",
            hf_revision=self.hf_revision.strip() if self.hf_revision else None,
            skip_olive=bool(self.skip_olive),
        )

    def cache_identity(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "task": self.task.value,
            "task_profile": self.task_profile,
            "hf_revision": self.hf_revision,
            "skip_olive": self.skip_olive,
        }


@dataclass(frozen=True)
class IdempotencyRecord:
    body_sha256: str
    job_id: str


class BuildStageRunner(Protocol):
    def run(
        self,
        job: BuildJob,
        *,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        ...


class TextInferenceBackend(Protocol):
    def infer(
        self,
        *,
        artifact: BuildArtifact,
        job: BuildJob,
        prompt: str,
        max_tokens: int,
    ) -> str:
        ...


class AsrInferenceBackend(Protocol):
    def infer(
        self,
        *,
        artifact: BuildArtifact,
        job: BuildJob,
        audio_bytes: bytes,
        filename: str,
    ) -> str:
        ...


class UnverifiedBuildStageRunner:
    def run(
        self,
        job: BuildJob,
        *,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        if cancellation_event.is_set():
            return
        transition(job, JobState.DOWNLOADING, "Download stage reserved for adapter integration.")
        persist()
        if cancellation_event.is_set():
            return
        transition(job, JobState.MOBIUS_BUILDING, "Mobius build stage requested.")
        persist()
        fail_job(
            job,
            FailureInfo(
                stage=JobState.MOBIUS_BUILDING,
                classification=FailureClassification.NOT_VERIFIED,
                message=(
                    "Real Mobius/Olive/Foundry build adapters are not verified in this "
                    "local service configuration."
                ),
            ),
        )
        job.validations.append(
            ValidationResult(
                stage=JobState.MOBIUS_BUILDING,
                status=ValidationStatus.NOT_VERIFIED,
                checks=("adapter-verification=missing",),
                failure=job.failure,
            )
        )
        job.finished_utc = datetime.now(timezone.utc)
        persist()


class SQLiteStateStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        ensure_dir(self._db_path.parent)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    started_utc TEXT NOT NULL,
                    finished_utc TEXT,
                    failure_json TEXT,
                    result_artifact_id TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    state TEXT NOT NULL,
                    message TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    job_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    artifact_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS validations (
                    job_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    validation_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    idempotency_key TEXT PRIMARY KEY,
                    body_sha256 TEXT NOT NULL,
                    job_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS preflight_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL
                );
                """
            )

    def load_jobs(self) -> dict[str, BuildJob]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, state, request_json, started_utc, finished_utc, failure_json, result_artifact_id
                FROM jobs
                ORDER BY started_utc ASC
                """
            ).fetchall()
            out: dict[str, BuildJob] = {}
            for row in rows:
                request = _deserialize_build_request(json.loads(row["request_json"]))
                job = BuildJob(
                    job_id=row["job_id"],
                    request=request,
                    state=JobState(row["state"]),
                    started_utc=_parse_datetime(row["started_utc"]),
                    finished_utc=_parse_datetime(row["finished_utc"]) if row["finished_utc"] else None,
                )
                if row["failure_json"]:
                    job.failure = _deserialize_failure_info(json.loads(row["failure_json"]))
                job.result_artifact_id = row["result_artifact_id"]

                event_rows = connection.execute(
                    """
                    SELECT sequence, timestamp_utc, state, message
                    FROM events
                    WHERE job_id = ?
                    ORDER BY sequence ASC
                    """,
                    (job.job_id,),
                ).fetchall()
                job.events = [
                    JobEvent(
                        sequence=int(event["sequence"]),
                        timestamp_utc=_parse_datetime(event["timestamp_utc"]),
                        state=JobState(event["state"]),
                        message=event["message"],
                    )
                    for event in event_rows
                ]

                artifact_rows = connection.execute(
                    """
                    SELECT artifact_json
                    FROM artifacts
                    WHERE job_id = ?
                    ORDER BY ordinal ASC
                    """,
                    (job.job_id,),
                ).fetchall()
                job.artifacts = [
                    _deserialize_build_artifact(json.loads(artifact["artifact_json"]))
                    for artifact in artifact_rows
                ]

                validation_rows = connection.execute(
                    """
                    SELECT validation_json
                    FROM validations
                    WHERE job_id = ?
                    ORDER BY ordinal ASC
                    """,
                    (job.job_id,),
                ).fetchall()
                job.validations = [
                    _deserialize_validation_result(json.loads(validation["validation_json"]))
                    for validation in validation_rows
                ]
                out[job.job_id] = job
            return out

    def save_job(self, job: BuildJob) -> None:
        request_payload = _serialize_build_request(job.request)
        failure_payload = _serialize_failure_info(job.failure) if job.failure else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (job_id, state, request_json, started_utc, finished_utc, failure_json, result_artifact_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    state = excluded.state,
                    request_json = excluded.request_json,
                    started_utc = excluded.started_utc,
                    finished_utc = excluded.finished_utc,
                    failure_json = excluded.failure_json,
                    result_artifact_id = excluded.result_artifact_id
                """,
                (
                    job.job_id,
                    job.state.value,
                    json.dumps(request_payload, separators=(",", ":")),
                    _format_datetime(job.started_utc),
                    _format_datetime(job.finished_utc) if job.finished_utc else None,
                    json.dumps(failure_payload, separators=(",", ":")) if failure_payload else None,
                    job.result_artifact_id,
                ),
            )
            connection.execute("DELETE FROM events WHERE job_id = ?", (job.job_id,))
            connection.execute("DELETE FROM artifacts WHERE job_id = ?", (job.job_id,))
            connection.execute("DELETE FROM validations WHERE job_id = ?", (job.job_id,))
            connection.executemany(
                """
                INSERT INTO events (job_id, sequence, timestamp_utc, state, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        job.job_id,
                        event.sequence,
                        _format_datetime(event.timestamp_utc),
                        event.state.value,
                        event.message,
                    )
                    for event in job.events
                ],
            )
            connection.executemany(
                """
                INSERT INTO artifacts (job_id, ordinal, artifact_json)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        job.job_id,
                        idx,
                        json.dumps(_serialize_build_artifact(artifact), separators=(",", ":")),
                    )
                    for idx, artifact in enumerate(job.artifacts)
                ],
            )
            connection.executemany(
                """
                INSERT INTO validations (job_id, ordinal, validation_json)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        job.job_id,
                        idx,
                        json.dumps(_serialize_validation_result(validation), separators=(",", ":")),
                    )
                    for idx, validation in enumerate(job.validations)
                ],
            )

    def load_idempotency(self) -> dict[str, IdempotencyRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT idempotency_key, body_sha256, job_id FROM idempotency"
            ).fetchall()
            return {
                row["idempotency_key"]: IdempotencyRecord(
                    body_sha256=row["body_sha256"],
                    job_id=row["job_id"],
                )
                for row in rows
            }

    def save_idempotency(self, key: str, record: IdempotencyRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO idempotency (idempotency_key, body_sha256, job_id)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    body_sha256 = excluded.body_sha256,
                    job_id = excluded.job_id
                """,
                (key, record.body_sha256, record.job_id),
            )

    def load_preflight_cache(self) -> dict[str, PreflightResult]:
        with self._connect() as connection:
            rows = connection.execute("SELECT cache_key, result_json FROM preflight_cache").fetchall()
            out: dict[str, PreflightResult] = {}
            for row in rows:
                payload = json.loads(row["result_json"])
                out[row["cache_key"]] = _deserialize_preflight_result(payload)
            return out

    def save_preflight_cache(self, cache_key: str, result: PreflightResult) -> None:
        payload = json.dumps(_serialize_preflight_result(result), separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO preflight_cache (cache_key, result_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    result_json = excluded.result_json
                """,
                (cache_key, payload),
            )


class LocalOnboardingService:
    def __init__(
        self,
        *,
        db_path: Path | None = None,
        workspace_base: Path | None = None,
        model_cache_dir: Path | None = None,
        hf_metadata: HuggingFaceMetadataClient | None = None,
        foundry_catalog: FoundryCatalogClient | None = None,
        process_runner: ProcessRunner | None = None,
        preflight_inspector: PreflightInspector | None = None,
        process_registry: ProcessOwnershipRegistry | None = None,
        build_stage_runner: BuildStageRunner | None = None,
        text_inference_backend: TextInferenceBackend | None = None,
        asr_inference_backend: AsrInferenceBackend | None = None,
    ) -> None:
        data_root = default_data_root() if (db_path is None or model_cache_dir is None) else None
        self._workspace_base = (workspace_base or default_workspace_base()).resolve()
        if model_cache_dir is not None:
            resolved_cache_dir = model_cache_dir.resolve()
        else:
            assert data_root is not None
            resolved_cache_dir = (data_root / "cache").resolve()
        self._model_cache_dir = ensure_dir(resolved_cache_dir)
        if db_path is not None:
            resolved_db_path = db_path.resolve()
        else:
            assert data_root is not None
            resolved_db_path = (data_root / "state" / "service.sqlite3").resolve()
        self._store = SQLiteStateStore(resolved_db_path)
        self._lock = threading.RLock()
        self._jobs = self._store.load_jobs()
        self._idempotency = self._store.load_idempotency()
        self._preflight_cache = self._store.load_preflight_cache()
        self._artifact_to_job: dict[str, str] = {}
        self._cancel_events: dict[str, Event] = {}
        self._queue: Queue[str | None] = Queue()
        self._shutdown = Event()
        self._closed = False

        self._process_runner = process_runner or SafeSubprocessRunner()
        self._hf_metadata = hf_metadata or HuggingFaceMetadataAdapter()
        self._foundry_catalog = foundry_catalog or FoundryCliCatalogAdapter(self._process_runner)
        self._preflight_inspector = preflight_inspector or PreflightInspector(
            runner=self._process_runner,
            foundry=self._foundry_catalog,
            hf_metadata=self._hf_metadata,
        )
        self._process_registry = process_registry or ProcessOwnershipRegistry()
        self._build_stage_runner = build_stage_runner or UnverifiedBuildStageRunner()
        self._text_inference_backend = text_inference_backend
        self._asr_inference_backend = asr_inference_backend

        for job in self._jobs.values():
            for artifact in job.artifacts:
                self._artifact_to_job[artifact.artifact_id] = job.job_id
            if job.state in CANCELLABLE_STATES:
                self._cancel_events[job.job_id] = Event()
                self._queue.put(job.job_id)

        self._worker = Thread(target=self._worker_loop, name="fl-onboard-worker", daemon=True)
        self._worker.start()

    @property
    def db_path(self) -> Path:
        return self._store.db_path

    @property
    def cors_origin_regex(self) -> str:
        return _DEFAULT_CORS_ORIGIN_REGEX

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._shutdown.set()
            self._queue.put(None)
        self._worker.join(timeout=5)

    def health(self) -> dict[str, object]:
        with self._lock:
            active_job_id = next(
                (job_id for job_id, job in self._jobs.items() if job.state in CANCELLABLE_STATES),
                None,
            )
            return {
                "status": "ok",
                "active_job_id": active_job_id,
                "jobs_total": len(self._jobs),
                "storage_path": str(self.db_path),
            }

    def search_models(self, *, query: str, limit: int) -> dict[str, object]:
        q = query.strip()
        if not q:
            raise ServiceError(
                code="INVALID_QUERY",
                message="Query must not be empty.",
                status_code=400,
            )
        bounded_limit = max(1, min(limit, 50))
        results = self._hf_metadata.search_models(query=q, limit=bounded_limit, sort="downloads")
        return {
            "query": q,
            "limit": bounded_limit,
            "results": [to_jsonable(result) for result in results],
        }

    def model_detail(self, *, model_id: str) -> dict[str, object]:
        normalized = model_id.strip()
        if not normalized:
            raise ServiceError(
                code="INVALID_MODEL_ID",
                message="Model id must not be empty.",
                status_code=400,
            )
        try:
            metadata = self._hf_metadata.get_metadata(model_id=normalized, files_metadata=False)
        except Exception as exc:
            raise ServiceError(
                code="MODEL_LOOKUP_FAILED",
                message=_sanitize_text(str(exc) or "Model lookup failed."),
                status_code=502,
            ) from exc
        requires_remote_code = config_requires_remote_code(metadata.config)
        buildable = not bool(metadata.is_gated) and not requires_remote_code
        blockers: list[str] = []
        if metadata.is_gated:
            blockers.append("gated_model_blocked")
        if requires_remote_code:
            blockers.append("remote_code_blocked")

        matches = ()
        warnings: list[str] = []
        try:
            matches = self._foundry_catalog.list_matches(normalized.split("/")[-1])
        except Exception as exc:
            warnings.append(_sanitize_text(f"Foundry catalog probe failed: {exc}"))

        task_hints = _derive_task_hints(model_id=normalized, config=metadata.config)
        return {
            "model_id": metadata.model_id,
            "revision": metadata.revision,
            "sha": metadata.sha,
            "private": metadata.is_private,
            "gated": metadata.is_gated,
            "requires_remote_code": requires_remote_code,
            "buildable": buildable,
            "build_blockers": blockers,
            "task_hints": task_hints,
            "last_modified": metadata.last_modified,
            "safetensors_total_bytes": metadata.safetensors_total_bytes,
            "safetensors_parameter_count": metadata.safetensors_parameter_count,
            "card_data": metadata.card_data,
            "foundry_catalog_matches": [to_jsonable(item) for item in matches],
            "warnings": warnings,
        }

    def preflight(self, submission: BuildSubmission) -> dict[str, object]:
        normalized = submission.normalized()
        request = self._build_request(normalized, job_id="_preflight")
        output, cached, cache_key = self._inspect_preflight_with_cache_state(
            request,
            fallback_cache_payload=normalized.cache_identity(),
        )
        payload = to_jsonable(output)
        payload["cache_key"] = cache_key
        return {
            "cache_key": cache_key,
            "ok": output.ok,
            "cached": cached,
            "result": payload,
        }

    def create_build(self, submission: BuildSubmission, idempotency_key: str) -> tuple[BuildJob, bool]:
        key = idempotency_key.strip()
        if not key:
            raise ServiceError(
                code="IDEMPOTENCY_KEY_REQUIRED",
                message="Idempotency-Key is required.",
                status_code=400,
            )
        normalized = submission.normalized()
        body_sha256 = _sha256_json(normalized.cache_identity())
        with self._lock:
            existing = self._idempotency.get(key)
            if existing is not None:
                if existing.body_sha256 != body_sha256:
                    raise ServiceError(
                        code="IDEMPOTENCY_BODY_CONFLICT",
                        message="Idempotency-Key was reused with a different request body.",
                        status_code=409,
                    )
                replay_job = self._jobs.get(existing.job_id)
                if replay_job is None:
                    raise ServiceError(
                        code="IDEMPOTENCY_DANGLING_JOB",
                        message="Idempotency record points to a missing job.",
                        status_code=500,
                    )
                return replay_job, True
            active_job = next(
                (job for job in self._jobs.values() if job.state in CANCELLABLE_STATES),
                None,
            )
            if active_job is not None:
                raise ServiceError(
                    code="ACTIVE_BUILD_EXISTS",
                    message=(
                        f"Build job '{active_job.job_id}' is still active in state '{active_job.state.value}'. "
                        "Only one active build is supported."
                    ),
                    status_code=409,
                    detail={"active_job_id": active_job.job_id},
                )

            job_id = str(uuid.uuid4())
            request = self._build_request(normalized, job_id=job_id)
            job = BuildJob(job_id=job_id, request=request)
            job.add_event("Build job queued.")
            self._jobs[job_id] = job
            self._cancel_events[job_id] = Event()
            self._idempotency[key] = IdempotencyRecord(body_sha256=body_sha256, job_id=job_id)
            self._store.save_job(job)
            self._store.save_idempotency(key, self._idempotency[key])
            self._queue.put(job_id)
            return job, False

    def record_artifact(self, job_id: str, artifact: BuildArtifact) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ServiceError(
                    code="JOB_NOT_FOUND",
                    message=f"Build job '{job_id}' was not found.",
                    status_code=404,
                )
            job.register_artifact(artifact)
            self._artifact_to_job[artifact.artifact_id] = job_id
            self._store.save_job(job)

    def get_build(self, job_id: str) -> BuildJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ServiceError(
                    code="JOB_NOT_FOUND",
                    message=f"Build job '{job_id}' was not found.",
                    status_code=404,
                )
            return job

    def get_events(self, job_id: str, after: int = 0) -> tuple[JobEvent, ...]:
        if after < 0:
            raise ServiceError(
                code="INVALID_CURSOR",
                message="Event cursor must be >= 0.",
                status_code=400,
            )
        job = self.get_build(job_id)
        return job.events_after(after)

    def cancel_build(self, job_id: str, reason: str = "Cancelled by client request.") -> tuple[BuildJob, Path | None]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ServiceError(
                    code="JOB_NOT_FOUND",
                    message=f"Build job '{job_id}' was not found.",
                    status_code=404,
                )
            if job.state not in CANCELLABLE_STATES:
                raise ServiceError(
                    code="JOB_NOT_CANCELLABLE",
                    message=f"Build job '{job_id}' is not cancellable in state '{job.state.value}'.",
                    status_code=409,
                )
            cancellation_event = self._cancel_events.get(job_id)
            if cancellation_event:
                cancellation_event.set()
            quarantine = self._process_registry.cancel(job, reason=reason)
            job.finished_utc = datetime.now(timezone.utc)
            self._store.save_job(job)
            return job, quarantine

    def infer_text(self, *, artifact_id: str, prompt: str, max_tokens: int) -> dict[str, object]:
        artifact, job = self._resolve_inference_target(
            artifact_id=artifact_id,
            expected_modality=CandidateModality.LLM,
        )
        if self._text_inference_backend is None:
            raise ServiceError(
                code="INFERENCE_NOT_IMPLEMENTED",
                message="Text inference backend is not available in this service configuration.",
                status_code=501,
            )
        output = self._text_inference_backend.infer(
            artifact=artifact,
            job=job,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return {"artifact_id": artifact_id, "output": _sanitize_text(output)}

    def infer_asr(self, *, artifact_id: str, audio_bytes: bytes, filename: str) -> dict[str, object]:
        artifact, job = self._resolve_inference_target(
            artifact_id=artifact_id,
            expected_modality=CandidateModality.ASR,
        )
        if self._asr_inference_backend is None:
            raise ServiceError(
                code="INFERENCE_NOT_IMPLEMENTED",
                message="ASR inference backend is not available in this service configuration.",
                status_code=501,
            )
        transcript = self._asr_inference_backend.infer(
            artifact=artifact,
            job=job,
            audio_bytes=audio_bytes,
            filename=filename,
        )
        return {"artifact_id": artifact_id, "transcript": _sanitize_text(transcript)}

    def _resolve_inference_target(
        self,
        *,
        artifact_id: str,
        expected_modality: CandidateModality,
    ) -> tuple[BuildArtifact, BuildJob]:
        with self._lock:
            job_id = self._artifact_to_job.get(artifact_id)
            if job_id is None:
                raise ServiceError(
                    code="ARTIFACT_NOT_FOUND",
                    message=f"Artifact '{artifact_id}' was not found.",
                    status_code=404,
                )
            job = self._jobs[job_id]
            if job.state != JobState.SUCCEEDED:
                raise ServiceError(
                    code="ARTIFACT_NOT_READY",
                    message=(
                        f"Artifact '{artifact_id}' belongs to job '{job_id}' in state '{job.state.value}'. "
                        "Only succeeded jobs are inferable."
                    ),
                    status_code=409,
                )
            if job.request.candidate.modality != expected_modality:
                raise ServiceError(
                    code="ARTIFACT_TASK_MISMATCH",
                    message=(
                        f"Artifact '{artifact_id}' is '{job.request.candidate.modality.value}' and cannot be used for "
                        f"'{expected_modality.value}' inference."
                    ),
                    status_code=409,
                )
            for artifact in job.artifacts:
                if artifact.artifact_id == artifact_id:
                    return artifact, job
            raise ServiceError(
                code="ARTIFACT_NOT_FOUND",
                message=f"Artifact '{artifact_id}' metadata is missing from job '{job_id}'.",
                status_code=404,
            )

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
                return
            cancellation_event = self._cancel_events.setdefault(job_id, Event())
            if cancellation_event.is_set():
                return
            transition(job, JobState.PREFLIGHT, "Starting preflight checks.")
            self._store.save_job(job)

        try:
            preflight = self._inspect_preflight(job.request)
        except Exception as exc:
            classified = FailureInfo(
                stage=JobState.PREFLIGHT,
                classification=FailureClassification.UNKNOWN,
                message=_sanitize_text(str(exc) or "Unexpected preflight failure."),
            )
            with self._lock:
                live = self._jobs.get(job_id)
                if live is None or live.state == JobState.CANCELLED:
                    return
                fail_job(live, classified)
                live.finished_utc = datetime.now(timezone.utc)
                self._store.save_job(live)
            return

        with self._lock:
            live = self._jobs.get(job_id)
            if live is None or live.state == JobState.CANCELLED:
                return
            validation_status = ValidationStatus.PASSED if preflight.ok else ValidationStatus.FAILED
            validation_failure = preflight.blockers[0] if preflight.blockers else None
            live.validations.append(
                ValidationResult(
                    stage=JobState.PREFLIGHT,
                    status=validation_status,
                    checks=tuple(preflight.warnings),
                    failure=validation_failure,
                )
            )
            self._store.save_job(live)

            if not preflight.ok:
                fail_job(live, _sanitize_failure(preflight.blockers[0]))
                live.finished_utc = datetime.now(timezone.utc)
                self._store.save_job(live)
                return
            cancellation_event = self._cancel_events.setdefault(job_id, Event())
            if cancellation_event.is_set() or live.state == JobState.CANCELLED:
                return

            def persist() -> None:
                for artifact in live.artifacts:
                    self._artifact_to_job[artifact.artifact_id] = live.job_id
                self._store.save_job(live)

            self._build_stage_runner.run(
                live,
                persist=persist,
                cancellation_event=cancellation_event,
            )
            if live.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED} and live.finished_utc is None:
                live.finished_utc = datetime.now(timezone.utc)
            persist()

    def _inspect_preflight(self, request: BuildRequest) -> PreflightResult:
        result, _, _ = self._inspect_preflight_with_cache_state(
            request,
            fallback_cache_payload={
                "model": request.candidate.huggingface_model_id,
                "task_profile": request.task_profile,
                "revision": request.hf_revision,
                "skip_olive": request.skip_olive,
                "runtime": request.runtime,
            },
        )
        return result

    def _inspect_preflight_with_cache_state(
        self,
        request: BuildRequest,
        *,
        fallback_cache_payload: dict[str, object],
    ) -> tuple[PreflightResult, bool, str]:
        result = self._preflight_inspector.inspect(request)
        cache_key = result.cache_key or _sha256_json(
            {
                **fallback_cache_payload,
            }
        )
        with self._lock:
            cached = self._preflight_cache.get(cache_key)
            if cached is not None:
                return cached, True, cache_key
            self._preflight_cache[cache_key] = result
            self._store.save_preflight_cache(cache_key, result)
            return result, False, cache_key

    def _build_request(self, submission: BuildSubmission, *, job_id: str) -> BuildRequest:
        workspace = workspace_root_for_job(job_id=job_id, base_dir=self._workspace_base)
        ensure_dir(workspace)
        output_dir = ensure_dir(workspace / "output")
        candidate = ModelCandidate(
            key=_normalize_model_key(submission.model_id, submission.task),
            huggingface_model_id=submission.model_id,
            modality=submission.task,
            recommended_mobius_dtype="f16" if submission.task == CandidateModality.LLM else None,
            recommended_olive_precision="int4" if submission.task == CandidateModality.LLM else "fp16",
            notes="Derived from local service request.",
        )
        return BuildRequest(
            candidate=candidate,
            workspace_root=workspace,
            model_cache_dir=self._model_cache_dir,
            output_dir=output_dir,
            task_profile=submission.task_profile,
            hf_revision=submission.hf_revision,
            skip_olive=submission.skip_olive,
            dry_run=False,
        )


def _normalize_model_key(model_id: str, modality: CandidateModality) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id.strip().lower())
    return f"{modality.value}-{sanitized}"


def _derive_task_hints(model_id: str, config: dict[str, object] | None) -> list[str]:
    values = [model_id.lower()]
    if isinstance(config, dict):
        model_type = config.get("model_type")
        if isinstance(model_type, str):
            values.append(model_type.lower())
    joined = " ".join(values)
    if "whisper" in joined or "asr" in joined:
        return [CandidateModality.ASR.value]
    return [CandidateModality.LLM.value]


def _serialize_build_request(value: BuildRequest) -> dict[str, object]:
    return {
        "candidate": {
            "key": value.candidate.key,
            "huggingface_model_id": value.candidate.huggingface_model_id,
            "modality": value.candidate.modality.value,
            "recommended_mobius_dtype": value.candidate.recommended_mobius_dtype,
            "recommended_olive_precision": value.candidate.recommended_olive_precision,
            "notes": value.candidate.notes,
        },
        "workspace_root": str(value.workspace_root),
        "model_cache_dir": str(value.model_cache_dir),
        "output_dir": str(value.output_dir),
        "task_profile": value.task_profile,
        "hf_revision": value.hf_revision,
        "runtime": value.runtime,
        "external_data_format": value.external_data_format,
        "max_shard_size": value.max_shard_size,
        "enforce_cpu_target": value.enforce_cpu_target,
        "skip_olive": value.skip_olive,
        "dry_run": value.dry_run,
    }


def _deserialize_build_request(value: dict[str, object]) -> BuildRequest:
    candidate_raw = value["candidate"]
    if not isinstance(candidate_raw, dict):
        raise ValueError("Invalid build request payload: missing candidate object.")
    candidate = ModelCandidate(
        key=str(candidate_raw["key"]),
        huggingface_model_id=str(candidate_raw["huggingface_model_id"]),
        modality=CandidateModality(str(candidate_raw["modality"])),
        recommended_mobius_dtype=_optional_str(candidate_raw.get("recommended_mobius_dtype")),
        recommended_olive_precision=_optional_str(candidate_raw.get("recommended_olive_precision")),
        notes=str(candidate_raw.get("notes", "")),
    )
    return BuildRequest(
        candidate=candidate,
        workspace_root=Path(str(value["workspace_root"])),
        model_cache_dir=Path(str(value["model_cache_dir"])),
        output_dir=Path(str(value["output_dir"])),
        task_profile=str(value.get("task_profile", "default")),
        hf_revision=_optional_str(value.get("hf_revision")),
        runtime=str(value.get("runtime", "ort-genai")),
        external_data_format=str(value.get("external_data_format", "safetensors")),
        max_shard_size=str(value.get("max_shard_size", "5GB")),
        enforce_cpu_target=bool(value.get("enforce_cpu_target", True)),
        skip_olive=bool(value.get("skip_olive", False)),
        dry_run=bool(value.get("dry_run", False)),
    )


def _serialize_failure_info(value: FailureInfo) -> dict[str, object]:
    return {
        "stage": value.stage.value,
        "classification": value.classification.value,
        "message": value.message,
        "detail": dict(value.detail),
    }


def _deserialize_failure_info(value: dict[str, object]) -> FailureInfo:
    detail_raw = value.get("detail")
    detail: dict[str, str] = {}
    if isinstance(detail_raw, dict):
        detail = {str(k): str(v) for k, v in detail_raw.items()}
    return FailureInfo(
        stage=JobState(str(value["stage"])),
        classification=FailureClassification(str(value["classification"])),
        message=str(value["message"]),
        detail=detail,
    )


def _serialize_build_artifact(value: BuildArtifact) -> dict[str, object]:
    return {
        "artifact_id": value.artifact_id,
        "kind": value.kind.value,
        "path": str(value.path),
        "description": value.description,
        "size_bytes": value.size_bytes,
        "sha256": value.sha256,
    }


def _deserialize_build_artifact(value: dict[str, object]) -> BuildArtifact:
    return BuildArtifact(
        artifact_id=str(value["artifact_id"]),
        kind=ArtifactKind(str(value["kind"])),
        path=Path(str(value["path"])),
        description=str(value["description"]),
        size_bytes=int(value["size_bytes"]) if value.get("size_bytes") is not None else None,
        sha256=_optional_str(value.get("sha256")),
    )


def _serialize_validation_result(value: ValidationResult) -> dict[str, object]:
    return {
        "stage": value.stage.value,
        "status": value.status.value,
        "checks": list(value.checks),
        "failure": _serialize_failure_info(value.failure) if value.failure else None,
    }


def _deserialize_validation_result(value: dict[str, object]) -> ValidationResult:
    checks_raw = value.get("checks")
    checks = tuple(str(item) for item in checks_raw) if isinstance(checks_raw, list) else ()
    failure_raw = value.get("failure")
    failure_value = (
        _deserialize_failure_info(failure_raw)
        if isinstance(failure_raw, dict)
        else None
    )
    return ValidationResult(
        stage=JobState(str(value["stage"])),
        status=ValidationStatus(str(value["status"])),
        checks=checks,
        failure=failure_value,
    )


def _serialize_tool_availability(value: ToolAvailability) -> dict[str, object]:
    return {
        "name": value.name,
        "kind": value.kind,
        "available": value.available,
        "version": value.version,
        "detail": value.detail,
    }


def _deserialize_tool_availability(value: dict[str, object]) -> ToolAvailability:
    return ToolAvailability(
        name=str(value["name"]),
        kind=str(value["kind"]),
        available=bool(value["available"]),
        version=_optional_str(value.get("version")),
        detail=str(value.get("detail", "")),
    )


def _serialize_catalog_match(value) -> dict[str, object]:
    return {
        "alias": value.alias,
        "model_or_variant_id": value.model_or_variant_id,
        "source_schema": value.source_schema,
        "confidence": value.confidence.value,
        "reason": value.reason,
        "cached": value.cached,
        "model_type": value.model_type,
    }


def _deserialize_catalog_match(value: dict[str, object]):
    from .contracts import CatalogMatchAssessment

    return CatalogMatchAssessment(
        alias=str(value["alias"]),
        model_or_variant_id=str(value["model_or_variant_id"]),
        source_schema=str(value["source_schema"]),
        confidence=MatchConfidence(str(value["confidence"])),
        reason=str(value["reason"]),
        cached=value.get("cached") if isinstance(value.get("cached"), bool) else None,
        model_type=_optional_str(value.get("model_type")),
    )


def _serialize_preflight_result(value: PreflightResult) -> dict[str, object]:
    return {
        "candidate": {
            "key": value.candidate.key,
            "huggingface_model_id": value.candidate.huggingface_model_id,
            "modality": value.candidate.modality.value,
            "recommended_mobius_dtype": value.candidate.recommended_mobius_dtype,
            "recommended_olive_precision": value.candidate.recommended_olive_precision,
            "notes": value.candidate.notes,
        },
        "workspace_root": str(value.workspace_root),
        "model_cache_dir": str(value.model_cache_dir),
        "output_dir": str(value.output_dir),
        "disk_free_gb_workspace": value.disk_free_gb_workspace,
        "disk_free_gb_cache": value.disk_free_gb_cache,
        "tools": [_serialize_tool_availability(tool) for tool in value.tools],
        "foundry_catalog_matches": [_serialize_catalog_match(item) for item in value.foundry_catalog_matches],
        "huggingface_revision": value.huggingface_revision,
        "huggingface_sha": value.huggingface_sha,
        "huggingface_private": value.huggingface_private,
        "huggingface_gated": value.huggingface_gated,
        "cache_key": value.cache_key,
        "blockers": [_serialize_failure_info(item) for item in value.blockers],
        "warnings": list(value.warnings),
    }


def _deserialize_preflight_result(value: dict[str, object]) -> PreflightResult:
    candidate_raw = value["candidate"]
    if not isinstance(candidate_raw, dict):
        raise ValueError("Invalid preflight cache payload: candidate missing.")
    candidate = ModelCandidate(
        key=str(candidate_raw["key"]),
        huggingface_model_id=str(candidate_raw["huggingface_model_id"]),
        modality=CandidateModality(str(candidate_raw["modality"])),
        recommended_mobius_dtype=_optional_str(candidate_raw.get("recommended_mobius_dtype")),
        recommended_olive_precision=_optional_str(candidate_raw.get("recommended_olive_precision")),
        notes=str(candidate_raw.get("notes", "")),
    )
    tools_raw = value.get("tools")
    tools = (
        tuple(_deserialize_tool_availability(item) for item in tools_raw if isinstance(item, dict))
        if isinstance(tools_raw, list)
        else ()
    )
    matches_raw = value.get("foundry_catalog_matches")
    matches = (
        tuple(_deserialize_catalog_match(item) for item in matches_raw if isinstance(item, dict))
        if isinstance(matches_raw, list)
        else ()
    )
    blockers_raw = value.get("blockers")
    blockers = (
        tuple(_deserialize_failure_info(item) for item in blockers_raw if isinstance(item, dict))
        if isinstance(blockers_raw, list)
        else ()
    )
    warnings_raw = value.get("warnings")
    warnings = tuple(str(item) for item in warnings_raw) if isinstance(warnings_raw, list) else ()
    return PreflightResult(
        candidate=candidate,
        workspace_root=Path(str(value["workspace_root"])),
        model_cache_dir=Path(str(value["model_cache_dir"])),
        output_dir=Path(str(value["output_dir"])),
        disk_free_gb_workspace=float(value.get("disk_free_gb_workspace", 0.0)),
        disk_free_gb_cache=float(value.get("disk_free_gb_cache", 0.0)),
        tools=tools,
        foundry_catalog_matches=matches,
        huggingface_revision=_optional_str(value.get("huggingface_revision")),
        huggingface_sha=_optional_str(value.get("huggingface_sha")),
        huggingface_private=_optional_bool(value.get("huggingface_private")),
        huggingface_gated=_optional_bool(value.get("huggingface_gated")),
        cache_key=_optional_str(value.get("cache_key")),
        blockers=blockers,
        warnings=warnings,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _sha256_json(payload: dict[str, object]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sanitize_failure(value: FailureInfo) -> FailureInfo:
    return failure(
        stage=value.stage,
        classification=value.classification,
        message=_sanitize_text(value.message),
        detail={key: _sanitize_text(content) for key, content in value.detail.items()},
    )


def _sanitize_text(text: str) -> str:
    cleaned = _HF_TOKEN_RE.sub("[REDACTED]", text)
    cleaned = _BEARER_RE.sub("Bearer [REDACTED]", cleaned)
    cleaned = _API_KEY_RE.sub(r"\1[REDACTED]", cleaned)
    escaped = html.escape(cleaned, quote=False)
    return escaped[:4000]
