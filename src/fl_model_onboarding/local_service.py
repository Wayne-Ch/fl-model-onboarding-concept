from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Protocol

from .adapters.foundry_cli import FoundryCliCatalogAdapter
from .adapters.huggingface_metadata import HuggingFaceMetadataAdapter
from .adapters.interfaces import (
    FoundryCatalogClient,
    HuggingFaceAcquisitionClient,
    HuggingFaceMetadata,
    HuggingFaceMetadataClient,
    ProcessRunner,
)
from .architecture_capabilities import (
    ArgumentEvidenceConfidence,
    ArchitectureCapabilityRegistry,
    CapabilityStatus,
    ResolutionOutcome,
    load_architecture_capability_registry,
    normalize_huggingface_metadata,
)
from .cancellation import ProcessOwnershipRegistry
from .contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    BuildRequest,
    CandidateModality,
    FailureClassification,
    FailureInfo,
    GeneratedRecipeAttemptBinding,
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
from .production_runner import (
    FoundrySdkTextInferenceBackend,
    PreOliveArtifactDescriptor,
    PreOliveGenerationIdentity,
    PreOliveReuseError,
    ProductionBuildStageRunner,
    capture_pre_olive_artifact,
    pre_olive_generation_identity_from_generated_record,
    production_invocation_evidence_to_candidate_counters,
    production_package_paths,
    revalidate_pre_olive_source,
)
from .quality_validation import (
    GateState,
    PromptExecutionRecord,
    QualityRetryEvaluation,
    QualityValidationProfileRegistry,
    evaluate_quality_validation,
    load_quality_validation_profile_registry,
)
from .recipe_attempt_store import (
    AttemptFailure,
    AttemptFailureClassification,
    AttemptGate,
    AttemptGateSequenceError,
    AttemptGateStatus,
    AttemptIdempotencyConflictError,
    CandidateReuseIntegrityError,
    CandidateSelectionReuseQuery,
    AttemptState,
    AttemptStateTransitionError,
    CandidateAttemptRecord,
    CandidateInvocationCounters,
    CandidateLineageSelectionState,
    CandidatePlanValidationError,
    CandidateWinnerStatus,
    RecipeAttemptStore,
    RecipeAttemptStoreError,
    TERMINAL_ATTEMPT_STATES,
    build_attempt_request_fingerprint,
    build_attempt_request_from_generated,
    build_reuse_query_from_generated,
)
from .recipe_compiler import (
    GeneratedRecipe,
    GeneratedRecipeCompileError,
    RecipeArgumentProvenance,
    PromotionGateCheck,
    PromotionGateEvidence,
    RecipeCompilerInput,
    RecipeCompilerToolchain,
    RecipeGenerationProvenance,
    RecipeInputMetadata,
    RecipePromotionRecord,
    TrustedCandidateCompilationError,
    compile_generated_recipe,
    compile_trusted_candidate_recipe,
    promote_generated_recipe,
)
from .recipe_selection_policy import (
    DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY,
    DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY,
    RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
    RecipeSelectionPolicy,
)
from .recipes import (
    AncillaryFileRule,
    DEFAULT_RECIPE_REGISTRY,
    DISTIL_WHISPER_BLOCKED_REVISION,
    DISTIL_WHISPER_MODEL_ID,
    MobiusRecipeArgs,
    ModelRecipe,
    OliveRecipeArgs,
    OptimizationChoice,
    RecipeRegistry,
    RecipeResolution,
    RecipeStatus,
)
from .serialization import to_jsonable
from .state_machine import CANCELLABLE_STATES, fail_job, transition
from .subprocess_runner import SafeSubprocessRunner
from .workspace_layout import default_workspace_base, workspace_root_for_job

_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{8,})")
_API_KEY_RE = re.compile(r"(?i)\b(api[-_ ]?key\s*[=:]\s*)(\S+)")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|authorization|secret)\s*[:=]\s*([^\s,;]+)"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:(?:\\|/(?!/))[^\"'\r\n]*")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_DEFAULT_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
_ASR_MODEL_ID = DISTIL_WHISPER_MODEL_ID
_ASR_REVISION = DISTIL_WHISPER_BLOCKED_REVISION
_RECIPE_DEFAULT_TASK = CandidateModality.LLM.value
_RECIPE_DEFAULT_DEVICE = "cpu"
_RECIPE_DEFAULT_PRECISION = "auto"
_RECIPE_QUALITY_PROFILE_ID = "textgen-basic-quality-v1"
_AUTOMATIC_RECIPE_ATTEMPT_CONFIRMATION_PROVENANCE = "api.confirm_automatic_recipe_attempt"
_FALLBACK_CANDIDATE_ATTEMPT_CONFIRMATION_PROVENANCE = "internal.fallback_candidate_attempt"
_ATTEMPT_FAILURE_MESSAGE_MAX = 2048
_QUALITY_EVIDENCE_SCHEMA_VERSION = "1.0.0"
_QUALITY_EVIDENCE_FILENAME = "quality-validation-evidence.json"
_QUALITY_EVIDENCE_MAX_OUTPUT_CHARS = 512
_QUALITY_EVIDENCE_MAX_PROMPT_CHARS = 256
_QUALITY_METRICS_REF_RE = re.compile(
    r"^quality-metrics://(?P<job_id>[0-9a-fA-F-]+)/(?P<filename>[^/\\]+)$"
)
_RECIPE_TOOLCHAIN = RecipeCompilerToolchain(
    mobius_version="0.1.0",
    olive_version="0.13.0",
    onnx_version="1.22.0",
    ort_version="1.29.0",
    oga_version="0.15.2",
    foundry_sdk_version="1.2.4",
    foundry_cli_version="0.11.0",
)
_RECIPE_ATTEMPT_GATE_SEQUENCE: tuple[AttemptGate, ...] = (
    AttemptGate.MOBIUS_BUILD,
    AttemptGate.OLIVE_OPTIMIZE,
    AttemptGate.ONNX_VALIDATION,
    AttemptGate.ORT_VALIDATION,
    AttemptGate.OGA_VALIDATION,
    AttemptGate.FL_SDK_INFERENCE,
    AttemptGate.QUALITY_VALIDATION,
)
_ASR_RUNTIME_BLOCKER_DETAIL: dict[str, str] = {
    "required_input": "position_ids",
    "runtime_component": "WhisperDecoderState",
    "runtime_gap": "position_ids_not_bound_or_updated",
    "error_signature": "Missing Input: position_ids",
}


def _asr_candidate_outcome() -> dict[str, object]:
    return {
        "model_id": _ASR_MODEL_ID,
        "revision": _ASR_REVISION,
        "profile": "cpu/ort-genai; mobius=f32; deterministic-adapter=parser+model-load",
        "status": "blocked",
        "tested_status": "not_verified",
        "failed_stage": JobState.INFERENCING.value,
        "classification": FailureClassification.SOURCE_RUNTIME_CONTRACT_INCOMPATIBLE.value,
        "error_summary": (
            "Decoder ONNX requires position_ids, but OGA WhisperDecoderState does not bind/update it; "
            "OGA and Foundry Local transcription fail with Missing Input: position_ids."
        ),
        "versions": {
            "mobius": "0.1.0",
            "olive": "0.13.0",
            "onnx": "1.22.0",
            "onnxruntime": "1.29.0",
            "onnxruntime_genai": "0.15.2",
            "foundry_local_sdk": "1.2.4",
            "foundry_cli": "0.11.0",
        },
        "gate_outcomes": [
            {
                "stage": JobState.MOBIUS_BUILDING.value,
                "status": "passed",
                "summary": "Mobius CPU ort-genai f32 build succeeded.",
            },
            {
                "stage": JobState.RUNTIME_VALIDATING.value,
                "status": "passed",
                "summary": "ONNX checker and ORT CPU load succeeded.",
            },
            {
                "stage": JobState.FL_LOADING.value,
                "status": "passed",
                "summary": "Deterministic config adaptation advanced OGA parser/model-load gates.",
            },
            {
                "stage": JobState.INFERENCING.value,
                "status": "failed",
                "summary": (
                    "OGA and Foundry Local transcription fail with Missing Input: position_ids "
                    "(WhisperDecoderState does not bind/update position_ids)."
                ),
            },
        ],
        "evidence_reference": (
            "docs/asr-contract-repair.md#irreducible-failure-boundary "
            "(run 20260831-124030-fc016713)"
        ),
        "capability_owner": (
            "Primary owner: microsoft/onnxruntime-genai Whisper runtime; "
            "coordinate Mobius Whisper regression coverage."
        ),
        "next_action": (
            "Implement optional position_ids binding/updates from prompt + past sequence length, "
            "regression-test a Mobius-exported Whisper package, then rerun OGA + Foundry Local SDK "
            "transcription."
        ),
    }


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
    allow_experimental: bool = False
    optimization_strategy: str | None = None
    optimization_precision: str | None = None

    def normalized(self) -> "BuildSubmission":
        return BuildSubmission(
            model_id=self.model_id.strip(),
            task=self.task,
            task_profile=self.task_profile.strip() or "default",
            hf_revision=self.hf_revision.strip() if self.hf_revision else None,
            skip_olive=bool(self.skip_olive),
            allow_experimental=bool(self.allow_experimental),
            optimization_strategy=self.optimization_strategy.strip()
            if self.optimization_strategy
            else None,
            optimization_precision=self.optimization_precision.strip()
            if self.optimization_precision
            else None,
        )

    def cache_identity(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "task": self.task.value,
            "task_profile": self.task_profile,
            "hf_revision": self.hf_revision,
            "skip_olive": self.skip_olive,
            "allow_experimental": self.allow_experimental,
            "optimization_strategy": self.optimization_strategy,
            "optimization_precision": self.optimization_precision,
        }


@dataclass(frozen=True)
class IdempotencyRecord:
    body_sha256: str
    job_id: str


@dataclass(frozen=True)
class GeneratedRecipePreviewContext:
    model_id: str
    revision: str
    task: CandidateModality
    requested_device: str
    requested_precision: str
    generated_recipe: GeneratedRecipe | None
    compile_error: str | None
    capability: dict[str, object]
    files: tuple[str, ...]
    config_files: tuple[str, ...]
    tokenizer_files: tuple[str, ...]
    catalog_matches: tuple[object, ...]
    eligible_for_automatic_attempt: bool
    verified_reuse: dict[str, object] | None
    candidate_selection_reuse: dict[str, object] | None
    candidate_plan: dict[str, object] | None


@dataclass(frozen=True)
class CandidateSelectionReuseResolution:
    winner_candidate: CandidateAttemptRecord
    winner_attempt: Any
    winner_generated_record: Any
    winner_job_id: str


@dataclass(frozen=True)
class QualityValidationOutcome:
    passed: bool
    gate_status: AttemptGateStatus
    message: str
    evidence_ref: str
    metrics_ref: str | None = None
    # Slice 3B1: the typed quality-retry disposition/reasons derived from real gate
    # evidence by `evaluate_quality_validation`, carried through so the candidate
    # orchestration layer can decide whether a narrow, declarative fallback retry is
    # allowed -- never inferred from message text. Stays `None` whenever quality
    # validation never actually ran (missing/unavailable baseline, etc.), which the
    # orchestration layer must always treat the same as "not retryable".
    quality_retry_evaluation: QualityRetryEvaluation | None = None


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
                CREATE TABLE IF NOT EXISTS tested_models (
                    model_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    verified_utc TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    revision TEXT NOT NULL DEFAULT '',
                    task_profile TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS tested_model_profiles (
                    model_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    task_profile TEXT NOT NULL,
                    task TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    verified_utc TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    PRIMARY KEY (model_id, revision, task_profile)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tested_models)").fetchall()
            }
            if "revision" not in columns:
                connection.execute(
                    "ALTER TABLE tested_models ADD COLUMN revision TEXT NOT NULL DEFAULT ''"
                )
            if "task_profile" not in columns:
                connection.execute(
                    "ALTER TABLE tested_models ADD COLUMN task_profile TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO tested_model_profiles (
                    model_id, revision, task_profile, task, artifact_id, verified_utc, evidence
                )
                SELECT model_id, revision, task_profile, task, artifact_id, verified_utc, evidence
                FROM tested_models
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

    def load_tested_models(self) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT model_id, task, artifact_id, verified_utc, evidence, revision, task_profile
                FROM tested_model_profiles
                ORDER BY verified_utc DESC, model_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def save_tested_model(
        self,
        *,
        model_id: str,
        task: CandidateModality,
        artifact_id: str,
        revision: str | None,
        task_profile: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tested_model_profiles (
                    model_id, task, artifact_id, verified_utc, evidence, revision, task_profile
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, revision, task_profile) DO UPDATE SET
                    task = excluded.task,
                    artifact_id = excluded.artifact_id,
                    verified_utc = excluded.verified_utc,
                    evidence = excluded.evidence,
                    revision = excluded.revision,
                    task_profile = excluded.task_profile
                """,
                (
                    model_id,
                    task.value,
                    artifact_id,
                    datetime.now(timezone.utc).isoformat(),
                    "successful_fl_inference",
                    revision or "",
                    task_profile,
                ),
            )


class _AttemptSyncGuard:
    """Per-attempt serialization primitive for `_safe_sync_generated_attempt`.

    `lock` serializes duplicate concurrent syncs of the *same* generated
    attempt (e.g. the background worker finishing a job racing a
    `get_recipe_attempt` poll's own lazy re-sync) without ever holding the
    service-wide `LocalOnboardingService._lock` across the expensive work a
    sync performs (manifest hashing, process execution, quality validation).

    `waiters` is a refcount of callers currently holding a reference to this
    exact guard object, always mutated only while the service-wide lock is
    held (see `_acquire_attempt_sync_guard`/`_release_attempt_sync_guard`).
    It lets a finished caller reclaim the map entry once nobody references it
    any more, so the guard map stays bounded by the number of generated
    attempts *currently* syncing rather than growing for the lifetime of the
    service.
    """

    __slots__ = ("lock", "waiters")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.waiters = 0


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
        recipe_registry: RecipeRegistry | None = None,
        capability_registry: ArchitectureCapabilityRegistry | None = None,
        quality_profile_registry: QualityValidationProfileRegistry | None = None,
        recipe_attempt_store: RecipeAttemptStore | None = None,
        enable_production_runner: bool = False,
        runtime_python_executable: Path | str | None = None,
        model_acquisition: HuggingFaceAcquisitionClient | None = None,
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
        self._recipe_registry = recipe_registry or DEFAULT_RECIPE_REGISTRY
        self._capability_registry = capability_registry or load_architecture_capability_registry()
        self._quality_profile_registry = quality_profile_registry or load_quality_validation_profile_registry()
        self._quality_profile = self._quality_profile_registry.get(_RECIPE_QUALITY_PROFILE_ID)
        self._recipe_attempt_store = recipe_attempt_store or RecipeAttemptStore(
            resolved_db_path.parent / "recipe-attempts.sqlite3"
        )
        self._build_job_to_attempt: dict[str, str] = {}
        self._attempt_to_build_job: dict[str, str] = {}
        # Slice 3B1: the approved default CPU INT4 candidate-selection policy this
        # service resolves for every eligible generated attempt. Fixed for this
        # slice -- not user-selectable -- matching the "approved selection
        # policy/plan" scope described for Slice 3B1.
        self._recipe_selection_policy: RecipeSelectionPolicy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
        # Slice 3B1: process-memory-only cache of a captured pre-Olive Mobius
        # artifact descriptor for a default candidate's attempt_id, populated by
        # `_capture_pre_olive_descriptor_if_eligible` (the `on_mobius_ready` hook)
        # and consumed at most once by `_launch_fallback_candidate_attempt`.
        # Deliberately never persisted: it is always lost on a service restart,
        # which is why restart recovery below fails closed instead of silently
        # rebuilding Mobius to resume a fallback candidate.
        self._pre_olive_descriptors: dict[str, PreOliveArtifactDescriptor] = {}
        # Slice 3B1: job_id -> (descriptor, fallback_generation_identity) for a
        # fallback candidate job that has been enqueued but not yet dispatched to
        # the build stage runner. Consumed exactly once by `_run_job`, which then
        # calls `ProductionBuildStageRunner.run_fallback_with_pre_olive_reuse`
        # instead of the ordinary `run()` for that job only.
        self._fallback_launch_context: dict[str, tuple[PreOliveArtifactDescriptor, PreOliveGenerationIdentity]] = {}
        # Slice 3B1 revision: per-attempt serialization guards for
        # `_safe_sync_generated_attempt` (see `_AttemptSyncGuard` and
        # `_acquire_attempt_sync_guard`/`_release_attempt_sync_guard`).
        # Refcounted and torn down once unused, so this never grows without
        # bound across the service's lifetime.
        self._attempt_sync_guards: dict[str, _AttemptSyncGuard] = {}

        self._process_runner = process_runner or SafeSubprocessRunner()
        self._hf_metadata = hf_metadata or HuggingFaceMetadataAdapter()
        self._foundry_catalog = foundry_catalog or FoundryCliCatalogAdapter(self._process_runner)
        self._preflight_inspector = preflight_inspector or PreflightInspector(
            runner=self._process_runner,
            foundry=self._foundry_catalog,
            hf_metadata=self._hf_metadata,
        )
        self._process_registry = process_registry or ProcessOwnershipRegistry()
        self._build_stage_runner = build_stage_runner or (
            ProductionBuildStageRunner(
                self._process_runner,
                model_acquisition=model_acquisition,
                recipe_registry=self._recipe_registry,
                recipe_attempt_store=self._recipe_attempt_store,
                runtime_python_executable=runtime_python_executable,
                on_mobius_ready=self._capture_pre_olive_descriptor_if_eligible,
            )
            if enable_production_runner
            else UnverifiedBuildStageRunner()
        )
        self._text_inference_backend = text_inference_backend or (
            FoundrySdkTextInferenceBackend(
                self._process_runner,
                runtime_python_executable=runtime_python_executable,
            )
            if enable_production_runner
            else None
        )
        self._asr_inference_backend = asr_inference_backend

        for job in self._jobs.values():
            generated_attempt = job.request.generated_recipe_attempt
            if generated_attempt is not None:
                attempt_id = generated_attempt.attempt_id.strip()
                if attempt_id:
                    self._build_job_to_attempt[job.job_id] = attempt_id
                    self._attempt_to_build_job[attempt_id] = job.job_id
            for artifact in job.artifacts:
                self._artifact_to_job[artifact.artifact_id] = job.job_id
            if job.state in CANCELLABLE_STATES:
                for package_path in production_package_paths(job, recipe_registry=self._recipe_registry):
                    if package_path.exists():
                        shutil.rmtree(package_path)
                job.state = JobState.FAILED
                job.failure = FailureInfo(
                    stage=JobState.PREFLIGHT,
                    classification=FailureClassification.NOT_VERIFIED,
                    message=(
                        "Interrupted active job was not resumed after service restart; "
                        "submit a new idempotency key to retry."
                    ),
                )
                job.finished_utc = datetime.now(timezone.utc)
                job.add_event("Interrupted job marked failed during service recovery.")
                self._store.save_job(job)

        self._recover_orphaned_candidate_lineages()
        self._recover_abandoned_reuse_attempts_at_startup()
        self._recover_legacy_reuse_dispatch_evidence_at_startup()

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
            self._shutdown.set()
            for cancellation_event in self._cancel_events.values():
                cancellation_event.set()
            self._queue.put(None)
        self._worker.join(timeout=30)
        if self._worker.is_alive():
            raise RuntimeError("Local onboarding worker did not stop after cancellation.")
        with self._lock:
            self._closed = True

    def health(self) -> dict[str, object]:
        with self._lock:
            active_job_id = next(
                (job_id for job_id, job in self._jobs.items() if job.state in CANCELLABLE_STATES),
                None,
            )
            return {
                "status": "ok",
                "service": "fl-model-onboarding",
                "active_job_id": active_job_id,
                "jobs_total": len(self._jobs),
                "storage_path": str(self.db_path),
                "compatibility_index": [
                    {
                        **record,
                        "display_name": record["model_id"].split("/")[-1],
                        "tested_status": "tested",
                    }
                    for record in self._store.load_tested_models()
                ],
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
            "results": [
                {
                    **to_jsonable(result),
                    "verification": self._verification_for_model(result.model_id),
                }
                for result in results
            ],
        }

    def _load_model_metadata(self, model_id: str) -> HuggingFaceMetadata:
        try:
            return self._hf_metadata.get_metadata(model_id=model_id, files_metadata=True)
        except Exception as exc:
            raise ServiceError(
                code="MODEL_LOOKUP_FAILED",
                message=_sanitize_text(str(exc) or "Model lookup failed."),
                status_code=502,
            ) from exc

    def _probe_foundry_catalog(self, model_id: str) -> tuple[tuple[object, ...], str, list[str]]:
        warnings: list[str] = []
        try:
            matches = tuple(self._foundry_catalog.list_matches(model_id.split("/")[-1]))
            return matches, "live", warnings
        except Exception as exc:
            warnings.append(_sanitize_text(f"Foundry catalog probe failed: {exc}"))
            return (), "unavailable", warnings

    @staticmethod
    def _extract_model_files(metadata: HuggingFaceMetadata) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        files = tuple(sorted({str(path) for path in (metadata.sibling_files or ())}))
        config_files = tuple(
            path
            for path in files
            if PurePosixPath(path).name.lower() == "config.json"
        )
        tokenizer_files = tuple(
            path
            for path in files
            if PurePosixPath(path).name.lower() in {"tokenizer.json", "tokenizer_config.json", "tokenizer.model"}
        )
        return files, config_files, tokenizer_files

    def _capability_resolution_payload(self, *, resolution: Any) -> dict[str, object]:
        payload: dict[str, object] = {
            "outcome": resolution.outcome.value,
            "reason_code": resolution.reason_code.value,
            "reason": resolution.reason,
            "matched_aliases": list(resolution.matched_aliases),
            "evidence": [to_jsonable(item) for item in resolution.evidence],
            "capability": None,
        }
        if resolution.capability is None:
            return payload
        capability = resolution.capability
        payload["capability"] = {
            "capability_id": capability.capability_id,
            "version": capability.version,
            "family": capability.family,
            "status": capability.status.value,
            "modality": capability.modality.value,
            "supported_tasks": [
                row.value for row in capability.static_eligibility_constraints.supported_tasks
            ],
            "supported_devices": [
                row.value for row in capability.static_eligibility_constraints.supported_devices
            ],
            "supported_request_precisions": [
                row.value
                for row in capability.static_eligibility_constraints.supported_request_precisions
            ],
            "known_blockers": [to_jsonable(item) for item in capability.known_blockers],
        }
        return payload

    @staticmethod
    def _verified_reuse_payload(record: Any) -> dict[str, object]:
        return {
            "available": True,
            "verified_fingerprint": record.verified_fingerprint,
            "source_recipe_fingerprint": record.source_recipe_fingerprint,
            "attempt_id": record.attempt_id,
            "promoted_utc": record.promoted_utc.isoformat(),
            "recipe": record.payload(),
        }

    @staticmethod
    def _candidate_role(candidate_index: int) -> str:
        """Stable, user-understandable role code for a policy candidate
        index: ``"default"`` for the always-eligible index 0, and
        ``"quality_retry"`` for every conditional fallback candidate (index
        >= 1). Frontend (Slice 3C2) is expected to translate these into
        plain-language labels (e.g. "First recipe" / "Automatic quality
        retry") rather than surfacing the machine code directly.
        """
        return "default" if candidate_index == 0 else "quality_retry"

    @classmethod
    def _candidate_plan_payload(cls, policy: RecipeSelectionPolicy) -> dict[str, object]:
        """Public, additive candidate-plan summary for the generated-recipe
        preview response (Slice 3C1). Pure projection of the approved,
        already-validated selection policy: never touches the store, never
        depends on any particular model/attempt, and is always the exact
        same shape for every CPU INT4-eligible preview. Candidate order is
        the policy's own already-validated, index-ordered tuple, so this is
        always deterministic.
        """
        return {
            "policy_id": policy.policy_id,
            "policy_version": policy.version,
            "policy_fingerprint": policy.fingerprint,
            "max_candidates": policy.max_candidates,
            "candidates": [
                {
                    "candidate_index": candidate.candidate_index,
                    "candidate_id": candidate.candidate_id,
                    "role": cls._candidate_role(candidate.candidate_index),
                    "quantization_override": (
                        {"block_size": candidate.quantization_override.block_size}
                        if candidate.quantization_override is not None
                        else None
                    ),
                    "eligibility_trigger": candidate.eligibility_trigger,
                }
                for candidate in policy.candidates
            ],
        }

    @staticmethod
    def _candidate_selection_reuse_payload(
        resolution: CandidateSelectionReuseResolution,
    ) -> dict[str, object]:
        winner = resolution.winner_candidate
        return {
            "available": True,
            "source_parent_attempt_id": winner.parent_attempt_id,
            "winner_candidate_attempt_id": winner.candidate_attempt_id,
            "winner_attempt_id": winner.attempt_id,
            "winner_candidate_index": winner.candidate_index,
            "winner_candidate_id": winner.candidate_id,
            "winner_recipe_fingerprint": winner.recipe_fingerprint,
            "selection_reason": winner.selection_reason,
            "policy_fingerprint": winner.policy_fingerprint,
            "quality_profile_fingerprint": winner.quality_profile_fingerprint,
            "winner_job_id": resolution.winner_job_id,
        }

    @staticmethod
    def _selection_scope_identity_from_generated_record(
        record: Any,
    ) -> tuple[str, str, str, str] | None:
        payload = record.payload()
        recipe_payload = payload.get("recipe") if isinstance(payload, dict) else None
        if not isinstance(recipe_payload, dict):
            return None
        olive_payload = recipe_payload.get("olive")
        mobius_payload = recipe_payload.get("mobius")
        if not isinstance(olive_payload, dict) or not isinstance(mobius_payload, dict):
            return None
        target_device = olive_payload.get("device")
        target_ep = mobius_payload.get("ep")
        if not isinstance(target_device, str) or not target_device.strip():
            return None
        if not isinstance(target_ep, str) or not target_ep.strip():
            return None
        toolchain = record.toolchain_fingerprint
        if not isinstance(toolchain, str) or not toolchain.strip():
            return None
        provider = olive_payload.get("provider")
        environment_scope = (
            f"foundry-local-onboarding:{provider}"
            if isinstance(provider, str) and provider.strip()
            else "foundry-local-onboarding"
        )
        return (target_device.strip().lower(), target_ep.strip(), toolchain.strip().lower(), environment_scope)

    def _build_candidate_selection_reuse_query(
        self,
        *,
        record: Any,
    ) -> CandidateSelectionReuseQuery | None:
        quality_fingerprint = self._quality_profile.fingerprint.strip().lower()
        policy_fingerprint = self._recipe_selection_policy.fingerprint.strip().lower()
        identities = (
            record.model_id,
            record.revision_sha,
            record.requested_device,
            record.requested_precision,
            record.compiler_version,
            record.capability_fingerprint,
            record.toolchain_fingerprint,
            record.profile_fingerprint,
            quality_fingerprint,
            policy_fingerprint,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            # Missing identity is always a safe miss: never wildcard any field.
            return None
        return CandidateSelectionReuseQuery(
            model_id=record.model_id,
            revision_sha=record.revision_sha,
            requested_device=record.requested_device,
            requested_precision=record.requested_precision,
            compiler_version=record.compiler_version,
            capability_fingerprint=record.capability_fingerprint,
            toolchain_fingerprint=record.toolchain_fingerprint,
            profile_fingerprint=record.profile_fingerprint,
            quality_profile_fingerprint=quality_fingerprint,
            policy_fingerprint=policy_fingerprint,
        )

    def _raise_candidate_selection_reuse_integrity_error(self, message: str) -> None:
        raise ServiceError(
            code="CANDIDATE_SELECTION_REUSE_INTEGRITY_ERROR",
            message=_sanitize_text(message),
            status_code=500,
        )

    def _resolve_reusable_candidate_selection(
        self,
        *,
        record: Any,
    ) -> CandidateSelectionReuseResolution | None:
        query = self._build_candidate_selection_reuse_query(record=record)
        if query is None:
            return None
        expected_scope = self._selection_scope_identity_from_generated_record(record)
        if expected_scope is None:
            return None
        expected_device, expected_ep, expected_toolchain, expected_environment = expected_scope
        try:
            winner = self._recipe_attempt_store.find_reusable_candidate_selection(query)
        except CandidateReuseIntegrityError as exc:
            self._raise_candidate_selection_reuse_integrity_error(str(exc))
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc
        if winner is None:
            return None
        if winner.selection_status != CandidateWinnerStatus.SELECTED:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Candidate winner '{winner.candidate_attempt_id}' is not marked selected."
            )
        if winner.attempt_state != AttemptState.SUCCEEDED:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Candidate winner '{winner.candidate_attempt_id}' references non-succeeded attempt "
                f"'{winner.attempt_id}' ({winner.attempt_state.value})."
            )
        if not winner.has_fully_validated_selection_scope:
            return None
        if (
            (winner.validated_target_device or "").strip().lower() != expected_device
            or (winner.validated_target_ep or "").strip() != expected_ep
            or (winner.validated_toolchain_fingerprint or "").strip().lower() != expected_toolchain
            or (winner.validated_environment_scope or "").strip() != expected_environment
        ):
            return None

        try:
            lineage = self._recipe_attempt_store.get_candidate_lineage(winner.parent_attempt_id)
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc
        if lineage is None:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Selected winner '{winner.candidate_attempt_id}' has no parent lineage "
                f"'{winner.parent_attempt_id}'."
            )
        if lineage.selection_state != CandidateLineageSelectionState.SELECTED:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner '{winner.candidate_attempt_id}' parent lineage '{winner.parent_attempt_id}' "
                f"is '{lineage.selection_state.value}', expected 'selected'."
            )
        if lineage.selected_candidate_attempt_id != winner.candidate_attempt_id:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Lineage '{winner.parent_attempt_id}' selected candidate id mismatch."
            )
        if lineage.policy_fingerprint != self._recipe_selection_policy.fingerprint:
            return None
        if lineage.quality_profile_fingerprint != self._quality_profile.fingerprint:
            return None

        try:
            winner_attempt = self._recipe_attempt_store.get_attempt(winner.attempt_id)
        except KeyError:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner attempt '{winner.attempt_id}' does not exist."
            )
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc
        if winner_attempt.state != AttemptState.SUCCEEDED:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner attempt '{winner.attempt_id}' is '{winner_attempt.state.value}', expected 'succeeded'."
            )
        try:
            parent_attempt = self._recipe_attempt_store.get_attempt(winner.parent_attempt_id)
        except KeyError:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner parent attempt '{winner.parent_attempt_id}' does not exist."
            )
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc
        parent_record = self._recipe_attempt_store.get_generated_recipe(parent_attempt.recipe_fingerprint)
        if parent_record is None:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner parent generated recipe '{parent_attempt.recipe_fingerprint}' is missing."
            )
        winner_record = self._recipe_attempt_store.get_generated_recipe(winner.recipe_fingerprint)
        if winner_record is None:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner generated recipe '{winner.recipe_fingerprint}' is missing."
            )

        try:
            trusted_policy_candidate = self._recipe_selection_policy.candidates[winner.candidate_index]
        except IndexError:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner candidate index '{winner.candidate_index}' is out of range for current policy "
                f"'{self._recipe_selection_policy.policy_id}'."
            )
        if trusted_policy_candidate.candidate_id != winner.candidate_id:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner candidate id '{winner.candidate_id}' does not match trusted policy candidate "
                f"'{trusted_policy_candidate.candidate_id}' at index {winner.candidate_index}."
            )

        try:
            default_recipe = self._recompile_generated_recipe_record(record=parent_record)
        except ServiceError as exc:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner parent recipe '{parent_record.recipe_fingerprint}' failed trusted recompile: "
                f"{exc.message}"
            )
        if winner.candidate_index == 0:
            expected_winner = default_recipe
        else:
            try:
                expected_winner = compile_trusted_candidate_recipe(
                    default_recipe,
                    policy=self._recipe_selection_policy,
                    candidate=trusted_policy_candidate,
                )
            except TrustedCandidateCompilationError as exc:
                self._raise_candidate_selection_reuse_integrity_error(
                    f"Unable to re-derive trusted candidate {winner.candidate_index} under policy "
                    f"'{self._recipe_selection_policy.policy_id}': {exc}"
                )
        if winner.recipe_fingerprint != expected_winner.fingerprint:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner recipe fingerprint '{winner.recipe_fingerprint}' does not match re-derived trusted "
                f"candidate fingerprint '{expected_winner.fingerprint}'."
            )
        if winner_record.recipe_fingerprint != expected_winner.fingerprint:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner generated recipe '{winner_record.recipe_fingerprint}' does not match re-derived trusted "
                f"candidate fingerprint '{expected_winner.fingerprint}'."
            )
        try:
            self._recompile_generated_recipe_record(record=winner_record)
        except ServiceError as exc:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner generated recipe '{winner_record.recipe_fingerprint}' failed trusted recompile: "
                f"{exc.message}"
            )

        with self._lock:
            winner_job_id = self._attempt_to_build_job.get(winner.attempt_id)
            winner_job = self._jobs.get(winner_job_id) if winner_job_id is not None else None
        if winner_job_id is None or winner_job is None:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner attempt '{winner.attempt_id}' has no corresponding build job."
            )
        if winner_job.result_artifact_id is None:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner job '{winner_job_id}' has no result artifact id."
            )
        expected_artifact_ref = f"job://{winner_job_id}/artifact/{winner_job.result_artifact_id}"
        expected_package_ref = f"job://{winner_job_id}/package"
        if winner.artifact_ref != expected_artifact_ref or winner.package_ref != expected_package_ref:
            self._raise_candidate_selection_reuse_integrity_error(
                f"Winner '{winner.candidate_attempt_id}' artifact/package refs do not match job '{winner_job_id}'."
            )

        return CandidateSelectionReuseResolution(
            winner_candidate=winner,
            winner_attempt=winner_attempt,
            winner_generated_record=winner_record,
            winner_job_id=winner_job_id,
        )

    def _materialize_reused_generated_attempt(
        self,
        *,
        attempt_id: str,
        winner_attempt: Any,
    ) -> Any:
        """Serialize and idempotently drive the multi-step (get -> start ->
        copy gates -> finish) materialization of a candidate-selection-reuse
        attempt.

        Concurrent `create_generated_recipe_attempt` calls that resolve the
        SAME idempotency key -- and therefore the same `attempt_id` -- must
        never race each other through this sequence: `start_attempt`,
        `record_attempt_gate`, and `finish_attempt_succeeded` each open and
        commit their own store-level transaction, so without additional
        serialization here a second concurrent caller could read a stale
        `GENERATED` snapshot and call `start_attempt` again after a first
        caller already transitioned the row to `RUNNING`, surfacing an
        `AttemptStateTransitionError` (mapped to a 500) or the generic
        `RECIPE_ATTEMPT_ALREADY_STARTED` 409 instead of quietly joining the
        in-flight (or already finished) materialization.

        This uses the exact same per-attempt guard primitive as
        `_safe_sync_generated_attempt` (`_acquire_attempt_sync_guard`/
        `_release_attempt_sync_guard`): a `threading.Lock` scoped to this
        single `attempt_id`, acquired and released without ever holding the
        service-wide `self._lock` across the guarded work, matching the
        approved 3B1 lock-ordering design (no call site here blocks on the
        per-attempt guard while already holding `self._lock`). Every
        concurrent caller for the same `attempt_id` serializes on that
        per-attempt lock, re-reads the attempt's *current* persisted state
        once it acquires the lock, and only then decides whether
        materialization is still necessary -- so at most one caller ever
        performs a state transition and every other caller observes and
        returns the already-materialized result instead of racing it.

        Fresh-process crash/interrupt recovery (e.g. a restart between
        `start_attempt` and `finish_attempt_succeeded`) is handled by the
        durable `reuse_source_attempt_id` marker written atomically with the
        `RUNNING` transition in `start_attempt`: a `RUNNING` attempt whose
        marker names this exact winner is a recoverable, in-progress reuse
        materialization -- this path never dispatches a build job or touches
        `ProductionBuildStageRunner`, so there is never a runner to race or
        re-invoke -- and is resumed deterministically by copying only the
        winner gates not yet recorded, then finishing. A `RUNNING` attempt
        with no marker, or a marker naming a different winner, is never
        assumed to be a safe reuse resume: it fails closed with a typed 409,
        exactly like any other in-flight or terminal-but-mismatched state.
        """
        guard = self._acquire_attempt_sync_guard(attempt_id)
        try:
            with guard.lock:
                attempt = self._recipe_attempt_store.get_attempt(attempt_id)

                if attempt.state == AttemptState.SUCCEEDED:
                    reuse_source = self._recipe_attempt_store.get_attempt_reuse_source(attempt_id)
                    if reuse_source == winner_attempt.attempt_id:
                        self._backfill_legacy_reuse_dispatch_evidence(attempt_id)
                        return attempt
                    self._raise_candidate_selection_reuse_integrity_error(
                        f"Recipe attempt '{attempt_id}' already succeeded without a matching reuse "
                        f"marker for winner attempt '{winner_attempt.attempt_id}'."
                    )

                if attempt.state == AttemptState.RUNNING:
                    reuse_source = self._recipe_attempt_store.get_attempt_reuse_source(attempt_id)
                    if reuse_source != winner_attempt.attempt_id:
                        raise ServiceError(
                            code="RECIPE_ATTEMPT_ALREADY_STARTED",
                            message=(
                                f"Recipe attempt '{attempt_id}' is already running without a recoverable "
                                "reuse-materialization marker for this winner."
                            ),
                            status_code=409,
                        )
                    if winner_attempt.state != AttemptState.SUCCEEDED:
                        self._raise_candidate_selection_reuse_integrity_error(
                            "Cannot resume reuse materialization from non-succeeded winner attempt "
                            f"'{winner_attempt.attempt_id}'."
                        )
                    for index, recorded_gate in enumerate(attempt.gate_results):
                        if (
                            index >= len(winner_attempt.gate_results)
                            or recorded_gate.gate != winner_attempt.gate_results[index].gate
                        ):
                            self._raise_candidate_selection_reuse_integrity_error(
                                f"Recipe attempt '{attempt_id}' has a partial gate history that does not "
                                f"match winner attempt '{winner_attempt.attempt_id}'."
                            )
                    for gate in winner_attempt.gate_results[len(attempt.gate_results) :]:
                        self._recipe_attempt_store.record_attempt_gate(
                            attempt_id=attempt_id,
                            gate=gate.gate,
                            status=gate.status,
                            evidence_ref=gate.evidence_ref,
                            metrics_ref=gate.metrics_ref,
                        )
                    return self._recipe_attempt_store.finish_reused_attempt_succeeded_with_dispatch_evidence(
                        attempt_id,
                        source_attempt_id=winner_attempt.attempt_id,
                    )

                if attempt.state != AttemptState.GENERATED:
                    raise ServiceError(
                        code="RECIPE_ATTEMPT_ALREADY_STARTED",
                        message=(
                            f"Recipe attempt '{attempt.attempt_id}' is already in state '{attempt.state.value}' "
                            "without a recoverable build job mapping."
                        ),
                        status_code=409,
                    )
                if winner_attempt.state != AttemptState.SUCCEEDED:
                    self._raise_candidate_selection_reuse_integrity_error(
                        f"Cannot materialize reuse from non-succeeded winner attempt '{winner_attempt.attempt_id}'."
                    )
                self._recipe_attempt_store.start_attempt(
                    attempt.attempt_id,
                    reuse_source_attempt_id=winner_attempt.attempt_id,
                )
                for gate in winner_attempt.gate_results:
                    self._recipe_attempt_store.record_attempt_gate(
                        attempt_id=attempt.attempt_id,
                        gate=gate.gate,
                        status=gate.status,
                        evidence_ref=gate.evidence_ref,
                        metrics_ref=gate.metrics_ref,
                    )
                return self._recipe_attempt_store.finish_reused_attempt_succeeded_with_dispatch_evidence(
                    attempt.attempt_id,
                    source_attempt_id=winner_attempt.attempt_id,
                )
        finally:
            self._release_attempt_sync_guard(attempt_id, guard)

    def _backfill_legacy_reuse_dispatch_evidence(self, attempt_id: str) -> None:
        """Legacy/crash backfill (reuse success/evidence atomicity revision):
        opportunistically heal an already-``SUCCEEDED`` candidate-selection-
        reuse attempt that is missing its durable dispatch evidence -- only
        reachable from the pre-fix code path that recorded success and
        evidence in two separate transactions (or a crash between them),
        never from current writes, which always use
        `finish_reused_attempt_succeeded_with_dispatch_evidence`.

        Called opportunistically on every observed path that can see such an
        attempt again (resubmission here, plus `get_recipe_attempt` polling
        and fresh-startup recovery -- see `_recover_legacy_reuse_dispatch_evidence_at_startup`),
        guarded by the same per-attempt lock used everywhere else in this
        file so it can never race a concurrent caller into inserting
        conflicting evidence. Never raises: if the source cannot be safely
        revalidated, records an explicit, sanitized integrity-failure audit
        row instead (see `RecipeAttemptStore.record_reuse_evidence_backfill_integrity_failure`)
        so the gap stays detectable rather than silently persisting forever.
        The already-terminal attempt row itself is never touched, no tool is
        ever rerun, and no gate/source invocation counter is ever rewritten.
        """
        if self._recipe_attempt_store.get_reuse_dispatch_evidence(attempt_id) is not None:
            return
        try:
            reuse_source_attempt_id = self._recipe_attempt_store.get_attempt_reuse_source(attempt_id)
        except KeyError:
            return
        if reuse_source_attempt_id is None:
            return
        try:
            self._recipe_attempt_store.backfill_reused_attempt_dispatch_evidence(
                attempt_id,
                source_attempt_id=reuse_source_attempt_id,
            )
        except RecipeAttemptStoreError as exc:
            try:
                self._recipe_attempt_store.record_reuse_evidence_backfill_integrity_failure(
                    attempt_id=attempt_id,
                    reuse_source_attempt_id=reuse_source_attempt_id,
                    reason=_sanitize_attempt_failure_message(str(exc)),
                )
            except RecipeAttemptStoreError:
                # Recording the audit row itself is best-effort: even an
                # unredactable reason must never turn this opportunistic
                # healing call into a hard failure for an unrelated
                # poll/resubmit/startup caller. The gap remains detectable
                # via `find_legacy_succeeded_reuse_attempts_missing_evidence`
                # regardless.
                pass

    def _recover_legacy_reuse_dispatch_evidence_at_startup(self) -> None:
        """Restart-time backfill sweep (reuse success/evidence atomicity
        revision) mirroring `_recover_abandoned_reuse_attempts_at_startup`:
        proactively heals every already-``SUCCEEDED``, reuse-marked attempt
        missing its durable dispatch evidence using the store's bounded,
        indexed `find_legacy_succeeded_reuse_attempts_missing_evidence` query,
        so a fresh service startup alone -- with no client poll or
        resubmission at all -- is enough to resolve every such row a
        pre-fix service instance could have left behind."""
        for attempt_id in self._recipe_attempt_store.find_legacy_succeeded_reuse_attempts_missing_evidence():
            guard = self._acquire_attempt_sync_guard(attempt_id)
            try:
                with guard.lock:
                    self._backfill_legacy_reuse_dispatch_evidence(attempt_id)
            finally:
                self._release_attempt_sync_guard(attempt_id, guard)

    def _finalize_abandoned_reuse_attempt_failed(
        self,
        *,
        attempt_id: str,
        reuse_source_attempt_id: str,
        reason: str,
    ) -> Any:
        """Fail a candidate-selection-reuse attempt terminally with an
        explicit, sanitized reason (Slice 3B2b abandoned-recovery), instead
        of ever leaving it `RUNNING` forever. Never swallows/hides the
        failure as a success: this is the fail-closed counterpart of
        `_recover_abandoned_reuse_attempt`'s successful-completion path."""
        try:
            return self._recipe_attempt_store.finish_attempt_failed(
                attempt_id,
                failure=AttemptFailure(
                    classification=AttemptFailureClassification.INTERNAL_ERROR,
                    stage="candidate_selection_reuse_recovery",
                    message=_sanitize_attempt_failure_message(
                        "Abandoned candidate-selection-reuse attempt could not be safely recovered: "
                        f"{reason}."
                    ),
                    evidence_refs=(f"attempt://{reuse_source_attempt_id}",),
                    source_owner="fl-onboarding",
                    next_action="Resubmit the original recipe fingerprint with a new idempotency key to retry.",
                ),
            )
        except AttemptStateTransitionError:
            # Another concurrent caller (poll or resubmission) already moved this
            # attempt to a terminal state first; return whatever it now is
            # rather than raising -- never leave the caller without a result.
            return self._recipe_attempt_store.get_attempt(attempt_id)

    def _recover_abandoned_reuse_attempt(self, attempt_id: str) -> Any:
        """Bounded recovery (Slice 3B2b) for a marked `RUNNING`
        candidate-selection-reuse attempt whose client never resubmitted the
        original Idempotency-Key and only polls `get_recipe_attempt`, or a
        fresh service startup after a crash left it `RUNNING` with no
        in-memory guard/job mapping ever able to resume it (candidate-
        selection reuse never creates a `BuildJob`, so the ordinary
        `get_recipe_attempt` job-sync path can never reach it either).

        Reads the durable `reuse_source_attempt_id` marker directly and never
        trusts identity match alone: it revalidates that the source is still
        a genuinely registered, `SELECTED` candidate winner with a fully
        validated selection scope, that its own linked attempt actually
        reached `SUCCEEDED`, and that this attempt's own recorded gate
        history is a strict prefix of the source's gate history -- the same
        defense-in-depth check `_materialize_reused_generated_attempt`
        already performs for its own `RUNNING`-resume branch. If every check
        passes, copies only the remaining genuine source gate evidence,
        finishes this attempt as `SUCCEEDED`, and records measured-zero
        dispatch evidence -- never dispatching to the build stage runner. If
        the source is missing/corrupt/untrusted, fails this attempt
        terminally with an explicit, sanitized reason instead of ever
        leaving it `RUNNING` forever.

        Uses the exact same per-attempt guard as
        `_materialize_reused_generated_attempt`, keyed by `attempt_id`, so a
        concurrent resubmission of the same Idempotency-Key and a concurrent
        `get_recipe_attempt` poll can never race each other into double-
        finishing (or double-failing) this attempt.

        Returns the (possibly just-completed) attempt, or `None` if
        `attempt_id` does not exist, is not currently `RUNNING`, or has no
        reuse marker at all -- an ordinary non-reuse `RUNNING` attempt, which
        this must never auto-resume."""
        guard = self._acquire_attempt_sync_guard(attempt_id)
        try:
            with guard.lock:
                try:
                    attempt = self._recipe_attempt_store.get_attempt(attempt_id)
                except KeyError:
                    return None
                if attempt.state != AttemptState.RUNNING:
                    return attempt
                try:
                    reuse_source_attempt_id = self._recipe_attempt_store.get_attempt_reuse_source(attempt_id)
                except KeyError:
                    return None
                if reuse_source_attempt_id is None:
                    return None  # Ordinary non-reuse RUNNING attempt: never auto-resumed here.

                source_candidate = self._recipe_attempt_store.find_candidate_attempt_by_attempt_id(
                    reuse_source_attempt_id
                )
                if source_candidate is None:
                    return self._finalize_abandoned_reuse_attempt_failed(
                        attempt_id=attempt_id,
                        reuse_source_attempt_id=reuse_source_attempt_id,
                        reason=(
                            f"reuse source attempt '{reuse_source_attempt_id}' is not a registered "
                            "candidate winner"
                        ),
                    )
                if source_candidate.selection_status != CandidateWinnerStatus.SELECTED:
                    return self._finalize_abandoned_reuse_attempt_failed(
                        attempt_id=attempt_id,
                        reuse_source_attempt_id=reuse_source_attempt_id,
                        reason=(
                            f"reuse source candidate '{source_candidate.candidate_attempt_id}' is not "
                            "marked selected"
                        ),
                    )
                if not source_candidate.has_fully_validated_selection_scope:
                    return self._finalize_abandoned_reuse_attempt_failed(
                        attempt_id=attempt_id,
                        reuse_source_attempt_id=reuse_source_attempt_id,
                        reason=(
                            f"reuse source candidate '{source_candidate.candidate_attempt_id}' has an "
                            "incomplete validated selection scope"
                        ),
                    )
                try:
                    winner_attempt = self._recipe_attempt_store.get_attempt(reuse_source_attempt_id)
                except KeyError:
                    return self._finalize_abandoned_reuse_attempt_failed(
                        attempt_id=attempt_id,
                        reuse_source_attempt_id=reuse_source_attempt_id,
                        reason=f"reuse source attempt '{reuse_source_attempt_id}' no longer exists",
                    )
                if winner_attempt.state != AttemptState.SUCCEEDED:
                    return self._finalize_abandoned_reuse_attempt_failed(
                        attempt_id=attempt_id,
                        reuse_source_attempt_id=reuse_source_attempt_id,
                        reason=(
                            f"reuse source attempt '{reuse_source_attempt_id}' is "
                            f"'{winner_attempt.state.value}', expected 'succeeded'"
                        ),
                    )
                for index, recorded_gate in enumerate(attempt.gate_results):
                    if (
                        index >= len(winner_attempt.gate_results)
                        or recorded_gate.gate != winner_attempt.gate_results[index].gate
                    ):
                        return self._finalize_abandoned_reuse_attempt_failed(
                            attempt_id=attempt_id,
                            reuse_source_attempt_id=reuse_source_attempt_id,
                            reason=(
                                "abandoned reuse attempt gate history does not match the reuse "
                                "source attempt"
                            ),
                        )
                try:
                    for gate in winner_attempt.gate_results[len(attempt.gate_results) :]:
                        self._recipe_attempt_store.record_attempt_gate(
                            attempt_id=attempt_id,
                            gate=gate.gate,
                            status=gate.status,
                            evidence_ref=gate.evidence_ref,
                            metrics_ref=gate.metrics_ref,
                        )
                    return self._recipe_attempt_store.finish_reused_attempt_succeeded_with_dispatch_evidence(
                        attempt_id,
                        source_attempt_id=reuse_source_attempt_id,
                    )
                except RecipeAttemptStoreError as exc:
                    return self._finalize_abandoned_reuse_attempt_failed(
                        attempt_id=attempt_id,
                        reuse_source_attempt_id=reuse_source_attempt_id,
                        reason=f"failed to complete abandoned reuse attempt: {exc}",
                    )
        finally:
            self._release_attempt_sync_guard(attempt_id, guard)

    def _recover_abandoned_reuse_attempts_at_startup(self) -> None:
        """Restart-time recovery (Slice 3B2b) mirroring `get_recipe_attempt`'s
        lazy poll-triggered recovery: any attempt still `RUNNING` with a
        durable `reuse_source_attempt_id` marker and no build-job mapping
        (candidate-selection reuse never creates one) is proactively
        completed or failed terminally here, so a fresh service startup
        alone -- with no client poll or resubmission at all -- is enough to
        resolve it. Ordinary non-reuse `RUNNING` attempts are untouched here:
        the store's own `_recover_interrupted_attempts` already marks those
        `FAILED` at construction time."""
        for attempt in self._recipe_attempt_store.list_attempts():
            if attempt.state != AttemptState.RUNNING:
                continue
            with self._lock:
                already_mapped = attempt.attempt_id in self._attempt_to_build_job
            if already_mapped:
                continue
            try:
                reuse_source = self._recipe_attempt_store.get_attempt_reuse_source(attempt.attempt_id)
            except KeyError:
                continue
            if reuse_source is None:
                continue
            self._recover_abandoned_reuse_attempt(attempt.attempt_id)

    def _resolve_generated_recipe_preview(
        self,
        *,
        metadata: HuggingFaceMetadata,
        task: CandidateModality,
        requested_precision: str = _RECIPE_DEFAULT_PRECISION,
        catalog_matches: tuple[object, ...] = (),
    ) -> GeneratedRecipePreviewContext:
        normalized_metadata = normalize_huggingface_metadata(
            model_id=metadata.model_id,
            config=metadata.config,
            is_gated=metadata.is_gated,
            is_private=metadata.is_private,
        )
        capability_resolution = self._capability_registry.resolve(
            metadata=normalized_metadata,
            task=task.value,
            device=_RECIPE_DEFAULT_DEVICE,
            requested_precision=requested_precision,
        )
        capability_payload = self._capability_resolution_payload(resolution=capability_resolution)
        files, config_files, tokenizer_files = self._extract_model_files(metadata)

        revision = (metadata.sha or metadata.revision or "").strip().lower()
        compile_error: str | None = None
        generated_recipe: GeneratedRecipe | None = None
        verified_reuse: dict[str, object] | None = None
        candidate_selection_reuse: dict[str, object] | None = None
        candidate_plan: dict[str, object] | None = None
        eligible = False

        if task != CandidateModality.LLM:
            compile_error = "Automatic generated recipe attempts are restricted to LLM/text-generation tasks."
        elif catalog_matches:
            compile_error = (
                "Foundry catalog already has a matching model entry. Automatic generated recipe attempts are blocked."
            )
        elif not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            compile_error = (
                "Model metadata did not include a full pinned 40-character revision SHA required for deterministic recipes."
            )
        else:
            try:
                generated_recipe = compile_generated_recipe(
                    RecipeCompilerInput(
                        model_id=metadata.model_id,
                        revision_sha=revision,
                        model_type=normalized_metadata.model_type,
                        architectures=tuple(normalized_metadata.architecture_aliases),
                        task=task.value,
                        requested_device=_RECIPE_DEFAULT_DEVICE,
                        requested_precision=requested_precision,
                        is_gated=bool(metadata.is_gated),
                        requires_remote_code=bool(normalized_metadata.requires_remote_code),
                        config_files=config_files,
                        tokenizer_files=tokenizer_files,
                        available_files=files,
                        capability_resolution=capability_resolution,
                        toolchain=_RECIPE_TOOLCHAIN,
                    )
                )
            except GeneratedRecipeCompileError as exc:
                compile_error = _sanitize_text(str(exc) or "Generated recipe compilation failed.")
            if generated_recipe is not None:
                generated_record = self._recipe_attempt_store.upsert_generated_recipe(generated_recipe)
                if self._generated_record_is_cpu_int4_eligible(generated_record):
                    candidate_plan = self._candidate_plan_payload(self._recipe_selection_policy)
                reusable_candidate_selection = self._resolve_reusable_candidate_selection(record=generated_record)
                if reusable_candidate_selection is not None:
                    candidate_selection_reuse = self._candidate_selection_reuse_payload(
                        reusable_candidate_selection
                    )
                reusable = self._recipe_attempt_store.find_reusable_verified_recipe(
                    build_reuse_query_from_generated(generated_record)
                )
                if reusable is not None:
                    verified_reuse = self._verified_reuse_payload(reusable)
                elif reusable_candidate_selection is None:
                    eligible = True

        return GeneratedRecipePreviewContext(
            model_id=metadata.model_id,
            revision=revision,
            task=task,
            requested_device=_RECIPE_DEFAULT_DEVICE,
            requested_precision=requested_precision,
            generated_recipe=generated_recipe,
            compile_error=compile_error,
            capability=capability_payload,
            files=files,
            config_files=config_files,
            tokenizer_files=tokenizer_files,
            catalog_matches=catalog_matches,
            eligible_for_automatic_attempt=eligible,
            verified_reuse=verified_reuse,
            candidate_selection_reuse=candidate_selection_reuse,
            candidate_plan=candidate_plan,
        )

    def _generated_recipe_payload(self, context: GeneratedRecipePreviewContext) -> dict[str, object]:
        candidate = context.generated_recipe
        validation_gates = [gate.value for gate in _RECIPE_ATTEMPT_GATE_SEQUENCE]
        if candidate is None:
            return {
                "eligible_for_automatic_recipe_attempt": False,
                "requires_explicit_attempt_confirmation": True,
                "experimental_until_verified": True,
                "compile_error": context.compile_error,
                "capability": context.capability,
                "fingerprint": None,
                "recipe": None,
                "argument_confidence": None,
                "toolchain": to_jsonable(_RECIPE_TOOLCHAIN),
                "validation_gates": validation_gates,
                "verified_reuse": context.verified_reuse,
                "candidate_selection_reuse": context.candidate_selection_reuse,
                "candidate_plan": context.candidate_plan,
            }
        provenance = candidate.provenance
        return {
            "eligible_for_automatic_recipe_attempt": context.eligible_for_automatic_attempt,
            "requires_explicit_attempt_confirmation": True,
            "experimental_until_verified": True,
            "compile_error": context.compile_error,
            "capability": context.capability,
            "fingerprint": candidate.fingerprint,
            "recipe": candidate.payload(),
            "argument_confidence": {
                "mobius_dtype_confidence": provenance.argument_provenance.mobius_dtype_confidence.value,
                "olive_precision_confidence": provenance.argument_provenance.olive_precision_confidence.value,
                "contains_unverified_arguments": provenance.argument_provenance.contains_unverified_arguments,
            },
            "toolchain": to_jsonable(provenance.toolchain),
            "validation_gates": validation_gates,
            "verified_reuse": context.verified_reuse,
            "candidate_selection_reuse": context.candidate_selection_reuse,
            "candidate_plan": context.candidate_plan,
        }

    def model_detail(self, *, model_id: str) -> dict[str, object]:
        normalized = model_id.strip()
        if not normalized:
            raise ServiceError(
                code="INVALID_MODEL_ID",
                message="Model id must not be empty.",
                status_code=400,
            )
        metadata = self._load_model_metadata(normalized)
        requires_remote_code = config_requires_remote_code(metadata.config)
        blockers: list[str] = []
        if metadata.is_gated:
            blockers.append("gated_model_blocked")
        if requires_remote_code:
            blockers.append("remote_code_blocked")

        matches, catalog_status, warnings = self._probe_foundry_catalog(normalized)

        task_hints = _derive_task_hints(model_id=normalized, config=metadata.config)
        hinted_task = task_hints[0] if task_hints else CandidateModality.LLM.value
        hinted_modality = CandidateModality(hinted_task)
        generated_preview = self._resolve_generated_recipe_preview(
            metadata=metadata,
            task=hinted_modality,
            catalog_matches=matches,
        )
        generated_payload = self._generated_recipe_payload(generated_preview)
        recipe_match = self._recipe_registry.resolve(
            model_id=metadata.model_id,
            modality=hinted_modality,
            task_profile="default",
            allow_experimental=False,
        )
        generated_buildable = bool(
            generated_payload["eligible_for_automatic_recipe_attempt"] or generated_payload["verified_reuse"]
        )
        if not recipe_match.buildable and not generated_buildable:
            blockers.append(_recipe_blocker_code(recipe_match))
        if generated_preview.compile_error and not generated_buildable:
            blockers.append("generated_recipe_unavailable")
        buildable = (
            not bool(metadata.is_gated)
            and not requires_remote_code
            and (recipe_match.buildable or generated_buildable)
        )
        recipe = recipe_match.recipe
        recipe_payload = self._recipe_payload(recipe_match)
        supported_optimizations = (
            recipe_payload.get("supported_optimizations", []) if recipe_payload else []
        )
        buildable_with_experimental_opt_in = (
            recipe is not None
            and recipe.status == RecipeStatus.EXPERIMENTAL
            and not bool(metadata.is_gated)
            and not requires_remote_code
        )
        return {
            "model_id": metadata.model_id,
            "revision": metadata.revision,
            "sha": metadata.sha,
            "private": metadata.is_private,
            "gated": metadata.is_gated,
            "requires_remote_code": requires_remote_code,
            "config": metadata.config,
            "model_type": (
                str(metadata.config.get("model_type"))
                if isinstance(metadata.config, dict) and isinstance(metadata.config.get("model_type"), str)
                else None
            ),
            "task": hinted_modality.value,
            "buildable": buildable,
            "build_blockers": blockers,
            "task_hints": task_hints,
            "last_modified": metadata.last_modified,
            "safetensors_total_bytes": metadata.safetensors_total_bytes,
            "safetensors_parameter_count": metadata.safetensors_parameter_count,
            "card_data": metadata.card_data,
            "files": list(generated_preview.files),
            "config_files": list(generated_preview.config_files),
            "tokenizer_files": list(generated_preview.tokenizer_files),
            "foundry_catalog_status": catalog_status,
            "foundry_catalog_matches": [to_jsonable(item) for item in matches],
            "warnings": warnings,
            "verification": self._verification_for_model(metadata.model_id),
            "candidate_outcome": (
                _asr_candidate_outcome() if metadata.model_id == _ASR_MODEL_ID else None
            ),
            "recipe": recipe_payload,
            "recipe_status": recipe_match.status,
            "recipe_reason": recipe_match.reason,
            "requires_experimental_opt_in": recipe_match.requires_experimental_opt_in,
            "buildable_with_experimental_opt_in": buildable_with_experimental_opt_in,
            "supported_optimizations": supported_optimizations,
            "generated_recipe": generated_payload,
        }

    def preflight(self, submission: BuildSubmission) -> dict[str, object]:
        normalized = submission.normalized()
        metadata = self._load_model_metadata(normalized.model_id)
        catalog_matches, catalog_status, warnings = self._probe_foundry_catalog(metadata.model_id)
        generated_preview = self._resolve_generated_recipe_preview(
            metadata=metadata,
            task=normalized.task,
            catalog_matches=catalog_matches,
        )
        generated_payload = self._generated_recipe_payload(generated_preview)
        recipe_match = self._recipe_registry.resolve(
            model_id=normalized.model_id,
            modality=normalized.task,
            task_profile=normalized.task_profile,
            allow_experimental=normalized.allow_experimental,
        )
        request = self._build_request(
            normalized,
            job_id="_preflight",
            recipe_match=recipe_match,
            enforce_recipe_buildable=False,
        )
        recipe_payload = self._recipe_payload(recipe_match)
        supported_optimizations = (
            recipe_payload.get("supported_optimizations", []) if recipe_payload else []
        )
        candidate_outcome = (
            _asr_candidate_outcome()
            if normalized.task == CandidateModality.ASR and normalized.model_id == _ASR_MODEL_ID
            else None
        )
        if not recipe_match.buildable:
            blockers = [
                failure(
                    stage=JobState.PREFLIGHT,
                    classification=FailureClassification.INVALID_REQUEST,
                    message=recipe_match.reason,
                )
            ]
            if candidate_outcome is not None:
                blockers.insert(
                    0,
                    FailureInfo(
                        stage=JobState(str(candidate_outcome["failed_stage"])),
                        classification=FailureClassification(str(candidate_outcome["classification"])),
                        message=str(candidate_outcome["error_summary"]),
                        detail=dict(_ASR_RUNTIME_BLOCKER_DETAIL),
                    ),
                )
            blocked = PreflightResult(
                candidate=request.candidate,
                workspace_root=request.workspace_root,
                model_cache_dir=request.model_cache_dir,
                output_dir=request.output_dir,
                disk_free_gb_workspace=0.0,
                disk_free_gb_cache=0.0,
                tools=(),
                foundry_catalog_matches=catalog_matches,
                huggingface_revision=metadata.revision or request.hf_revision,
                huggingface_sha=metadata.sha,
                huggingface_private=metadata.is_private,
                huggingface_gated=metadata.is_gated,
                cache_key=None,
                blockers=tuple(blockers),
                warnings=tuple(warnings),
            )
            cache_key = _sha256_json(
                {
                    **normalized.cache_identity(),
                    "recipe_status": recipe_match.status,
                    "recipe_reason": recipe_match.reason,
                    "generated_recipe_fingerprint": generated_payload.get("fingerprint"),
                }
            )
            return {
                "cache_key": cache_key,
                "ok": False,
                "cached": False,
                "result": to_jsonable(blocked),
                "candidate_outcome": candidate_outcome,
                "recipe": recipe_payload,
                "recipe_status": recipe_match.status,
                "recipe_reason": recipe_match.reason,
                "requires_experimental_opt_in": recipe_match.requires_experimental_opt_in,
                "supported_optimizations": supported_optimizations,
                "foundry_catalog_status": catalog_status,
                "foundry_catalog_matches": [to_jsonable(item) for item in catalog_matches],
                "warnings": warnings,
                "model_detail": {
                    "model_id": metadata.model_id,
                    "revision": metadata.revision,
                    "sha": metadata.sha,
                    "private": metadata.is_private,
                    "gated": metadata.is_gated,
                    "requires_remote_code": config_requires_remote_code(metadata.config),
                    "config": metadata.config,
                    "files": list(generated_preview.files),
                    "config_files": list(generated_preview.config_files),
                    "tokenizer_files": list(generated_preview.tokenizer_files),
                },
                "generated_recipe": generated_payload,
            }

        output, cached, cache_key = self._inspect_preflight_with_cache_state(
            request,
            fallback_cache_payload=normalized.cache_identity(),
        )
        payload = to_jsonable(output)
        payload["cache_key"] = cache_key
        supported = output.ok
        if (
            normalized.task == CandidateModality.ASR
            and normalized.model_id == _ASR_MODEL_ID
        ):
            payload["blockers"] = [
                {
                    "stage": candidate_outcome["failed_stage"],
                    "classification": candidate_outcome["classification"],
                    "message": candidate_outcome["error_summary"],
                    "detail": dict(_ASR_RUNTIME_BLOCKER_DETAIL),
                },
                *payload.get("blockers", []),
            ]
            supported = False
        return {
            "cache_key": cache_key,
            "ok": supported,
            "cached": cached,
            "result": payload,
            "candidate_outcome": candidate_outcome,
            "recipe": recipe_payload,
            "recipe_status": recipe_match.status,
            "recipe_reason": recipe_match.reason,
            "requires_experimental_opt_in": recipe_match.requires_experimental_opt_in,
            "supported_optimizations": supported_optimizations,
            "foundry_catalog_status": catalog_status,
            "foundry_catalog_matches": [to_jsonable(item) for item in catalog_matches],
            "warnings": warnings,
            "model_detail": {
                "model_id": metadata.model_id,
                "revision": metadata.revision,
                "sha": metadata.sha,
                "private": metadata.is_private,
                "gated": metadata.is_gated,
                "requires_remote_code": config_requires_remote_code(metadata.config),
                "config": metadata.config,
                "files": list(generated_preview.files),
                "config_files": list(generated_preview.config_files),
                "tokenizer_files": list(generated_preview.tokenizer_files),
            },
            "generated_recipe": generated_payload,
        }

    def generated_recipe_preview(
        self,
        *,
        model_id: str,
        task: CandidateModality = CandidateModality.LLM,
    ) -> dict[str, object]:
        normalized = model_id.strip()
        if not normalized:
            raise ServiceError(
                code="INVALID_MODEL_ID",
                message="Model id must not be empty.",
                status_code=400,
            )
        metadata = self._load_model_metadata(normalized)
        matches, catalog_status, warnings = self._probe_foundry_catalog(metadata.model_id)
        preview = self._resolve_generated_recipe_preview(
            metadata=metadata,
            task=task,
            catalog_matches=matches,
        )
        return {
            "model_id": metadata.model_id,
            "revision": metadata.revision,
            "sha": metadata.sha,
            "gated": metadata.is_gated,
            "private": metadata.is_private,
            "requires_remote_code": config_requires_remote_code(metadata.config),
            "task": task.value,
            "config": metadata.config,
            "model_type": (
                str(metadata.config.get("model_type"))
                if isinstance(metadata.config, dict) and isinstance(metadata.config.get("model_type"), str)
                else None
            ),
            "files": list(preview.files),
            "config_files": list(preview.config_files),
            "tokenizer_files": list(preview.tokenizer_files),
            "foundry_catalog_status": catalog_status,
            "foundry_catalog_matches": [to_jsonable(item) for item in matches],
            "warnings": warnings,
            "generated_recipe": self._generated_recipe_payload(preview),
        }

    def _build_request_for_generated_attempt(
        self,
        *,
        record: Any,
        attempt_id: str,
        job_id: str,
    ) -> BuildRequest:
        payload = record.payload()
        recipe_payload = payload.get("recipe")
        if not isinstance(recipe_payload, dict):
            raise ServiceError(
                code="GENERATED_RECIPE_INVALID",
                message="Generated recipe payload is missing a recipe object.",
                status_code=500,
            )
        modality_raw = recipe_payload.get("modality")
        try:
            modality = CandidateModality(str(modality_raw))
        except ValueError as exc:
            raise ServiceError(
                code="GENERATED_RECIPE_INVALID",
                message=f"Generated recipe has unsupported modality '{modality_raw}'.",
                status_code=500,
            ) from exc
        if modality != CandidateModality.LLM:
            raise ServiceError(
                code="GENERATED_RECIPE_TASK_UNSUPPORTED",
                message="Automatic generated recipe attempts currently support LLM/text-generation only.",
                status_code=400,
            )
        workspace = workspace_root_for_job(job_id=job_id, base_dir=self._workspace_base)
        ensure_dir(workspace)
        output_dir = ensure_dir(workspace / "output")
        optimization_choices_raw = recipe_payload.get("optimization_choices")
        optimization_choices = (
            [row for row in optimization_choices_raw if isinstance(row, dict)]
            if isinstance(optimization_choices_raw, list)
            else []
        )
        selected_choice = next(
            (row for row in optimization_choices if row.get("default") is True),
            optimization_choices[0] if optimization_choices else None,
        )
        olive_payload = recipe_payload.get("olive")
        olive_dict = olive_payload if isinstance(olive_payload, dict) else None
        skip_olive = (
            bool(selected_choice.get("skip_olive"))
            if isinstance(selected_choice, dict) and "skip_olive" in selected_choice
            else olive_dict is None
        )
        mobius_payload = recipe_payload.get("mobius")
        mobius_dtype = (
            str(mobius_payload.get("dtype"))
            if isinstance(mobius_payload, dict) and isinstance(mobius_payload.get("dtype"), str)
            else None
        )
        candidate = ModelCandidate(
            key=_normalize_model_key(record.model_id, modality),
            huggingface_model_id=record.model_id,
            modality=modality,
            recommended_mobius_dtype=mobius_dtype,
            recommended_olive_precision=(
                str(olive_dict.get("precision"))
                if olive_dict is not None and isinstance(olive_dict.get("precision"), str)
                else None
            ),
            notes=str(recipe_payload.get("status_reason") or "Deterministic generated recipe candidate."),
        )
        return BuildRequest(
            candidate=candidate,
            workspace_root=workspace,
            model_cache_dir=self._model_cache_dir,
            output_dir=output_dir,
            task_profile=str(recipe_payload.get("task_profile") or "llm-cpu-default"),
            hf_revision=record.revision_sha,
            skip_olive=skip_olive,
            dry_run=False,
            recipe_id=str(recipe_payload.get("id") or ""),
            recipe_version=str(recipe_payload.get("version") or ""),
            recipe_status=str(recipe_payload.get("status") or RecipeStatus.EXPERIMENTAL.value),
            recipe_reason=(
                "Generated recipe attempt execution confirmed via explicit "
                "confirm_automatic_recipe_attempt signal."
            ),
            generated_recipe_attempt=GeneratedRecipeAttemptBinding(
                attempt_id=attempt_id,
                recipe_fingerprint=record.recipe_fingerprint,
                confirmed=True,
                confirmation_provenance=_AUTOMATIC_RECIPE_ATTEMPT_CONFIRMATION_PROVENANCE,
            ),
            recipe_artifact_cache_prefix=(
                str(recipe_payload.get("artifact_cache_prefix"))
                if isinstance(recipe_payload.get("artifact_cache_prefix"), str)
                else None
            ),
            recipe_model_name_prefix=(
                str(recipe_payload.get("model_name_prefix"))
                if isinstance(recipe_payload.get("model_name_prefix"), str)
                else None
            ),
            allow_experimental=True,
            optimization_strategy=(
                str(selected_choice.get("strategy"))
                if isinstance(selected_choice, dict) and isinstance(selected_choice.get("strategy"), str)
                else None
            ),
            optimization_precision=(
                str(selected_choice.get("precision"))
                if isinstance(selected_choice, dict) and isinstance(selected_choice.get("precision"), str)
                else None
            ),
        )

    def _serialize_recipe_attempt(self, attempt: Any) -> dict[str, object]:
        with self._lock:
            job_id = self._attempt_to_build_job.get(attempt.attempt_id)
            workspace_root = (
                self._jobs[job_id].request.workspace_root
                if job_id is not None and job_id in self._jobs
                else None
            )
        quality_validation = self._quality_validation_summary_for_attempt(
            attempt=attempt,
            job_id=job_id,
            workspace_root=workspace_root,
        )
        workflow_outcome, candidate_selection = self._candidate_selection_summary_for_attempt(attempt)
        return {
            "attempt_id": attempt.attempt_id,
            "idempotency_key": attempt.idempotency_key,
            "request_fingerprint": attempt.request_fingerprint,
            "recipe_fingerprint": attempt.recipe_fingerprint,
            "model_id": attempt.model_id,
            "revision_sha": attempt.revision_sha,
            "requested_device": attempt.requested_device,
            "requested_precision": attempt.requested_precision,
            "compiler_version": attempt.compiler_version,
            "capability_fingerprint": attempt.capability_fingerprint,
            "toolchain_fingerprint": attempt.toolchain_fingerprint,
            "profile_fingerprint": attempt.profile_fingerprint,
            "state": attempt.state.value,
            "created_utc": attempt.created_utc.isoformat(),
            "finished_utc": attempt.finished_utc.isoformat() if attempt.finished_utc is not None else None,
            "build_job_id": job_id,
            "gates": [
                {
                    "sequence": gate.sequence,
                    "gate": gate.gate.value,
                    "status": gate.status.value,
                    "evidence_ref": gate.evidence_ref,
                    "metrics_ref": gate.metrics_ref,
                    "started_utc": gate.started_utc.isoformat(),
                    "finished_utc": gate.finished_utc.isoformat(),
                }
                for gate in attempt.gate_results
            ],
            "failure": (
                {
                    "classification": attempt.failure.classification.value,
                    "stage": attempt.failure.stage,
                    "message": attempt.failure.message,
                    "evidence_refs": list(attempt.failure.evidence_refs),
                    "source_owner": attempt.failure.source_owner,
                    "next_action": attempt.failure.next_action,
                }
                if attempt.failure is not None
                else None
            ),
            "quality_validation": quality_validation,
            "workflow_outcome": workflow_outcome,
            "candidate_selection": candidate_selection,
        }

    def _candidate_selection_summary_for_attempt(
        self,
        attempt: Any,
    ) -> tuple[str, dict[str, object] | None]:
        """Additive, backward-compatible candidate-selection/reuse summary
        for one recipe attempt (Slice 3C1).

        Returns ``(workflow_outcome, candidate_selection)`` where
        ``workflow_outcome`` is always one of the five stable machine-
        readable codes ``not_applicable`` / ``pending`` / ``selected`` /
        ``exhausted`` / ``reused``, and ``candidate_selection`` is ``None``
        exactly when ``workflow_outcome == "not_applicable"`` -- i.e. this
        attempt is neither part of any registered candidate-selection
        lineage (legacy/static/non-CPU-INT4-eligible generated attempts) nor
        itself a candidate-selection-reuse materialization. Every existing
        legacy attempt payload therefore still validates: the two new keys
        are purely additive, and the richer ``candidate_selection`` object
        is only ever populated once real candidate-plan/timeline/reuse
        evidence exists for this exact attempt.

        Never rewrites or relabels any candidate's own terminal
        ``attempt_state``: a failed default candidate is always reported as
        ``failed`` here even when ``workflow_outcome`` is ``selected``
        because a *different* (fallback) candidate was the one actually
        verified and selected.
        """
        candidate = self._recipe_attempt_store.find_candidate_attempt_by_attempt_id(attempt.attempt_id)
        if candidate is None:
            reuse_evidence = self._recipe_attempt_store.get_reuse_dispatch_evidence(attempt.attempt_id)
            if reuse_evidence is None:
                return "not_applicable", None
            return "reused", {
                "policy_id": reuse_evidence.policy_id,
                "policy_version": reuse_evidence.policy_version,
                "policy_fingerprint": reuse_evidence.policy_fingerprint,
                "max_candidates": None,
                "lineage_selection_state": None,
                "selected_candidate": None,
                "candidates": [],
                "aggregate_invocation_counters": None,
                "reuse": self._candidate_reuse_evidence_payload(reuse_evidence),
            }

        try:
            lineage = self._recipe_attempt_store.get_candidate_lineage(candidate.parent_attempt_id)
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc
        if lineage is None:
            # A registered candidate row must always belong to a lineage row
            # (the store never allows one to exist without the other); this
            # can only mean on-disk corruption. Never present that silently
            # as a normal "not_applicable" summary.
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(
                    f"Candidate '{candidate.candidate_attempt_id}' has no parent lineage "
                    f"'{candidate.parent_attempt_id}'; candidate-selection store state is corrupt."
                ),
                status_code=500,
            )

        try:
            candidates = self._recipe_attempt_store.list_candidate_attempts(candidate.parent_attempt_id)
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc
        selected_candidate_payload: dict[str, object] | None = None
        if (
            lineage.selection_state == CandidateLineageSelectionState.SELECTED
            and lineage.selected_candidate_attempt_id is not None
        ):
            winner = next(
                (row for row in candidates if row.candidate_attempt_id == lineage.selected_candidate_attempt_id),
                None,
            )
            if winner is None:
                raise ServiceError(
                    code="RECIPE_ATTEMPT_STORE_ERROR",
                    message=_sanitize_text(
                        f"Lineage '{lineage.parent_attempt_id}' selected candidate "
                        f"'{lineage.selected_candidate_attempt_id}' is not among its registered candidates."
                    ),
                    status_code=500,
                )
            selected_candidate_payload = {
                "candidate_attempt_id": winner.candidate_attempt_id,
                "attempt_id": winner.attempt_id,
                "candidate_index": winner.candidate_index,
                "candidate_id": winner.candidate_id,
                "selected_by": winner.selected_by,
                "selection_reason": winner.selection_reason,
                "selected_utc": winner.selected_utc.isoformat() if winner.selected_utc is not None else None,
            }

        candidate_selection = {
            "policy_id": lineage.policy_id,
            "policy_version": lineage.policy_version,
            "policy_fingerprint": lineage.policy_fingerprint,
            "max_candidates": lineage.policy_max_candidates,
            "lineage_selection_state": lineage.selection_state.value,
            "selected_candidate": selected_candidate_payload,
            "candidates": [self._candidate_timeline_entry_payload(row) for row in candidates],
            "aggregate_invocation_counters": self._aggregate_candidate_invocation_counters(candidates),
            "reuse": None,
        }
        return lineage.selection_state.value, candidate_selection

    @classmethod
    def _candidate_timeline_entry_payload(cls, candidate: CandidateAttemptRecord) -> dict[str, object]:
        """Per-candidate timeline entry: this never rewrites or infers the
        candidate's own terminal lifecycle state -- ``attempt_state`` is
        always the linked attempt's real, persisted state (e.g. ``failed``
        for a default candidate that regressed, even once a fallback
        candidate is later verified and selected)."""
        counters = candidate.invocation_counters
        return {
            "candidate_attempt_id": candidate.candidate_attempt_id,
            "attempt_id": candidate.attempt_id,
            "candidate_index": candidate.candidate_index,
            "candidate_id": candidate.candidate_id,
            "role": cls._candidate_role(candidate.candidate_index),
            "attempt_state": candidate.attempt_state.value,
            "recipe_fingerprint": candidate.recipe_fingerprint,
            "quantization_override": (
                {"block_size": candidate.quantization_override_block_size}
                if candidate.quantization_override_block_size is not None
                else None
            ),
            "eligibility_trigger": candidate.eligibility_trigger,
            "disposition": candidate.disposition,
            "disposition_reasons": list(candidate.disposition_reasons),
            "selection_status": candidate.selection_status.value,
            "artifact_ref": candidate.artifact_ref,
            "package_ref": candidate.package_ref,
            "invocation_counters": {
                "mobius_build_invocation_count": counters.mobius_build_invocation_count,
                "olive_optimize_invocation_count": counters.olive_optimize_invocation_count,
                "total_invocation_count": counters.total_invocation_count,
                "wall_clock_seconds": counters.wall_clock_seconds,
                "estimated_cost_usd": counters.estimated_cost_usd,
            },
            "validated_scope": {
                "target_device": candidate.validated_target_device,
                "target_ep": candidate.validated_target_ep,
                "toolchain_fingerprint": candidate.validated_toolchain_fingerprint,
                "environment_scope": candidate.validated_environment_scope,
            },
        }

    @staticmethod
    def _aggregate_candidate_invocation_counters(
        candidates: tuple[CandidateAttemptRecord, ...],
    ) -> dict[str, object]:
        """Sum real, persisted per-candidate invocation evidence across every
        registered candidate in a lineage. Never a hardcoded/inferred
        constant: each field is derived purely from whatever
        `CandidateInvocationCounters` values are already durably recorded.
        A field stays ``None`` ("unknown") whenever *no* candidate in the
        lineage has recorded that metric yet; it is never coerced to ``0``.
        Once at least one candidate has recorded a real value, only the
        known values are summed (an as-yet-unmeasured sibling candidate is
        excluded from the sum rather than either dropping the whole
        aggregate to null or silently treating it as zero).
        """

        def _sum_known(values: list[float | int | None]) -> float | int | None:
            known = [value for value in values if value is not None]
            if not known:
                return None
            return sum(known)

        return {
            "mobius_build_invocation_count": _sum_known(
                [row.invocation_counters.mobius_build_invocation_count for row in candidates]
            ),
            "olive_optimize_invocation_count": _sum_known(
                [row.invocation_counters.olive_optimize_invocation_count for row in candidates]
            ),
            "total_invocation_count": _sum_known(
                [row.invocation_counters.total_invocation_count for row in candidates]
            ),
            "wall_clock_seconds": _sum_known(
                [row.invocation_counters.wall_clock_seconds for row in candidates]
            ),
            "estimated_cost_usd": _sum_known(
                [row.invocation_counters.estimated_cost_usd for row in candidates]
            ),
        }

    @staticmethod
    def _candidate_reuse_evidence_payload(reuse_evidence: Any) -> dict[str, object]:
        """Durable, measured-zero candidate-selection-reuse dispatch
        evidence for an attempt that was returned as an alias of a
        previously selected/verified winner's own build/artifact -- never a
        new build. ``reused_without_build`` is always ``True`` and every
        invocation count is always the real, persisted ``0`` recorded at
        dispatch time (never inferred or synthesized here)."""
        return {
            "reused_without_build": reuse_evidence.reused_without_build,
            "source_attempt_id": reuse_evidence.source_attempt_id,
            "source_candidate_attempt_id": reuse_evidence.source_candidate_attempt_id,
            "source_parent_attempt_id": reuse_evidence.parent_attempt_id,
            "policy_id": reuse_evidence.policy_id,
            "policy_version": reuse_evidence.policy_version,
            "policy_fingerprint": reuse_evidence.policy_fingerprint,
            "quality_profile_fingerprint": reuse_evidence.quality_profile_fingerprint,
            "runner_dispatch_count": reuse_evidence.runner_dispatch_count,
            "mobius_invocation_count": reuse_evidence.mobius_invocation_count,
            "olive_invocation_count": reuse_evidence.olive_invocation_count,
            "recorded_utc": reuse_evidence.recorded_utc.isoformat(),
        }

    @staticmethod
    def _recipe_integrity_status_from_gate_status(status: AttemptGateStatus) -> str:
        if status == AttemptGateStatus.PASSED:
            return "verified"
        if status == AttemptGateStatus.FAILED:
            return "blocked"
        return "inconclusive"

    def _quality_validation_summary_for_attempt(
        self,
        *,
        attempt: Any,
        job_id: str | None,
        workspace_root: Path | None,
    ) -> dict[str, object] | None:
        quality_gate = next(
            (row for row in attempt.gate_results if row.gate == AttemptGate.QUALITY_VALIDATION),
            None,
        )
        if quality_gate is None:
            return None
        fallback_integrity_status = self._recipe_integrity_status_from_gate_status(
            quality_gate.status
        )
        summary: dict[str, object] = {
            "recipe_integrity": {"status": fallback_integrity_status},
            "model_capability": None,
        }
        metrics_ref = quality_gate.metrics_ref
        if metrics_ref is None:
            return summary
        payload = self._load_quality_validation_evidence_payload(
            metrics_ref=metrics_ref,
            expected_job_id=job_id,
            workspace_root=workspace_root,
        )
        if payload is None:
            return summary
        summary["recipe_integrity"] = self._recipe_integrity_summary_from_payload(
            payload=payload,
            fallback_status=fallback_integrity_status,
        )
        summary["model_capability"] = self._model_capability_summary_from_payload(payload)
        return summary

    def _load_quality_validation_evidence_payload(
        self,
        *,
        metrics_ref: str,
        expected_job_id: str | None,
        workspace_root: Path | None,
    ) -> dict[str, object] | None:
        match = _QUALITY_METRICS_REF_RE.fullmatch(metrics_ref.strip())
        if match is None:
            return None
        ref_job_id = match.group("job_id")
        if expected_job_id is not None and ref_job_id != expected_job_id:
            return None
        filename = match.group("filename")
        if filename != _QUALITY_EVIDENCE_FILENAME:
            return None
        target_workspace = (
            workspace_root
            if workspace_root is not None
            else workspace_root_for_job(ref_job_id, base_dir=self._workspace_base)
        )
        evidence_path = target_workspace / filename
        if not evidence_path.is_file():
            return None
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _recipe_integrity_summary_from_payload(
        *,
        payload: dict[str, object],
        fallback_status: str,
    ) -> dict[str, object]:
        summary: dict[str, object] = {"status": fallback_status}
        recipe_raw = payload.get("recipe_verification")
        if not isinstance(recipe_raw, dict):
            return summary
        status_raw = recipe_raw.get("status")
        if isinstance(status_raw, str):
            normalized = status_raw.strip().lower()
            if normalized in {"verified", "blocked", "inconclusive"}:
                summary["status"] = normalized
        gate_status = recipe_raw.get("gate_status")
        if isinstance(gate_status, str):
            summary["gate_status"] = gate_status.strip().lower()
        for field in (
            "runtime_functional",
            "baseline_available",
            "regression_free",
            "can_promote",
        ):
            value = recipe_raw.get(field)
            if isinstance(value, bool):
                summary[field] = value
        failures = recipe_raw.get("integrity_failures")
        if isinstance(failures, list):
            summary["integrity_failures"] = [
                str(entry) for entry in failures if isinstance(entry, str)
            ]
        return summary

    @staticmethod
    def _model_capability_summary_from_payload(
        payload: dict[str, object],
    ) -> dict[str, object] | None:
        model_raw = payload.get("model_capability")
        if isinstance(model_raw, dict):
            checks_passed_raw = model_raw.get("checks_passed")
            total_checks_raw = model_raw.get("total_checks")
            if (
                type(checks_passed_raw) is int
                and checks_passed_raw >= 0
                and type(total_checks_raw) is int
                and total_checks_raw >= 0
            ):
                summary: dict[str, object] = {
                    "checks_passed": checks_passed_raw,
                    "total_checks": total_checks_raw,
                    "warnings": [
                        str(entry)
                        for entry in model_raw.get("warnings", [])
                        if isinstance(entry, str)
                    ]
                    if isinstance(model_raw.get("warnings"), list)
                    else [],
                }
                confidence_raw = model_raw.get("confidence")
                if isinstance(confidence_raw, dict):
                    confidence: dict[str, object] = {}
                    level = confidence_raw.get("level")
                    if isinstance(level, str):
                        normalized = level.strip().lower()
                        if normalized in {"high", "low"}:
                            confidence["level"] = normalized
                    determinism_supported = confidence_raw.get("determinism_supported")
                    if isinstance(determinism_supported, bool):
                        confidence["determinism_supported"] = determinism_supported
                    reasons_raw = confidence_raw.get("reasons")
                    if isinstance(reasons_raw, list):
                        confidence["reasons"] = [
                            str(entry) for entry in reasons_raw if isinstance(entry, str)
                        ]
                    if confidence:
                        summary["confidence"] = confidence
                return summary

        prompt_rows = payload.get("per_prompt")
        if not isinstance(prompt_rows, list):
            return None
        checks_passed = 0
        total_checks = 0
        for row in prompt_rows:
            if not isinstance(row, dict):
                continue
            optimized = row.get("optimized")
            if not isinstance(optimized, dict):
                continue
            checks = optimized.get("checks")
            if not isinstance(checks, dict):
                continue
            passed = checks.get("passed")
            if not isinstance(passed, bool):
                continue
            total_checks += 1
            if passed:
                checks_passed += 1
        if total_checks == 0:
            return None
        return {
            "checks_passed": checks_passed,
            "total_checks": total_checks,
            "warnings": [],
        }

    def get_recipe_attempt(self, *, attempt_id: str) -> dict[str, object]:
        normalized = attempt_id.strip()
        if not normalized:
            raise ServiceError(
                code="RECIPE_ATTEMPT_NOT_FOUND",
                message="Attempt id must not be empty.",
                status_code=404,
            )
        try:
            attempt = self._recipe_attempt_store.get_attempt(normalized)
        except KeyError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_NOT_FOUND",
                message=f"Recipe attempt '{normalized}' was not found.",
                status_code=404,
            ) from exc
        if attempt.state in {AttemptState.RUNNING, AttemptState.SUCCEEDED}:
            with self._lock:
                mapped_job_id = self._attempt_to_build_job.get(normalized)
                mapped_job = self._jobs.get(mapped_job_id) if mapped_job_id is not None else None
            if mapped_job is not None and mapped_job.state in {
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
            }:
                self._safe_sync_generated_attempt(job=mapped_job)
                attempt = self._recipe_attempt_store.get_attempt(normalized)
            elif mapped_job is None and attempt.state == AttemptState.RUNNING:
                # Slice 3B2b: a RUNNING attempt with no mapped BuildJob can never
                # be synced by the ordinary job-sync path above -- candidate-
                # selection reuse never creates one. If it is a marked, abandoned
                # reuse materialization (the client only polled and never
                # resubmitted the original Idempotency-Key), complete or fail it
                # here instead of leaving it RUNNING forever. A no-op for every
                # other RUNNING attempt (returns `None`, and `attempt` is left
                # untouched): an ordinary in-flight build with no reuse marker is
                # never auto-resumed.
                recovered = self._recover_abandoned_reuse_attempt(normalized)
                if recovered is not None:
                    attempt = recovered
            elif mapped_job is None and attempt.state == AttemptState.SUCCEEDED:
                # Reuse success/evidence atomicity revision: a legacy/crash
                # attempt from before atomic finish+evidence recording may
                # already be terminally SUCCEEDED with a reuse marker but no
                # dispatch evidence. Opportunistically heal it on poll too,
                # not only at startup/resubmission -- never touches the
                # already-terminal attempt row itself.
                self._backfill_legacy_reuse_dispatch_evidence(normalized)
        return self._serialize_recipe_attempt(attempt)

    def create_generated_recipe_attempt(
        self,
        *,
        recipe_fingerprint: str,
        idempotency_key: str,
        model_id: str | None = None,
    ) -> tuple[BuildJob, bool, dict[str, object]]:
        key = idempotency_key.strip()
        if not key:
            raise ServiceError(
                code="IDEMPOTENCY_KEY_REQUIRED",
                message="Idempotency-Key is required.",
                status_code=400,
            )
        fingerprint = recipe_fingerprint.strip().lower()
        if not fingerprint:
            raise ServiceError(
                code="RECIPE_FINGERPRINT_REQUIRED",
                message="Generated recipe fingerprint is required.",
                status_code=400,
            )
        generated_record = self._recipe_attempt_store.get_generated_recipe(fingerprint)
        if generated_record is None:
            raise ServiceError(
                code="GENERATED_RECIPE_NOT_FOUND",
                message=f"Generated recipe '{fingerprint}' was not found.",
                status_code=404,
            )
        if model_id is not None and model_id.strip() and model_id.strip() != generated_record.model_id:
            raise ServiceError(
                code="GENERATED_RECIPE_MODEL_MISMATCH",
                message=(
                    f"Generated recipe '{fingerprint}' belongs to '{generated_record.model_id}', "
                    f"received '{model_id.strip()}'."
                ),
                status_code=409,
            )
        reusable_candidate_selection = self._resolve_reusable_candidate_selection(record=generated_record)
        attempt_request_record = (
            reusable_candidate_selection.winner_generated_record
            if reusable_candidate_selection is not None
            else generated_record
        )
        attempt_request = build_attempt_request_from_generated(attempt_request_record)
        request_fingerprint = build_attempt_request_fingerprint(attempt_request)
        try:
            attempt, replay = self._recipe_attempt_store.create_attempt(
                idempotency_key=key,
                request=attempt_request,
                request_fingerprint=request_fingerprint,
            )
        except AttemptIdempotencyConflictError as exc:
            raise ServiceError(
                code="IDEMPOTENCY_BODY_CONFLICT",
                message="Idempotency-Key was reused with a different generated attempt payload.",
                status_code=409,
            ) from exc
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc

        with self._lock:
            mapped_job_id = self._attempt_to_build_job.get(attempt.attempt_id)
            if mapped_job_id is not None and mapped_job_id in self._jobs:
                return self._jobs[mapped_job_id], True, self._serialize_recipe_attempt(attempt)

        if reusable_candidate_selection is not None:
            # `_materialize_reused_generated_attempt` serializes this call against
            # any other concurrent caller resolving the SAME `attempt.attempt_id`
            # (i.e. the same Idempotency-Key) via a per-attempt guard, and is
            # itself idempotent: every caller -- whether it performs the
            # materialization or joins an already-materialized/in-flight one --
            # observes and returns the same succeeded attempt. This never creates
            # a new BuildJob for `attempt.attempt_id`: `_attempt_to_build_job`/
            # `_build_job_to_attempt` are deliberately left without an entry for
            # this attempt_id (reuse never dispatches to the build stage runner),
            # so the `job_id` returned below is an alias of the selected winner's
            # existing build, not a distinct job for this attempt. Current
            # contracts key a returned build purely by `job_id` for polling, so
            # this aliasing is safe as long as callers do not assume
            # `job.request.generated_recipe_attempt` identifies *this* attempt.
            try:
                materialized_attempt = self._materialize_reused_generated_attempt(
                    attempt_id=attempt.attempt_id,
                    winner_attempt=reusable_candidate_selection.winner_attempt,
                )
            except RecipeAttemptStoreError as exc:
                raise ServiceError(
                    code="RECIPE_ATTEMPT_STORE_ERROR",
                    message=_sanitize_text(str(exc)),
                    status_code=500,
                ) from exc
            with self._lock:
                winner_job = self._jobs.get(reusable_candidate_selection.winner_job_id)
            if winner_job is None:
                self._raise_candidate_selection_reuse_integrity_error(
                    f"Winner job '{reusable_candidate_selection.winner_job_id}' no longer exists."
                )
            attempt_payload = self._serialize_recipe_attempt(materialized_attempt)
            attempt_payload["candidate_selection_reuse"] = self._candidate_selection_reuse_payload(
                reusable_candidate_selection
            )
            return winner_job, replay, attempt_payload

        with self._lock:
            if attempt.state != AttemptState.GENERATED:
                raise ServiceError(
                    code="RECIPE_ATTEMPT_ALREADY_STARTED",
                    message=(
                        f"Recipe attempt '{attempt.attempt_id}' is already in state '{attempt.state.value}' "
                        "without a recoverable build job mapping."
                    ),
                    status_code=409,
                )

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

            started_attempt = self._recipe_attempt_store.start_attempt(attempt.attempt_id)
            self._register_default_candidate_lineage_if_eligible(
                attempt_id=attempt.attempt_id,
                record=generated_record,
            )
            job_id = str(uuid.uuid4())
            request = self._build_request_for_generated_attempt(
                record=generated_record,
                attempt_id=attempt.attempt_id,
                job_id=job_id,
            )
            job = BuildJob(job_id=job_id, request=request)
            job.add_event(
                "Generated recipe attempt queued; automatic recipe execution remains experimental until verified."
            )
            self._jobs[job_id] = job
            self._cancel_events[job_id] = Event()
            self._build_job_to_attempt[job_id] = attempt.attempt_id
            self._attempt_to_build_job[attempt.attempt_id] = job_id
            self._store.save_job(job)
            self._queue.put(job_id)
            return job, replay, self._serialize_recipe_attempt(started_attempt)

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
            recipe_match = self._recipe_registry.resolve(
                model_id=normalized.model_id,
                modality=normalized.task,
                task_profile=normalized.task_profile,
                allow_experimental=normalized.allow_experimental,
            )
            request = self._build_request(
                normalized,
                job_id=job_id,
                recipe_match=recipe_match,
                enforce_recipe_buildable=True,
            )
            job = BuildJob(job_id=job_id, request=request)
            job.add_event("Build job queued.")
            self._jobs[job_id] = job
            self._cancel_events[job_id] = Event()
            self._idempotency[key] = IdempotencyRecord(body_sha256=body_sha256, job_id=job_id)
            self._store.save_job(job)
            self._store.save_idempotency(key, self._idempotency[key])
            self._queue.put(job_id)
            return job, False

    def _attempt_id_for_job(self, job_id: str) -> str | None:
        with self._lock:
            attempt_id = self._build_job_to_attempt.get(job_id)
            if attempt_id is not None:
                return attempt_id
            job = self._jobs.get(job_id)
            if job is None:
                return None
            generated_attempt = job.request.generated_recipe_attempt
            if generated_attempt is None:
                return None
            recovered_attempt_id = generated_attempt.attempt_id.strip()
            if not recovered_attempt_id:
                return None
            self._build_job_to_attempt[job_id] = recovered_attempt_id
            self._attempt_to_build_job[recovered_attempt_id] = job_id
            return recovered_attempt_id

    @staticmethod
    def _failure_gate_for_job(job: BuildJob) -> tuple[AttemptGate, AttemptGateStatus]:
        stage = job.failure.stage if job.failure is not None else job.state
        if stage in {
            JobState.PREFLIGHT,
            JobState.DOWNLOADING,
        }:
            return AttemptGate.MOBIUS_BUILD, AttemptGateStatus.NOT_RUN
        if stage in {
            JobState.MOBIUS_BUILDING,
            JobState.MOBIUS_VALIDATING,
        }:
            return AttemptGate.MOBIUS_BUILD, AttemptGateStatus.FAILED
        if stage in {JobState.OLIVE_OPTIMIZING, JobState.PACKAGING}:
            return AttemptGate.OLIVE_OPTIMIZE, AttemptGateStatus.FAILED
        if stage == JobState.RUNTIME_VALIDATING:
            return AttemptGate.ONNX_VALIDATION, AttemptGateStatus.FAILED
        if stage == JobState.FL_LOADING:
            return AttemptGate.OGA_VALIDATION, AttemptGateStatus.FAILED
        if stage == JobState.INFERENCING:
            return AttemptGate.FL_SDK_INFERENCE, AttemptGateStatus.FAILED
        return AttemptGate.QUALITY_VALIDATION, AttemptGateStatus.FAILED

    def _record_attempt_gate(
        self,
        *,
        attempt_id: str,
        gate: AttemptGate,
        status: AttemptGateStatus,
        evidence_ref: str,
        metrics_ref: str | None = None,
    ) -> None:
        try:
            self._recipe_attempt_store.record_attempt_gate(
                attempt_id=attempt_id,
                gate=gate,
                status=status,
                evidence_ref=evidence_ref,
                metrics_ref=metrics_ref,
            )
        except AttemptStateTransitionError:
            existing = self._recipe_attempt_store.get_attempt(attempt_id)
            if existing.state not in {AttemptState.SUCCEEDED, AttemptState.FAILED, AttemptState.CANCELLED}:
                raise
            matched = next((row for row in existing.gate_results if row.gate == gate), None)
            if (
                matched is not None
                and matched.status == status
                and matched.evidence_ref == evidence_ref
                and matched.metrics_ref == metrics_ref
            ):
                return
            return
        except AttemptGateSequenceError:
            existing = self._recipe_attempt_store.get_attempt(attempt_id)
            matched = next((row for row in existing.gate_results if row.gate == gate), None)
            if (
                matched is not None
                and matched.status == status
                and matched.evidence_ref == evidence_ref
                and matched.metrics_ref == metrics_ref
            ):
                return
            raise ServiceError(
                code="RECIPE_ATTEMPT_GATE_CONFLICT",
                message=(
                    f"Attempt '{attempt_id}' gate '{gate.value}' already recorded with "
                    "different status or evidence."
                ),
                status_code=409,
            )
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc

    def _finalize_attempt_failed(
        self,
        *,
        attempt_id: str,
        classification: AttemptFailureClassification,
        job: BuildJob,
        message: str,
        source_owner: str,
        next_action: str,
    ) -> None:
        try:
            self._recipe_attempt_store.finish_attempt_failed(
                attempt_id,
                failure=AttemptFailure(
                    classification=classification,
                    stage=(job.failure.stage.value if job.failure is not None else job.state.value),
                    message=_sanitize_attempt_failure_message(message),
                    evidence_refs=(f"job://{job.job_id}",),
                    source_owner=source_owner,
                    next_action=next_action,
                ),
            )
        except AttemptStateTransitionError:
            terminal = self._recipe_attempt_store.get_attempt(attempt_id)
            if terminal.state in {AttemptState.FAILED, AttemptState.CANCELLED, AttemptState.SUCCEEDED}:
                return
            raise
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc

    def _finalize_attempt_cancelled(self, *, attempt_id: str, job: BuildJob, reason: str) -> None:
        try:
            self._recipe_attempt_store.cancel_attempt(
                attempt_id,
                failure=AttemptFailure(
                    classification=AttemptFailureClassification.CANCELLED,
                    stage=(job.failure.stage.value if job.failure is not None else job.state.value),
                    message=_sanitize_attempt_failure_message(reason),
                    evidence_refs=(f"job://{job.job_id}",),
                    source_owner="fl-onboarding",
                    next_action="Resubmit the same recipe fingerprint with a new idempotency key to retry.",
                ),
            )
        except AttemptStateTransitionError:
            terminal = self._recipe_attempt_store.get_attempt(attempt_id)
            if terminal.state in {AttemptState.CANCELLED, AttemptState.FAILED, AttemptState.SUCCEEDED}:
                return
            raise
        except RecipeAttemptStoreError as exc:
            raise ServiceError(
                code="RECIPE_ATTEMPT_STORE_ERROR",
                message=_sanitize_text(str(exc)),
                status_code=500,
            ) from exc

    def _record_failure_gates(self, *, attempt_id: str, job: BuildJob) -> AttemptGate:
        failure_gate, failure_gate_status = self._failure_gate_for_job(job)
        failure_status_suffix = "failed" if failure_gate_status == AttemptGateStatus.FAILED else "not-run"
        for gate in _RECIPE_ATTEMPT_GATE_SEQUENCE:
            if gate == AttemptGate.QUALITY_VALIDATION:
                break
            if gate == failure_gate:
                self._record_attempt_gate(
                    attempt_id=attempt_id,
                    gate=gate,
                    status=failure_gate_status,
                    evidence_ref=f"job://{job.job_id}/{gate.value}/{failure_status_suffix}",
                )
                break
            self._record_attempt_gate(
                attempt_id=attempt_id,
                gate=gate,
                status=AttemptGateStatus.PASSED,
                evidence_ref=f"job://{job.job_id}/{gate.value}/passed",
            )
        return failure_gate

    def _record_success_gates(self, *, attempt_id: str, job: BuildJob) -> None:
        for gate in _RECIPE_ATTEMPT_GATE_SEQUENCE:
            if gate == AttemptGate.QUALITY_VALIDATION:
                return
            self._record_attempt_gate(
                attempt_id=attempt_id,
                gate=gate,
                status=AttemptGateStatus.PASSED,
                evidence_ref=f"job://{job.job_id}/{gate.value}/passed",
            )

    @staticmethod
    def _model_capability_advisory_message(model_capability: Any) -> str:
        summary = (
            "Model capability advisory: "
            f"{model_capability.checks_passed}/{model_capability.total_checks} checks passed."
        )
        warnings = list(model_capability.warnings)
        if warnings:
            preview = ", ".join(warnings[:3])
            if len(warnings) > 3:
                preview = f"{preview}, ..."
            summary += f" Warnings: {preview}."
        confidence = model_capability.confidence
        if confidence.level.value == "low":
            summary += " Confidence: low (determinism support is partial)."
        return summary

    def _run_quality_validation(
        self,
        *,
        job: BuildJob,
        artifact: BuildArtifact | None,
    ) -> QualityValidationOutcome:
        evidence_prefix = f"quality://{job.job_id}/{AttemptGate.QUALITY_VALIDATION.value}"
        if artifact is None:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.FAILED,
                message="Missing result artifact required for quality validation.",
                evidence_ref=f"{evidence_prefix}/optimized-artifact-missing",
            )
        if self._text_inference_backend is None:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.UNAVAILABLE,
                message="Text inference backend is unavailable for deterministic quality validation.",
                evidence_ref=f"{evidence_prefix}/runtime-unavailable",
            )

        baseline_artifact, baseline_resolution = self._resolve_quality_baseline_artifact(
            job=job,
            optimized_artifact=artifact,
        )
        if baseline_resolution is not None:
            return baseline_resolution
        if baseline_artifact is None:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.NOT_RUN,
                message="Quality baseline not run for this generated attempt.",
                evidence_ref=f"{evidence_prefix}/baseline-not-run",
            )
        try:
            baseline, baseline_batch_worker = self._execute_quality_prompts(
                job=job,
                artifact=baseline_artifact,
                execution_label="Baseline",
            )
        except RuntimeError as exc:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.UNAVAILABLE,
                message=str(exc),
                evidence_ref=f"{evidence_prefix}/baseline-unavailable",
            )
        try:
            optimized, optimized_batch_worker = self._execute_quality_prompts(
                job=job,
                artifact=artifact,
                execution_label="Optimized",
            )
        except RuntimeError as exc:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.FAILED,
                message=str(exc),
                evidence_ref=f"{evidence_prefix}/optimized-execution-failed",
            )

        result = evaluate_quality_validation(
            profile=self._quality_profile,
            model_task="text-generation",
            optimized_outputs=optimized,
            baseline_outputs=baseline,
            require_baseline_comparison=True,
        )
        try:
            metrics_ref = self._persist_quality_validation_metrics(
                job=job,
                baseline_outputs=baseline,
                optimized_outputs=optimized,
                quality_result=result,
                baseline_batch_worker=baseline_batch_worker,
                optimized_batch_worker=optimized_batch_worker,
            )
        except (OSError, ValueError, TypeError) as exc:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.FAILED,
                message=f"Quality evidence capture failed: {exc}",
                evidence_ref=f"{evidence_prefix}/metrics-persist-failed",
            )
        recipe_verification = result.recipe_verification
        if recipe_verification.gate_status == GateState.MISSING:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.NOT_RUN,
                message="Recipe integrity is inconclusive because quality baseline comparison was not recorded.",
                evidence_ref=f"{evidence_prefix}/baseline-not-run",
                metrics_ref=metrics_ref,
            )
        if recipe_verification.gate_status == GateState.UNAVAILABLE:
            return QualityValidationOutcome(
                passed=False,
                gate_status=AttemptGateStatus.UNAVAILABLE,
                message="Recipe integrity is inconclusive because quality baseline comparison was unavailable.",
                evidence_ref=f"{evidence_prefix}/baseline-unavailable",
                metrics_ref=metrics_ref,
            )
        advisory = self._model_capability_advisory_message(result.model_capability)
        if recipe_verification.can_promote:
            return QualityValidationOutcome(
                passed=True,
                gate_status=AttemptGateStatus.PASSED,
                message=(
                    "Recipe integrity verified: "
                    f"{result.promotion_evidence.profile_id}@{result.promotion_evidence.profile_version}. "
                    f"{advisory}"
                ),
                evidence_ref=f"{evidence_prefix}/baseline-passed",
                metrics_ref=metrics_ref,
                quality_retry_evaluation=result.quality_retry_evaluation,
            )

        integrity_failures = list(recipe_verification.integrity_failures)
        if not integrity_failures:
            integrity_failures.append("recipe_integrity_blocked")
        if any(
            failure.startswith("baseline_passed_optimized_failed:")
            or failure.startswith("optimized_structural_regression:")
            for failure in integrity_failures
        ):
            evidence_ref = f"{evidence_prefix}/baseline-regression"
        elif not recipe_verification.runtime_functional:
            evidence_ref = f"{evidence_prefix}/runtime-integrity-failed"
        else:
            evidence_ref = f"{evidence_prefix}/optimized-validation-failed"
        return QualityValidationOutcome(
            passed=False,
            gate_status=AttemptGateStatus.FAILED,
            message=f"Recipe integrity blocked: {'; '.join(integrity_failures)}. {advisory}",
            evidence_ref=evidence_ref,
            metrics_ref=metrics_ref,
            quality_retry_evaluation=result.quality_retry_evaluation,
        )

    @staticmethod
    def _quality_metrics_ref_for_job(job_id: str) -> str:
        return f"quality-metrics://{job_id}/{_QUALITY_EVIDENCE_FILENAME}"

    @staticmethod
    def _sanitize_quality_capture_text(text: str, *, max_chars: int) -> str:
        cleaned = _CONTROL_CHAR_RE.sub("", text)
        cleaned = _HF_TOKEN_RE.sub("[REDACTED]", cleaned)
        cleaned = _BEARER_RE.sub("******", cleaned)
        cleaned = _API_KEY_RE.sub(r"\1[REDACTED]", cleaned)
        cleaned = _SECRET_ASSIGNMENT_RE.sub(r"\1=[REDACTED]", cleaned)
        cleaned = _WINDOWS_ABSOLUTE_PATH_RE.sub("<redacted-absolute-path>", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) <= max_chars:
            return cleaned
        if max_chars <= 3:
            return cleaned[:max_chars]
        return cleaned[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _deterministic_inference_payload(value: Any) -> dict[str, object] | None:
        if value is None:
            return None
        return {
            "temperature": value.temperature,
            "seed": value.seed,
            "max_tokens": value.max_tokens,
        }

    @classmethod
    def _prompt_execution_payload(
        cls,
        *,
        record: PromptExecutionRecord | None,
        check: Any,
    ) -> dict[str, object] | None:
        if record is None and check is None:
            return None
        payload: dict[str, object] = {}
        if record is not None:
            payload["output_text"] = cls._sanitize_quality_capture_text(
                record.output_text,
                max_chars=_QUALITY_EVIDENCE_MAX_OUTPUT_CHARS,
            )
            payload["applied_determinism"] = cls._deterministic_inference_payload(record.applied_determinism)
            payload["unsupported_determinism_fields"] = list(record.unsupported_determinism_fields)
        if check is not None:
            determinism = check.determinism
            payload["checks"] = {
                "passed": check.passed,
                "failures": list(check.failures),
                "determinism": {
                    "recorded": determinism.recorded,
                    "fully_enforced": determinism.fully_enforced,
                    "unsupported_fields": list(determinism.unsupported_fields),
                    "mismatched_fields": list(determinism.mismatched_fields),
                },
            }
        return payload

    @staticmethod
    def _batch_worker_diagnostics_payload(value: Any) -> dict[str, object] | None:
        if not isinstance(value, dict):
            return None
        payload: dict[str, object] = {}

        request_raw = value.get("request")
        if isinstance(request_raw, dict):
            request_payload: dict[str, object] = {}
            for field in (
                "mode",
                "prompt_ids",
                "prompt_count",
                "max_tokens",
                "model_name",
                "expected_model_load_count",
                "per_prompt_timeout_seconds",
                "batch_timeout_seconds",
                "outer_command_timeout_seconds",
                "outer_timeout_grace_seconds",
            ):
                if field in request_raw:
                    request_payload[field] = request_raw[field]
            if request_payload:
                payload["request"] = request_payload

        response_raw = value.get("response")
        if isinstance(response_raw, dict):
            response_payload: dict[str, object] = {}
            for field in (
                "ok",
                "failure_stage",
                "failed_prompt_id",
                "completed_prompt_ids",
                "duration_seconds",
                "prompt_order_preserved",
            ):
                if field in response_raw:
                    response_payload[field] = response_raw[field]
            results_raw = response_raw.get("results")
            if isinstance(results_raw, list):
                results_payload: list[dict[str, object]] = []
                for row in results_raw:
                    if not isinstance(row, dict):
                        continue
                    results_payload.append(
                        {
                            "prompt_id": row.get("prompt_id"),
                            "duration_seconds": row.get("duration_seconds"),
                            "timed_out": row.get("timed_out"),
                        }
                    )
                response_payload["results"] = results_payload
            if response_payload:
                payload["response"] = response_payload

        return payload or None

    def _persist_quality_validation_metrics(
        self,
        *,
        job: BuildJob,
        baseline_outputs: tuple[PromptExecutionRecord, ...],
        optimized_outputs: tuple[PromptExecutionRecord, ...],
        quality_result: Any,
        baseline_batch_worker: dict[str, object] | None,
        optimized_batch_worker: dict[str, object] | None,
    ) -> str:
        baseline_by_id = {row.prompt_id: row for row in baseline_outputs}
        optimized_by_id = {row.prompt_id: row for row in optimized_outputs}
        baseline_checks = (
            {row.prompt_id: row for row in quality_result.baseline_functional.prompt_results}
            if quality_result.baseline_functional is not None
            else {}
        )
        optimized_checks = {
            row.prompt_id: row for row in quality_result.optimized_functional.prompt_results
        }
        prompt_rows: list[dict[str, object]] = []
        for prompt in self._quality_profile.prompts:
            prompt_rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category.value,
                    "prompt": self._sanitize_quality_capture_text(
                        prompt.prompt,
                        max_chars=_QUALITY_EVIDENCE_MAX_PROMPT_CHARS,
                    ),
                    "baseline": self._prompt_execution_payload(
                        record=baseline_by_id.get(prompt.prompt_id),
                        check=baseline_checks.get(prompt.prompt_id),
                    ),
                    "optimized": self._prompt_execution_payload(
                        record=optimized_by_id.get(prompt.prompt_id),
                        check=optimized_checks.get(prompt.prompt_id),
                    ),
                }
            )

        baseline_comparison = quality_result.baseline_comparison
        recipe_verification = quality_result.recipe_verification
        model_capability = quality_result.model_capability
        model_capability_prompt_rows = [
            {
                "prompt_id": row.prompt_id,
                "category": row.category.value,
                "baseline_status": row.baseline_status.value,
                "optimized_status": row.optimized_status.value,
                "comparison": row.comparison.value,
                "baseline_failures": list(row.baseline_failures),
                "optimized_failures": list(row.optimized_failures),
                "warnings": list(row.warnings),
            }
            for row in model_capability.prompt_results
        ]
        evidence_payload: dict[str, object] = {
            "schema_version": _QUALITY_EVIDENCE_SCHEMA_VERSION,
            "job_id": job.job_id,
            "profile_id": quality_result.promotion_evidence.profile_id,
            "profile_version": quality_result.promotion_evidence.profile_version,
            "deterministic_inference": self._deterministic_inference_payload(
                self._quality_profile.deterministic_inference
            ),
            "unsupported_determinism_fields_reported_by_runtime": ["temperature", "seed"],
            "recipe_verification": {
                "runtime_functional": recipe_verification.runtime_functional,
                "baseline_available": recipe_verification.baseline_available,
                "regression_free": recipe_verification.regression_free,
                "integrity_failures": list(recipe_verification.integrity_failures),
                "gate_status": recipe_verification.gate_status.value,
                "status": recipe_verification.status.value,
                "can_promote": recipe_verification.can_promote,
            },
            "model_capability": {
                "checks_passed": model_capability.checks_passed,
                "total_checks": model_capability.total_checks,
                "warnings": list(model_capability.warnings),
                "confidence": {
                    "level": model_capability.confidence.level.value,
                    "determinism_supported": model_capability.confidence.determinism_supported,
                    "reasons": list(model_capability.confidence.reasons),
                },
                "per_prompt": model_capability_prompt_rows,
            },
            "promotion_evidence": {
                "can_promote": quality_result.promotion_evidence.can_promote,
                "functional_gate": quality_result.promotion_evidence.functional_gate.value,
                "baseline_comparison_gate": quality_result.promotion_evidence.baseline_comparison_gate.value,
                "metrics_capture_gate": quality_result.promotion_evidence.metrics_gate.value,
                "notes": list(quality_result.promotion_evidence.notes),
            },
            "baseline_comparison": (
                {
                    "passed": baseline_comparison.passed,
                    "regressions": list(baseline_comparison.regressions),
                }
                if baseline_comparison is not None
                else None
            ),
            "batch_worker": {
                "prompt_order": [prompt.prompt_id for prompt in self._quality_profile.prompts],
                "baseline": self._batch_worker_diagnostics_payload(baseline_batch_worker),
                "optimized": self._batch_worker_diagnostics_payload(optimized_batch_worker),
            },
            "per_prompt": prompt_rows,
        }
        evidence_path = job.request.workspace_root / _QUALITY_EVIDENCE_FILENAME
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(evidence_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self._quality_metrics_ref_for_job(job.job_id)

    def _resolve_quality_baseline_artifact(
        self,
        *,
        job: BuildJob,
        optimized_artifact: BuildArtifact,
    ) -> tuple[BuildArtifact | None, QualityValidationOutcome | None]:
        evidence_prefix = f"quality://{job.job_id}/{AttemptGate.QUALITY_VALIDATION.value}"
        baseline_path = job.request.workspace_root / "mobius"
        if not baseline_path.is_dir():
            return (
                None,
                QualityValidationOutcome(
                    passed=False,
                    gate_status=AttemptGateStatus.NOT_RUN,
                    message=(
                        "Quality baseline not run: pre-Olive Mobius baseline package was not produced "
                        "for this attempt."
                    ),
                    evidence_ref=f"{evidence_prefix}/baseline-not-run-mobius-missing",
                ),
            )
        try:
            baseline_resolved = baseline_path.resolve()
            optimized_resolved = optimized_artifact.path.resolve()
        except OSError:
            baseline_resolved = baseline_path
            optimized_resolved = optimized_artifact.path
        if baseline_resolved == optimized_resolved:
            return (
                None,
                QualityValidationOutcome(
                    passed=False,
                    gate_status=AttemptGateStatus.NOT_RUN,
                    message=(
                        "Quality baseline not run: baseline and optimized artifacts resolved to the "
                        "same package identity."
                    ),
                    evidence_ref=f"{evidence_prefix}/baseline-not-run-self-comparison",
                ),
            )
        descriptor_path = baseline_path / "inference_model.json"
        if not descriptor_path.is_file():
            return (
                None,
                QualityValidationOutcome(
                    passed=False,
                    gate_status=AttemptGateStatus.UNAVAILABLE,
                    message=(
                        "Quality baseline unavailable: pre-Olive Mobius package is missing "
                        "inference_model.json."
                    ),
                    evidence_ref=f"{evidence_prefix}/baseline-unavailable-descriptor-missing",
                ),
            )
        return (
            BuildArtifact(
                artifact_id=f"baseline-{job.job_id}",
                kind=ArtifactKind.MODEL,
                path=baseline_path,
                description="Pre-Olive Mobius baseline package",
            ),
            None,
        )

    def _execute_quality_prompts(
        self,
        *,
        job: BuildJob,
        artifact: BuildArtifact,
        execution_label: str,
    ) -> tuple[tuple[PromptExecutionRecord, ...], dict[str, object] | None]:
        backend = self._text_inference_backend
        if backend is None:
            raise RuntimeError("Text inference backend is unavailable for quality validation.")
        max_tokens = self._quality_profile.deterministic_inference.max_tokens
        prompt_rows = tuple((prompt.prompt_id, prompt.prompt) for prompt in self._quality_profile.prompts)
        batch_infer = getattr(backend, "infer_batch", None)
        if callable(batch_infer):
            try:
                batch_outputs = batch_infer(
                    artifact=artifact,
                    job=job,
                    prompts=prompt_rows,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                raise RuntimeError(f"{execution_label} quality prompt batch failed: {exc}") from exc
            if not isinstance(batch_outputs, (list, tuple)):
                raise RuntimeError(
                    f"{execution_label} quality prompt batch returned invalid output container."
                )
            if len(batch_outputs) != len(prompt_rows):
                raise RuntimeError(
                    f"{execution_label} quality prompt batch returned {len(batch_outputs)} outputs for "
                    f"{len(prompt_rows)} prompts."
                )
            diagnostics: dict[str, object] | None = None
            consume_diagnostics = getattr(backend, "consume_last_batch_diagnostics", None)
            if callable(consume_diagnostics):
                payload = consume_diagnostics()
                diagnostics = payload if isinstance(payload, dict) else None
            return (
                tuple(
                    PromptExecutionRecord(
                        prompt_id=prompt_id,
                        output_text=str(output_text),
                        applied_determinism=self._quality_profile.deterministic_inference,
                        unsupported_determinism_fields=("temperature", "seed"),
                    )
                    for (prompt_id, _), output_text in zip(prompt_rows, batch_outputs)
                ),
                diagnostics,
            )
        outputs: list[PromptExecutionRecord] = []
        prompt_timing_rows: list[dict[str, object]] = []
        batch_started = time.monotonic()
        for prompt in self._quality_profile.prompts:
            prompt_started = time.monotonic()
            try:
                response = backend.infer(
                    artifact=artifact,
                    job=job,
                    prompt=prompt.prompt,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{execution_label} quality prompt '{prompt.prompt_id}' failed: {exc}"
                ) from exc
            duration_seconds = round(time.monotonic() - prompt_started, 3)
            prompt_timing_rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "duration_seconds": duration_seconds,
                    "timed_out": False,
                }
            )
            outputs.append(
                PromptExecutionRecord(
                    prompt_id=prompt.prompt_id,
                    output_text=response,
                    applied_determinism=self._quality_profile.deterministic_inference,
                    unsupported_determinism_fields=("temperature", "seed"),
                )
            )
        return (
            tuple(outputs),
            {
                "request": {
                    "mode": "single-prompt-loop",
                    "prompt_ids": [prompt_id for prompt_id, _ in prompt_rows],
                    "prompt_count": len(prompt_rows),
                    "max_tokens": max_tokens,
                    "per_prompt_timeout_seconds": None,
                    "batch_timeout_seconds": None,
                    "outer_command_timeout_seconds": None,
                    "outer_timeout_grace_seconds": None,
                },
                "response": {
                    "ok": True,
                    "failure_stage": None,
                    "failed_prompt_id": None,
                    "completed_prompt_ids": [row["prompt_id"] for row in prompt_timing_rows],
                    "duration_seconds": round(time.monotonic() - batch_started, 3),
                    "prompt_order_preserved": True,
                    "results": prompt_timing_rows,
                },
            },
        )

    def _promotion_evidence_from_attempt(self, *, attempt: Any) -> PromotionGateEvidence:
        gate_map = {row.gate: row for row in attempt.gate_results}

        def check(gate: AttemptGate) -> PromotionGateCheck:
            row = gate_map.get(gate)
            if row is None:
                return PromotionGateCheck(passed=False, evidence=f"{gate.value}:missing")
            return PromotionGateCheck(
                passed=row.status == AttemptGateStatus.PASSED,
                evidence=row.evidence_ref,
                metrics_ref=row.metrics_ref,
            )

        return PromotionGateEvidence(
            mobius_build=check(AttemptGate.MOBIUS_BUILD),
            olive_optimize=check(AttemptGate.OLIVE_OPTIMIZE),
            onnx_validation=check(AttemptGate.ONNX_VALIDATION),
            ort_validation=check(AttemptGate.ORT_VALIDATION),
            oga_validation=check(AttemptGate.OGA_VALIDATION),
            fl_sdk_inference=check(AttemptGate.FL_SDK_INFERENCE),
            quality_validation=check(AttemptGate.QUALITY_VALIDATION),
        )

    def _recompile_generated_recipe_record(self, *, record: Any) -> GeneratedRecipe:
        payload = record.payload()
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ServiceError(
                code="GENERATED_RECIPE_INVALID",
                message="Generated recipe payload is missing provenance metadata.",
                status_code=500,
            )
        input_metadata = provenance.get("input_metadata")
        toolchain_payload = provenance.get("toolchain")
        if not isinstance(input_metadata, dict) or not isinstance(toolchain_payload, dict):
            raise ServiceError(
                code="GENERATED_RECIPE_INVALID",
                message="Generated recipe payload is missing deterministic input metadata/toolchain metadata.",
                status_code=500,
            )
        model_id = str(input_metadata.get("model_id") or record.model_id)
        revision_sha = str(input_metadata.get("revision_sha") or record.revision_sha)
        model_type_value = input_metadata.get("model_type")
        model_type = str(model_type_value) if isinstance(model_type_value, str) else None
        architectures_raw = input_metadata.get("architectures")
        architectures = tuple(
            str(item) for item in architectures_raw
        ) if isinstance(architectures_raw, list) else ()
        config_files_raw = input_metadata.get("config_files")
        tokenizer_files_raw = input_metadata.get("tokenizer_files")
        available_files_raw = input_metadata.get("available_files")
        config_files = tuple(str(item) for item in config_files_raw) if isinstance(config_files_raw, list) else ()
        tokenizer_files = (
            tuple(str(item) for item in tokenizer_files_raw) if isinstance(tokenizer_files_raw, list) else ()
        )
        available_files = tuple(str(item) for item in available_files_raw) if isinstance(available_files_raw, list) else ()
        task = str(input_metadata.get("task") or _RECIPE_DEFAULT_TASK)
        requested_device = str(input_metadata.get("requested_device") or _RECIPE_DEFAULT_DEVICE)
        requested_precision = str(input_metadata.get("requested_precision") or _RECIPE_DEFAULT_PRECISION)
        is_gated = bool(input_metadata.get("is_gated"))
        requires_remote_code = bool(input_metadata.get("requires_remote_code"))

        capability_config: dict[str, object] = {}
        if model_type is not None:
            capability_config["model_type"] = model_type
        if architectures:
            capability_config["architectures"] = list(architectures)
        if requires_remote_code:
            capability_config["auto_map"] = {"AutoModel": "remote"}
        normalized_metadata = normalize_huggingface_metadata(
            model_id=model_id,
            config=capability_config,
            is_gated=is_gated,
            is_private=False,
        )
        capability_resolution = self._capability_registry.resolve(
            metadata=normalized_metadata,
            task=task,
            device=requested_device,
            requested_precision=requested_precision,
        )
        try:
            candidate = compile_generated_recipe(
                RecipeCompilerInput(
                    model_id=model_id,
                    revision_sha=revision_sha,
                    model_type=model_type,
                    architectures=architectures,
                    task=task,
                    requested_device=requested_device,
                    requested_precision=requested_precision,
                    is_gated=is_gated,
                    requires_remote_code=requires_remote_code,
                    config_files=config_files,
                    tokenizer_files=tokenizer_files,
                    available_files=available_files,
                    capability_resolution=capability_resolution,
                    toolchain=RecipeCompilerToolchain(
                        mobius_version=str(toolchain_payload.get("mobius_version") or _RECIPE_TOOLCHAIN.mobius_version),
                        olive_version=str(toolchain_payload.get("olive_version") or _RECIPE_TOOLCHAIN.olive_version),
                        onnx_version=str(toolchain_payload.get("onnx_version") or _RECIPE_TOOLCHAIN.onnx_version),
                        ort_version=str(toolchain_payload.get("ort_version") or _RECIPE_TOOLCHAIN.ort_version),
                        oga_version=str(toolchain_payload.get("oga_version") or _RECIPE_TOOLCHAIN.oga_version),
                        foundry_sdk_version=str(
                            toolchain_payload.get("foundry_sdk_version") or _RECIPE_TOOLCHAIN.foundry_sdk_version
                        ),
                        foundry_cli_version=str(
                            toolchain_payload.get("foundry_cli_version") or _RECIPE_TOOLCHAIN.foundry_cli_version
                        ),
                    ),
                )
            )
        except GeneratedRecipeCompileError as exc:
            raise ServiceError(
                code="GENERATED_RECIPE_RECOMPILE_FAILED",
                message=_sanitize_text(str(exc)),
                status_code=409,
            ) from exc

        # Slice 3B1: a trusted candidate recipe (e.g. the block64 fallback) layers
        # a policy-approved quantization override on top of the base compile via
        # `compile_trusted_candidate_recipe`, which also folds its
        # `TrustedCandidateProvenance` into the fingerprinted payload. Recompiling
        # such a record must re-apply that exact same override -- by exact policy/
        # candidate identity, never by raw block_size alone -- or the recompiled
        # candidate's fingerprint (and therefore its Olive arguments) would never
        # match the persisted record, and promotion would silently fail closed.
        trusted_candidate_payload = provenance.get("trusted_candidate")
        if isinstance(trusted_candidate_payload, dict):
            policy_id = str(trusted_candidate_payload.get("policy_id") or "")
            candidate_index_raw = trusted_candidate_payload.get("candidate_index")
            try:
                trusted_policy = DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY.get(policy_id)
            except ValueError as exc:
                raise ServiceError(
                    code="GENERATED_RECIPE_INVALID",
                    message=f"Generated recipe references unknown trusted candidate policy '{policy_id}'.",
                    status_code=500,
                ) from exc
            if (
                not isinstance(candidate_index_raw, int)
                or isinstance(candidate_index_raw, bool)
                or candidate_index_raw < 0
                or candidate_index_raw >= len(trusted_policy.candidates)
            ):
                raise ServiceError(
                    code="GENERATED_RECIPE_INVALID",
                    message="Generated recipe references an out-of-range trusted candidate index.",
                    status_code=500,
                )
            trusted_candidate_selection = trusted_policy.candidates[candidate_index_raw]
            try:
                candidate = compile_trusted_candidate_recipe(
                    candidate,
                    policy=trusted_policy,
                    candidate=trusted_candidate_selection,
                )
            except TrustedCandidateCompilationError as exc:
                raise ServiceError(
                    code="GENERATED_RECIPE_RECOMPILE_FAILED",
                    message=_sanitize_text(str(exc)),
                    status_code=409,
                ) from exc

        if candidate.fingerprint != record.recipe_fingerprint:
            raise ServiceError(
                code="GENERATED_RECIPE_IDENTITY_MISMATCH",
                message=(
                    f"Stored recipe fingerprint '{record.recipe_fingerprint}' no longer matches deterministic compile "
                    f"output '{candidate.fingerprint}'."
                ),
                status_code=409,
            )
        return candidate

    # --- Slice 3B1: candidate lineage/fallback orchestration -----------------
    #
    # Everything below wires the durable candidate lineage/evidence/selection/
    # exhaustion API Slice 2 already added to `RecipeAttemptStore` -- plus Slice
    # 3A1's trusted block64 compilation and Slice 3A2's pre-Olive Mobius reuse --
    # into the generated-recipe-attempt lifecycle. It only ever activates for a
    # generated attempt whose *actual compiled* Olive device/precision match the
    # approved CPU INT4 selection policy's declared scope; every other generated
    # attempt (a different device/precision) and every static-recipe build is
    # completely untouched and keeps its pre-3B1 behavior exactly. See
    # `docs/candidate-orchestration.md` for the full internal design and the
    # exact Slice 3B2 handoff.

    @staticmethod
    def _generated_record_is_cpu_int4_eligible(record: Any) -> bool:
        """Whether `record`'s actual compiled Olive device/precision fall
        inside the approved CPU INT4 candidate-selection policy's declared
        scope. Deliberately mirrors exactly the same device/precision check
        `compile_trusted_candidate_recipe` itself enforces before it will
        ever layer a trusted quantization override onto a default recipe, so
        eligibility here can never silently drift from what a fallback
        compile would actually accept."""
        try:
            payload = record.payload()
        except RecipeAttemptStoreError:
            return False
        recipe_payload = payload.get("recipe")
        if not isinstance(recipe_payload, dict):
            return False
        olive_payload = recipe_payload.get("olive")
        if not isinstance(olive_payload, dict):
            return False
        policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
        return (
            olive_payload.get("device") == policy.target_device.value
            and olive_payload.get("precision") == policy.quantization.value
        )

    def _register_default_candidate_lineage_if_eligible(self, *, attempt_id: str, record: Any) -> None:
        """Idempotently register candidate 0 (the default, parent attempt)
        under the approved policy before the default candidate ever
        executes. A no-op for an ineligible device/precision, or when a
        lineage is already registered for `attempt_id` (a benign replay/race:
        `register_candidate_attempt` itself would otherwise reject a second
        registration of index 0)."""
        if not self._generated_record_is_cpu_int4_eligible(record):
            return
        try:
            existing = self._recipe_attempt_store.get_candidate_lineage(attempt_id)
        except RecipeAttemptStoreError:
            return
        if existing is not None:
            return
        try:
            self._recipe_attempt_store.register_candidate_attempt(
                parent_attempt_id=attempt_id,
                attempt_id=attempt_id,
                candidate_index=0,
                policy=self._recipe_selection_policy,
                quality_profile_fingerprint=self._quality_profile.fingerprint,
            )
        except CandidatePlanValidationError:
            return

    def _capture_pre_olive_descriptor_if_eligible(self, job: BuildJob, mobius_dir: Path) -> None:
        """`ProductionBuildStageRunner`'s `on_mobius_ready` result hook.

        Captures an immutable `PreOliveArtifactDescriptor` for a *default*
        candidate's just-succeeded Mobius output while `mobius_dir` is
        guaranteed to still exist on disk, so a later retryable structural-
        regression fallback candidate can reuse it instead of rebuilding
        Mobius. A completely inert no-op for every job that is not a CPU
        INT4-eligible generated attempt's default candidate (index 0) under
        a still-PENDING lineage: static recipes, legacy generated attempts
        without a registered policy, and the fallback candidate's own job
        (which never runs Mobius at all, so this hook is never even invoked
        for it) all leave `self._pre_olive_descriptors` untouched. Any
        failure here (a malformed payload, a symlink/reparse point, tampered
        content) is swallowed: it only ever means a later fallback trigger
        finds no usable descriptor and correctly refuses to retry, never
        that this build itself fails.
        """
        generated_attempt = job.request.generated_recipe_attempt
        if generated_attempt is None:
            return
        attempt_id = generated_attempt.attempt_id.strip()
        if not attempt_id:
            return
        try:
            candidate = self._recipe_attempt_store.find_candidate_attempt_by_attempt_id(attempt_id)
        except RecipeAttemptStoreError:
            return
        if candidate is None or candidate.candidate_index != 0 or candidate.parent_attempt_id != attempt_id:
            return
        try:
            lineage = self._recipe_attempt_store.get_candidate_lineage(attempt_id)
        except RecipeAttemptStoreError:
            return
        if lineage is None or lineage.selection_state != CandidateLineageSelectionState.PENDING:
            return
        generated_record = self._recipe_attempt_store.get_generated_recipe(generated_attempt.recipe_fingerprint)
        if generated_record is None:
            return
        try:
            default_candidate_recipe = self._recompile_generated_recipe_record(record=generated_record)
        except ServiceError:
            return
        identity = pre_olive_generation_identity_from_generated_record(generated_record)
        try:
            descriptor = capture_pre_olive_artifact(
                mobius_source_dir=mobius_dir,
                authorized_root=job.request.workspace_root,
                generation_identity=identity,
                mobius_args=default_candidate_recipe.recipe.mobius,
                source_attempt_id=attempt_id,
                source_candidate_id=candidate.candidate_attempt_id,
            )
        except PreOliveReuseError:
            return
        with self._lock:
            self._pre_olive_descriptors[attempt_id] = descriptor

    def _on_candidate_attempt_terminal(self, *, job: BuildJob, attempt_id: str) -> CandidateAttemptRecord | None:
        """Idempotently record terminal evidence/counters for `attempt_id`'s
        candidate-plan row -- if `attempt_id` is part of any Slice 3B1
        candidate lineage at all -- and return that row. Returns ``None``
        when `attempt_id` is not a tracked candidate, which every caller
        uses as the uniform signal to leave the pre-3B1 behavior completely
        untouched (static recipes and legacy generated attempts without a
        registered policy)."""
        try:
            candidate = self._recipe_attempt_store.find_candidate_attempt_by_attempt_id(attempt_id)
        except RecipeAttemptStoreError:
            return None
        if candidate is None:
            return None
        self._finalize_candidate_terminal_evidence(job=job, candidate=candidate)
        return candidate

    def _finalize_candidate_terminal_evidence(self, *, job: BuildJob, candidate: CandidateAttemptRecord) -> None:
        """Persist real, actual (never inferred/constant) invocation
        counters plus compact artifact/package references for one terminal
        candidate attempt. Write-once and idempotent: a repeated call with
        identical values (e.g. duplicate worker delivery re-syncing an
        already-terminal candidate) is always safe."""
        counters: CandidateInvocationCounters | None = None
        if job.production_invocation_evidence is not None:
            counters = production_invocation_evidence_to_candidate_counters(job.production_invocation_evidence)
        artifact_ref = f"job://{job.job_id}/artifact/{job.result_artifact_id}" if job.result_artifact_id else None
        package_ref = f"job://{job.job_id}/package" if job.result_artifact_id else None
        try:
            self._recipe_attempt_store.finalize_candidate_attempt_evidence(
                candidate.candidate_attempt_id,
                artifact_ref=artifact_ref,
                package_ref=package_ref,
                invocation_counters=counters,
            )
        except RecipeAttemptStoreError:
            return

    def _finalize_lineage_exhausted_if_applicable(self, *, candidate: CandidateAttemptRecord, reason: str) -> None:
        """Finalize `candidate`'s lineage exhausted -- no winner -- unless it
        is already finalized (selected or exhausted), which makes every call
        site safe to call unconditionally/idempotently without first
        re-deriving whether exhaustion is legal; the store itself enforces
        every remaining precondition (every candidate terminal, none
        verified-but-unselected) and this simply no-ops on any conflict
        rather than raising into the sync path."""
        try:
            lineage = self._recipe_attempt_store.get_candidate_lineage(candidate.parent_attempt_id)
        except RecipeAttemptStoreError:
            return
        if lineage is None or lineage.selection_state != CandidateLineageSelectionState.PENDING:
            return
        try:
            self._recipe_attempt_store.finalize_exhausted_candidate_lineage(
                candidate.parent_attempt_id,
                reason=reason,
            )
        except RecipeAttemptStoreError:
            return

    def _select_verified_candidate(
        self,
        *,
        candidate: CandidateAttemptRecord,
        generated_record: Any,
    ) -> None:
        """Atomically select `candidate` as its lineage's single verified
        winner, with real target device/EP/toolchain/environment scope
        derived from the candidate's own actually-compiled recipe and
        persisted generation identity -- never left implicitly "verified for
        every scope" by omission (see
        `CandidateAttemptRecord.has_fully_validated_selection_scope`)."""
        try:
            lineage = self._recipe_attempt_store.get_candidate_lineage(candidate.parent_attempt_id)
        except RecipeAttemptStoreError:
            return
        if lineage is None or lineage.selection_state != CandidateLineageSelectionState.PENDING:
            return
        payload = generated_record.payload()
        recipe_payload = payload.get("recipe") if isinstance(payload, dict) else None
        olive_payload = recipe_payload.get("olive") if isinstance(recipe_payload, dict) else None
        mobius_payload = recipe_payload.get("mobius") if isinstance(recipe_payload, dict) else None
        validated_target_device = (
            str(olive_payload.get("device"))
            if isinstance(olive_payload, dict) and olive_payload.get("device")
            else None
        )
        validated_target_ep = (
            str(mobius_payload.get("ep"))
            if isinstance(mobius_payload, dict) and mobius_payload.get("ep")
            else None
        )
        validated_toolchain_fingerprint = generated_record.toolchain_fingerprint
        validated_environment_scope = (
            f"foundry-local-onboarding:{olive_payload.get('provider')}"
            if isinstance(olive_payload, dict) and olive_payload.get("provider")
            else "foundry-local-onboarding"
        )
        try:
            self._recipe_attempt_store.select_verified_candidate_attempt(
                parent_attempt_id=candidate.parent_attempt_id,
                candidate_attempt_id=candidate.candidate_attempt_id,
                reason=(
                    f"Candidate {candidate.candidate_index} ('{candidate.candidate_id}') verified: recipe "
                    "integrity passed strict baseline/optimized quality validation."
                ),
                validated_target_device=validated_target_device,
                validated_target_ep=validated_target_ep,
                validated_toolchain_fingerprint=validated_toolchain_fingerprint,
                validated_environment_scope=validated_environment_scope,
            )
        except RecipeAttemptStoreError:
            return

    def _maybe_launch_fallback_candidate(
        self,
        *,
        job: BuildJob,
        candidate: CandidateAttemptRecord,
        quality_outcome: QualityValidationOutcome,
    ) -> bool:
        """Evaluate every Slice 3B1 fallback-trigger precondition and, only
        if every single one holds, compile/register/launch exactly one
        trusted block64 fallback candidate. Returns ``True`` once the
        fallback candidate has been durably registered and enqueued (the
        caller must leave the lineage PENDING); ``False`` whenever any
        precondition fails, unknown/malformed/partial evidence included --
        the caller then finalizes the lineage exhausted instead."""
        if candidate.candidate_index != 0:
            # Only the default candidate may ever trigger a fallback.
            return False
        try:
            lineage = self._recipe_attempt_store.get_candidate_lineage(candidate.parent_attempt_id)
        except RecipeAttemptStoreError:
            return False
        if lineage is None or lineage.selection_state != CandidateLineageSelectionState.PENDING:
            return False
        if lineage.policy_max_candidates <= 1:
            return False
        retry_evaluation = quality_outcome.quality_retry_evaluation
        if retry_evaluation is None or not retry_evaluation.is_retryable:
            # Unknown/malformed/partial evidence, or any disposition other than
            # the sole allowlisted retryable one, is always non-retryable.
            return False
        if job.state != JobState.SUCCEEDED:
            # Defense in depth: Mobius, Olive, ONNX/ORT/OGA/FL runtime must all
            # have actually succeeded; only a quality structural regression may
            # ever block promotion for a retryable default candidate.
            return False
        generated_record = self._recipe_attempt_store.get_generated_recipe(candidate.recipe_fingerprint)
        if generated_record is None:
            return False
        descriptor = self._pre_olive_descriptors.get(candidate.parent_attempt_id)
        if descriptor is None:
            # No captured pre-Olive descriptor available (never captured, lost
            # across a restart, or already consumed) -- never rebuild Mobius to
            # recover one; fail closed to "not retryable".
            return False
        try:
            revalidate_pre_olive_source(descriptor)
        except PreOliveReuseError:
            return False
        try:
            default_candidate_recipe = self._recompile_generated_recipe_record(record=generated_record)
        except ServiceError:
            return False

        # `revalidate_pre_olive_source` above can take an unbounded amount of
        # wall-clock time hashing a potentially many-GB Mobius artifact, and
        # deliberately runs without holding `self._lock` (see
        # `_safe_sync_generated_attempt`). Re-check every precondition that
        # could have changed while we were outside the lock -- the job's own
        # cancellation signal, the candidate's linked attempt, and the
        # lineage's selection state -- immediately before committing to
        # consume the descriptor and launch the fallback candidate, so a
        # cancellation or any other concurrent mutation that landed during
        # the revalidation can never be silently raced past.
        cancellation_event = self._cancel_events.get(job.job_id)
        if cancellation_event is not None and cancellation_event.is_set():
            return False
        try:
            refreshed_candidate = self._recipe_attempt_store.get_candidate_attempt(candidate.candidate_attempt_id)
        except (KeyError, RecipeAttemptStoreError):
            return False
        if refreshed_candidate.attempt_state not in TERMINAL_ATTEMPT_STATES or refreshed_candidate.is_verified:
            # The default candidate's own terminal, non-verified quality-gate
            # failure -- the precondition that got us here in the first
            # place -- must still hold unchanged.
            return False
        try:
            refreshed_lineage = self._recipe_attempt_store.get_candidate_lineage(candidate.parent_attempt_id)
        except RecipeAttemptStoreError:
            return False
        if refreshed_lineage is None or refreshed_lineage.selection_state != CandidateLineageSelectionState.PENDING:
            return False

        # Every precondition holds; consume the descriptor only now that we
        # are committed to actually launching the fallback candidate. The
        # final cancellation re-check and descriptor consumption happen
        # together under `self._lock` so a cancellation landing in the
        # narrow window above cannot race past this commit point either.
        with self._lock:
            cancellation_event = self._cancel_events.get(job.job_id)
            if cancellation_event is not None and cancellation_event.is_set():
                return False
            current_descriptor = self._pre_olive_descriptors.get(candidate.parent_attempt_id)
            if current_descriptor is not descriptor:
                # Consumed, replaced, or evicted by another actor while this
                # revalidation was outstanding; never launch against a stale
                # descriptor reference.
                return False
            self._pre_olive_descriptors.pop(candidate.parent_attempt_id, None)
        try:
            self._launch_fallback_candidate_attempt(
                parent_attempt_id=candidate.parent_attempt_id,
                default_record=generated_record,
                default_candidate_recipe=default_candidate_recipe,
                descriptor=descriptor,
                retry_evaluation=retry_evaluation,
            )
        except (
            CandidatePlanValidationError,
            TrustedCandidateCompilationError,
            GeneratedRecipeCompileError,
            RecipeAttemptStoreError,
            ServiceError,
        ):
            return False
        return True

    def _launch_fallback_candidate_attempt(
        self,
        *,
        parent_attempt_id: str,
        default_record: Any,
        default_candidate_recipe: GeneratedRecipe,
        descriptor: PreOliveArtifactDescriptor,
        retry_evaluation: QualityRetryEvaluation,
    ) -> None:
        """Compile, persist, register, and enqueue the single trusted
        block64 fallback candidate (candidate index 1) the approved CPU
        INT4 selection policy permits for `parent_attempt_id`'s default
        candidate. Idempotent: if candidate 1 is already registered or
        already has a live build job for `parent_attempt_id` (a duplicate
        worker delivery or a defensive re-entrant call), this returns
        without creating a second candidate/attempt/job or launching any
        tool twice.
        """
        policy = self._recipe_selection_policy
        if any(row.candidate_index == 1 for row in self._recipe_attempt_store.list_candidate_attempts(parent_attempt_id)):
            return

        fallback_recipe = compile_trusted_candidate_recipe(
            default_candidate_recipe,
            policy=policy,
            candidate=policy.candidates[1],
        )
        fallback_record = self._recipe_attempt_store.upsert_generated_recipe(fallback_recipe)

        attempt_request = build_attempt_request_from_generated(fallback_record)
        request_fingerprint = build_attempt_request_fingerprint(attempt_request)
        # Deterministic per-parent idempotency key: repeated worker delivery for
        # the same parent attempt always resolves to the exact same fallback
        # attempt row, never a second one.
        idempotency_key = f"fallback-candidate-1::{parent_attempt_id}"
        fallback_attempt, _replay = self._recipe_attempt_store.create_attempt(
            idempotency_key=idempotency_key,
            request=attempt_request,
            request_fingerprint=request_fingerprint,
        )

        with self._lock:
            existing_job_id = self._attempt_to_build_job.get(fallback_attempt.attempt_id)
            already_launched = existing_job_id is not None and existing_job_id in self._jobs
        if already_launched:
            return

        if fallback_attempt.state == AttemptState.GENERATED:
            self._recipe_attempt_store.start_attempt(fallback_attempt.attempt_id)

        if not any(
            row.candidate_index == 1
            for row in self._recipe_attempt_store.list_candidate_attempts(parent_attempt_id)
        ):
            self._recipe_attempt_store.register_candidate_attempt(
                parent_attempt_id=parent_attempt_id,
                attempt_id=fallback_attempt.attempt_id,
                candidate_index=1,
                policy=policy,
                quality_profile_fingerprint=self._quality_profile.fingerprint,
                trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
                retry_evaluation=retry_evaluation,
            )

        with self._lock:
            existing_job_id = self._attempt_to_build_job.get(fallback_attempt.attempt_id)
            if existing_job_id is not None and existing_job_id in self._jobs:
                return
            job_id = str(uuid.uuid4())
            request = self._build_request_for_generated_attempt(
                record=fallback_record,
                attempt_id=fallback_attempt.attempt_id,
                job_id=job_id,
            )
            job = BuildJob(job_id=job_id, request=request)
            job.add_event(
                "Trusted block64 fallback candidate queued after a retryable optimized-only structural "
                "regression on the default candidate; reusing the default candidate's captured pre-Olive "
                "Mobius artifact instead of rebuilding it."
            )
            self._jobs[job_id] = job
            self._cancel_events[job_id] = Event()
            self._build_job_to_attempt[job_id] = fallback_attempt.attempt_id
            self._attempt_to_build_job[fallback_attempt.attempt_id] = job_id
            self._fallback_launch_context[job_id] = (
                descriptor,
                pre_olive_generation_identity_from_generated_record(fallback_record),
            )
            self._store.save_job(job)
            self._queue.put(job_id)

    def _recover_orphaned_candidate_lineages(self) -> None:
        """Restart-time fail-closed recovery.

        Finalizes as exhausted any PENDING candidate lineage whose every
        registered candidate has already reached a terminal, non-verified
        state before this restart -- regardless of whether that candidate
        count is below or exactly at the policy's `policy_max_candidates`
        (a crash between committing the last candidate's terminal state and
        this exact call finalizing the lineage must not orphan it forever:
        the parent attempt is itself already terminal by then, so the
        ordinary lazy-sync path -- a `get_recipe_attempt` poll -- never
        revisits it again to heal it). The in-memory pre-Olive descriptor
        required to safely resume a fallback candidate (see
        `self._pre_olive_descriptors`) never survives a restart, so resuming
        by silently rebuilding Mobius is never attempted here regardless of
        candidate count; the lineage is instead closed with an actionable
        reason.

        A verified-but-unselected candidate (its linked attempt reached
        `AttemptState.SUCCEEDED` but a crash landed between that commit and
        `select_verified_candidate_attempt`) is never exhausted/discarded:
        recovery instead selects it here using the exact same trusted
        selection logic (`_select_verified_candidate`) the live sync path
        uses. If the generated recipe backing it cannot be located, the
        lineage is left PENDING -- an explicit, still-actionable state --
        rather than ever exhausting a verified winner out from under it (the
        store itself also independently fails closed against this: see
        `RecipeAttemptStore.finalize_exhausted_candidate_lineage`).

        A lineage that is still genuinely in flight (any candidate not yet
        terminal) is left completely untouched -- the ordinary lazy-sync
        path (a `get_recipe_attempt` poll, or the interrupted-job recovery
        pass immediately above this call) finalizes it exactly once doing so
        is safe, and may still launch the fallback candidate normally if that
        candidate's own gates were never reached. An already-finalized
        lineage (`SELECTED`/`EXHAUSTED`) is always a no-op, making repeated
        restart recovery idempotent.
        """
        seen_parents: set[str] = set()
        for job in self._jobs.values():
            generated_attempt = job.request.generated_recipe_attempt
            if generated_attempt is None:
                continue
            parent_attempt_id = generated_attempt.attempt_id.strip()
            if not parent_attempt_id or parent_attempt_id in seen_parents:
                continue
            seen_parents.add(parent_attempt_id)
            self._recover_one_orphaned_candidate_lineage(parent_attempt_id)

    def _recover_one_orphaned_candidate_lineage(self, parent_attempt_id: str) -> None:
        try:
            lineage = self._recipe_attempt_store.get_candidate_lineage(parent_attempt_id)
        except RecipeAttemptStoreError:
            return
        if lineage is None or lineage.selection_state != CandidateLineageSelectionState.PENDING:
            # No lineage at all, or already finalized (selected/exhausted):
            # always a no-op, which is exactly what makes repeated restart
            # recovery idempotent.
            return
        try:
            candidates = self._recipe_attempt_store.list_candidate_attempts(parent_attempt_id)
        except RecipeAttemptStoreError:
            return
        if not candidates:
            return
        if any(row.attempt_state not in TERMINAL_ATTEMPT_STATES for row in candidates):
            # Still genuinely in flight; leave it completely untouched.
            return

        verified = [row for row in candidates if row.is_verified]
        if verified:
            # Never discard a verified candidate by exhausting the lineage
            # around it. Recover it using the exact same trusted selection
            # logic the live sync path uses; at most one candidate in a
            # lineage can ever be verified (a candidate's own attempt only
            # ever runs after every earlier candidate has already reached a
            # terminal, non-verified state), so selecting the first (only)
            # verified row is unambiguous.
            candidate = verified[0]
            generated_record = self._recipe_attempt_store.get_generated_recipe(candidate.recipe_fingerprint)
            if generated_record is None:
                # Cannot safely re-derive the validated selection scope
                # without the generated recipe; leave the lineage PENDING
                # (explicit and still actionable via lazy sync) rather than
                # ever exhausting a verified winner.
                return
            self._select_verified_candidate(candidate=candidate, generated_record=generated_record)
            return

        if len(candidates) < lineage.policy_max_candidates:
            reason = (
                "restart_fail_closed: a fallback candidate was never registered before a service "
                "restart, and the in-memory pre-Olive artifact descriptor required to resume it "
                "safely does not survive a restart; refusing to silently rebuild Mobius to resume."
            )
        else:
            reason = (
                "restart_recovery: every registered candidate in this lineage reached a terminal, "
                "non-verified state before a service restart interrupted lineage finalization; "
                "finalizing exhausted now that recovery has confirmed no candidate can be verified."
            )
        try:
            self._recipe_attempt_store.finalize_exhausted_candidate_lineage(
                parent_attempt_id,
                reason=reason,
            )
        except RecipeAttemptStoreError:
            # Leaves the lineage exactly as it was (still PENDING): a
            # transient store/transaction failure here never partially
            # commits, and the next restart's recovery pass (or a future
            # lazy-sync heal) simply retries from the same consistent state.
            return

    def _sync_generated_attempt_with_job(self, *, job: BuildJob) -> None:
        attempt_id = self._attempt_id_for_job(job.job_id)
        if attempt_id is None:
            return
        try:
            attempt = self._recipe_attempt_store.get_attempt(attempt_id)
        except KeyError:
            return
        if attempt.state in {AttemptState.FAILED, AttemptState.CANCELLED}:
            return

        if job.state == JobState.CANCELLED:
            self._record_failure_gates(attempt_id=attempt_id, job=job)
            self._finalize_attempt_cancelled(
                attempt_id=attempt_id,
                job=job,
                reason=(job.failure.message if job.failure is not None else "Build cancelled."),
            )
            lineage_candidate = self._on_candidate_attempt_terminal(job=job, attempt_id=attempt_id)
            if lineage_candidate is not None:
                self._finalize_lineage_exhausted_if_applicable(
                    candidate=lineage_candidate,
                    reason="Candidate attempt cancelled; cancellation stops the entire candidate lineage.",
                )
            return

        if job.state == JobState.FAILED:
            failure_gate = self._record_failure_gates(attempt_id=attempt_id, job=job)
            failure_message = (
                job.failure.message
                if job.failure is not None
                else f"Build failed during {failure_gate.value}."
            )
            source_owner = (
                "upstream-model"
                if (
                    job.failure is not None
                    and job.failure.classification
                    == FailureClassification.SOURCE_RUNTIME_CONTRACT_INCOMPATIBLE
                )
                else "fl-onboarding"
            )
            classification = AttemptFailureClassification.GATE_FAILED
            next_action = "Inspect the failed gate evidence and rerun with a fresh generated-attempt idempotency key."
            if job.failure is not None and (
                job.failure.classification == FailureClassification.INVALID_REQUEST
                or job.failure.stage in {JobState.PREFLIGHT, JobState.DOWNLOADING}
            ):
                classification = AttemptFailureClassification.VALIDATION_FAILED
                source_owner = "fl-onboarding"
                next_action = (
                    "Resolve generated-attempt identity/preflight mismatch and rerun with a fresh "
                    "generated-attempt idempotency key."
                )
            self._finalize_attempt_failed(
                attempt_id=attempt_id,
                classification=classification,
                job=job,
                message=failure_message,
                source_owner=source_owner,
                next_action=next_action,
            )
            # A generic gate failure (Mobius/Olive/runtime/preflight/etc.) never
            # triggers a fallback: the trigger requires Mobius, Olive, and every
            # runtime gate to have already succeeded, with only the quality
            # structural regression blocking promotion.
            lineage_candidate = self._on_candidate_attempt_terminal(job=job, attempt_id=attempt_id)
            if lineage_candidate is not None:
                self._finalize_lineage_exhausted_if_applicable(
                    candidate=lineage_candidate,
                    reason=(
                        "Candidate attempt failed before quality validation could run; no retry trigger "
                        "applies."
                    ),
                )
            return

        if job.state != JobState.SUCCEEDED:
            return

        if attempt.state == AttemptState.RUNNING:
            self._record_success_gates(attempt_id=attempt_id, job=job)
            artifact = next(
                (row for row in job.artifacts if row.artifact_id == job.result_artifact_id),
                None,
            )
            quality_outcome = self._run_quality_validation(job=job, artifact=artifact)
            self._record_attempt_gate(
                attempt_id=attempt_id,
                gate=AttemptGate.QUALITY_VALIDATION,
                status=quality_outcome.gate_status,
                evidence_ref=quality_outcome.evidence_ref,
                metrics_ref=quality_outcome.metrics_ref,
            )
            if not quality_outcome.passed:
                next_action = "Resolve quality prompt failures before attempting promotion."
                if quality_outcome.gate_status in {
                    AttemptGateStatus.NOT_RUN,
                    AttemptGateStatus.UNAVAILABLE,
                }:
                    next_action = (
                        "Ensure a pre-Olive Mobius baseline package can run deterministic prompt "
                        "validation before retrying promotion."
                    )
                self._finalize_attempt_failed(
                    attempt_id=attempt_id,
                    classification=AttemptFailureClassification.VALIDATION_FAILED,
                    job=job,
                    message=quality_outcome.message,
                    source_owner="fl-onboarding",
                    next_action=next_action,
                )
                lineage_candidate = self._on_candidate_attempt_terminal(job=job, attempt_id=attempt_id)
                if lineage_candidate is not None:
                    triggered = self._maybe_launch_fallback_candidate(
                        job=job,
                        candidate=lineage_candidate,
                        quality_outcome=quality_outcome,
                    )
                    if not triggered:
                        self._finalize_lineage_exhausted_if_applicable(
                            candidate=lineage_candidate,
                            reason=(
                                "Quality validation failed and no retryable fallback trigger applied; the "
                                "default candidate's own failure is preserved unchanged."
                            ),
                        )
                return

            try:
                self._recipe_attempt_store.finish_attempt_succeeded(attempt_id)
            except AttemptStateTransitionError:
                refreshed = self._recipe_attempt_store.get_attempt(attempt_id)
                if refreshed.state == AttemptState.RUNNING:
                    raise ServiceError(
                        code="RECIPE_ATTEMPT_STORE_ERROR",
                        message=(
                            f"Attempt '{attempt_id}' could not transition to succeeded while build "
                            f"'{job.job_id}' is terminal."
                        ),
                        status_code=500,
                    )
                if refreshed.state in {AttemptState.FAILED, AttemptState.CANCELLED}:
                    return
            except RecipeAttemptStoreError as exc:
                raise ServiceError(
                    code="RECIPE_ATTEMPT_STORE_ERROR",
                    message=_sanitize_text(str(exc)),
                    status_code=500,
                ) from exc
            else:
                refreshed = self._recipe_attempt_store.get_attempt(attempt_id)
        elif attempt.state == AttemptState.SUCCEEDED:
            refreshed = attempt
        else:
            return
        generated_record = self._recipe_attempt_store.get_generated_recipe(refreshed.recipe_fingerprint)
        if generated_record is None:
            self._finalize_attempt_failed(
                attempt_id=attempt_id,
                classification=AttemptFailureClassification.INTERNAL_ERROR,
                job=job,
                message=f"Generated recipe '{refreshed.recipe_fingerprint}' was missing at promotion time.",
                source_owner="fl-onboarding",
                next_action="Recompile and persist the generated recipe before retrying promotion.",
            )
            return

        lineage_candidate = self._on_candidate_attempt_terminal(job=job, attempt_id=attempt_id)
        if lineage_candidate is not None:
            self._select_verified_candidate(candidate=lineage_candidate, generated_record=generated_record)

        candidate = self._recompile_generated_recipe_record(record=generated_record)
        evidence = self._promotion_evidence_from_attempt(attempt=refreshed)
        promoted = promote_generated_recipe(
            candidate,
            evidence,
            new_version="1.0.0",
            status_reason=(
                "Verified recipe promoted after deterministic generated-attempt gates passed, "
                "with recipe integrity verified and model capability advisory recorded in quality evidence."
            ),
        )
        self._recipe_attempt_store.promote_verified_recipe(
            attempt_id=attempt_id,
            promoted_recipe=promoted,
        )
        if job.result_artifact_id is not None:
            self._store.save_tested_model(
                model_id=job.request.candidate.huggingface_model_id,
                task=job.request.candidate.modality,
                artifact_id=job.result_artifact_id,
                revision=job.request.hf_revision,
                task_profile=job.request.task_profile,
            )

    def _acquire_attempt_sync_guard(self, attempt_id: str) -> _AttemptSyncGuard:
        """Return the shared per-attempt guard for `attempt_id`, creating it
        under a brief `self._lock` critical section if this is the first
        concurrent caller. Every caller must pair this with exactly one
        `_release_attempt_sync_guard` call (in a `finally` block) once it is
        done using the returned guard's `.lock`."""
        with self._lock:
            guard = self._attempt_sync_guards.get(attempt_id)
            if guard is None:
                guard = _AttemptSyncGuard()
                self._attempt_sync_guards[attempt_id] = guard
            guard.waiters += 1
            return guard

    def _release_attempt_sync_guard(self, attempt_id: str, guard: _AttemptSyncGuard) -> None:
        """Drop this caller's reference to `guard` and, if it was the last
        live reference, remove it from `self._attempt_sync_guards` so that
        map never grows without bound across a long-running service's
        lifetime. Safe even if another caller concurrently created a *new*
        guard for the same `attempt_id` in between: the identity check below
        only ever deletes the exact guard object this caller released."""
        with self._lock:
            guard.waiters -= 1
            if guard.waiters == 0 and self._attempt_sync_guards.get(attempt_id) is guard:
                del self._attempt_sync_guards[attempt_id]

    def _safe_sync_generated_attempt(self, *, job: BuildJob) -> None:
        # Locking order/design (Slice 3B1 revision): a brief `self._lock`
        # critical section only ever creates/looks up/tears down the
        # per-attempt guard below; the (potentially slow) sync body itself
        # runs holding only that per-attempt `threading.Lock`, never the
        # service-wide `self._lock`. `_sync_generated_attempt_with_job` and
        # its callees (candidate lineage bookkeeping,
        # `_launch_fallback_candidate_attempt`) take `self._lock` themselves,
        # briefly, only around actual shared in-memory map/queue mutations --
        # always nested *inside* an already-held per-attempt guard, never the
        # reverse. No call site ever holds `self._lock` while blocking to
        # acquire a per-attempt guard (every caller of this method releases
        # `self._lock` first -- see `cancel_build` and `_run_job`), so there
        # is no lock-order inversion between the two: a thread can never be
        # holding `self._lock` and waiting on a per-attempt guard while a
        # second thread holds that guard and waits on `self._lock`. This
        # keeps unrelated `get_build`/`get_recipe_attempt`/`cancel_build`/new-
        # submission calls -- which only ever need brief `self._lock`
        # sections -- responsive while one attempt's sync revalidates a
        # potentially many-GB pre-Olive artifact or otherwise performs
        # unbounded I/O (manifest hashing, copying, process execution,
        # quality validation).
        #
        # A job with no generated-attempt mapping (a static/legacy recipe)
        # needs no serialization at all: `_sync_generated_attempt_with_job`
        # itself no-ops immediately for it, so skip the guard entirely.
        attempt_id = self._attempt_id_for_job(job.job_id)
        if attempt_id is None:
            try:
                self._sync_generated_attempt_with_job(job=job)
            except ServiceError as exc:
                job.add_event(f"Recipe attempt synchronization error: {_sanitize_text(str(exc))}")
                self._store.save_job(job)
            return

        guard = self._acquire_attempt_sync_guard(attempt_id)
        try:
            with guard.lock:
                try:
                    self._sync_generated_attempt_with_job(job=job)
                except ServiceError as exc:
                    job.add_event(f"Recipe attempt synchronization error: {_sanitize_text(str(exc))}")
                    self._store.save_job(job)
        finally:
            self._release_attempt_sync_guard(attempt_id, guard)

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
        # Outside `self._lock`: see `_safe_sync_generated_attempt` for why it
        # must never be called while holding the service-wide lock -- a
        # concurrent per-attempt sync already in flight for a *different*
        # attempt (e.g. revalidating a pre-Olive artifact) must never stall
        # this or any other `self._lock` user until it finishes.
        self._safe_sync_generated_attempt(job=job)
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
        self._record_successful_inference(job=job, artifact_id=artifact_id)
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
        self._record_successful_inference(job=job, artifact_id=artifact_id)
        return {"artifact_id": artifact_id, "transcript": _sanitize_text(transcript)}

    def _record_successful_inference(self, *, job: BuildJob, artifact_id: str) -> None:
        attempt_id = self._attempt_id_for_job(job.job_id)
        if attempt_id is not None:
            promoted = any(
                record.attempt_id == attempt_id
                for record in self._recipe_attempt_store.list_verified_recipes()
            )
            if not promoted:
                return
        self._store.save_tested_model(
            model_id=job.request.candidate.huggingface_model_id,
            task=job.request.candidate.modality,
            artifact_id=artifact_id,
            revision=job.request.hf_revision,
            task_profile=job.request.task_profile,
        )

    def _verification_for_model(self, model_id: str) -> dict[str, object]:
        record = next(
            (item for item in self._store.load_tested_models() if item["model_id"] == model_id),
            None,
        )
        if record is None:
            return {
                "status": "not_verified",
                "evidence": "none",
                "artifact_id": None,
                "verified_utc": None,
            }
        return {
            "status": "tested",
            "evidence": record["evidence"],
            "artifact_id": record["artifact_id"],
            "verified_utc": record["verified_utc"],
        }

    def _recipe_payload(self, match: RecipeResolution) -> dict[str, object] | None:
        if match.recipe is None:
            return None
        payload = self._recipe_registry.describe_recipe(match.recipe)
        payload["status"] = match.status
        payload["reason"] = match.reason
        payload["requires_experimental_opt_in"] = match.requires_experimental_opt_in
        return payload

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
                try:
                    self._run_job(job_id)
                except Exception as exc:
                    failed_job: BuildJob | None = None
                    with self._lock:
                        job = self._jobs.get(job_id)
                        if job is not None and job.state not in {
                            JobState.SUCCEEDED,
                            JobState.FAILED,
                            JobState.CANCELLED,
                        }:
                            failed_stage = job.state
                            job.state = JobState.FAILED
                            job.failure = FailureInfo(
                                stage=failed_stage,
                                classification=FailureClassification.UNKNOWN,
                                message=_sanitize_text(str(exc) or "Unexpected worker failure."),
                            )
                            job.finished_utc = datetime.now(timezone.utc)
                            job.add_event("Unexpected worker failure.")
                            self._store.save_job(job)
                            failed_job = job
                    if failed_job is not None:
                        # Outside `self._lock`: see `_safe_sync_generated_attempt`.
                        self._safe_sync_generated_attempt(job=failed_job)
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
            # Outside `self._lock`: see `_safe_sync_generated_attempt`.
            self._safe_sync_generated_attempt(job=live)
            return

        needs_terminal_sync = False
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
                needs_terminal_sync = True
            else:
                pinned_revision = preflight.huggingface_sha or preflight.huggingface_revision
                if pinned_revision:
                    live.request = replace(live.request, hf_revision=pinned_revision)
                    self._store.save_job(live)
                cancellation_event = self._cancel_events.setdefault(job_id, Event())
                if cancellation_event.is_set() or live.state == JobState.CANCELLED:
                    needs_terminal_sync = True

        if needs_terminal_sync:
            # Outside `self._lock`: see `_safe_sync_generated_attempt`.
            self._safe_sync_generated_attempt(job=live)
            return

        def persist() -> None:
            with self._lock:
                for artifact in live.artifacts:
                    self._artifact_to_job[artifact.artifact_id] = live.job_id
                self._store.save_job(live)

        # Slice 3B1: a fallback candidate job dispatches through
        # `run_fallback_with_pre_olive_reuse` instead of the ordinary `run()`,
        # reusing the default candidate's captured pre-Olive Mobius artifact
        # instead of invoking Mobius again. Every other job (static recipe,
        # legacy generated attempt, or a CPU INT4-eligible default candidate)
        # is completely unaffected.
        with self._lock:
            fallback_context = self._fallback_launch_context.pop(job_id, None)
        if fallback_context is not None:
            descriptor, fallback_identity = fallback_context
            runner = self._build_stage_runner
            if isinstance(runner, ProductionBuildStageRunner):
                runner.run_fallback_with_pre_olive_reuse(
                    live,
                    descriptor=descriptor,
                    fallback_generation_identity=fallback_identity,
                    persist=persist,
                    cancellation_event=cancellation_event,
                )
            else:
                fail_job(
                    live,
                    FailureInfo(
                        stage=live.state,
                        classification=FailureClassification.NOT_VERIFIED,
                        message=(
                            "Fallback candidate execution requires the production build stage runner."
                        ),
                    ),
                )
                live.finished_utc = datetime.now(timezone.utc)
                persist()
        else:
            self._build_stage_runner.run(
                live,
                persist=persist,
                cancellation_event=cancellation_event,
            )
        self._safe_sync_generated_attempt(job=live)
        with self._lock:
            is_generated_attempt = self._attempt_id_for_job(live.job_id) is not None
            inference_verified = any(
                validation.stage == JobState.INFERENCING
                and validation.status == ValidationStatus.PASSED
                for validation in live.validations
            )
            if (
                live.state == JobState.SUCCEEDED
                and live.result_artifact_id
                and inference_verified
                and not is_generated_attempt
            ):
                self._record_successful_inference(
                    job=live,
                    artifact_id=live.result_artifact_id,
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
                "recipe_id": request.recipe_id,
                "recipe_version": request.recipe_version,
                "recipe_status": request.recipe_status,
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

    def _build_request(
        self,
        submission: BuildSubmission,
        *,
        job_id: str,
        recipe_match: RecipeResolution | None = None,
        enforce_recipe_buildable: bool = True,
    ) -> BuildRequest:
        workspace = workspace_root_for_job(job_id=job_id, base_dir=self._workspace_base)
        ensure_dir(workspace)
        output_dir = ensure_dir(workspace / "output")
        resolved = recipe_match or self._recipe_registry.resolve(
            model_id=submission.model_id,
            modality=submission.task,
            task_profile=submission.task_profile,
            allow_experimental=submission.allow_experimental,
        )
        if enforce_recipe_buildable and not resolved.buildable:
            raise ServiceError(
                code=_recipe_error_code(resolved),
                message=resolved.reason,
                status_code=400,
                detail={"recipe_status": resolved.status},
            )

        recipe = resolved.recipe
        if recipe is not None:
            default_aliases = {
                "",
                "default",
                f"{submission.task.value}-cpu-default",
            }
            task_profile = (
                recipe.task_profile
                if submission.task_profile.strip().lower() in default_aliases
                else submission.task_profile
            )
            if task_profile.lower() != recipe.task_profile.lower():
                raise ServiceError(
                    code="UNSUPPORTED_TASK_PROFILE",
                    message=(
                        f"Task profile '{task_profile}' is not supported by recipe '{recipe.id}'. "
                        f"Expected '{recipe.task_profile}'."
                    ),
                    status_code=400,
                )
            task_profile = recipe.task_profile
            skip_olive = submission.skip_olive
            selected = recipe.choice_for_profile(task_profile, skip_olive)
            if (
                selected is None
                and submission.task_profile.strip().lower() in default_aliases
                and not submission.skip_olive
            ):
                selected = recipe.default_optimization()
                if selected is not None:
                    task_profile = selected.task_profile
                    skip_olive = selected.skip_olive
            if selected is None and recipe.optimization_choices:
                raise ServiceError(
                    code="UNSUPPORTED_OPTIMIZATION",
                    message=(
                        f"Recipe '{recipe.id}' does not support task_profile={task_profile} "
                        f"with skip_olive={skip_olive}."
                    ),
                    status_code=400,
                )
            if submission.optimization_strategy and selected is not None:
                if submission.optimization_strategy.lower() != selected.strategy.lower():
                    raise ServiceError(
                        code="UNSUPPORTED_OPTIMIZATION",
                        message=(
                            f"Optimization strategy '{submission.optimization_strategy}' is not supported "
                            f"for recipe '{recipe.id}'."
                        ),
                        status_code=400,
                    )
            if submission.optimization_precision and selected is not None:
                if submission.optimization_precision.lower() != selected.precision.lower():
                    raise ServiceError(
                        code="UNSUPPORTED_OPTIMIZATION",
                        message=(
                            f"Optimization precision '{submission.optimization_precision}' is not supported "
                            f"for recipe '{recipe.id}'."
                        ),
                        status_code=400,
                    )

            hf_revision = submission.hf_revision
            if recipe.verified_revision:
                if hf_revision and hf_revision != recipe.verified_revision:
                    raise ServiceError(
                        code="RECIPE_REVISION_MISMATCH",
                        message=(
                            f"Recipe '{recipe.id}' requires revision '{recipe.verified_revision}', "
                            f"received '{hf_revision}'."
                        ),
                        status_code=400,
                    )
                hf_revision = recipe.verified_revision
            elif not hf_revision and recipe.preferred_revision:
                hf_revision = recipe.preferred_revision
            candidate = recipe.to_candidate(task_profile=task_profile, skip_olive=skip_olive)
            return BuildRequest(
                candidate=candidate,
                workspace_root=workspace,
                model_cache_dir=self._model_cache_dir,
                output_dir=output_dir,
                task_profile=task_profile,
                hf_revision=hf_revision,
                skip_olive=skip_olive,
                dry_run=False,
                recipe_id=recipe.id,
                recipe_version=recipe.version,
                recipe_status=recipe.status.value,
                recipe_reason=resolved.reason,
                recipe_artifact_cache_prefix=recipe.artifact_cache_prefix,
                recipe_model_name_prefix=recipe.model_name_prefix,
                allow_experimental=submission.allow_experimental,
                optimization_strategy=selected.strategy if selected else submission.optimization_strategy,
                optimization_precision=selected.precision if selected else submission.optimization_precision,
            )

        candidate = ModelCandidate(
            key=_normalize_model_key(submission.model_id, submission.task),
            huggingface_model_id=submission.model_id,
            modality=submission.task,
            recommended_mobius_dtype=None,
            recommended_olive_precision=None,
            notes=f"No matched recipe ({resolved.status}).",
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
            recipe_id=None,
            recipe_version=None,
            recipe_status=resolved.status,
            recipe_reason=resolved.reason,
            recipe_artifact_cache_prefix=None,
            recipe_model_name_prefix=None,
            allow_experimental=submission.allow_experimental,
            optimization_strategy=submission.optimization_strategy,
            optimization_precision=submission.optimization_precision,
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


def _recipe_blocker_code(match: RecipeResolution) -> str:
    if match.requires_experimental_opt_in:
        return "recipe_experimental_opt_in_required"
    if match.status == RecipeStatus.BLOCKED.value:
        return "recipe_blocked"
    return "recipe_unregistered"


def _recipe_error_code(match: RecipeResolution) -> str:
    if match.requires_experimental_opt_in:
        return "EXPERIMENTAL_RECIPE_OPT_IN_REQUIRED"
    if match.status == RecipeStatus.BLOCKED.value:
        return "MODEL_RECIPE_BLOCKED"
    return "MODEL_RECIPE_NOT_FOUND"


def _serialize_generated_recipe_attempt_binding(
    value: GeneratedRecipeAttemptBinding | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "attempt_id": value.attempt_id,
        "recipe_fingerprint": value.recipe_fingerprint,
        "confirmed": value.confirmed,
        "confirmation_provenance": value.confirmation_provenance,
    }


def _deserialize_generated_recipe_attempt_binding(
    value: object,
) -> GeneratedRecipeAttemptBinding | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid build request payload: generated_recipe_attempt must be an object.")
    attempt_id = _optional_str(value.get("attempt_id"))
    recipe_fingerprint = _optional_str(value.get("recipe_fingerprint"))
    confirmation_provenance = _optional_str(value.get("confirmation_provenance"))
    if not attempt_id or not recipe_fingerprint or confirmation_provenance is None:
        raise ValueError(
            "Invalid build request payload: generated_recipe_attempt requires attempt_id, "
            "recipe_fingerprint, and confirmation_provenance."
        )
    return GeneratedRecipeAttemptBinding(
        attempt_id=attempt_id,
        recipe_fingerprint=recipe_fingerprint.lower(),
        confirmed=bool(value.get("confirmed")),
        confirmation_provenance=confirmation_provenance,
    )


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
        "recipe_id": value.recipe_id,
        "recipe_version": value.recipe_version,
        "recipe_status": value.recipe_status,
        "recipe_reason": value.recipe_reason,
        "generated_recipe_attempt": _serialize_generated_recipe_attempt_binding(
            value.generated_recipe_attempt
        ),
        "recipe_artifact_cache_prefix": value.recipe_artifact_cache_prefix,
        "recipe_model_name_prefix": value.recipe_model_name_prefix,
        "allow_experimental": value.allow_experimental,
        "optimization_strategy": value.optimization_strategy,
        "optimization_precision": value.optimization_precision,
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
        recipe_id=_optional_str(value.get("recipe_id")),
        recipe_version=_optional_str(value.get("recipe_version")),
        recipe_status=_optional_str(value.get("recipe_status")),
        recipe_reason=_optional_str(value.get("recipe_reason")),
        generated_recipe_attempt=_deserialize_generated_recipe_attempt_binding(
            value.get("generated_recipe_attempt")
        ),
        recipe_artifact_cache_prefix=_optional_str(value.get("recipe_artifact_cache_prefix")),
        recipe_model_name_prefix=_optional_str(value.get("recipe_model_name_prefix")),
        allow_experimental=bool(value.get("allow_experimental", False)),
        optimization_strategy=_optional_str(value.get("optimization_strategy")),
        optimization_precision=_optional_str(value.get("optimization_precision")),
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


def _sanitize_attempt_failure_message(text: str) -> str:
    compact = re.sub(r"\s+", " ", _sanitize_text(text)).strip()
    if not compact:
        return "Attempt failed without a detailed message; see job evidence reference."
    if len(compact) <= _ATTEMPT_FAILURE_MESSAGE_MAX:
        return compact
    return compact[: _ATTEMPT_FAILURE_MESSAGE_MAX - 3].rstrip() + "..."
