from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


class CandidateModality(StrEnum):
    LLM = "llm"
    ASR = "asr"


class JobState(StrEnum):
    QUEUED = "queued"
    PREFLIGHT = "preflight"
    DOWNLOADING = "downloading"
    MOBIUS_BUILDING = "mobius_building"
    MOBIUS_VALIDATING = "mobius_validating"
    OLIVE_OPTIMIZING = "olive_optimizing"
    PACKAGING = "packaging"
    RUNTIME_VALIDATING = "runtime_validating"
    FL_LOADING = "fl_loading"
    INFERENCING = "inferencing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class FailureClassification(StrEnum):
    NOT_VERIFIED = "not_verified"
    TOOL_UNAVAILABLE = "tool_unavailable"
    MISSING_DEPENDENCY = "missing_dependency"
    INVALID_REQUEST = "invalid_request"
    PATH_CONTAINMENT = "path_containment"
    NETWORK = "network"
    PROCESS_FAILED = "process_failed"
    COMPATIBILITY = "compatibility"
    SOURCE_RUNTIME_CONTRACT_INCOMPATIBLE = "source_runtime_contract_incompatible"
    OGA_RUNTIME_CONTRACT_INCOMPATIBLE = "oga_runtime_contract_incompatible"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_VERIFIED = "not_verified"


class ArtifactKind(StrEnum):
    MODEL = "model"
    CONFIG = "config"
    TOKENIZER = "tokenizer"
    DESCRIPTOR = "descriptor"
    RUNTIME_COMPATIBILITY = "runtime_compatibility"
    LOG = "log"
    OTHER = "other"


class MatchConfidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolInvocationTerminalStage(StrEnum):
    """Terminal outcome of one real external tool (Mobius/Olive) launch
    attempt within a single build/candidate execution."""

    NOT_RUN = "not_run"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolInvocationEvidence:
    """Real, per-tool-launch instrumentation for one external tool boundary
    (Mobius or Olive) within a single build/candidate execution.

    ``invocation_count`` only increments when the external process launch is
    actually attempted -- including a subsequent failure/timeout/cancel after
    that launch -- and never when upstream validation prevents the launch
    from happening at all. Every field stays at its not-run default
    (``invocation_count=0``, ``terminal_stage=NOT_RUN``, other fields
    ``None``) until this tool is actually launched for this job/candidate;
    ``0``/``NOT_RUN`` here must never be read as "definitely zero launches
    occurred for this recipe/candidate in general", only "not launched by
    this specific runner invocation so far".
    """

    invocation_count: int = 0
    terminal_stage: ToolInvocationTerminalStage = ToolInvocationTerminalStage.NOT_RUN
    success: bool | None = None
    started_utc: datetime | None = None
    finished_utc: datetime | None = None
    wall_seconds: float | None = None


@dataclass(frozen=True)
class ProductionInvocationEvidence:
    """Per-build/candidate-execution real invocation evidence for the Mobius
    and Olive external tool launch boundaries in ``ProductionBuildStageRunner``.

    A fresh instance is created for every ``run()``/``BuildJob`` and stored on
    that job, never shared as mutable state on the runner instance itself, so
    concurrent jobs handled by the same runner never share or race on
    counters.
    """

    mobius: ToolInvocationEvidence = field(default_factory=ToolInvocationEvidence)
    olive: ToolInvocationEvidence = field(default_factory=ToolInvocationEvidence)

    def sanitized_payload(self) -> dict[str, object]:
        """A persistable/loggable shape: counts, terminal stage/success, and
        wall-clock timing only. Deliberately excludes raw commands, argv,
        secrets, and absolute filesystem paths."""
        return {
            "mobius_invocation_count": self.mobius.invocation_count,
            "mobius_terminal_stage": self.mobius.terminal_stage.value,
            "mobius_success": self.mobius.success,
            "mobius_wall_seconds": self.mobius.wall_seconds,
            "olive_invocation_count": self.olive.invocation_count,
            "olive_terminal_stage": self.olive.terminal_stage.value,
            "olive_success": self.olive.success,
            "olive_wall_seconds": self.olive.wall_seconds,
        }


@dataclass(frozen=True)
class FailureInfo:
    stage: JobState
    classification: FailureClassification
    message: str
    detail: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCandidate:
    key: str
    huggingface_model_id: str
    modality: CandidateModality
    recommended_mobius_dtype: str | None
    recommended_olive_precision: str | None
    notes: str = ""


@dataclass(frozen=True)
class GeneratedRecipeAttemptBinding:
    attempt_id: str
    recipe_fingerprint: str
    confirmed: bool
    confirmation_provenance: str


@dataclass(frozen=True)
class BuildRequest:
    candidate: ModelCandidate
    workspace_root: Path
    model_cache_dir: Path
    output_dir: Path
    task_profile: str = "default"
    hf_revision: str | None = None
    runtime: str = "ort-genai"
    external_data_format: str = "safetensors"
    max_shard_size: str = "5GB"
    enforce_cpu_target: bool = True
    skip_olive: bool = False
    dry_run: bool = False
    recipe_id: str | None = None
    recipe_version: str | None = None
    recipe_status: str | None = None
    recipe_reason: str | None = None
    generated_recipe_attempt: GeneratedRecipeAttemptBinding | None = None
    recipe_artifact_cache_prefix: str | None = None
    recipe_model_name_prefix: str | None = None
    allow_experimental: bool = False
    optimization_strategy: str | None = None
    optimization_precision: str | None = None


@dataclass(frozen=True)
class BuildArtifact:
    artifact_id: str
    kind: ArtifactKind
    path: Path
    description: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class CatalogMatchAssessment:
    alias: str
    model_or_variant_id: str
    source_schema: str
    confidence: MatchConfidence
    reason: str
    cached: bool | None = None
    model_type: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    stage: JobState
    status: ValidationStatus
    checks: tuple[str, ...]
    failure: FailureInfo | None = None


@dataclass(frozen=True)
class ToolAvailability:
    name: str
    kind: str
    available: bool
    version: str | None
    detail: str = ""


@dataclass(frozen=True)
class PreflightResult:
    candidate: ModelCandidate
    workspace_root: Path
    model_cache_dir: Path
    output_dir: Path
    disk_free_gb_workspace: float
    disk_free_gb_cache: float
    tools: tuple[ToolAvailability, ...]
    foundry_catalog_matches: tuple[CatalogMatchAssessment, ...]
    huggingface_revision: str | None
    huggingface_sha: str | None
    huggingface_private: bool | None
    huggingface_gated: bool | None
    cache_key: str | None
    blockers: tuple[FailureInfo, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return len(self.blockers) == 0


@dataclass(frozen=True)
class JobEvent:
    sequence: int
    timestamp_utc: datetime
    state: JobState
    message: str


@dataclass
class BuildJob:
    job_id: str
    request: BuildRequest
    state: JobState = JobState.QUEUED
    events: list[JobEvent] = field(default_factory=list)
    artifacts: list[BuildArtifact] = field(default_factory=list)
    validations: list[ValidationResult] = field(default_factory=list)
    failure: FailureInfo | None = None
    result_artifact_id: str | None = None
    started_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_utc: datetime | None = None
    # Real per-job Mobius/Olive invocation instrumentation (Slice 3A1). ``None``
    # until `ProductionBuildStageRunner` starts this job's run; never shared
    # across concurrent jobs because each `BuildJob` is its own instance.
    production_invocation_evidence: ProductionInvocationEvidence | None = None

    def add_event(self, message: str) -> None:
        self.events.append(
            JobEvent(
                sequence=len(self.events) + 1,
                timestamp_utc=datetime.now(timezone.utc),
                state=self.state,
                message=message,
            )
        )

    def events_after(self, sequence: int) -> tuple[JobEvent, ...]:
        return tuple(event for event in self.events if event.sequence > sequence)

    def register_artifact(self, artifact: BuildArtifact) -> None:
        if artifact in self.artifacts:
            return
        if self.result_artifact_id and self.result_artifact_id != artifact.artifact_id:
            raise ValueError(
                "Build job already finalized with a different artifact_id; artifact identity is immutable."
            )
        self.artifacts.append(artifact)
        self.result_artifact_id = artifact.artifact_id
