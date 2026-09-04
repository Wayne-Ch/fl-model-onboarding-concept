from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Protocol

from ..contracts import (
    BuildArtifact,
    BuildRequest,
    CatalogMatchAssessment,
    JobState,
    ValidationResult,
)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path | None = None
    timeout_seconds: int = 900
    max_capture_bytes: int = 2_000_000


@dataclass(frozen=True)
class CommandResult:
    spec: CommandSpec
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class HuggingFaceMetadata:
    model_id: str
    revision: str | None
    sha: str | None
    is_private: bool | None
    is_gated: bool | None
    last_modified: str | None
    config: dict[str, object] | None
    safetensors_total_bytes: int | None
    safetensors_parameter_count: int | None
    card_data: dict[str, object] | None
    sibling_count: int | None
    sibling_files: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HuggingFaceSearchResult:
    model_id: str
    downloads: int | None
    likes: int | None
    last_modified: str | None


@dataclass(frozen=True)
class StageRunResult:
    stage: JobState
    message: str
    artifacts: tuple[BuildArtifact, ...] = ()
    validations: tuple[ValidationResult, ...] = ()


class ProcessRunner(Protocol):
    def run(self, spec: CommandSpec, cancel_event: Event | None = None) -> CommandResult:
        ...


class HuggingFaceMetadataClient(Protocol):
    def search_models(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
    ) -> tuple[HuggingFaceSearchResult, ...]:
        ...

    def get_metadata(
        self,
        model_id: str,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> HuggingFaceMetadata:
        ...


class HuggingFaceAcquisitionClient(Protocol):
    def acquire_snapshot(
        self,
        model_id: str,
        local_dir: Path,
        revision: str | None = None,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        ...


class FoundryCatalogClient(Protocol):
    def list_matches(self, search_query: str) -> tuple[CatalogMatchAssessment, ...]:
        ...

    def model_info(self, model_ref: str) -> dict[str, object]:
        ...

    def cache_location(self) -> Path:
        ...

    def status(self) -> dict[str, object]:
        ...


class MobiusBuildClient(Protocol):
    def build_command(
        self,
        request: BuildRequest,
        output_dir: Path,
        no_weights: bool = False,
    ) -> CommandSpec:
        ...


class OliveOptimizeClient(Protocol):
    def auto_opt_command(
        self,
        input_model_or_dir: Path,
        output_dir: Path,
        precision: str | None,
        device: str = "cpu",
        provider: str = "CPUExecutionProvider",
    ) -> CommandSpec:
        ...


class ArtifactAssemblerClient(Protocol):
    def package_for_foundry_cache(
        self,
        artifact_id: str,
        model_name: str,
        source_dir: Path,
        model_cache_dir: Path,
        prompt_template: dict[str, str] | None = None,
    ) -> tuple[BuildArtifact, ...]:
        ...


class OgaValidatorClient(Protocol):
    def validate(self, model_dir: Path) -> ValidationResult:
        ...


class FoundryInferenceClient(Protocol):
    def load_and_infer(
        self,
        model_name: str,
        model_cache_dir: Path,
        prompt: str,
        max_tokens: int = 64,
    ) -> ValidationResult:
        ...


StageHandler = Callable[[BuildRequest, Event | None], StageRunResult]
