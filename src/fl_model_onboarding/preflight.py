from __future__ import annotations

import re
import shutil

from dataclasses import dataclass
from pathlib import Path

from .adapters.interfaces import (
    CommandSpec,
    FoundryCatalogClient,
    HuggingFaceMetadataClient,
    ProcessRunner,
)
from .contracts import (
    BuildRequest,
    FailureClassification,
    FailureInfo,
    JobState,
    MatchConfidence,
    PreflightResult,
    ToolAvailability,
)
from .failures import classify_exception, failure
from .paths import ensure_within


_SEMVER_RE = re.compile(r"(?P<version>\d+\.\d+\.\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class PreflightPolicy:
    min_workspace_free_gb: float = 10.0
    min_cache_free_gb: float = 20.0


class PreflightInspector:
    def __init__(
        self,
        runner: ProcessRunner,
        foundry: FoundryCatalogClient,
        hf_metadata: HuggingFaceMetadataClient,
        policy: PreflightPolicy | None = None,
    ) -> None:
        self._runner = runner
        self._foundry = foundry
        self._hf_metadata = hf_metadata
        self._policy = policy or PreflightPolicy()

    def inspect(self, request: BuildRequest) -> PreflightResult:
        blockers: list[FailureInfo] = []
        warnings: list[str] = []

        try:
            ensure_within(request.workspace_root, request.output_dir)
        except Exception as exc:
            blockers.append(classify_exception(JobState.PREFLIGHT, exc))

        workspace_usage = shutil.disk_usage(request.workspace_root)
        cache_usage = shutil.disk_usage(request.model_cache_dir)
        workspace_free_gb = workspace_usage.free / (1024**3)
        cache_free_gb = cache_usage.free / (1024**3)
        if workspace_free_gb < self._policy.min_workspace_free_gb:
            blockers.append(
                failure(
                    JobState.PREFLIGHT,
                    FailureClassification.INVALID_REQUEST,
                    (
                        f"Workspace free space {workspace_free_gb:.1f}GB is below policy "
                        f"{self._policy.min_workspace_free_gb:.1f}GB."
                    ),
                )
            )
        if cache_free_gb < self._policy.min_cache_free_gb:
            blockers.append(
                failure(
                    JobState.PREFLIGHT,
                    FailureClassification.INVALID_REQUEST,
                    (
                        f"Cache free space {cache_free_gb:.1f}GB is below policy "
                        f"{self._policy.min_cache_free_gb:.1f}GB."
                    ),
                )
            )

        tools = (
            self._probe_command("foundry", ("--version",)),
            self._probe_command("mobius", ("--version",)),
            self._probe_command("olive", ("--help",)),
            self._probe_python_module("onnxruntime", "onnxruntime"),
            self._probe_python_module("onnxruntime-genai", "onnxruntime_genai"),
            self._probe_python_module("foundry-local-sdk", "foundry_local_sdk"),
            self._probe_python_module("huggingface_hub", "huggingface_hub"),
        )
        by_name = {tool.name: tool for tool in tools}
        foundry_version = by_name["foundry"].version
        if foundry_version and _version_lt(foundry_version, "0.11.0"):
            warnings.append(
                f"Foundry CLI {foundry_version} is older than 0.11.0; command contracts may differ."
            )
        if not by_name["foundry"].available:
            blockers.append(
                failure(
                    JobState.PREFLIGHT,
                    FailureClassification.MISSING_DEPENDENCY,
                    "foundry CLI is required for cache, status, and catalog probes.",
                )
            )
        if not by_name["mobius"].available:
            blockers.append(
                failure(
                    JobState.MOBIUS_BUILDING,
                    FailureClassification.MISSING_DEPENDENCY,
                    "mobius CLI is required to build BYOM ONNX packages.",
                )
            )
        if not by_name["olive"].available and not request.skip_olive:
            blockers.append(
                failure(
                    JobState.OLIVE_OPTIMIZING,
                    FailureClassification.MISSING_DEPENDENCY,
                    "olive CLI is required for optimization stage unless skip_olive is enabled.",
                )
            )
        if not by_name["onnxruntime-genai"].available:
            blockers.append(
                failure(
                    JobState.RUNTIME_VALIDATING,
                    FailureClassification.MISSING_DEPENDENCY,
                    "onnxruntime-genai is required for OGA model-loading validation.",
                )
            )
        if not by_name["foundry-local-sdk"].available:
            blockers.append(
                failure(
                    JobState.FL_LOADING,
                    FailureClassification.MISSING_DEPENDENCY,
                    "foundry-local-sdk is required for SDK loading/inference checks.",
                )
            )

        catalog_matches = ()
        try:
            query = request.candidate.huggingface_model_id.split("/")[-1]
            matches = self._foundry.list_matches(query)
            catalog_matches = matches
            if len(catalog_matches) == 0:
                warnings.append(
                    f"Candidate '{request.candidate.huggingface_model_id}' was not found in Foundry catalog."
                )
            elif all(match.confidence == MatchConfidence.LOW for match in catalog_matches):
                warnings.append(
                    "Foundry catalog returned weak alias hits only; compatibility remains unverified."
                )
        except Exception as exc:
            warnings.append(f"Foundry catalog probe failed: {exc}")

        hf_revision: str | None = None
        hf_sha: str | None = None
        hf_private: bool | None = None
        hf_gated: bool | None = None
        try:
            metadata = self._hf_metadata.get_metadata(
                request.candidate.huggingface_model_id,
                revision=request.hf_revision,
            )
            hf_revision = metadata.revision
            hf_sha = metadata.sha
            hf_private = metadata.is_private
            hf_gated = metadata.is_gated
        except Exception as exc:
            classified = classify_exception(JobState.PREFLIGHT, exc)
            if classified.classification == FailureClassification.INVALID_REQUEST:
                blockers.append(classified)
            else:
                warnings.append(f"Hugging Face metadata probe failed: {classified.message}")

        return PreflightResult(
            candidate=request.candidate,
            workspace_root=request.workspace_root,
            model_cache_dir=request.model_cache_dir,
            output_dir=request.output_dir,
            disk_free_gb_workspace=workspace_free_gb,
            disk_free_gb_cache=cache_free_gb,
            tools=tools,
            foundry_catalog_matches=catalog_matches,
            huggingface_revision=hf_revision,
            huggingface_sha=hf_sha,
            huggingface_private=hf_private,
            huggingface_gated=hf_gated,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )

    def _probe_command(self, name: str, args: tuple[str, ...]) -> ToolAvailability:
        spec = CommandSpec(argv=(name, *args), timeout_seconds=30)
        try:
            result = self._runner.run(spec)
            output = (result.stdout or result.stderr or "").strip()
            first_line = output.splitlines()[0] if output else ""
            parsed_version = _extract_version(first_line) if first_line else None
            return ToolAvailability(
                name=name,
                kind="command",
                available=result.ok,
                version=parsed_version,
                detail=first_line or f"exit_code={result.exit_code}",
            )
        except FileNotFoundError:
            return ToolAvailability(
                name=name,
                kind="command",
                available=False,
                version=None,
                detail="command-not-found",
            )
        except Exception as exc:
            return ToolAvailability(
                name=name,
                kind="command",
                available=False,
                version=None,
                detail=str(exc),
            )

    def _probe_python_module(self, distribution_name: str, import_name: str) -> ToolAvailability:
        python_code = (
            "import importlib,importlib.metadata,sys;"
            f"importlib.import_module('{import_name}');"
            f"print(importlib.metadata.version('{distribution_name}'))"
        )
        spec = CommandSpec(argv=("python", "-c", python_code), timeout_seconds=30)
        try:
            result = self._runner.run(spec)
            if result.ok:
                version = (result.stdout or "").strip().splitlines()[-1]
                return ToolAvailability(
                    name=distribution_name,
                    kind="python-package",
                    available=True,
                    version=version or None,
                    detail=f"import:{import_name}",
                )
            return ToolAvailability(
                name=distribution_name,
                kind="python-package",
                available=False,
                version=None,
                detail=(result.stderr or result.stdout).strip(),
            )
        except Exception as exc:
            return ToolAvailability(
                name=distribution_name,
                kind="python-package",
                available=False,
                version=None,
                detail=str(exc),
            )


def _extract_version(text: str) -> str | None:
    match = _SEMVER_RE.search(text)
    return match.group("version") if match else None


def _version_lt(lhs: str, rhs: str) -> bool:
    def parse(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".") if part.isdigit())

    return parse(lhs) < parse(rhs)
