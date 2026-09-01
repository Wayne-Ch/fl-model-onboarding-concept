from __future__ import annotations

import re
import shutil
import sys

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
from .hf_policy import config_requires_remote_code
from .paths import ensure_within
from .preflight_cache import PreflightResultCache, build_preflight_cache_key


_SEMVER_RE = re.compile(r"(?P<version>\d+\.\d+\.\d+(?:\.\d+)?)")
_COMMAND_NOT_FOUND_DETAIL = "command-not-found"
_MISSING_MODULE_MARKER = "no module named"


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
        cache: PreflightResultCache | None = None,
    ) -> None:
        self._runner = runner
        self._foundry = foundry
        self._hf_metadata = hf_metadata
        self._policy = policy or PreflightPolicy()
        self._cache = cache or PreflightResultCache.create()

    def inspect(self, request: BuildRequest) -> PreflightResult:
        environment_blockers: list[FailureInfo] = []
        compatibility_blockers: list[FailureInfo] = []
        environment_warnings: list[str] = []
        warnings: list[str] = []

        try:
            ensure_within(request.workspace_root, request.output_dir)
        except Exception as exc:
            environment_blockers.append(classify_exception(JobState.PREFLIGHT, exc))

        workspace_free_gb = 0.0
        cache_free_gb = 0.0
        try:
            workspace_usage = shutil.disk_usage(request.workspace_root)
            workspace_free_gb = workspace_usage.free / (1024**3)
        except FileNotFoundError:
            environment_blockers.append(
                failure(
                    JobState.PREFLIGHT,
                    FailureClassification.INVALID_REQUEST,
                    f"Workspace root does not exist: {request.workspace_root}",
                )
            )
        try:
            cache_usage = shutil.disk_usage(request.model_cache_dir)
            cache_free_gb = cache_usage.free / (1024**3)
        except FileNotFoundError:
            environment_blockers.append(
                failure(
                    JobState.PREFLIGHT,
                    FailureClassification.INVALID_REQUEST,
                    f"Model cache directory does not exist: {request.model_cache_dir}",
                )
            )
        if workspace_free_gb < self._policy.min_workspace_free_gb:
            environment_blockers.append(
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
            environment_blockers.append(
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
            self._probe_command("mobius", ("--help",), version_args=("--version",)),
            self._probe_command("olive", ("--help",), version_args=("--version",)),
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
        self._append_required_tool_blocker(
            blockers=compatibility_blockers,
            tool=by_name["foundry"],
            stage=JobState.PREFLIGHT,
            missing_message="foundry CLI is required for cache, status, and catalog probes.",
            probe_failure_message="foundry CLI probe failed; unable to confirm command availability.",
        )
        self._append_required_tool_blocker(
            blockers=compatibility_blockers,
            tool=by_name["mobius"],
            stage=JobState.MOBIUS_BUILDING,
            missing_message="mobius CLI is required to build BYOM ONNX packages.",
            probe_failure_message="mobius CLI probe failed; availability could not be verified from this environment.",
        )
        if not request.skip_olive:
            self._append_required_tool_blocker(
                blockers=compatibility_blockers,
                tool=by_name["olive"],
                stage=JobState.OLIVE_OPTIMIZING,
                missing_message="olive CLI is required for optimization stage unless skip_olive is enabled.",
                probe_failure_message="olive CLI probe failed; availability could not be verified from this environment.",
            )
        self._append_required_tool_blocker(
            blockers=compatibility_blockers,
            tool=by_name["onnxruntime-genai"],
            stage=JobState.RUNTIME_VALIDATING,
            missing_message="onnxruntime-genai is required for OGA model-loading validation.",
            probe_failure_message="onnxruntime-genai probe failed; package could not be validated in this runtime.",
        )
        self._append_required_tool_blocker(
            blockers=compatibility_blockers,
            tool=by_name["foundry-local-sdk"],
            stage=JobState.FL_LOADING,
            missing_message="foundry-local-sdk is required for SDK loading/inference checks.",
            probe_failure_message="foundry-local-sdk probe failed; package could not be validated in this runtime.",
        )

        catalog_matches = ()

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
            if hf_gated:
                compatibility_blockers.append(
                    failure(
                        JobState.PREFLIGHT,
                        FailureClassification.INVALID_REQUEST,
                        "POC phase-0 rejects gated Hugging Face models (no token flow in scope).",
                    )
                )
            if config_requires_remote_code(metadata.config):
                compatibility_blockers.append(
                    failure(
                        JobState.PREFLIGHT,
                        FailureClassification.INVALID_REQUEST,
                        "Model config requires remote code (`auto_map`/`trust_remote_code`), which POC policy blocks.",
                    )
                )
        except Exception as exc:
            classified = classify_exception(JobState.PREFLIGHT, exc)
            if classified.classification == FailureClassification.INVALID_REQUEST:
                compatibility_blockers.append(classified)
            else:
                warnings.append(f"Hugging Face metadata probe failed: {classified.message}")

        cache_key: str | None = None
        if hf_sha:
            try:
                cache_key = build_preflight_cache_key(request, huggingface_sha=hf_sha, tools=tools)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return self._merge_with_environment(
                        request=request,
                        cache_key=cache_key,
                        workspace_free_gb=workspace_free_gb,
                        cache_free_gb=cache_free_gb,
                        compatibility=cached,
                        environment_blockers=tuple(environment_blockers),
                        environment_warnings=tuple(environment_warnings),
                        add_cache_hit_warning=True,
                    )
            except Exception as exc:
                warnings.append(f"Preflight cache key unavailable: {exc}")
        else:
            warnings.append("Hugging Face revision SHA unavailable; preflight result not cacheable.")

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

        compatibility_result = PreflightResult(
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
            cache_key=cache_key,
            blockers=tuple(compatibility_blockers),
            warnings=tuple(warnings),
        )
        if cache_key:
            self._cache.put(cache_key, compatibility_result)
        return self._merge_with_environment(
            request=request,
            cache_key=cache_key,
            workspace_free_gb=workspace_free_gb,
            cache_free_gb=cache_free_gb,
            compatibility=compatibility_result,
            environment_blockers=tuple(environment_blockers),
            environment_warnings=tuple(environment_warnings),
        )

    def _merge_with_environment(
        self,
        request: BuildRequest,
        cache_key: str | None,
        workspace_free_gb: float,
        cache_free_gb: float,
        compatibility: PreflightResult,
        environment_blockers: tuple[FailureInfo, ...],
        environment_warnings: tuple[str, ...],
        add_cache_hit_warning: bool = False,
    ) -> PreflightResult:
        merged_warnings = list(environment_warnings)
        merged_warnings.extend(compatibility.warnings)
        if add_cache_hit_warning:
            merged_warnings.append("preflight-cache-hit")
        return PreflightResult(
            candidate=request.candidate,
            workspace_root=request.workspace_root,
            model_cache_dir=request.model_cache_dir,
            output_dir=request.output_dir,
            disk_free_gb_workspace=workspace_free_gb,
            disk_free_gb_cache=cache_free_gb,
            tools=compatibility.tools,
            foundry_catalog_matches=compatibility.foundry_catalog_matches,
            huggingface_revision=compatibility.huggingface_revision,
            huggingface_sha=compatibility.huggingface_sha,
            huggingface_private=compatibility.huggingface_private,
            huggingface_gated=compatibility.huggingface_gated,
            cache_key=cache_key,
            blockers=tuple(environment_blockers) + tuple(compatibility.blockers),
            warnings=tuple(merged_warnings),
        )

    def _append_required_tool_blocker(
        self,
        *,
        blockers: list[FailureInfo],
        tool: ToolAvailability,
        stage: JobState,
        missing_message: str,
        probe_failure_message: str,
    ) -> None:
        if tool.available:
            return
        if self._is_missing_dependency(tool):
            blockers.append(
                failure(
                    stage,
                    FailureClassification.MISSING_DEPENDENCY,
                    missing_message,
                )
            )
            return
        blockers.append(
            failure(
                stage,
                FailureClassification.TOOL_UNAVAILABLE,
                probe_failure_message,
                detail={"tool": tool.name, "probe_detail": tool.detail},
            )
        )

    @staticmethod
    def _is_missing_dependency(tool: ToolAvailability) -> bool:
        detail = tool.detail.lower()
        return detail == _COMMAND_NOT_FOUND_DETAIL or _MISSING_MODULE_MARKER in detail

    def _probe_command(
        self,
        name: str,
        availability_args: tuple[str, ...],
        *,
        version_args: tuple[str, ...] | None = None,
    ) -> ToolAvailability:
        spec = CommandSpec(argv=(name, *availability_args), timeout_seconds=30)
        try:
            result = self._runner.run(spec)
            output = (result.stdout or result.stderr or "").strip()
            first_line = output.splitlines()[0] if output else ""
            if not result.ok:
                detail = first_line or f"exit_code={result.exit_code}"
                return ToolAvailability(
                    name=name,
                    kind="command",
                    available=False,
                    version=None,
                    detail=f"probe-failed:{detail}",
                )
            parsed_version = _extract_version(first_line) if first_line else None
            detail = first_line or "probe-ok"
            if version_args:
                version, version_probe_error = self._probe_command_version(name, version_args)
                if version:
                    parsed_version = version
                elif version_probe_error:
                    detail = f"{detail}; version-probe-failed:{version_probe_error}"
            return ToolAvailability(
                name=name,
                kind="command",
                available=True,
                version=parsed_version,
                detail=detail,
            )
        except FileNotFoundError:
            return ToolAvailability(
                name=name,
                kind="command",
                available=False,
                version=None,
                detail=_COMMAND_NOT_FOUND_DETAIL,
            )
        except Exception as exc:
            return ToolAvailability(
                name=name,
                kind="command",
                available=False,
                version=None,
                detail=f"probe-error:{exc}",
            )

    def _probe_command_version(
        self,
        name: str,
        version_args: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        spec = CommandSpec(argv=(name, *version_args), timeout_seconds=30)
        try:
            result = self._runner.run(spec)
        except Exception as exc:
            return None, str(exc)
        output = (result.stdout or result.stderr or "").strip()
        first_line = output.splitlines()[0] if output else ""
        if not result.ok:
            return None, first_line or f"exit_code={result.exit_code}"
        return _extract_version(first_line), None

    def _probe_python_module(self, distribution_name: str, import_name: str) -> ToolAvailability:
        python_code = (
            "import importlib,importlib.metadata,sys;"
            f"importlib.import_module('{import_name}');"
            f"print(importlib.metadata.version('{distribution_name}'))"
        )
        spec = CommandSpec(argv=(sys.executable, "-c", python_code), timeout_seconds=30)
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
        except FileNotFoundError:
            return ToolAvailability(
                name=distribution_name,
                kind="python-package",
                available=False,
                version=None,
                detail=_COMMAND_NOT_FOUND_DETAIL,
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
