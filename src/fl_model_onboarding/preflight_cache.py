from __future__ import annotations

from dataclasses import dataclass

from .cache_identity import ToolchainFingerprint, build_cache_key
from .contracts import BuildRequest, PreflightResult, ToolAvailability


def build_preflight_cache_key(
    request: BuildRequest,
    huggingface_sha: str,
    tools: tuple[ToolAvailability, ...],
) -> str:
    versions = {tool.name: tool.version for tool in tools}
    fingerprint = ToolchainFingerprint(
        mobius_version=versions.get("mobius") or "missing",
        olive_version=versions.get("olive") or "missing",
        oga_version=versions.get("onnxruntime-genai") or "missing",
        ort_version=versions.get("onnxruntime") or "missing",
        foundry_cli_version=versions.get("foundry") or "missing",
        foundry_sdk_version=versions.get("foundry-local-sdk") or "missing",
    )
    return build_cache_key(
        model_id=request.candidate.huggingface_model_id,
        hf_revision_sha=huggingface_sha,
        task_profile=request.task_profile,
        toolchain=fingerprint,
    )


@dataclass
class PreflightResultCache:
    _entries: dict[str, PreflightResult]

    @classmethod
    def create(cls) -> "PreflightResultCache":
        return cls(_entries={})

    def get(self, key: str) -> PreflightResult | None:
        return self._entries.get(key)

    def put(self, key: str, result: PreflightResult) -> None:
        self._entries[key] = result
