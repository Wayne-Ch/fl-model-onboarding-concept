from __future__ import annotations

from dataclasses import dataclass

from .cache_identity import ToolchainFingerprint, build_cache_key
from .contracts import BuildRequest, PreflightResult, ToolAvailability


def build_preflight_cache_key(
    request: BuildRequest,
    huggingface_sha: str,
    tools: tuple[ToolAvailability, ...],
) -> str:
    by_name = {tool.name: tool for tool in tools}
    fingerprint = ToolchainFingerprint(
        mobius_version=_tool_fingerprint_value(by_name.get("mobius")),
        olive_version=_tool_fingerprint_value(by_name.get("olive")),
        oga_version=_tool_fingerprint_value(by_name.get("onnxruntime-genai")),
        ort_version=_tool_fingerprint_value(by_name.get("onnxruntime")),
        foundry_cli_version=_tool_fingerprint_value(by_name.get("foundry")),
        foundry_sdk_version=_tool_fingerprint_value(by_name.get("foundry-local-sdk")),
    )
    semantic_profile = (
        f"{request.task_profile}"
        f"|skip_olive={int(request.skip_olive)}"
        f"|runtime={request.runtime}"
        f"|cpu={int(request.enforce_cpu_target)}"
    )
    return build_cache_key(
        model_id=request.candidate.huggingface_model_id,
        hf_revision_sha=huggingface_sha,
        task_profile=semantic_profile,
        toolchain=fingerprint,
    )


def _tool_fingerprint_value(tool: ToolAvailability | None) -> str:
    if tool is None:
        return "missing"
    status = "available" if tool.available else "unavailable"
    version = tool.version or "unknown"
    return f"{status}-{version}"


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
