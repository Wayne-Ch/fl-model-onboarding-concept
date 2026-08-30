from __future__ import annotations

from pathlib import Path

from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import BuildRequest, ToolAvailability
from fl_model_onboarding.preflight_cache import PreflightResultCache, build_preflight_cache_key


def test_preflight_cache_key_uses_full_sha_and_profile(tmp_path: Path) -> None:
    request = BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=tmp_path,
        model_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        task_profile="llm-cpu-int4",
    )
    tools = (
        ToolAvailability("foundry", "command", True, "0.11.0"),
        ToolAvailability("mobius", "command", True, "0.1.0"),
        ToolAvailability("olive", "command", True, "0.13.0"),
        ToolAvailability("onnxruntime", "python-package", True, "1.26.0"),
        ToolAvailability("onnxruntime-genai", "python-package", True, "0.14.0"),
        ToolAvailability("foundry-local-sdk", "python-package", True, "1.2.0"),
        ToolAvailability("huggingface_hub", "python-package", True, "1.22.0"),
    )
    full_sha = "1234567890abcdef1234567890abcdef12345678"
    key = build_preflight_cache_key(request=request, huggingface_sha=full_sha, tools=tools)
    assert full_sha in key
    assert "llm-cpu-int4" in key
    request_skip = BuildRequest(
        candidate=request.candidate,
        workspace_root=request.workspace_root,
        model_cache_dir=request.model_cache_dir,
        output_dir=request.output_dir,
        task_profile=request.task_profile,
        skip_olive=True,
    )
    skip_key = build_preflight_cache_key(request=request_skip, huggingface_sha=full_sha, tools=tools)
    assert key != skip_key


def test_preflight_result_cache_put_get() -> None:
    cache = PreflightResultCache.create()
    assert cache.get("x") is None
