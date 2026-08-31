from __future__ import annotations

import pytest

from fl_model_onboarding.cache_identity import ToolchainFingerprint, build_cache_key


def test_cache_key_includes_full_sha_and_versions() -> None:
    full_sha = "1234567890abcdef1234567890abcdef12345678"
    key = build_cache_key(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        hf_revision_sha=full_sha,
        task_profile="llm-cpu-int4",
        toolchain=ToolchainFingerprint(
            mobius_version="0.1.0",
            olive_version="0.13.0",
            oga_version="0.15.2",
            ort_version="1.29.0",
            foundry_cli_version="0.11.0",
            foundry_sdk_version="1.2.4",
        ),
    )
    assert full_sha in key
    assert "mobius-0.1.0" in key
    assert "flsdk-1.2.4" in key


def test_cache_key_rejects_short_sha() -> None:
    with pytest.raises(ValueError):
        build_cache_key(
            model_id="x",
            hf_revision_sha="1234",
            task_profile="profile",
            toolchain=ToolchainFingerprint(
                mobius_version="0",
                olive_version="0",
                oga_version="0",
                ort_version="0",
                foundry_cli_version="0",
                foundry_sdk_version="0",
            ),
        )
