from __future__ import annotations

import re

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolchainFingerprint:
    mobius_version: str
    olive_version: str
    oga_version: str
    ort_version: str
    foundry_cli_version: str
    foundry_sdk_version: str


def build_cache_key(
    model_id: str,
    hf_revision_sha: str,
    task_profile: str,
    toolchain: ToolchainFingerprint,
) -> str:
    """Build a persistent cache identity without truncating the HF revision SHA."""
    if not hf_revision_sha or len(hf_revision_sha) < 16:
        raise ValueError("hf_revision_sha must contain the full immutable revision SHA.")
    parts = [
        _sanitize(model_id),
        f"sha-{_sanitize(hf_revision_sha)}",
        _sanitize(task_profile),
        f"mobius-{_sanitize(toolchain.mobius_version)}",
        f"olive-{_sanitize(toolchain.olive_version)}",
        f"oga-{_sanitize(toolchain.oga_version)}",
        f"ort-{_sanitize(toolchain.ort_version)}",
        f"flcli-{_sanitize(toolchain.foundry_cli_version)}",
        f"flsdk-{_sanitize(toolchain.foundry_sdk_version)}",
    ]
    return "__".join(parts)


def _sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
