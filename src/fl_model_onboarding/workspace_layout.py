from __future__ import annotations

import os

from pathlib import Path

from .paths import ensure_within


def default_workspace_base() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise EnvironmentError("LOCALAPPDATA is required to build a Windows-safe short workspace path.")
    return Path(local_app_data) / "fl-onboard" / "w"


def workspace_root_for_job(job_id: str, base_dir: Path | None = None) -> Path:
    base = (base_dir or default_workspace_base()).resolve()
    candidate = (base / job_id).resolve()
    ensure_within(base, candidate)
    ensure_short_path(candidate)
    return candidate


def ensure_short_path(path: Path, max_chars: int = 180) -> None:
    if os.name == "nt" and len(str(path)) > max_chars:
        raise ValueError(
            f"Workspace path '{path}' exceeds {max_chars} chars; choose a shorter base or job id."
        )
