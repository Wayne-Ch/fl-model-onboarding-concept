from __future__ import annotations

from pathlib import Path

import pytest

from fl_model_onboarding.workspace_layout import ensure_short_path, workspace_root_for_job


def test_workspace_root_for_job_is_contained(tmp_path: Path) -> None:
    root = workspace_root_for_job("job-123", base_dir=tmp_path)
    assert root == (tmp_path / "job-123").resolve()


def test_ensure_short_path_rejects_long_value() -> None:
    fake_path = Path("C:\\" + ("a" * 300))
    with pytest.raises(ValueError):
        ensure_short_path(fake_path, max_chars=20)
