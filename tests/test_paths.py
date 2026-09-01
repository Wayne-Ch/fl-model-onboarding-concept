from __future__ import annotations

from pathlib import Path

import pytest

from fl_model_onboarding.paths import PathContainmentError, ensure_within


def test_ensure_within_accepts_nested_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "nested" / "file.txt"
    child.parent.mkdir(parents=True)
    child.write_text("ok", encoding="utf-8")
    assert ensure_within(root, child) == child.resolve()


def test_ensure_within_rejects_escape_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    escaping = root / ".." / "outside.txt"
    with pytest.raises(PathContainmentError):
        ensure_within(root, escaping)
