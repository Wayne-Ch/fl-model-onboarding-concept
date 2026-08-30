from __future__ import annotations

from pathlib import Path

import pytest

from fl_model_onboarding.adapters.huggingface_acquisition import HuggingFaceAcquisitionAdapter
from fl_model_onboarding.adapters.interfaces import HuggingFaceMetadata


class StubMetadata:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def get_metadata(
        self, model_id: str, revision: str | None = None, files_metadata: bool = False
    ) -> HuggingFaceMetadata:
        return HuggingFaceMetadata(
            model_id=model_id,
            revision=revision,
            sha="1234567890abcdef1234567890abcdef12345678",
            is_private=False,
            is_gated=False,
            last_modified="2026-01-01T00:00:00Z",
            config=self._config,
            safetensors_total_bytes=None,
            safetensors_parameter_count=None,
            card_data=None,
            sibling_count=3,
        )


def test_acquisition_blocks_remote_code_models(monkeypatch, tmp_path: Path) -> None:
    called = {"snapshot": False}

    def fake_snapshot_download(**kwargs):  # noqa: ANN001
        called["snapshot"] = True
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    adapter = HuggingFaceAcquisitionAdapter(
        metadata=StubMetadata({"auto_map": {"AutoModel": "x.y.Model"}})  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError):
        adapter.acquire_snapshot("owner/model", tmp_path)
    assert called["snapshot"] is False


def test_acquisition_uses_safe_default_allow_patterns(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return str(tmp_path / "downloaded")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    adapter = HuggingFaceAcquisitionAdapter(
        metadata=StubMetadata({"model_type": "llama"})  # type: ignore[arg-type]
    )
    path = adapter.acquire_snapshot("owner/model", tmp_path)
    assert path == (tmp_path / "downloaded")
    allow_patterns = captured["allow_patterns"]
    assert isinstance(allow_patterns, list)
    assert "*.json" in allow_patterns
