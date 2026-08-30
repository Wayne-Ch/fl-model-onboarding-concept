from __future__ import annotations

from pathlib import Path


class HuggingFaceAcquisitionAdapter:
    def acquire_snapshot(
        self,
        model_id: str,
        local_dir: Path,
        revision: str | None = None,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        downloaded = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
        )
        return Path(downloaded)
