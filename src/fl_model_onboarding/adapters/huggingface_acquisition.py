from __future__ import annotations

from pathlib import Path

from .huggingface_metadata import HuggingFaceMetadataAdapter
from ..hf_policy import config_requires_remote_code


SAFE_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "*.jinja",
)


class HuggingFaceAcquisitionAdapter:
    def __init__(self, metadata: HuggingFaceMetadataAdapter | None = None) -> None:
        self._metadata = metadata or HuggingFaceMetadataAdapter()

    def acquire_snapshot(
        self,
        model_id: str,
        local_dir: Path,
        revision: str | None = None,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        metadata = self._metadata.get_metadata(model_id=model_id, revision=revision, files_metadata=False)
        if config_requires_remote_code(metadata.config):
            raise ValueError(
                "Model config includes `auto_map`/remote-code requirement and is blocked by policy."
            )

        downloaded = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=list(allow_patterns or SAFE_ALLOW_PATTERNS),
        )
        return Path(downloaded)
