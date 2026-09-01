from __future__ import annotations

import json
import shutil

from pathlib import Path

from .interfaces import ArtifactAssemblerClient
from ..contracts import ArtifactKind, BuildArtifact
from ..paths import ensure_dir, ensure_within


DEFAULT_CHATML_TEMPLATE = {
    "system": "<|im_start|>system\n{Content}<|im_end|>\n",
    "user": "<|im_start|>user\n{Content}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{Content}<|im_end|>\n",
    "prompt": "{Messages}<|im_start|>assistant\n",
}


class FoundryArtifactAssembler(ArtifactAssemblerClient):
    def package_for_foundry_cache(
        self,
        artifact_id: str,
        model_name: str,
        source_dir: Path,
        model_cache_dir: Path,
        prompt_template: dict[str, str] | None = None,
    ) -> tuple[BuildArtifact, ...]:
        source = source_dir.resolve()
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Source model directory does not exist: {source}")

        custom_root = ensure_dir(model_cache_dir.resolve() / "Custom")
        target = ensure_within(custom_root, custom_root / model_name)
        if target.exists():
            raise FileExistsError(f"Target model path already exists: {target}")
        shutil.copytree(source, target)

        inference_model_path = target / "inference_model.json"
        inference_payload = {
            "Name": model_name,
            "PromptTemplate": prompt_template or DEFAULT_CHATML_TEMPLATE,
        }
        inference_model_path.write_text(json.dumps(inference_payload, indent=2), encoding="utf-8")

        artifacts: list[BuildArtifact] = [
            BuildArtifact(
                artifact_id=artifact_id,
                kind=ArtifactKind.DESCRIPTOR,
                path=inference_model_path,
                description="Foundry Local inference descriptor",
                size_bytes=inference_model_path.stat().st_size,
            )
        ]
        for candidate in ("genai_config.json", "runtime_compatibility.json"):
            artifact_path = target / candidate
            if artifact_path.exists():
                artifacts.append(
                    BuildArtifact(
                        artifact_id=artifact_id,
                        kind=ArtifactKind.CONFIG
                        if candidate == "genai_config.json"
                        else ArtifactKind.RUNTIME_COMPATIBILITY,
                        path=artifact_path,
                        description=candidate,
                        size_bytes=artifact_path.stat().st_size,
                    )
                )
        return tuple(artifacts)
