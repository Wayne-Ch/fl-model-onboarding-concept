from __future__ import annotations

import json

from pathlib import Path

import pytest

from fl_model_onboarding.adapters.artifact_assembler import FoundryArtifactAssembler
from fl_model_onboarding.adapters.mobius_cli import MobiusCliAdapter
from fl_model_onboarding.adapters.olive_cli import OliveCliAdapter
from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import BuildRequest
from fl_model_onboarding.paths import PathContainmentError


def _request(tmp_path: Path) -> BuildRequest:
    return BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=tmp_path,
        model_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


def test_mobius_command_contains_expected_arguments(tmp_path: Path) -> None:
    adapter = MobiusCliAdapter()
    cmd = adapter.build_command(_request(tmp_path), tmp_path / "out")
    assert cmd.argv[0:2] == ("mobius", "build")
    assert "--runtime" in cmd.argv
    assert "ort-genai" in cmd.argv
    assert "--external-data" in cmd.argv
    assert "safetensors" in cmd.argv
    assert "--model" in cmd.argv
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in cmd.argv


def test_olive_command_contains_expected_arguments(tmp_path: Path) -> None:
    adapter = OliveCliAdapter()
    cmd = adapter.auto_opt_command(
        input_model_or_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        precision="int4",
    )
    assert cmd.argv[0:2] == ("olive", "auto-opt")
    assert "--model_name_or_path" in cmd.argv
    assert "--output_path" in cmd.argv
    assert "--use_ort_genai" in cmd.argv
    assert "--precision" in cmd.argv
    assert "int4" in cmd.argv


def test_artifact_assembler_copies_files_and_writes_descriptor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "genai_config.json").write_text("{}", encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()

    assembler = FoundryArtifactAssembler()
    artifacts = assembler.package_for_foundry_cache(
        artifact_id="artifact-1",
        model_name="my-model",
        source_dir=source,
        model_cache_dir=cache,
    )
    target = cache / "Custom" / "my-model"
    assert target.exists()
    assert (target / "genai_config.json").exists()
    descriptor = json.loads((target / "inference_model.json").read_text(encoding="utf-8"))
    assert descriptor["Name"] == "my-model"
    assert all(a.artifact_id == "artifact-1" for a in artifacts)
    assert any(a.path.name == "inference_model.json" for a in artifacts)


def test_artifact_assembler_rejects_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "genai_config.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    assembler = FoundryArtifactAssembler()
    with pytest.raises(PathContainmentError):
        assembler.package_for_foundry_cache(
            artifact_id="artifact-1",
            model_name="..\\escape",
            source_dir=source,
            model_cache_dir=cache,
        )
