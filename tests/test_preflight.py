from __future__ import annotations

from pathlib import Path

from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec, HuggingFaceMetadata
from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import CatalogMatchAssessment, MatchConfidence
from fl_model_onboarding.contracts import BuildRequest, FailureClassification, JobState
from fl_model_onboarding.preflight import PreflightInspector


class FakeRunner:
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        cmd = spec.argv[0]
        if cmd == "foundry":
            return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
        if cmd == "mobius":
            raise FileNotFoundError("mobius not found")
        if cmd == "olive":
            return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
        if cmd == "python":
            source = spec.argv[2]
            if "version('onnxruntime')" in source:
                return CommandResult(spec=spec, exit_code=0, stdout="1.26.0\n", stderr="")
            if "version('onnxruntime-genai')" in source:
                return CommandResult(spec=spec, exit_code=0, stdout="0.14.0\n", stderr="")
            if "version('foundry-local-sdk')" in source:
                return CommandResult(spec=spec, exit_code=0, stdout="1.2.0\n", stderr="")
            if "version('huggingface_hub')" in source:
                return CommandResult(spec=spec, exit_code=0, stdout="1.22.0\n", stderr="")
        raise RuntimeError(f"Unhandled fake command: {spec.argv}")


class FakeFoundry:
    def list_matches(self, search_query: str) -> tuple[CatalogMatchAssessment, ...]:
        return ()

    def model_info(self, model_ref: str) -> dict[str, object]:
        return {"alias": model_ref}

    def cache_location(self) -> Path:
        raise NotImplementedError

    def status(self) -> dict[str, object]:
        return {}


class FakeHF:
    def search_models(self, query: str, limit: int = 20, sort: str = "downloads"):  # noqa: ANN001
        return ()

    def get_metadata(
        self,
        model_id: str,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> HuggingFaceMetadata:
        return HuggingFaceMetadata(
            model_id=model_id,
            revision=revision or "1234567890abcdef1234567890abcdef12345678",
            sha="1234567890abcdef1234567890abcdef12345678",
            is_private=False,
            is_gated=False,
            last_modified="2026-01-01T00:00:00Z",
            config={"model_type": "llama"},
            safetensors_total_bytes=1024,
            safetensors_parameter_count=256,
            card_data={"license": "apache-2.0"},
            sibling_count=12,
        )


def _request(workspace: Path, output: Path) -> BuildRequest:
    return BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=workspace,
        model_cache_dir=workspace / "cache",
        output_dir=output,
    )


def test_preflight_reports_missing_mobius(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), FakeHF())
    result = inspector.inspect(request)
    assert not result.ok
    assert any(
        blocker.stage == JobState.MOBIUS_BUILDING
        and blocker.classification == FailureClassification.MISSING_DEPENDENCY
        for blocker in result.blockers
    )
    assert any("not found in Foundry catalog" in warning for warning in result.warnings)


def test_preflight_reports_path_containment_blocker(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    outside = tmp_path.parent / "outside-output"
    request = _request(tmp_path, outside)
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), FakeHF())
    result = inspector.inspect(request)
    assert not result.ok
    assert any(
        blocker.classification == FailureClassification.PATH_CONTAINMENT
        for blocker in result.blockers
    )


def test_preflight_keeps_likely_match_semantics(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class FoundryLikely(FakeFoundry):
        def list_matches(self, search_query: str) -> tuple[CatalogMatchAssessment, ...]:
            return (
                CatalogMatchAssessment(
                    alias="qwen3.5-0.8b",
                    model_or_variant_id="qwen3.5-0.8b-generic-gpu:2",
                    source_schema="models",
                    confidence=MatchConfidence.MEDIUM,
                    reason="Likely alias match by name similarity only",
                ),
            )

    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FoundryLikely(), FakeHF())
    result = inspector.inspect(request)
    assert result.foundry_catalog_matches
    assert result.foundry_catalog_matches[0].confidence == MatchConfidence.MEDIUM
    assert result.cache_key is not None


def test_preflight_rejects_gated_models(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class GatedHF(FakeHF):
        def get_metadata(  # type: ignore[override]
            self, model_id: str, revision: str | None = None, files_metadata: bool = False
        ) -> HuggingFaceMetadata:
            base = super().get_metadata(model_id, revision=revision, files_metadata=files_metadata)
            return HuggingFaceMetadata(
                model_id=base.model_id,
                revision=base.revision,
                sha=base.sha,
                is_private=base.is_private,
                is_gated=True,
                last_modified=base.last_modified,
                config=base.config,
                safetensors_total_bytes=base.safetensors_total_bytes,
                safetensors_parameter_count=base.safetensors_parameter_count,
                card_data=base.card_data,
                sibling_count=base.sibling_count,
            )

    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), GatedHF())
    result = inspector.inspect(request)
    assert any("rejects gated" in blocker.message for blocker in result.blockers)


def test_preflight_rejects_remote_code_models(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class AutoMapHF(FakeHF):
        def get_metadata(  # type: ignore[override]
            self, model_id: str, revision: str | None = None, files_metadata: bool = False
        ) -> HuggingFaceMetadata:
            base = super().get_metadata(model_id, revision=revision, files_metadata=files_metadata)
            return HuggingFaceMetadata(
                model_id=base.model_id,
                revision=base.revision,
                sha=base.sha,
                is_private=base.is_private,
                is_gated=base.is_gated,
                last_modified=base.last_modified,
                config={"auto_map": {"AutoModelForCausalLM": "remote.module.Model"}},
                safetensors_total_bytes=base.safetensors_total_bytes,
                safetensors_parameter_count=base.safetensors_parameter_count,
                card_data=base.card_data,
                sibling_count=base.sibling_count,
            )

    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), AutoMapHF())
    result = inspector.inspect(request)
    assert any("requires remote code" in blocker.message for blocker in result.blockers)
