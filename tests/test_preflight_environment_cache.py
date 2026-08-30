from __future__ import annotations

from pathlib import Path

from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec, HuggingFaceMetadata
from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import BuildRequest, CatalogMatchAssessment, FailureClassification, MatchConfidence
from fl_model_onboarding.preflight import PreflightInspector, PreflightPolicy
from fl_model_onboarding.preflight_cache import PreflightResultCache


class AllToolsRunner:
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        cmd = spec.argv[0]
        if cmd == "foundry":
            return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
        if cmd == "mobius":
            return CommandResult(spec=spec, exit_code=0, stdout="0.1.0\n", stderr="")
        if cmd == "olive":
            return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
        if cmd == "python":
            source = spec.argv[2]
            for pkg, version in (
                ("onnxruntime", "1.26.0"),
                ("onnxruntime-genai", "0.14.0"),
                ("foundry-local-sdk", "1.2.0"),
                ("huggingface_hub", "1.22.0"),
            ):
                if f"version('{pkg}')" in source:
                    return CommandResult(spec=spec, exit_code=0, stdout=f"{version}\n", stderr="")
        raise RuntimeError(f"Unhandled command: {spec.argv}")


class StableFoundry:
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

    def model_info(self, model_ref: str) -> dict[str, object]:
        return {"alias": model_ref}

    def cache_location(self) -> Path:
        raise NotImplementedError

    def status(self) -> dict[str, object]:
        return {}


class StableHF:
    def search_models(self, query: str, limit: int = 20, sort: str = "downloads"):  # noqa: ANN001
        return ()

    def get_metadata(  # noqa: ANN001
        self, model_id: str, revision: str | None = None, files_metadata: bool = False
    ) -> HuggingFaceMetadata:
        return HuggingFaceMetadata(
            model_id=model_id,
            revision=revision or "1234567890abcdef1234567890abcdef12345678",
            sha="1234567890abcdef1234567890abcdef12345678",
            is_private=False,
            is_gated=False,
            last_modified="2026-01-01T00:00:00Z",
            config={"model_type": "llama"},
            safetensors_total_bytes=10,
            safetensors_parameter_count=5,
            card_data={},
            sibling_count=1,
        )


def test_environment_blockers_are_recomputed_even_on_cache_hit(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    request = BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=tmp_path,
        model_cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    cache = PreflightResultCache.create()
    inspector_ok = PreflightInspector(
        runner=AllToolsRunner(),
        foundry=StableFoundry(),
        hf_metadata=StableHF(),
        policy=PreflightPolicy(min_workspace_free_gb=0.0, min_cache_free_gb=0.0),
        cache=cache,
    )
    first = inspector_ok.inspect(request)
    assert first.cache_key is not None
    assert not any(
        blocker.classification == FailureClassification.INVALID_REQUEST
        and "free space" in blocker.message
        for blocker in first.blockers
    )

    inspector_strict = PreflightInspector(
        runner=AllToolsRunner(),
        foundry=StableFoundry(),
        hf_metadata=StableHF(),
        policy=PreflightPolicy(min_workspace_free_gb=999999.0, min_cache_free_gb=999999.0),
        cache=cache,
    )
    second = inspector_strict.inspect(request)
    assert any("free space" in blocker.message for blocker in second.blockers)
    assert "preflight-cache-hit" in second.warnings
