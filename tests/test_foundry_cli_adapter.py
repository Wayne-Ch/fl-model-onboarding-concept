from __future__ import annotations

import json

from fl_model_onboarding.adapters.foundry_cli import FoundryCliCatalogAdapter
from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec
from fl_model_onboarding.contracts import MatchConfidence


class FakeRunner:
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        if spec.argv[:3] == ("foundry", "model", "list"):
            if "--variants" in spec.argv:
                payload = {
                    "variants": [
                        {
                            "alias": "qwen3.5-0.8b",
                            "variantId": "qwen3.5-0.8b-generic-gpu:2",
                            "type": "Unknown",
                            "cached": True,
                        }
                    ]
                }
            else:
                payload = {
                    "models": [
                        {
                            "alias": "qwen3.5-0.8b",
                            "id": "qwen3.5-0.8b-generic-gpu:2",
                            "type": "Unknown",
                            "cached": True,
                        }
                    ]
                }
            return CommandResult(spec=spec, exit_code=0, stdout=json.dumps(payload), stderr="")
        if spec.argv[:3] == ("foundry", "cache", "location"):
            return CommandResult(
                spec=spec,
                exit_code=0,
                stdout='{"path":"C:\\\\cache"}',
                stderr="",
            )
        if spec.argv[:2] == ("foundry", "status"):
            return CommandResult(spec=spec, exit_code=0, stdout='{"service":{"state":"ready"}}', stderr="")
        raise RuntimeError(f"Unhandled command: {spec.argv}")


def test_list_matches_parses_models_and_variants() -> None:
    adapter = FoundryCliCatalogAdapter(FakeRunner())  # type: ignore[arg-type]
    matches = adapter.list_matches("qwen3.5-0.8b")
    assert len(matches) == 2
    assert {m.source_schema for m in matches} == {"models", "variants"}
    assert all(m.confidence in {MatchConfidence.MEDIUM, MatchConfidence.LOW} for m in matches)


def test_list_matches_handles_missing_sections() -> None:
    class MissingSectionsRunner(FakeRunner):
        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            if spec.argv[:3] == ("foundry", "model", "list"):
                return CommandResult(spec=spec, exit_code=0, stdout='{"unexpected":[]}', stderr="")
            return super().run(spec, cancel_event=cancel_event)

    adapter = FoundryCliCatalogAdapter(MissingSectionsRunner())  # type: ignore[arg-type]
    matches = adapter.list_matches("anything")
    assert matches == ()
