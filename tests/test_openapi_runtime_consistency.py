from __future__ import annotations

from pathlib import Path

import yaml

from fl_model_onboarding.local_api import create_app
from fl_model_onboarding.local_service import LocalOnboardingService


def test_runtime_openapi_paths_match_contract(tmp_path: Path) -> None:
    service = LocalOnboardingService(
        db_path=tmp_path / "state.sqlite3",
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
    )
    try:
        app = create_app(service=service)
        runtime_spec = app.openapi()
        contract_path = Path("contracts") / "openapi.yaml"
        contract_spec = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

        runtime_paths = set(runtime_spec["paths"].keys())
        contract_paths = set(contract_spec["paths"].keys())
        assert runtime_paths == contract_paths

        runtime_asr_body = runtime_spec["paths"]["/api/artifacts/{artifact_id}/infer/asr"]["post"][
            "requestBody"
        ]["content"]
        assert "multipart/form-data" in runtime_asr_body

        runtime_build_params = runtime_spec["paths"]["/api/builds"]["post"]["parameters"]
        idempotency = next(param for param in runtime_build_params if param["name"] == "Idempotency-Key")
        assert idempotency["required"] is True
    finally:
        service.close()
