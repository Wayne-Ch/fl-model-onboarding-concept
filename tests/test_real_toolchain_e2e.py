from __future__ import annotations

import json
import os
import subprocess

from pathlib import Path

import pytest


@pytest.mark.e2e
def test_retained_smollm2_package_validates_and_infers(tmp_path: Path) -> None:
    model_dir_value = os.environ.get("FL_ONBOARDING_E2E_MODEL_DIR")
    python_value = os.environ.get("FL_ONBOARDING_E2E_PYTHON")
    if not model_dir_value or not python_value:
        pytest.skip(
            "Set FL_ONBOARDING_E2E_MODEL_DIR and FL_ONBOARDING_E2E_PYTHON "
            "to run the real retained-package E2E."
        )

    model_dir = Path(model_dir_value).resolve()
    python = Path(python_value).resolve()
    descriptor = json.loads((model_dir / "inference_model.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())

    validation = subprocess.run(
        [
            str(python),
            "-m",
            "fl_model_onboarding.runtime_worker",
            "validate-runtime",
            "--model-dir",
            str(model_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
        env=env,
    )
    assert validation.returncode == 0, validation.stderr or validation.stdout
    assert json.loads(validation.stdout)["ok"] is True

    request_file = tmp_path / "inference-request.json"
    request_file.write_text(
        json.dumps({"prompt": "Reply with: OK", "max_tokens": 64}),
        encoding="utf-8",
    )
    inference = subprocess.run(
        [
            str(python),
            "-m",
            "fl_model_onboarding.runtime_worker",
            "foundry-infer",
            "--model-dir",
            str(model_dir),
            "--model-name",
            descriptor["Name"],
            "--request-file",
            str(request_file),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
        env=env,
    )
    assert inference.returncode == 0, inference.stderr or inference.stdout
    result = json.loads(inference.stdout)
    assert result["ok"] is True
    assert str(result["output"]).strip()
