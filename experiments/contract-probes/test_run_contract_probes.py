from __future__ import annotations

import subprocess

import run_contract_probes
from run_contract_probes import Candidate, first_failed_stage, infer_mobius_style, is_happy_path, redact


def test_redact_hf_token_and_bearer() -> None:
    value = "hf_abcdefghijklmnopqrstuvwxyz123456\nAuthorization: Bearer abc.def.ghi"
    redacted = redact(value)
    assert "hf_" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "***REDACTED***" in redacted


def test_redact_json_token_value() -> None:
    value = '{"token":"very-secret-value"}'
    redacted = redact(value)
    assert "very-secret-value" not in redacted
    assert '{"token":"***REDACTED***"}' == redacted


def test_infer_mobius_style_prefer_subcommand() -> None:
    style = infer_mobius_style({"mobius": "C:\\mobius.exe", "mobiusbuild": "C:\\mobiusbuild.exe"})
    assert style == "subcommand"


def test_infer_mobius_style_split() -> None:
    style = infer_mobius_style({"mobius": None, "mobiusbuild": "C:\\mobiusbuild.exe"})
    assert style == "split"


def test_first_failed_stage() -> None:
    events = [
        {"stage": "a", "success": True},
        {"stage": "b", "success": False},
        {"stage": "c", "success": False},
    ]
    assert first_failed_stage(events) == "b"


def test_is_happy_path_requires_oga_for_speech() -> None:
    checker = {"success": True}
    ort = {"success": True}
    foundry_sdk = {"success": True}
    oga_failure = {"success": False, "error": "speech runtime parser error"}

    assert is_happy_path(checker, ort, oga_failure, foundry_sdk) is False


def test_foundry_sdk_probe_timeout_stage(monkeypatch, tmp_path) -> None:
    candidate = Candidate(
        key="asr",
        model_id="distil-whisper/distil-medium.en",
        task="speech",
        olive_precision="int8",
        byom_name="distil-whisper-contract-probe",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout", 0),
            output=b"partial-stdout",
            stderr=b"partial-stderr",
        )

    monkeypatch.setattr(run_contract_probes.subprocess, "run", fake_run)

    result = run_contract_probes._run_foundry_sdk_probe_subprocess(
        candidate=candidate,
        model_dir=model_dir,
        scratch_dir=scratch_dir,
        timeout_seconds=17,
    )
    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["stage"] == "fl_sdk_probe_timeout"
    assert result["timeout_seconds"] == 17
