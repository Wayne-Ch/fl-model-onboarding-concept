from __future__ import annotations

from run_contract_probes import first_failed_stage, infer_mobius_style, redact


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
