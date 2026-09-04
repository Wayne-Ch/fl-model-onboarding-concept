from __future__ import annotations

import importlib.util
import os
import sys

from pathlib import Path


def _load_round_runner_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "recipe-agent-v1"
        / "round-1"
        / "run_round1.py"
    )
    spec = importlib.util.spec_from_file_location("round1_runner", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_console_script_from_scripts_dir(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    mobius_exe = scripts_dir / "mobius.exe"
    mobius_exe.write_text("stub", encoding="utf-8")
    resolved = module._resolve_console_script_path(scripts_dir, "mobius")
    assert resolved == mobius_exe.resolve()


def test_with_scripts_on_path_prepends_when_parent_path_missing(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    base_env = {"PATH": os.pathsep.join([str(tmp_path / "bin"), str(tmp_path / "tools")])}
    updated_env, did_prefix = module._with_scripts_on_path(base_env, scripts_dir)
    assert did_prefix is True
    assert updated_env["PATH"].split(os.pathsep)[0] == str(scripts_dir)
    assert base_env["PATH"].split(os.pathsep)[0] != str(scripts_dir)


def test_with_scripts_on_path_skips_duplicate_entry(tmp_path: Path) -> None:
    module = _load_round_runner_module()
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    base_env = {"PATH": os.pathsep.join([str(scripts_dir), str(tmp_path / "bin")])}
    updated_env, did_prefix = module._with_scripts_on_path(base_env, scripts_dir)
    assert did_prefix is False
    assert updated_env["PATH"] == base_env["PATH"]


def test_required_environment_blockers_detect_missing_required() -> None:
    module = _load_round_runner_module()
    blockers = module._required_environment_blockers(
        {
            "probes": [
                {"kind": "command", "name": "foundry", "required": True, "available": True},
                {"kind": "command", "name": "mobius", "required": True, "available": False},
                {"kind": "python-package", "name": "onnxruntime", "required": True, "available": True},
            ]
        }
    )
    assert blockers == ["command:mobius"]


def test_expected_unregistered_recipe_preflight_is_not_environment_failure() -> None:
    module = _load_round_runner_module()
    payload = {
        "recipe_status": "unregistered",
        "generated_recipe": {
            "eligible_for_automatic_recipe_attempt": True,
        },
    }
    assert module._expected_unregistered_recipe_preflight(payload) is True


def test_classify_round_invalid_environment_sets_not_applicable_rate() -> None:
    module = _load_round_runner_module()
    outcome = module._classify_round_outcome(
        manifest_invariants={"ok": True},
        environment_blockers=["command:mobius", "command:olive"],
        model_results=[
            {"failure_summary": {"attempt_state": "failed"}},
            {"failure_summary": {"attempt_state": "failed"}},
            {"failure_summary": {"attempt_state": "failed"}},
            {"failure_summary": {"attempt_state": "failed"}},
            {"failure_summary": {"attempt_state": "failed"}},
        ],
        environment_fail_fast_triggered=True,
    )
    assert outcome["round_classification"] == "invalid_environment"
    assert outcome["baseline_valid"] is False
    assert outcome["model_success_rate_applicable"] is False
    assert outcome["success_rate"] == "not_applicable"
    assert outcome["attempt_success_rate"] == "0/5"
    assert outcome["environment_fail_fast_triggered"] is True


def test_sanitize_text_rewrites_known_roots_and_redacts_other_absolutes() -> None:
    module = _load_round_runner_module()
    replacements = (
        (r"C:\scratch\r1", "scratch://round-1/r1"),
        (r"C:\repo-root", "<repo-root>"),
    )
    raw = (
        r"artifacts=C:\scratch\r1\state\service.sqlite3 "
        r"repo=C:\repo-root "
        r"private=C:\Users\wayne\secrets\token.txt"
    )
    sanitized = module._sanitize_text(raw, replacements)
    assert "scratch://round-1/r1" in sanitized
    assert "<repo-root>" in sanitized
    assert r"C:\scratch\r1" not in sanitized
    assert r"C:\repo-root" not in sanitized
    assert r"C:\Users\wayne\secrets\token.txt" not in sanitized
    assert "<redacted-absolute-path>" in sanitized


def test_command_probe_uses_canonical_non_empty_probe_field() -> None:
    module = _load_round_runner_module()
    probe = module._probe_command_version("missing-command-for-round1-probe")
    assert probe["kind"] == "command"
    assert probe["required"] is True
    assert probe["available"] is False
    assert isinstance(probe["probe"], str) and probe["probe"]
