from __future__ import annotations

import importlib.util
import sys

from pathlib import Path


def _load_round5_runner_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "recipe-agent-v1"
        / "round-5"
        / "run_round5.py"
    )
    spec = importlib.util.spec_from_file_location("round5_runner", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_round5_runner_injects_round_specific_defaults(monkeypatch) -> None:
    module = _load_round5_runner_module()
    captured: dict[str, object] = {}

    def fake_round4_main() -> int:
        captured["argv"] = list(sys.argv[1:])
        return 0

    monkeypatch.setattr(module, "_load_round4_runner_main", lambda: fake_round4_main)
    monkeypatch.setattr(sys, "argv", ["run_round5.py"])

    assert module.main() == 0
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "--round-name" in argv
    assert "round-5" in argv
    assert "--output-dir" in argv
    assert "--scratch-root" in argv
    assert r"C:\fmo-r5" in argv


def test_round5_runner_preserves_explicit_overrides(monkeypatch) -> None:
    module = _load_round5_runner_module()
    captured: dict[str, object] = {}

    def fake_round4_main() -> int:
        captured["argv"] = list(sys.argv[1:])
        return 0

    monkeypatch.setattr(module, "_load_round4_runner_main", lambda: fake_round4_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_round5.py",
            "--round-name",
            "custom-round",
            "--output-dir",
            "X:\\custom-out",
            "--scratch-root",
            "X:\\scratch",
        ],
    )

    assert module.main() == 0
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv.count("--round-name") == 1
    assert argv.count("--output-dir") == 1
    assert argv.count("--scratch-root") == 1
