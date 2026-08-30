from __future__ import annotations

from fl_model_onboarding.cli import main


def test_version_command_outputs_version(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main(["version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "0.1.0"
