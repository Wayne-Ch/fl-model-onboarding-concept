from __future__ import annotations

from pathlib import Path

from .interfaces import CommandResult, CommandSpec
from ..subprocess_runner import SafeSubprocessRunner


class OliveCliAdapter:
    def __init__(self, runner: SafeSubprocessRunner | None = None) -> None:
        self._runner = runner or SafeSubprocessRunner()

    def auto_opt_command(
        self,
        input_model_or_dir: Path,
        output_dir: Path,
        precision: str | None,
        device: str = "cpu",
        provider: str = "CPUExecutionProvider",
    ) -> CommandSpec:
        argv: list[str] = [
            "olive",
            "auto-opt",
            "--model_name_or_path",
            str(input_model_or_dir),
            "--output_path",
            str(output_dir),
            "--device",
            device,
            "--provider",
            provider,
            "--use_ort_genai",
            "--log_level",
            "1",
        ]
        if precision:
            argv.extend(["--precision", precision])
        return CommandSpec(argv=tuple(argv), timeout_seconds=14_400)

    def run_auto_opt(
        self,
        input_model_or_dir: Path,
        output_dir: Path,
        precision: str | None,
        device: str = "cpu",
        provider: str = "CPUExecutionProvider",
    ) -> CommandResult:
        return self._runner.run(
            self.auto_opt_command(
                input_model_or_dir=input_model_or_dir,
                output_dir=output_dir,
                precision=precision,
                device=device,
                provider=provider,
            )
        )
