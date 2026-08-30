from __future__ import annotations

from pathlib import Path

from .interfaces import CommandResult, CommandSpec
from ..contracts import BuildRequest
from ..subprocess_runner import SafeSubprocessRunner


class MobiusCliAdapter:
    def __init__(self, runner: SafeSubprocessRunner | None = None) -> None:
        self._runner = runner or SafeSubprocessRunner()

    def build_command(
        self,
        request: BuildRequest,
        output_dir: Path,
        no_weights: bool = False,
    ) -> CommandSpec:
        argv: list[str] = [
            "mobius",
            "build",
            "--model",
            request.candidate.huggingface_model_id,
            "--output",
            str(output_dir),
            "--ep",
            "cpu" if request.enforce_cpu_target else "default",
            "--runtime",
            request.runtime,
            "--external-data",
            request.external_data_format,
            "--max-shard-size",
            request.max_shard_size,
        ]
        if request.candidate.recommended_mobius_dtype:
            argv.extend(["--dtype", request.candidate.recommended_mobius_dtype])
        if request.hf_revision:
            argv.extend(["--revision", request.hf_revision])
        if no_weights:
            argv.append("--no-weights")
        return CommandSpec(
            argv=tuple(argv),
            cwd=request.workspace_root,
            timeout_seconds=14_400,
        )

    def run_build(
        self,
        request: BuildRequest,
        output_dir: Path,
        no_weights: bool = False,
    ) -> CommandResult:
        return self._runner.run(self.build_command(request=request, output_dir=output_dir, no_weights=no_weights))
