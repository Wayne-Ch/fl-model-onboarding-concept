from __future__ import annotations

import subprocess
import time

from threading import Event

from .adapters.interfaces import CommandResult, CommandSpec


class SubprocessCancelledError(RuntimeError):
    pass


class SafeSubprocessRunner:
    """Runs subprocesses with argument arrays and shell disabled."""

    def run(self, spec: CommandSpec, cancel_event: Event | None = None) -> CommandResult:
        start = time.monotonic()
        process = subprocess.Popen(
            list(spec.argv),
            cwd=str(spec.cwd) if spec.cwd else None,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                process.kill()
                process.wait(timeout=5)
                raise SubprocessCancelledError("Process cancelled by caller.")
            if time.monotonic() - start > spec.timeout_seconds:
                process.kill()
                process.wait(timeout=5)
                raise TimeoutError(f"Command timed out after {spec.timeout_seconds}s: {spec.argv}")
            time.sleep(0.1)

        stdout, stderr = process.communicate()
        return CommandResult(spec=spec, exit_code=process.returncode, stdout=stdout, stderr=stderr)
