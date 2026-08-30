from __future__ import annotations

import threading

from fl_model_onboarding.adapters.interfaces import CommandSpec
from fl_model_onboarding.subprocess_runner import SafeSubprocessRunner, SubprocessCancelledError


def test_subprocess_runner_returns_stdout() -> None:
    runner = SafeSubprocessRunner()
    result = runner.run(CommandSpec(argv=("python", "-c", "print('hello')"), timeout_seconds=10))
    assert result.ok
    assert result.stdout.strip() == "hello"


def test_subprocess_runner_supports_cancellation() -> None:
    runner = SafeSubprocessRunner()
    event = threading.Event()
    timer = threading.Timer(0.2, event.set)
    timer.start()
    try:
        raised = False
        try:
            runner.run(
                CommandSpec(argv=("python", "-c", "import time; time.sleep(10)"), timeout_seconds=30),
                cancel_event=event,
            )
        except SubprocessCancelledError:
            raised = True
        assert raised
    finally:
        timer.cancel()


def test_subprocess_runner_drains_large_stdout_and_stderr() -> None:
    runner = SafeSubprocessRunner()
    script = (
        "import sys\n"
        "chunk='x'*65536\n"
        "for _ in range(32):\n"
        "    sys.stdout.write(chunk)\n"
        "    sys.stderr.write(chunk)\n"
        "sys.stdout.write('DONE\\n')\n"
        "sys.stderr.write('DONE_ERR\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stderr.flush()\n"
    )
    result = runner.run(
        CommandSpec(
            argv=("python", "-c", script),
            timeout_seconds=60,
            max_capture_bytes=4_000_000,
        )
    )
    assert result.ok
    assert "DONE" in result.stdout
    assert "DONE_ERR" in result.stderr
