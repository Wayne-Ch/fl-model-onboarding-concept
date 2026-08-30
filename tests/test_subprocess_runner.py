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
