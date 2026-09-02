from __future__ import annotations

import os
import threading

import pytest

import fl_model_onboarding.subprocess_runner as subprocess_runner_module
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only process-group behavior")
def test_posix_runner_starts_child_in_new_session() -> None:
    parent_pgid = os.getpgid(0)
    runner = SafeSubprocessRunner()
    result = runner.run(CommandSpec(argv=("python", "-c", "import os; print(os.getpgid(0))")))
    child_pgid = int(result.stdout.strip())
    assert child_pgid != parent_pgid


def test_windows_timeout_cleanup_prefers_taskkill_process_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    class _DummyProcess:
        pid = 4321

        def __init__(self) -> None:
            self.kill_called = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.kill_called = True

    def fake_run(argv, **kwargs):  # noqa: ANN001
        calls.append(list(argv))
        return _Completed()

    monkeypatch.setattr(subprocess_runner_module.os, "name", "nt", raising=False)
    monkeypatch.setattr(subprocess_runner_module.subprocess, "run", fake_run)
    process = _DummyProcess()

    subprocess_runner_module._terminate_process_tree(process)  # type: ignore[arg-type]

    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]
    assert process.kill_called is False


def test_windows_timeout_cleanup_falls_back_to_process_kill_when_taskkill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Completed:
        returncode = 1
        stdout = "failed"
        stderr = "failed"

    class _DummyProcess:
        pid = 4321

        def __init__(self) -> None:
            self.kill_called = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.kill_called = True

    monkeypatch.setattr(subprocess_runner_module.os, "name", "nt", raising=False)
    monkeypatch.setattr(subprocess_runner_module.subprocess, "run", lambda *args, **kwargs: _Completed())
    process = _DummyProcess()

    subprocess_runner_module._terminate_process_tree(process)  # type: ignore[arg-type]

    assert process.kill_called is True
