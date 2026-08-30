from __future__ import annotations

import subprocess
import time

from io import BufferedReader
from threading import Event
from threading import Lock, Thread

from .adapters.interfaces import CommandResult, CommandSpec


class SubprocessCancelledError(RuntimeError):
    pass


class _BoundedCapture:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._chunks: list[bytes] = []
        self._size = 0
        self._truncated = False
        self._lock = Lock()

    def add(self, chunk: bytes) -> None:
        with self._lock:
            if self._max_bytes <= 0:
                self._truncated = True
                return
            remaining = self._max_bytes - self._size
            if remaining <= 0:
                self._truncated = True
                return
            kept = chunk[:remaining]
            self._chunks.append(kept)
            self._size += len(kept)
            if len(kept) < len(chunk):
                self._truncated = True

    def render(self) -> str:
        joined = b"".join(self._chunks)
        text = joined.decode("utf-8", errors="replace")
        sanitized = _sanitize_text(text)
        if self._truncated:
            return f"{sanitized}\n...[output truncated]..."
        return sanitized


class SafeSubprocessRunner:
    """Runs subprocesses with argument arrays and shell disabled."""

    def run(self, spec: CommandSpec, cancel_event: Event | None = None) -> CommandResult:
        start = time.monotonic()
        stdout_capture = _BoundedCapture(spec.max_capture_bytes)
        stderr_capture = _BoundedCapture(spec.max_capture_bytes)
        process = subprocess.Popen(
            list(spec.argv),
            cwd=str(spec.cwd) if spec.cwd else None,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        assert isinstance(process.stdout, BufferedReader)
        assert isinstance(process.stderr, BufferedReader)

        stdout_thread = Thread(
            target=_drain_pipe,
            args=(process.stdout, stdout_capture),
            daemon=True,
        )
        stderr_thread = Thread(
            target=_drain_pipe,
            args=(process.stderr, stderr_capture),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                process.kill()
                process.wait(timeout=5)
                _join_reader_threads(stdout_thread, stderr_thread)
                raise SubprocessCancelledError(
                    f"Process cancelled by caller. command={spec.argv!r}"
                )
            if time.monotonic() - start > spec.timeout_seconds:
                process.kill()
                process.wait(timeout=5)
                _join_reader_threads(stdout_thread, stderr_thread)
                raise TimeoutError(f"Command timed out after {spec.timeout_seconds}s: {spec.argv}")
            time.sleep(0.1)

        process.wait(timeout=5)
        _join_reader_threads(stdout_thread, stderr_thread)
        return CommandResult(
            spec=spec,
            exit_code=process.returncode,
            stdout=stdout_capture.render(),
            stderr=stderr_capture.render(),
        )


def _drain_pipe(stream: BufferedReader, capture: _BoundedCapture) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            capture.add(chunk)
    finally:
        stream.close()


def _join_reader_threads(*threads: Thread) -> None:
    for thread in threads:
        thread.join(timeout=5)


def _sanitize_text(text: str) -> str:
    sanitized_chars: list[str] = []
    for char in text:
        code = ord(char)
        if char in ("\n", "\r", "\t") or code >= 32:
            sanitized_chars.append(char)
        else:
            sanitized_chars.append("?")
    return "".join(sanitized_chars)
