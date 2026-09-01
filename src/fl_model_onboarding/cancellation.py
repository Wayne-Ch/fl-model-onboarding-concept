from __future__ import annotations

import os
import shutil
import signal
import subprocess

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import BuildJob
from .paths import ensure_dir, ensure_within
from .state_machine import CANCELLABLE_STATES, StateTransitionError, cancel_job


@dataclass(frozen=True)
class OwnedProcess:
    pid: int
    output_dir: Path


class ProcessOwnershipRegistry:
    def __init__(self) -> None:
        self._owned: dict[str, OwnedProcess] = {}

    def register(self, job_id: str, pid: int, output_dir: Path) -> None:
        self._owned[job_id] = OwnedProcess(pid=pid, output_dir=output_dir)

    def cancel(self, job: BuildJob, reason: str) -> Path | None:
        if job.state not in CANCELLABLE_STATES:
            raise StateTransitionError(f"Job '{job.job_id}' in state '{job.state.value}' is not cancellable.")

        owned = self._owned.pop(job.job_id, None)
        if owned is not None:
            _terminate_process_tree(owned.pid)

        quarantine_path = quarantine_partial_output(
            workspace_root=job.request.workspace_root,
            output_dir=job.request.output_dir,
            job_id=job.job_id,
        )
        cancel_job(job, reason)
        return quarantine_path


def quarantine_partial_output(workspace_root: Path, output_dir: Path, job_id: str) -> Path | None:
    root = workspace_root.resolve()
    target = ensure_within(root, output_dir)
    if not target.exists():
        return None
    quarantine_root = ensure_dir(root / ".quarantine")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = quarantine_root / f"{job_id}-{stamp}"
    shutil.move(str(target), str(destination))
    return destination


def _terminate_process_tree(pid: int) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if result.returncode not in (0, 128, 255):
            raise RuntimeError(
                f"Failed to terminate process tree for pid={pid}: {result.stderr or result.stdout}"
            )
        return

    try:
        target_pgid = os.getpgid(pid)
        parent_pgid = os.getpgid(0)
        if target_pgid == parent_pgid:
            os.kill(pid, signal.SIGTERM)
        else:
            os.killpg(target_pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
