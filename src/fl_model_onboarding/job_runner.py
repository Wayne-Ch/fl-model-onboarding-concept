from __future__ import annotations

import uuid

from dataclasses import dataclass
from datetime import datetime, timezone

from .adapters.interfaces import CommandSpec
from .contracts import (
    BuildJob,
    BuildRequest,
    FailureClassification,
    FailureInfo,
    JobState,
    ValidationResult,
    ValidationStatus,
)
from .contracts import PreflightResult
from .preflight import PreflightInspector
from .state_machine import EXECUTION_ORDER, fail_job, transition


@dataclass(frozen=True)
class DryRunPlan:
    job: BuildJob
    preflight: PreflightResult
    commands: tuple[CommandSpec, ...]


class LocalJobRunner:
    def __init__(self, preflight: PreflightInspector) -> None:
        self._preflight = preflight

    def create_job(self, request: BuildRequest, job_id_override: str | None = None) -> BuildJob:
        return BuildJob(job_id=job_id_override or str(uuid.uuid4()), request=request)

    def run_dry(
        self,
        request: BuildRequest,
        commands: tuple[CommandSpec, ...] = (),
        job_id_override: str | None = None,
    ) -> DryRunPlan:
        job = self.create_job(request, job_id_override=job_id_override)
        transition(job, JobState.PREFLIGHT, "Starting preflight checks")
        preflight = self._preflight.inspect(request)
        preflight_status = (
            ValidationStatus.PASSED if preflight.ok else ValidationStatus.FAILED
        )
        preflight_failure = None
        if not preflight.ok:
            first = preflight.blockers[0]
            preflight_failure = FailureInfo(
                stage=first.stage,
                classification=first.classification,
                message=first.message,
                detail=first.detail,
            )
        job.validations.append(
            ValidationResult(
                stage=JobState.PREFLIGHT,
                status=preflight_status,
                checks=tuple(preflight.warnings),
                failure=preflight_failure,
            )
        )

        if not preflight.ok:
            fail_job(job, preflight.blockers[0])
            job.finished_utc = datetime.now(timezone.utc)
            return DryRunPlan(job=job, preflight=preflight, commands=commands)

        for stage in EXECUTION_ORDER[1:]:
            transition(job, stage, f"Dry run reached stage '{stage.value}'")
            job.validations.append(
                ValidationResult(
                    stage=stage,
                    status=ValidationStatus.NOT_VERIFIED,
                    checks=(f"dry-run stage: {stage.value}",),
                    failure=FailureInfo(
                        stage=stage,
                        classification=FailureClassification.NOT_VERIFIED,
                        message="Dry run does not execute external model operations.",
                    ),
                )
            )

        transition(job, JobState.SUCCEEDED, "Dry run finished deterministic state sequence.")
        job.finished_utc = datetime.now(timezone.utc)
        return DryRunPlan(job=job, preflight=preflight, commands=commands)
