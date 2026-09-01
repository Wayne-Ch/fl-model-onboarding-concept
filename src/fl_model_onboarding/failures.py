from __future__ import annotations

from .contracts import FailureClassification, FailureInfo, JobState
from .paths import PathContainmentError
from .subprocess_runner import SubprocessCancelledError


def failure(
    stage: JobState,
    classification: FailureClassification,
    message: str,
    detail: dict[str, str] | None = None,
) -> FailureInfo:
    return FailureInfo(
        stage=stage,
        classification=classification,
        message=message,
        detail=detail or {},
    )


def classify_exception(stage: JobState, exc: Exception) -> FailureInfo:
    if isinstance(exc, PathContainmentError):
        return failure(stage, FailureClassification.PATH_CONTAINMENT, str(exc))
    if isinstance(exc, SubprocessCancelledError):
        return failure(stage, FailureClassification.CANCELLED, str(exc))
    if isinstance(exc, TimeoutError):
        return failure(stage, FailureClassification.PROCESS_FAILED, str(exc))
    if isinstance(exc, FileNotFoundError):
        return failure(stage, FailureClassification.MISSING_DEPENDENCY, str(exc))
    message = str(exc)
    lowered = message.lower()
    if "no module named" in lowered:
        return failure(stage, FailureClassification.MISSING_DEPENDENCY, message)
    if "404" in lowered or "not found" in lowered:
        return failure(stage, FailureClassification.INVALID_REQUEST, message)
    if "connection" in lowered or "network" in lowered or "timed out" in lowered:
        return failure(stage, FailureClassification.NETWORK, message)
    return failure(stage, FailureClassification.UNKNOWN, message)
