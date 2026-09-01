from __future__ import annotations

from fl_model_onboarding.contracts import FailureClassification, JobState
from fl_model_onboarding.failures import classify_exception
from fl_model_onboarding.paths import PathContainmentError
from fl_model_onboarding.subprocess_runner import SubprocessCancelledError


def test_classify_path_containment() -> None:
    failure = classify_exception(JobState.PREFLIGHT, PathContainmentError("outside"))
    assert failure.classification == FailureClassification.PATH_CONTAINMENT


def test_classify_missing_module() -> None:
    failure = classify_exception(JobState.PREFLIGHT, ModuleNotFoundError("No module named x"))
    assert failure.classification == FailureClassification.MISSING_DEPENDENCY


def test_classify_cancelled() -> None:
    failure = classify_exception(JobState.PREFLIGHT, SubprocessCancelledError("cancelled"))
    assert failure.classification == FailureClassification.CANCELLED
