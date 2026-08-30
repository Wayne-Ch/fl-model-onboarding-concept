from __future__ import annotations

from pathlib import Path

from .interfaces import FoundryInferenceClient
from ..contracts import (
    FailureClassification,
    FailureInfo,
    JobState,
    ValidationResult,
    ValidationStatus,
)


class FoundrySdkInferenceAdapter(FoundryInferenceClient):
    def load_and_infer(
        self,
        model_name: str,
        model_cache_dir: Path,
        prompt: str,
        max_tokens: int = 64,
    ) -> ValidationResult:
        checks = [f"model_name={model_name}", f"model_cache_dir={model_cache_dir.resolve()}"]
        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            return ValidationResult(
                stage=JobState.INFERENCING,
                status=ValidationStatus.NOT_VERIFIED,
                checks=tuple(checks + ["foundry_local_sdk import missing"]),
                failure=FailureInfo(
                    stage=JobState.INFERENCING,
                    classification=FailureClassification.MISSING_DEPENDENCY,
                    message=str(exc),
                ),
            )

        if getattr(FoundryLocalManager, "instance", None) is None:
            FoundryLocalManager.initialize(
                Configuration(app_name="fl-model-onboarding", model_cache_dir=str(model_cache_dir))
            )
        manager = FoundryLocalManager.instance
        checks.append("manager_initialized=true")

        try:
            model = manager.catalog.get_model(model_name)
        except Exception as exc:
            return ValidationResult(
                stage=JobState.FL_LOADING,
                status=ValidationStatus.NOT_VERIFIED,
                checks=tuple(checks + ["catalog.get_model failed"]),
                failure=FailureInfo(
                    stage=JobState.FL_LOADING,
                    classification=FailureClassification.NOT_VERIFIED,
                    message=(
                        f"Model lookup failed in SDK catalog: {exc}. "
                        "Known limitation: SDK discovery may not include custom cache BYOM entries."
                    ),
                ),
            )

        try:
            model.load()
            checks.append("model.load succeeded")
        except Exception as exc:
            return ValidationResult(
                stage=JobState.FL_LOADING,
                status=ValidationStatus.FAILED,
                checks=tuple(checks + ["model.load failed"]),
                failure=FailureInfo(
                    stage=JobState.FL_LOADING,
                    classification=FailureClassification.COMPATIBILITY,
                    message=str(exc),
                ),
            )

        try:
            client = model.get_chat_client()
            if hasattr(client, "settings"):
                client.settings.max_tokens = max_tokens
                client.settings.temperature = 0.0
            response = client.complete_chat([{"role": "user", "content": prompt}])
            choices = getattr(response, "choices", [])
            if not choices:
                raise RuntimeError("No choices returned by Foundry Local SDK chat inference.")
            checks.append("chat completion returned at least one choice")
        except Exception as exc:
            return ValidationResult(
                stage=JobState.INFERENCING,
                status=ValidationStatus.FAILED,
                checks=tuple(checks + ["chat inference failed"]),
                failure=FailureInfo(
                    stage=JobState.INFERENCING,
                    classification=FailureClassification.PROCESS_FAILED,
                    message=str(exc),
                ),
            )
        finally:
            model.unload()

        return ValidationResult(
            stage=JobState.INFERENCING,
            status=ValidationStatus.PASSED,
            checks=tuple(checks),
        )
