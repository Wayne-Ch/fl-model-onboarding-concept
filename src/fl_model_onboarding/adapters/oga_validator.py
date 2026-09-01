from __future__ import annotations

from pathlib import Path

from .interfaces import OgaValidatorClient
from ..contracts import (
    FailureClassification,
    FailureInfo,
    JobState,
    ValidationResult,
    ValidationStatus,
)


class OgaValidator(OgaValidatorClient):
    def validate(self, model_dir: Path) -> ValidationResult:
        resolved = model_dir.resolve()
        checks: list[str] = [f"model_dir={resolved}"]

        config_path = resolved / "genai_config.json"
        if not config_path.exists():
            return ValidationResult(
                stage=JobState.RUNTIME_VALIDATING,
                status=ValidationStatus.FAILED,
                checks=tuple(checks + ["missing genai_config.json"]),
                failure=FailureInfo(
                    stage=JobState.RUNTIME_VALIDATING,
                    classification=FailureClassification.COMPATIBILITY,
                    message="genai_config.json is required for OGA runtime loading.",
                ),
            )

        tokenizer_candidates = (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "vocab.json",
        )
        has_tokenizer = any((resolved / file_name).exists() for file_name in tokenizer_candidates)
        checks.append(f"tokenizer_files_present={has_tokenizer}")
        if not has_tokenizer:
            return ValidationResult(
                stage=JobState.RUNTIME_VALIDATING,
                status=ValidationStatus.FAILED,
                checks=tuple(checks),
                failure=FailureInfo(
                    stage=JobState.RUNTIME_VALIDATING,
                    classification=FailureClassification.COMPATIBILITY,
                    message="Tokenizer files are required for runtime tokenization.",
                ),
            )

        try:
            import onnxruntime_genai as og  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            return ValidationResult(
                stage=JobState.RUNTIME_VALIDATING,
                status=ValidationStatus.NOT_VERIFIED,
                checks=tuple(checks + ["onnxruntime_genai import missing"]),
                failure=FailureInfo(
                    stage=JobState.RUNTIME_VALIDATING,
                    classification=FailureClassification.MISSING_DEPENDENCY,
                    message=str(exc),
                ),
            )

        try:
            _ = og.Model(str(resolved))
        except Exception as exc:
            return ValidationResult(
                stage=JobState.RUNTIME_VALIDATING,
                status=ValidationStatus.FAILED,
                checks=tuple(checks + ["onnxruntime_genai.Model load failed"]),
                failure=FailureInfo(
                    stage=JobState.RUNTIME_VALIDATING,
                    classification=FailureClassification.COMPATIBILITY,
                    message=str(exc),
                ),
            )

        checks.append("onnxruntime_genai.Model load succeeded")
        return ValidationResult(
            stage=JobState.RUNTIME_VALIDATING,
            status=ValidationStatus.PASSED,
            checks=tuple(checks),
        )
