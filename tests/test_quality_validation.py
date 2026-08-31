from __future__ import annotations

import json

from dataclasses import replace
from pathlib import Path

import pytest

from fl_model_onboarding.quality_validation import (
    DEFAULT_TEXT_GENERATION_QUALITY_PROFILE,
    DeterministicInferenceConfig,
    GateState,
    PromptExecutionRecord,
    QualityMetrics,
    evaluate_quality_validation,
    load_quality_validation_profile_registry,
    quality_validation_profiles_path,
    quality_validation_schema_path,
)

_UNCHANGED = object()


def _profile():
    return DEFAULT_TEXT_GENERATION_QUALITY_PROFILE


def _determinism(profile) -> DeterministicInferenceConfig:
    return DeterministicInferenceConfig(
        temperature=profile.deterministic_inference.temperature,
        seed=profile.deterministic_inference.seed,
        max_tokens=profile.deterministic_inference.max_tokens,
    )


def _passing_outputs(profile) -> tuple[PromptExecutionRecord, ...]:
    return (
        PromptExecutionRecord(
            prompt_id="arithmetic-addition-17-plus-28",
            output_text="45",
            applied_determinism=_determinism(profile),
        ),
        PromptExecutionRecord(
            prompt_id="factual-red-planet",
            output_text="Mars",
            applied_determinism=_determinism(profile),
        ),
        PromptExecutionRecord(
            prompt_id="instruction-two-words-blue-river",
            output_text="blue river",
            applied_determinism=_determinism(profile),
        ),
        PromptExecutionRecord(
            prompt_id="format-json-answer-unit",
            output_text='{"answer": 12, "unit": "cm"}',
            applied_determinism=_determinism(profile),
        ),
    )


def _replace_output(
    outputs: tuple[PromptExecutionRecord, ...],
    prompt_id: str,
    *,
    output_text: str | None = None,
    determinism: DeterministicInferenceConfig | None | object = _UNCHANGED,
    unsupported: tuple[str, ...] | None = None,
) -> tuple[PromptExecutionRecord, ...]:
    updated: list[PromptExecutionRecord] = []
    for row in outputs:
        if row.prompt_id != prompt_id:
            updated.append(row)
            continue
        resolved_determinism = row.applied_determinism
        if determinism is not _UNCHANGED:
            resolved_determinism = determinism if isinstance(determinism, DeterministicInferenceConfig) else None
        updated.append(
            PromptExecutionRecord(
                prompt_id=row.prompt_id,
                output_text=output_text if output_text is not None else row.output_text,
                applied_determinism=resolved_determinism,
                unsupported_determinism_fields=(
                    unsupported if unsupported is not None else row.unsupported_determinism_fields
                ),
            )
        )
    return tuple(updated)


def test_quality_validation_passes_and_emits_promotable_evidence() -> None:
    profile = _profile()
    optimized = _passing_outputs(profile)
    baseline = _passing_outputs(profile)
    result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=optimized,
        baseline_outputs=baseline,
        optimized_metrics=QualityMetrics(latency_ms=75.0, peak_memory_mb=842.0, package_size_mb=431.2),
        baseline_metrics=QualityMetrics(latency_ms=72.5, peak_memory_mb=910.0, package_size_mb=622.8),
        require_baseline_comparison=True,
    )
    assert result.optimized_functional.passed is True
    assert result.baseline_functional is not None
    assert result.baseline_functional.passed is True
    assert result.baseline_comparison is not None
    assert result.baseline_comparison.passed is True
    assert result.promotion_evidence.functional_gate == GateState.PASSED
    assert result.promotion_evidence.baseline_comparison_gate == GateState.PASSED
    assert result.promotion_evidence.metrics_gate == GateState.RECORDED
    assert result.promotion_evidence.can_promote is True


def test_quality_validation_detects_baseline_pass_optimized_fail_regression() -> None:
    profile = _profile()
    baseline = _passing_outputs(profile)
    optimized = _replace_output(
        _passing_outputs(profile),
        "instruction-two-words-blue-river",
        output_text="blue blue blue blue blue blue",
    )
    result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=optimized,
        baseline_outputs=baseline,
        require_baseline_comparison=True,
    )
    assert result.optimized_functional.passed is False
    assert result.baseline_comparison is not None
    assert result.baseline_comparison.passed is False
    assert "optimized_failed_prompt:instruction-two-words-blue-river" in result.baseline_comparison.regressions
    assert result.promotion_evidence.can_promote is False


@pytest.mark.parametrize(
    "output_text",
    [
        "",
        "ok ok ok ok ok ok ok ok ok",
        "@@@@ #### $$$$ !!!!",
    ],
)
def test_quality_validation_rejects_empty_repetitive_or_garbled_outputs(output_text: str) -> None:
    profile = _profile()
    optimized = _replace_output(
        _passing_outputs(profile),
        "factual-red-planet",
        output_text=output_text,
    )
    result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=optimized,
        baseline_outputs=_passing_outputs(profile),
        require_baseline_comparison=True,
    )
    assert result.optimized_functional.passed is False
    assert result.promotion_evidence.functional_gate == GateState.FAILED


def test_format_constraint_requires_valid_json_and_keys() -> None:
    profile = _profile()
    optimized = _replace_output(
        _passing_outputs(profile),
        "format-json-answer-unit",
        output_text='{"answer": 12, "unit": "cm"}',
    )
    passing = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=optimized,
        baseline_outputs=_passing_outputs(profile),
        require_baseline_comparison=True,
    )
    assert passing.optimized_functional.passed is True

    failing = _replace_output(
        _passing_outputs(profile),
        "format-json-answer-unit",
        output_text='{"answer": 12}',
    )
    failed_result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=failing,
        baseline_outputs=_passing_outputs(profile),
        require_baseline_comparison=True,
    )
    assert failed_result.optimized_functional.passed is False
    row = next(
        item
        for item in failed_result.optimized_functional.prompt_results
        if item.prompt_id == "format-json-answer-unit"
    )
    assert "json_key_missing:unit" in row.failures


def test_missing_determinism_record_fails_closed() -> None:
    profile = _profile()
    optimized = _replace_output(
        _passing_outputs(profile),
        "arithmetic-addition-17-plus-28",
        determinism=None,
    )
    result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=optimized,
        baseline_outputs=_passing_outputs(profile),
        require_baseline_comparison=True,
    )
    assert result.optimized_functional.passed is False
    row = next(
        item
        for item in result.optimized_functional.prompt_results
        if item.prompt_id == "arithmetic-addition-17-plus-28"
    )
    assert "determinism_not_recorded" in row.failures


def test_partial_determinism_support_is_recorded_not_silently_ignored() -> None:
    profile = _profile()
    applied = DeterministicInferenceConfig(
        temperature=profile.deterministic_inference.temperature,
        seed=999,
        max_tokens=profile.deterministic_inference.max_tokens,
    )
    optimized = _replace_output(
        _passing_outputs(profile),
        "arithmetic-addition-17-plus-28",
        determinism=applied,
        unsupported=("seed",),
    )
    result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=optimized,
        baseline_outputs=_passing_outputs(profile),
        require_baseline_comparison=True,
    )
    assert result.optimized_functional.passed is True
    row = next(
        item
        for item in result.optimized_functional.prompt_results
        if item.prompt_id == "arithmetic-addition-17-plus-28"
    )
    assert row.determinism.recorded is True
    assert row.determinism.fully_enforced is False
    assert row.determinism.unsupported_fields == ("seed",)
    assert any(
        "Runtime could not enforce all deterministic settings" in note
        for note in result.promotion_evidence.notes
    )


def test_metrics_are_optional_for_gate_pass() -> None:
    profile = _profile()
    result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=_passing_outputs(profile),
        baseline_outputs=_passing_outputs(profile),
        require_baseline_comparison=True,
    )
    assert result.metrics.gate_state == GateState.UNAVAILABLE
    assert result.promotion_evidence.metrics_gate == GateState.UNAVAILABLE
    assert result.promotion_evidence.can_promote is True


def test_missing_required_baseline_comparison_blocks_promotion() -> None:
    profile = _profile()
    result = evaluate_quality_validation(
        profile=profile,
        model_task="llm",
        optimized_outputs=_passing_outputs(profile),
        baseline_outputs=None,
        require_baseline_comparison=True,
    )
    assert result.baseline_comparison is None
    assert result.promotion_evidence.baseline_comparison_gate == GateState.MISSING
    assert result.promotion_evidence.can_promote is False


def test_schema_and_profile_data_integrity() -> None:
    schema = json.loads(quality_validation_schema_path().read_text(encoding="utf-8"))
    data = json.loads(quality_validation_profiles_path().read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == data["schema_version"]
    registry = load_quality_validation_profile_registry()
    assert len(registry.all()) >= 1
    profile = registry.get("textgen-basic-quality-v1")
    assert profile.task.value == "text-generation"
    assert len(profile.prompts) == 4


def test_profile_fingerprint_changes_when_profile_changes() -> None:
    profile = _profile()
    changed_version = replace(profile, version="1.0.1")
    changed_prompt = replace(
        profile,
        prompts=(
            replace(profile.prompts[0], prompt=profile.prompts[0].prompt + " now"),
            *profile.prompts[1:],
        ),
    )
    assert changed_version.fingerprint != profile.fingerprint
    assert changed_prompt.fingerprint != profile.fingerprint


def test_loader_rejects_duplicate_prompt_ids(tmp_path: Path) -> None:
    payload = json.loads(quality_validation_profiles_path().read_text(encoding="utf-8"))
    first_prompt = dict(payload["profiles"][0]["prompts"][0])
    payload["profiles"][0]["prompts"].append(first_prompt)
    data_path = tmp_path / "duplicate-prompts.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        load_quality_validation_profile_registry(data_path=data_path, schema_path=quality_validation_schema_path())


def test_loader_rejects_invalid_expected_constraints(tmp_path: Path) -> None:
    payload = json.loads(quality_validation_profiles_path().read_text(encoding="utf-8"))
    payload["profiles"][0]["prompts"][0]["expected"] = {}
    data_path = tmp_path / "invalid-expected.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one assessable constraint"):
        load_quality_validation_profile_registry(data_path=data_path, schema_path=quality_validation_schema_path())


def test_wrong_model_task_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported model task"):
        evaluate_quality_validation(
            profile=_profile(),
            model_task="asr",
            optimized_outputs=_passing_outputs(_profile()),
            baseline_outputs=_passing_outputs(_profile()),
        )


def test_missing_prompt_results_fail_closed() -> None:
    profile = _profile()
    missing = _passing_outputs(profile)[:-1]
    with pytest.raises(ValueError, match="missing prompt results"):
        evaluate_quality_validation(
            profile=profile,
            model_task="llm",
            optimized_outputs=missing,
            baseline_outputs=_passing_outputs(profile),
        )


def test_inconsistent_baseline_and_optimized_cases_fail_closed() -> None:
    profile = _profile()
    baseline = (
        PromptExecutionRecord(
            prompt_id="unexpected-prompt-id",
            output_text="45",
            applied_determinism=_determinism(profile),
        ),
        *_passing_outputs(profile)[1:],
    )
    with pytest.raises(ValueError, match="baseline outputs"):
        evaluate_quality_validation(
            profile=profile,
            model_task="llm",
            optimized_outputs=_passing_outputs(profile),
            baseline_outputs=tuple(baseline),
        )
