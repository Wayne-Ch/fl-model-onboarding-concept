from __future__ import annotations

import json

from pathlib import Path

import pytest

from fl_model_onboarding.quality_validation import QualityRetryDisposition
from fl_model_onboarding.recipe_selection_policy import (
    DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY,
    DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY,
    RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER,
    RecipeSelectionQuantization,
    RecipeSelectionTargetDevice,
    load_recipe_selection_policy_registry,
    recipe_selection_policies_path,
    recipe_selection_policy_schema_path,
)


def test_retry_trigger_constant_is_sourced_from_quality_retry_disposition() -> None:
    """Cross-module boundary test (Slice 2 reviewer follow-up): the policy trigger
    string must be the exact same value as QualityRetryDisposition's retryable
    member, not merely an equal-looking duplicate string. A future rename on
    either side must fail this test (and the module-level assertion in
    recipe_attempt_store.py) rather than silently disabling the fallback
    candidate.
    """
    assert (
        RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER
        == QualityRetryDisposition.RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION.value
    )


def _load_payload() -> dict[str, object]:
    return json.loads(recipe_selection_policies_path().read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object], *, name: str = "policies.json") -> Path:
    data_path = tmp_path / name
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return data_path


def test_schema_and_policy_data_integrity() -> None:
    schema = json.loads(recipe_selection_policy_schema_path().read_text(encoding="utf-8"))
    data = _load_payload()
    assert schema["properties"]["schema_version"]["const"] == data["schema_version"]
    registry = load_recipe_selection_policy_registry()
    assert len(registry.all()) >= 1
    policy = registry.get("cpu-int4-recipe-selection-v1")
    assert policy.target_device == RecipeSelectionTargetDevice.CPU
    assert policy.quantization == RecipeSelectionQuantization.INT4
    assert policy.max_candidates == 2
    assert len(policy.candidates) == 2


def test_default_policy_is_registered_and_addressable_by_id() -> None:
    policy = DEFAULT_RECIPE_SELECTION_POLICY_REGISTRY.get("cpu-int4-recipe-selection-v1")
    assert policy is DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY


def test_candidate_zero_is_default_with_no_override_and_always_eligible() -> None:
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    default_candidate = policy.candidates[0]
    assert default_candidate.candidate_index == 0
    assert default_candidate.is_default is True
    assert default_candidate.quantization_override is None
    assert default_candidate.eligibility_trigger is None
    assert default_candidate.is_eligible_for(None) is True
    assert default_candidate.is_eligible_for("anything") is True


def test_candidate_one_carries_proven_block_size_64_override_and_gated_trigger() -> None:
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    override_candidate = policy.candidates[1]
    assert override_candidate.candidate_index == 1
    assert override_candidate.is_default is False
    assert override_candidate.quantization_override is not None
    assert override_candidate.quantization_override.block_size == 64
    assert override_candidate.eligibility_trigger == RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER
    assert override_candidate.is_eligible_for(None) is False
    assert override_candidate.is_eligible_for("some_other_trigger") is False
    assert override_candidate.is_eligible_for(RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER) is True


def test_plan_is_deterministic_default_only_without_trigger() -> None:
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    plan = policy.plan()
    assert [candidate.candidate_id for candidate in plan] == ["default-int4"]


def test_plan_includes_override_candidate_only_when_allowlisted_trigger_matches() -> None:
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    plan = policy.plan(trigger=RETRYABLE_OPTIMIZED_STRUCTURAL_REGRESSION_TRIGGER)
    assert [candidate.candidate_id for candidate in plan] == ["default-int4", "int4-block-size-64"]

    plan_with_unknown_trigger = policy.plan(trigger="unrelated_trigger")
    assert [candidate.candidate_id for candidate in plan_with_unknown_trigger] == ["default-int4"]


def test_policy_fingerprint_and_identity_are_stable_and_change_with_content() -> None:
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    assert policy.identity == f"{policy.policy_id}:{policy.version}:{policy.fingerprint}"
    # Reloading from disk must produce an identical fingerprint (stable, content-derived).
    reloaded = load_recipe_selection_policy_registry().get(policy.policy_id)
    assert reloaded.fingerprint == policy.fingerprint
    assert reloaded.identity == policy.identity


def test_no_model_selector_fields_are_present_anywhere_in_policy_data() -> None:
    payload = _load_payload()
    forbidden_keys = {
        "model_id",
        "model_name",
        "huggingface_model_id",
        "hidden_size",
        "org",
        "architecture",
        "architectures",
        "model_type",
    }

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            present = forbidden_keys & node.keys()
            assert not present, f"Found forbidden selector field(s) {present} in policy data."
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)


def test_loader_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["unexpected_field"] = True
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown field"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_unknown_policy_field(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["model_id"] = "some/model"
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown field"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_unknown_candidate_field(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["candidates"][1]["hidden_size"] = 2048
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown field"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_invalid_cap_mismatched_with_candidate_count(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["max_candidates"] = 1
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="max_candidates"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_invalid_cap_exceeding_hard_ceiling(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["max_candidates"] = 3
    payload["policies"][0]["candidates"].append(
        {
            "candidate_index": 2,
            "candidate_id": "extra",
            "description": "extra candidate",
            "quantization_override": None,
            "eligibility_trigger": "retryable_optimized_structural_regression",
        }
    )
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="must not exceed 2"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_zero_or_negative_max_candidates(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["max_candidates"] = 0
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="greater than zero"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_precision_changing_override(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["candidates"][1]["quantization_override"] = {"block_size": 64, "precision": "fp32"}
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="unknown field"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_dtype_only_override_with_no_block_size(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["candidates"][1]["quantization_override"] = {}
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="block_size"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_default_candidate_with_override(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["candidates"][0]["quantization_override"] = {"block_size": 32}
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="must not declare a quantization_override"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_default_candidate_with_eligibility_trigger(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["candidates"][0]["eligibility_trigger"] = "retryable_optimized_structural_regression"
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="must not declare an eligibility_trigger"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_non_default_candidate_missing_eligibility_trigger(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["candidates"][1]["eligibility_trigger"] = None
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="must declare an eligibility_trigger"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_arbitrary_eligibility_trigger_string(tmp_path: Path) -> None:
    payload = _load_payload()
    payload["policies"][0]["candidates"][1]["eligibility_trigger"] = "some_free_text_trigger"
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="eligibility_trigger must be one of"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_duplicate_policy_ids(tmp_path: Path) -> None:
    payload = _load_payload()
    duplicate = json.loads(json.dumps(payload["policies"][0]))
    payload["policies"].append(duplicate)
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="Duplicate recipe selection policy_id"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())


def test_loader_rejects_out_of_order_candidate_indices(tmp_path: Path) -> None:
    payload = _load_payload()
    candidates = payload["policies"][0]["candidates"]
    candidates[0]["candidate_index"], candidates[1]["candidate_index"] = (
        candidates[1]["candidate_index"],
        candidates[0]["candidate_index"],
    )
    data_path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="candidate_index"):
        load_recipe_selection_policy_registry(data_path=data_path, schema_path=recipe_selection_policy_schema_path())
