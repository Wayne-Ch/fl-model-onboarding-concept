from __future__ import annotations

import json

from pathlib import Path

import yaml


def test_openapi_contains_required_paths() -> None:
    path = Path("contracts") / "openapi.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    paths = spec["paths"]
    assert "/api/health" in paths
    assert "/api/models/search" in paths
    assert "/api/models/detail" in paths
    assert "/api/models/preflight" in paths
    assert "/api/builds" in paths
    assert "/api/builds/{job_id}/events" in paths
    assert "/api/artifacts/{artifact_id}/infer/text" in paths
    assert "/api/artifacts/{artifact_id}/infer/asr" in paths
    parameters = paths["/api/builds"]["post"]["parameters"]
    idempotency = next(p for p in parameters if p["name"] == "Idempotency-Key")
    assert idempotency["required"] is True

    build_job = spec["components"]["schemas"]["BuildJob"]
    assert "request" in build_job["properties"]
    assert "started_utc" in build_job["properties"]
    assert "finished_utc" in build_job["properties"]
    assert "validations" in build_job["properties"]

    failure_classification = spec["components"]["schemas"]["FailureClassification"]["enum"]
    assert "not_verified" in failure_classification
    assert "tool_unavailable" in failure_classification
    assert "source_runtime_contract_incompatible" in failure_classification
    assert "oga_runtime_contract_incompatible" in failure_classification
    attempt_gate_status = spec["components"]["schemas"]["RecipeAttemptGate"]["properties"]["status"]["enum"]
    assert attempt_gate_status == ["passed", "failed", "not_run", "unavailable"]


def test_openapi_candidate_selection_response_contract() -> None:
    """Slice 3C1: candidate plan (preview) and candidate timeline/selection/
    reuse-evidence (attempt) schemas exactly match the runtime-serialized
    field names/enums the service actually emits."""
    path = Path("contracts") / "openapi.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]

    preview = schemas["GeneratedRecipePreview"]["properties"]
    assert "candidate_plan" in preview
    assert "candidate_selection_reuse" in preview

    plan = schemas["CandidateSelectionPlan"]
    assert set(plan["required"]) == {
        "policy_id",
        "policy_version",
        "policy_fingerprint",
        "max_candidates",
        "candidates",
    }
    role_enum = schemas["CandidateRole"]["enum"]
    assert role_enum == ["default", "quality_retry"]

    plan_entry_required = set(schemas["CandidatePlanEntry"]["required"])
    assert plan_entry_required == {"candidate_index", "candidate_id", "role", "eligibility_trigger"}

    attempt = schemas["RecipeAttempt"]["properties"]
    assert attempt["workflow_outcome"]["enum"] == [
        "not_applicable",
        "pending",
        "selected",
        "exhausted",
        "reused",
    ]
    assert "candidate_selection" in attempt

    selection = schemas["RecipeAttemptCandidateSelection"]["properties"]
    lineage_selection_state = selection["lineage_selection_state"]
    lineage_selection_state_variants = lineage_selection_state["oneOf"]
    assert {"type": "null"} in lineage_selection_state_variants
    string_variant = next(variant for variant in lineage_selection_state_variants if variant.get("type") == "string")
    assert string_variant["enum"] == ["pending", "selected", "exhausted"]
    assert "lineage_selection_state" in schemas["RecipeAttemptCandidateSelection"]["required"]
    assert set(selection.keys()) == {
        "policy_id",
        "policy_version",
        "policy_fingerprint",
        "max_candidates",
        "lineage_selection_state",
        "selected_candidate",
        "candidates",
        "aggregate_invocation_counters",
        "reuse",
    }

    timeline_entry_required = set(schemas["CandidateTimelineEntry"]["required"])
    assert {
        "candidate_attempt_id",
        "attempt_id",
        "candidate_index",
        "candidate_id",
        "role",
        "attempt_state",
        "selection_status",
        "invocation_counters",
        "validated_scope",
    }.issubset(timeline_entry_required)

    reuse_required = set(schemas["CandidateReuseEvidence"]["required"])
    assert reuse_required == {
        "reused_without_build",
        "source_attempt_id",
        "source_candidate_attempt_id",
        "source_parent_attempt_id",
        "policy_id",
        "policy_version",
        "policy_fingerprint",
        "quality_profile_fingerprint",
        "runner_dispatch_count",
        "mobius_invocation_count",
        "olive_invocation_count",
        "recorded_utc",
    }


def test_state_machine_contract_contains_cancellable_flags() -> None:
    path = Path("contracts") / "job-state-machine.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    states = {row["name"]: row for row in contract["states"]}
    assert states["mobius_building"]["cancellable"] is True
    assert states["succeeded"]["cancellable"] is False


def test_generated_recipe_schema_contains_required_top_level_keys() -> None:
    path = Path("contracts") / "generated-recipe.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "1.0.0"
    required = set(schema["required"])
    assert "recipe" in required
    assert "pinned_revision" in required
    assert "provenance" in required
    assert "fingerprint" in required


def test_recipe_attempt_schema_contains_core_record_definitions() -> None:
    path = Path("contracts") / "recipe-attempt.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    defs = schema["$defs"]
    assert "generated_recipe_record" in defs
    assert "recipe_attempt" in defs
    assert "verified_recipe_record" in defs
    assert defs["generated_recipe_record"]["properties"]["schema_version"]["const"] == "1.0.0"
    assert defs["attempt_state"]["enum"] == [
        "generated",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    ]
    assert defs["attempt_gate_status"]["enum"] == [
        "passed",
        "failed",
        "not_run",
        "unavailable",
    ]
