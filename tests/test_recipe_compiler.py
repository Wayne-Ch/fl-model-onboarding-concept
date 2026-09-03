from __future__ import annotations

import json

from dataclasses import replace

import pytest

from fl_model_onboarding.architecture_capabilities import (
    ArgumentEvidenceConfidence,
    CapabilityStatus,
    load_architecture_capability_registry,
    normalize_huggingface_metadata,
)
from fl_model_onboarding.recipe_compiler import (
    GENERATED_RECIPE_SCHEMA_VERSION,
    GeneratedRecipeCompileError,
    GeneratedRecipePromotionError,
    PromotionGateCheck,
    PromotionGateEvidence,
    RecipeCompilerInput,
    RecipeCompilerToolchain,
    TrustedCandidateCompilationError,
    compile_generated_recipe,
    compile_trusted_candidate_recipe,
    generated_recipe_schema_path,
    promote_generated_recipe,
)
from fl_model_onboarding.recipe_selection_policy import (
    DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY,
    RecipeQuantizationOverride,
)
from fl_model_onboarding.recipes import RecipeStatus

_REVISION_SHA = "0123456789abcdef0123456789abcdef01234567"
_ALT_REVISION_SHA = "89abcdef0123456789abcdef0123456789abcdef"


def _toolchain() -> RecipeCompilerToolchain:
    return RecipeCompilerToolchain(
        mobius_version="0.1.0",
        olive_version="0.13.0",
        onnx_version="1.22.0",
        ort_version="1.29.0",
        oga_version="0.15.2",
        foundry_sdk_version="1.2.4",
        foundry_cli_version="0.11.0",
    )


def _resolve_capability(
    *,
    model_id: str,
    model_type: str,
    architecture: str,
    requested_precision: str = "int4",
):
    registry = load_architecture_capability_registry()
    metadata = normalize_huggingface_metadata(
        model_id=model_id,
        config={
            "model_type": model_type,
            "architectures": [architecture],
        },
        is_gated=False,
        is_private=False,
    )
    return registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision=requested_precision,
    )


def _llm_input(
    *,
    model_id: str = "example-org/new-text-model",
    model_type: str = "llama",
    architecture: str = "LlamaForCausalLM",
    requested_precision: str = "auto",
) -> RecipeCompilerInput:
    resolution = _resolve_capability(
        model_id=model_id,
        model_type=model_type,
        architecture=architecture,
        requested_precision=requested_precision,
    )
    return RecipeCompilerInput(
        model_id=model_id,
        revision_sha=_REVISION_SHA,
        model_type=model_type,
        architectures=(architecture,),
        task="llm",
        requested_device="cpu",
        requested_precision=requested_precision,
        is_gated=False,
        requires_remote_code=False,
        config_files=("config.json",),
        tokenizer_files=("tokenizer.json",),
        available_files=("config.json", "tokenizer.json", "model.safetensors"),
        capability_resolution=resolution,
        toolchain=_toolchain(),
    )


def _passing_gate_evidence() -> PromotionGateEvidence:
    return PromotionGateEvidence(
        mobius_build=PromotionGateCheck(passed=True, evidence="mobius-log://job-100"),
        olive_optimize=PromotionGateCheck(passed=True, evidence="olive-log://job-100"),
        onnx_validation=PromotionGateCheck(passed=True, evidence="onnx-check://job-100"),
        ort_validation=PromotionGateCheck(passed=True, evidence="ort-load://job-100"),
        oga_validation=PromotionGateCheck(passed=True, evidence="oga-run://job-100"),
        fl_sdk_inference=PromotionGateCheck(passed=True, evidence="fl-sdk://job-100"),
        quality_validation=PromotionGateCheck(passed=True, evidence="quality://job-100"),
    )


def test_verified_capability_still_compiles_experimental_recipe() -> None:
    compiled = compile_generated_recipe(_llm_input())

    assert compiled.recipe.status == RecipeStatus.EXPERIMENTAL
    assert compiled.recipe.verified_revision is None
    assert compiled.recipe.preferred_revision == _REVISION_SHA
    assert compiled.provenance.capability_status == CapabilityStatus.VERIFIED
    assert compiled.provenance.argument_provenance.contains_unverified_arguments is False
    payload = compiled.payload()
    assert payload["schema_version"] == GENERATED_RECIPE_SCHEMA_VERSION


def test_unverified_capability_marks_candidate_argument_provenance() -> None:
    compiled = compile_generated_recipe(
        _llm_input(
            model_id="example-org/q-candidate",
            model_type="qwen2",
            architecture="Qwen2ForCausalLM",
            requested_precision="int4",
        )
    )
    assert compiled.recipe.status == RecipeStatus.EXPERIMENTAL
    assert compiled.recipe.mobius.dtype == "f32"
    assert compiled.recipe.olive is not None
    assert compiled.recipe.olive.precision == "int4"
    assert compiled.provenance.capability_status == CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED
    assert compiled.provenance.argument_provenance.contains_unverified_arguments is True
    assert (
        compiled.provenance.argument_provenance.mobius_dtype_confidence
        == ArgumentEvidenceConfidence.CANDIDATE_UNVERIFIED
    )
    assert (
        compiled.provenance.argument_provenance.olive_precision_confidence
        == ArgumentEvidenceConfidence.CANDIDATE_UNVERIFIED
    )


def test_qwen_mobius_dtype_participates_in_generated_fingerprint() -> None:
    request = _llm_input(
        model_id="example-org/qwen-candidate",
        model_type="qwen2",
        architecture="Qwen2ForCausalLM",
        requested_precision="int4",
    )
    compiled = compile_generated_recipe(request)
    capability = request.capability_resolution.capability
    assert capability is not None
    alternate_capability = replace(
        capability,
        mobius_rules=replace(
            capability.mobius_rules,
            dtype=None,
            dtype_rule="temporary test override",
        ),
    )
    alternate_resolution = replace(request.capability_resolution, capability=alternate_capability)
    alternate = compile_generated_recipe(replace(request, capability_resolution=alternate_resolution))

    assert compiled.recipe.mobius.dtype == "f32"
    assert alternate.recipe.mobius.dtype is None
    assert compiled.fingerprint != alternate.fingerprint
    assert compiled.provenance.capability_status == CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED
    assert alternate.provenance.capability_status == CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED


def test_compilation_is_byte_deterministic_for_same_inputs() -> None:
    request = _llm_input()
    first = compile_generated_recipe(request)
    second = compile_generated_recipe(request)

    assert first.canonical_json == second.canonical_json
    assert first.fingerprint == second.fingerprint
    payload = json.loads(first.canonical_json)
    assert payload["fingerprint"] == first.fingerprint


def test_revision_and_toolchain_change_fingerprint() -> None:
    baseline_request = _llm_input()
    baseline = compile_generated_recipe(baseline_request)

    revision_changed = compile_generated_recipe(
        replace(baseline_request, revision_sha=_ALT_REVISION_SHA)
    )
    toolchain_changed = compile_generated_recipe(
        replace(
            baseline_request,
            toolchain=replace(baseline_request.toolchain, ort_version="1.30.0"),
        )
    )
    assert revision_changed.fingerprint != baseline.fingerprint
    assert toolchain_changed.fingerprint != baseline.fingerprint


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda item: replace(item, is_gated=True), "Gated models are not eligible"),
        (lambda item: replace(item, requires_remote_code=True), "Remote-code models are not eligible"),
        (lambda item: replace(item, config_files=()), "config_files cannot be empty"),
        (lambda item: replace(item, tokenizer_files=()), "tokenizer_files cannot be empty"),
        (lambda item: replace(item, task="asr"), "Unsupported task"),
        (lambda item: replace(item, requested_device="cuda"), "Unsupported device"),
        (lambda item: replace(item, requested_precision="fp32"), "Unsupported requested precision"),
    ],
)
def test_fail_closed_for_ineligible_requests(mutator, message: str) -> None:
    request = _llm_input()
    with pytest.raises(GeneratedRecipeCompileError, match=message):
        compile_generated_recipe(mutator(request))


def test_source_change_required_capability_is_rejected() -> None:
    request = _llm_input()
    capability = request.capability_resolution.capability
    assert capability is not None
    source_gap_resolution = replace(
        request.capability_resolution,
        capability=replace(capability, status=CapabilityStatus.SOURCE_CHANGE_REQUIRED),
    )
    with pytest.raises(GeneratedRecipeCompileError, match="source-change-required"):
        compile_generated_recipe(replace(request, capability_resolution=source_gap_resolution))


def test_ambiguous_ancillary_rules_are_rejected() -> None:
    request = _llm_input()
    capability = request.capability_resolution.capability
    assert capability is not None
    duplicate_rules = (
        capability.required_artifacts[0],
        replace(capability.required_artifacts[1], relative_path="MODEL.ONNX"),
        *capability.required_artifacts[2:],
    )
    duplicate_resolution = replace(
        request.capability_resolution,
        capability=replace(capability, required_artifacts=duplicate_rules),
    )
    with pytest.raises(GeneratedRecipeCompileError, match="Ambiguous ancillary file rule"):
        compile_generated_recipe(replace(request, capability_resolution=duplicate_resolution))


def test_unsafe_ancillary_paths_are_rejected() -> None:
    request = _llm_input()
    capability = request.capability_resolution.capability
    assert capability is not None
    unsafe_rules = (
        replace(capability.required_artifacts[0], relative_path="../model.onnx"),
        *capability.required_artifacts[1:],
    )
    unsafe_resolution = replace(
        request.capability_resolution,
        capability=replace(capability, required_artifacts=unsafe_rules),
    )
    with pytest.raises(GeneratedRecipeCompileError, match="unsafe path component"):
        compile_generated_recipe(replace(request, capability_resolution=unsafe_resolution))


def test_promotion_requires_all_passed_gate_evidence() -> None:
    candidate = compile_generated_recipe(_llm_input())
    failing = _passing_gate_evidence()
    with pytest.raises(GeneratedRecipePromotionError, match="quality_validation"):
        promote_generated_recipe(
            candidate,
            replace(
                failing,
                quality_validation=PromotionGateCheck(passed=False, evidence="quality://missing"),
            ),
            new_version="1.0.1",
            status_reason="Quality gate failed.",
        )
    with pytest.raises(GeneratedRecipePromotionError, match="missing explicit evidence"):
        promote_generated_recipe(
            candidate,
            replace(
                failing,
                ort_validation=PromotionGateCheck(passed=True, evidence=" "),
            ),
            new_version="1.0.1",
            status_reason="Missing ORT evidence.",
        )


def test_promotion_creates_new_verified_version_without_mutating_candidate() -> None:
    candidate = compile_generated_recipe(_llm_input())
    promoted = promote_generated_recipe(
        candidate,
        _passing_gate_evidence(),
        new_version="1.0.1",
        status_reason="All required runtime and quality gates passed.",
    )

    assert candidate.recipe.status == RecipeStatus.EXPERIMENTAL
    assert promoted.recipe.status == RecipeStatus.VERIFIED
    assert promoted.recipe.version == "1.0.1"
    assert promoted.recipe.verified_revision == candidate.pinned_revision
    assert promoted.recipe.preferred_revision == candidate.pinned_revision
    assert promoted.provenance.promotion is not None
    assert promoted.provenance.promotion.promoted_from_fingerprint == candidate.fingerprint
    assert promoted.fingerprint != candidate.fingerprint


def test_model_id_only_affects_identity_not_capability_arguments() -> None:
    first = compile_generated_recipe(
        _llm_input(model_id="example-org/alpha-text-model")
    )
    second = compile_generated_recipe(
        _llm_input(model_id="another-org/beta-text-model")
    )

    assert first.recipe.status == RecipeStatus.EXPERIMENTAL
    assert second.recipe.status == RecipeStatus.EXPERIMENTAL
    assert first.recipe.mobius == second.recipe.mobius
    assert first.recipe.olive == second.recipe.olive
    assert first.recipe.ancillary_files == second.recipe.ancillary_files
    assert first.provenance.capability_id == second.provenance.capability_id
    assert first.recipe.id != second.recipe.id


def test_generated_recipe_schema_contract_is_present() -> None:
    schema = json.loads(generated_recipe_schema_path().read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == GENERATED_RECIPE_SCHEMA_VERSION
    assert "provenance" in schema["$defs"]


# --- Slice 3A1: trusted candidate compilation -------------------------------


def test_trusted_candidate_default_is_returned_unchanged() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY

    result = compile_trusted_candidate_recipe(
        default,
        policy=policy,
        candidate=policy.candidates[0],
    )

    assert result is default
    assert result.fingerprint == default.fingerprint
    assert result.recipe.olive is not None
    assert result.recipe.olive.block_size is None


def test_trusted_candidate_block64_applies_override_and_changes_fingerprint() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    fallback_candidate = policy.candidates[1]

    result = compile_trusted_candidate_recipe(
        default,
        policy=policy,
        candidate=fallback_candidate,
    )

    assert result.recipe.olive is not None
    assert default.recipe.olive is not None
    assert result.recipe.olive.block_size == 64
    assert result.recipe.olive.device == default.recipe.olive.device
    assert result.recipe.olive.precision == default.recipe.olive.precision
    assert result.recipe.olive.task == default.recipe.olive.task
    assert result.recipe.olive.provider == default.recipe.olive.provider
    assert result.fingerprint != default.fingerprint
    assert result.payload()["recipe"]["olive"]["block_size"] == 64

    # Source revision/toolchain/capability/profile identity is preserved.
    assert result.pinned_revision == default.pinned_revision
    assert result.provenance.toolchain == default.provenance.toolchain
    assert result.provenance.capability_id == default.provenance.capability_id
    assert result.provenance.capability_version == default.provenance.capability_version
    assert result.recipe.task_profile == default.recipe.task_profile
    assert result.recipe.mobius == default.recipe.mobius

    trusted = result.provenance.trusted_candidate
    assert trusted is not None
    assert trusted.policy_id == policy.policy_id
    assert trusted.policy_version == policy.version
    assert trusted.policy_fingerprint == policy.fingerprint
    assert trusted.candidate_index == 1
    assert trusted.candidate_id == fallback_candidate.candidate_id
    assert trusted.resolved_block_size == 64
    assert default.provenance.trusted_candidate is None


def test_trusted_candidate_is_byte_deterministic() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    first = compile_trusted_candidate_recipe(default, policy=policy, candidate=policy.candidates[1])
    second = compile_trusted_candidate_recipe(default, policy=policy, candidate=policy.candidates[1])

    assert first.canonical_json == second.canonical_json
    assert first.fingerprint == second.fingerprint


def test_trusted_candidate_does_not_branch_on_model_identity() -> None:
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    first_default = compile_generated_recipe(_llm_input(model_id="example-org/alpha-text-model"))
    second_default = compile_generated_recipe(_llm_input(model_id="another-org/beta-text-model"))

    first = compile_trusted_candidate_recipe(first_default, policy=policy, candidate=policy.candidates[1])
    second = compile_trusted_candidate_recipe(second_default, policy=policy, candidate=policy.candidates[1])

    assert first.recipe.olive is not None
    assert second.recipe.olive is not None
    assert first.recipe.olive.block_size == second.recipe.olive.block_size == 64
    assert first.provenance.trusted_candidate is not None
    assert second.provenance.trusted_candidate is not None
    assert first.provenance.trusted_candidate.candidate_id == second.provenance.trusted_candidate.candidate_id


def test_trusted_candidate_rejects_unapproved_policy_version() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    altered_policy = replace(policy, version="9.9.9")

    with pytest.raises(TrustedCandidateCompilationError, match="does not exactly match"):
        compile_trusted_candidate_recipe(default, policy=altered_policy, candidate=policy.candidates[1])


def test_trusted_candidate_rejects_unknown_policy_id() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    bogus_policy = replace(policy, policy_id="not-a-real-policy")

    with pytest.raises(TrustedCandidateCompilationError, match="Unknown recipe selection policy_id"):
        compile_trusted_candidate_recipe(default, policy=bogus_policy, candidate=policy.candidates[1])


def test_trusted_candidate_rejects_arbitrary_block_size() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    tampered_candidate = replace(
        policy.candidates[1],
        quantization_override=RecipeQuantizationOverride(block_size=128),
    )

    with pytest.raises(TrustedCandidateCompilationError, match="does not exactly match"):
        compile_trusted_candidate_recipe(default, policy=policy, candidate=tampered_candidate)


def test_trusted_candidate_rejects_out_of_range_candidate_index() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    out_of_range = replace(policy.candidates[1], candidate_index=5)

    with pytest.raises(TrustedCandidateCompilationError, match="out of range"):
        compile_trusted_candidate_recipe(default, policy=policy, candidate=out_of_range)


def test_trusted_candidate_rejects_device_change_attempt() -> None:
    default = compile_generated_recipe(_llm_input())
    assert default.recipe.olive is not None
    tampered_default = replace(
        default,
        recipe=replace(default.recipe, olive=replace(default.recipe.olive, device="cuda")),
    )
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY

    with pytest.raises(TrustedCandidateCompilationError, match="does not match policy target device"):
        compile_trusted_candidate_recipe(tampered_default, policy=policy, candidate=policy.candidates[1])


def test_trusted_candidate_rejects_precision_change_attempt() -> None:
    default = compile_generated_recipe(_llm_input())
    assert default.recipe.olive is not None
    tampered_default = replace(
        default,
        recipe=replace(default.recipe, olive=replace(default.recipe.olive, precision="fp32")),
    )
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY

    with pytest.raises(TrustedCandidateCompilationError, match="does not match policy quantization"):
        compile_trusted_candidate_recipe(tampered_default, policy=policy, candidate=policy.candidates[1])


def test_trusted_candidate_default_rejects_declared_override() -> None:
    default = compile_generated_recipe(_llm_input())
    policy = DEFAULT_CPU_INT4_RECIPE_SELECTION_POLICY
    tampered_default_candidate = replace(
        policy.candidates[0],
        quantization_override=RecipeQuantizationOverride(block_size=64),
    )

    # A tampered candidate 0 is caught by the byte-for-byte identity check
    # against the trusted registry before the "no override on candidate 0"
    # check is ever reached -- defense in depth, not a gap.
    with pytest.raises(TrustedCandidateCompilationError, match="does not exactly match"):
        compile_trusted_candidate_recipe(default, policy=policy, candidate=tampered_default_candidate)
