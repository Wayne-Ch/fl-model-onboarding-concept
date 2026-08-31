from __future__ import annotations

import hashlib
import json
import re

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .architecture_capabilities import (
    ArgumentEvidenceConfidence,
    ArchitectureCapability,
    CapabilityModality,
    CapabilityResolution,
    CapabilityStatus,
    ResolutionOutcome,
)
from .contracts import CandidateModality
from .recipes import (
    AncillaryFileRule,
    MobiusRecipeArgs,
    ModelRecipe,
    OliveRecipeArgs,
    OptimizationChoice,
    RecipeStatus,
)

RECIPE_AGENT_COMPILER_VERSION = "1.0.0"
GENERATED_RECIPE_SCHEMA_VERSION = "1.0.0"

_ALLOWED_TASK_ALIASES = frozenset({"llm", "textgeneration"})
_ALLOWED_DEVICE_ALIASES = frozenset({"cpu"})
_ALLOWED_REQUEST_PRECISIONS = frozenset({"auto", "int4"})
_ALLOWED_EFFECTIVE_PRECISIONS = frozenset({"int4"})
_TOKENIZER_FILE_NAMES = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    }
)

_REVISION_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


class GeneratedRecipeCompileError(ValueError):
    """Raised when deterministic candidate recipe compilation must fail closed."""


class GeneratedRecipePromotionError(ValueError):
    """Raised when candidate promotion requirements are not fully satisfied."""


@dataclass(frozen=True)
class RecipeCompilerToolchain:
    mobius_version: str
    olive_version: str
    onnx_version: str
    ort_version: str
    oga_version: str
    foundry_sdk_version: str
    foundry_cli_version: str


@dataclass(frozen=True)
class RecipeCompilerInput:
    model_id: str
    revision_sha: str
    model_type: str | None
    architectures: tuple[str, ...]
    task: str
    requested_device: str
    requested_precision: str
    is_gated: bool
    requires_remote_code: bool
    config_files: tuple[str, ...]
    tokenizer_files: tuple[str, ...]
    available_files: tuple[str, ...]
    capability_resolution: CapabilityResolution
    toolchain: RecipeCompilerToolchain


@dataclass(frozen=True)
class RecipeInputMetadata:
    model_id: str
    revision_sha: str
    model_type: str | None
    architectures: tuple[str, ...]
    task: str
    requested_device: str
    requested_precision: str
    is_gated: bool
    requires_remote_code: bool
    config_files: tuple[str, ...]
    tokenizer_files: tuple[str, ...]
    available_files: tuple[str, ...]


@dataclass(frozen=True)
class RecipeArgumentProvenance:
    mobius_dtype_rule: str
    mobius_dtype_confidence: ArgumentEvidenceConfidence
    olive_precision_rule: str
    olive_precision_confidence: ArgumentEvidenceConfidence
    contains_unverified_arguments: bool


@dataclass(frozen=True)
class PromotionGateCheck:
    passed: bool
    evidence: str


@dataclass(frozen=True)
class PromotionGateEvidence:
    mobius_build: PromotionGateCheck
    olive_optimize: PromotionGateCheck
    onnx_validation: PromotionGateCheck
    ort_validation: PromotionGateCheck
    oga_validation: PromotionGateCheck
    fl_sdk_inference: PromotionGateCheck
    quality_validation: PromotionGateCheck


@dataclass(frozen=True)
class RecipePromotionRecord:
    promoted_from_fingerprint: str
    new_version: str
    status_reason: str
    gate_evidence: PromotionGateEvidence


@dataclass(frozen=True)
class RecipeGenerationProvenance:
    compiler_version: str
    generation_kind: str
    capability_id: str
    capability_version: str
    capability_status: CapabilityStatus
    resolution_outcome: ResolutionOutcome
    resolution_reason_code: str
    matched_aliases: tuple[str, ...]
    argument_provenance: RecipeArgumentProvenance
    evidence: tuple[dict[str, str], ...]
    toolchain: RecipeCompilerToolchain
    input_metadata: RecipeInputMetadata
    promotion: RecipePromotionRecord | None = None


@dataclass(frozen=True)
class GeneratedRecipe:
    recipe: ModelRecipe
    pinned_revision: str
    provenance: RecipeGenerationProvenance
    fingerprint: str
    canonical_json: str

    def payload(self) -> dict[str, object]:
        parsed = json.loads(self.canonical_json)
        if not isinstance(parsed, dict):
            raise ValueError("Generated recipe payload must be an object.")
        return parsed


def generated_recipe_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts" / "generated-recipe.schema.json"


def compile_generated_recipe(
    request: RecipeCompilerInput,
    *,
    schema_path: Path | None = None,
) -> GeneratedRecipe:
    model_id = _coerce_str(request.model_id, "model_id")
    revision_sha = _normalize_revision_sha(request.revision_sha)
    model_type = _coerce_optional_str(request.model_type, "model_type")
    architectures = _normalize_strings(
        request.architectures,
        path="architectures",
        allow_empty=True,
    )
    if model_type is None and not architectures:
        raise GeneratedRecipeCompileError(
            "Either model_type or architectures must be present for deterministic compilation."
        )

    task = _normalize_task(request.task)
    device = _normalize_device(request.requested_device)
    requested_precision = _normalize_requested_precision(request.requested_precision)

    if request.is_gated:
        raise GeneratedRecipeCompileError("Gated models are not eligible for generated recipes.")
    if request.requires_remote_code:
        raise GeneratedRecipeCompileError("Remote-code models are not eligible for generated recipes.")

    resolution = request.capability_resolution
    if resolution.outcome != ResolutionOutcome.EXACT or resolution.capability is None:
        raise GeneratedRecipeCompileError(
            "Capability resolution must be exact with a resolved capability to compile a recipe."
        )

    capability = resolution.capability
    if capability.modality != CapabilityModality.LLM:
        raise GeneratedRecipeCompileError(
            "Only text-generation/LLM capabilities are eligible for deterministic generated recipes."
        )
    if capability.status == CapabilityStatus.SOURCE_CHANGE_REQUIRED:
        raise GeneratedRecipeCompileError(
            "Capability is source-change-required and cannot compile to a recipe candidate."
        )
    if capability.status not in {
        CapabilityStatus.VERIFIED,
        CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED,
    }:
        raise GeneratedRecipeCompileError(
            f"Unsupported capability status '{capability.status.value}' for recipe compilation."
        )

    supported_tasks = {item.value for item in capability.static_eligibility_constraints.supported_tasks}
    supported_devices = {item.value for item in capability.static_eligibility_constraints.supported_devices}
    supported_request_precisions = {
        item.value for item in capability.static_eligibility_constraints.supported_request_precisions
    }
    if task not in supported_tasks:
        raise GeneratedRecipeCompileError(
            f"Capability '{capability.capability_id}' does not support task '{task}'."
        )
    if device not in supported_devices:
        raise GeneratedRecipeCompileError(
            f"Capability '{capability.capability_id}' does not support device '{device}'."
        )
    if requested_precision not in supported_request_precisions:
        raise GeneratedRecipeCompileError(
            f"Capability '{capability.capability_id}' does not support requested precision '{requested_precision}'."
        )

    config_files = _normalize_paths(request.config_files, path="config_files", allow_empty=False)
    tokenizer_files = _normalize_paths(request.tokenizer_files, path="tokenizer_files", allow_empty=False)
    available_files = _normalize_paths(request.available_files, path="available_files", allow_empty=False)
    _require_subset(config_files, available_files, path="config_files")
    _require_subset(tokenizer_files, available_files, path="tokenizer_files")
    if not any(PurePosixPath(path).name.lower() == "config.json" for path in config_files):
        raise GeneratedRecipeCompileError(
            "Missing required config metadata file: config.json."
        )
    if not any(PurePosixPath(path).name.lower() in _TOKENIZER_FILE_NAMES for path in tokenizer_files):
        raise GeneratedRecipeCompileError(
            "Missing required tokenizer metadata file."
        )

    normalized_toolchain = _normalize_toolchain(request.toolchain)
    optimization_choices = _compile_optimization_choices(capability)
    effective_precision = _resolve_effective_precision(
        requested_precision=requested_precision,
        capability_olive_precision=capability.olive_rules.precision,
        optimization_choices=optimization_choices,
    )
    if capability.olive_rules.precision is not None:
        pinned_precision = _normalize_effective_precision(
            capability.olive_rules.precision,
            "capability.olive_rules.precision",
        )
        if pinned_precision != effective_precision:
            raise GeneratedRecipeCompileError(
                "Capability Olive precision conflicts with requested/effective precision."
            )
    selected_choice = _select_choice_for_precision(optimization_choices, effective_precision)
    ancillary_files = _compile_ancillary_file_rules(capability)
    cache_prefix = _build_cache_prefix(model_id)

    recipe = ModelRecipe(
        id=_build_recipe_id(model_id=model_id, effective_precision=effective_precision),
        version="0.1.0",
        status=RecipeStatus.EXPERIMENTAL,
        status_reason=_candidate_status_reason(
            capability_id=capability.capability_id,
            capability_status=capability.status,
        ),
        huggingface_model_id=model_id,
        modality=CandidateModality(capability.modality.value),
        task_profile=selected_choice.task_profile,
        verified_revision=None,
        preferred_revision=revision_sha,
        mobius=MobiusRecipeArgs(
            ep=capability.mobius_rules.ep,
            runtime=capability.mobius_rules.runtime,
            dtype=capability.mobius_rules.dtype,
            task=capability.mobius_rules.task,
        ),
        olive=OliveRecipeArgs(
            input_source="mobius-output-dir",
            task=capability.olive_rules.task,
            precision=effective_precision,
            device=capability.olive_rules.device,
            provider=capability.olive_rules.provider,
            log_level="1",
        ),
        ancillary_files=ancillary_files,
        runtime_validation=(
            f"{capability.oga_runtime_contract.validator_profile} "
            f"({capability.oga_runtime_contract.runtime}; loader={capability.oga_runtime_contract.loader})"
        ),
        inference_modality=CandidateModality(capability.modality.value),
        optimization_choices=optimization_choices,
        artifact_cache_prefix=cache_prefix,
        model_name_prefix=f"{cache_prefix}-onboarding",
        success_message=(
            "Candidate recipe compiled deterministically; promotion requires successful "
            "Mobius, Olive, ONNX, ORT, OGA, FL SDK inference, and quality gates."
        ),
    )

    provenance = RecipeGenerationProvenance(
        compiler_version=RECIPE_AGENT_COMPILER_VERSION,
        generation_kind="deterministic-recipe-agent-v1",
        capability_id=capability.capability_id,
        capability_version=capability.version,
        capability_status=capability.status,
        resolution_outcome=resolution.outcome,
        resolution_reason_code=resolution.reason_code.value,
        matched_aliases=tuple(sorted(resolution.matched_aliases)),
        argument_provenance=RecipeArgumentProvenance(
            mobius_dtype_rule=capability.mobius_rules.dtype_rule,
            mobius_dtype_confidence=capability.mobius_rules.dtype_confidence,
            olive_precision_rule=capability.olive_rules.precision_rule,
            olive_precision_confidence=capability.olive_rules.precision_confidence,
            contains_unverified_arguments=(
                capability.mobius_rules.dtype_confidence
                == ArgumentEvidenceConfidence.CANDIDATE_UNVERIFIED
                or capability.olive_rules.precision_confidence
                == ArgumentEvidenceConfidence.CANDIDATE_UNVERIFIED
            ),
        ),
        evidence=tuple(
            {
                "evidence_id": row.evidence_id,
                "source_type": row.source_type,
                "location": row.location,
                "summary": row.summary,
            }
            for row in sorted(capability.evidence, key=lambda item: item.evidence_id)
        ),
        toolchain=normalized_toolchain,
        input_metadata=RecipeInputMetadata(
            model_id=model_id,
            revision_sha=revision_sha,
            model_type=model_type,
            architectures=architectures,
            task=task,
            requested_device=device,
            requested_precision=requested_precision,
            is_gated=False,
            requires_remote_code=False,
            config_files=config_files,
            tokenizer_files=tokenizer_files,
            available_files=available_files,
        ),
    )
    return _materialize_generated_recipe(
        recipe=recipe,
        pinned_revision=revision_sha,
        provenance=provenance,
        schema_path=schema_path,
    )


def promote_generated_recipe(
    candidate: GeneratedRecipe,
    gate_evidence: PromotionGateEvidence,
    *,
    new_version: str,
    status_reason: str,
    schema_path: Path | None = None,
) -> GeneratedRecipe:
    if candidate.recipe.status != RecipeStatus.EXPERIMENTAL:
        raise GeneratedRecipePromotionError(
            f"Only experimental candidates can be promoted; got '{candidate.recipe.status.value}'."
        )
    normalized_new_version = _coerce_str(new_version, "new_version")
    if not _SEMVER_RE.fullmatch(normalized_new_version):
        raise GeneratedRecipePromotionError(
            "new_version must be an explicit semantic version like '1.2.3'."
        )
    if normalized_new_version == candidate.recipe.version:
        raise GeneratedRecipePromotionError(
            "Promotion must create a new recipe version and cannot reuse the candidate version."
        )
    normalized_status_reason = _coerce_str(status_reason, "status_reason")
    _validate_promotion_gate_evidence(gate_evidence)

    promoted_recipe = replace(
        candidate.recipe,
        version=normalized_new_version,
        status=RecipeStatus.VERIFIED,
        status_reason=normalized_status_reason,
        verified_revision=candidate.pinned_revision,
        preferred_revision=candidate.pinned_revision,
        success_message=(
            "Verified recipe promoted after Mobius, Olive, ONNX, ORT, OGA, FL SDK inference, "
            "and quality validation gates passed."
        ),
    )
    promoted_provenance = replace(
        candidate.provenance,
        promotion=RecipePromotionRecord(
            promoted_from_fingerprint=candidate.fingerprint,
            new_version=normalized_new_version,
            status_reason=normalized_status_reason,
            gate_evidence=gate_evidence,
        ),
    )
    return _materialize_generated_recipe(
        recipe=promoted_recipe,
        pinned_revision=candidate.pinned_revision,
        provenance=promoted_provenance,
        schema_path=schema_path,
    )


def _materialize_generated_recipe(
    *,
    recipe: ModelRecipe,
    pinned_revision: str,
    provenance: RecipeGenerationProvenance,
    schema_path: Path | None,
) -> GeneratedRecipe:
    normalized_revision = _normalize_revision_sha(pinned_revision)
    payload_without_fingerprint = {
        "schema_version": GENERATED_RECIPE_SCHEMA_VERSION,
        "recipe": _recipe_to_payload(recipe),
        "pinned_revision": normalized_revision,
        "provenance": _provenance_to_payload(provenance),
    }
    fingerprint = hashlib.sha256(
        _canonical_json(payload_without_fingerprint).encode("utf-8")
    ).hexdigest()
    payload = dict(payload_without_fingerprint)
    payload["fingerprint"] = fingerprint
    _validate_generated_recipe_payload(payload, schema_path=schema_path)
    canonical_json = _canonical_json(payload)
    return GeneratedRecipe(
        recipe=recipe,
        pinned_revision=normalized_revision,
        provenance=provenance,
        fingerprint=fingerprint,
        canonical_json=canonical_json,
    )


def _recipe_to_payload(recipe: ModelRecipe) -> dict[str, object]:
    return {
        "id": recipe.id,
        "version": recipe.version,
        "status": recipe.status.value,
        "status_reason": recipe.status_reason,
        "huggingface_model_id": recipe.huggingface_model_id,
        "modality": recipe.modality.value,
        "task_profile": recipe.task_profile,
        "verified_revision": recipe.verified_revision,
        "preferred_revision": recipe.preferred_revision,
        "mobius": {
            "task": recipe.mobius.task,
            "ep": recipe.mobius.ep,
            "runtime": recipe.mobius.runtime,
            "dtype": recipe.mobius.dtype,
        },
        "olive": (
            {
                "input_source": recipe.olive.input_source,
                "task": recipe.olive.task,
                "precision": recipe.olive.precision,
                "device": recipe.olive.device,
                "provider": recipe.olive.provider,
                "log_level": recipe.olive.log_level,
            }
            if recipe.olive is not None
            else None
        ),
        "ancillary_files": [
            {
                "relative_path": row.relative_path,
                "required": row.required,
                "source": row.source,
            }
            for row in recipe.ancillary_files
        ],
        "runtime_validation": recipe.runtime_validation,
        "inference_modality": recipe.inference_modality.value,
        "optimization_choices": [
            {
                "strategy": row.strategy,
                "precision": row.precision,
                "task_profile": row.task_profile,
                "skip_olive": row.skip_olive,
                "default": row.default,
            }
            for row in recipe.optimization_choices
        ],
        "artifact_cache_prefix": recipe.artifact_cache_prefix,
        "model_name_prefix": recipe.model_name_prefix,
        "success_message": recipe.success_message,
    }


def _provenance_to_payload(provenance: RecipeGenerationProvenance) -> dict[str, object]:
    return {
        "compiler_version": provenance.compiler_version,
        "generation_kind": provenance.generation_kind,
        "capability_id": provenance.capability_id,
        "capability_version": provenance.capability_version,
        "capability_status": provenance.capability_status.value,
        "resolution_outcome": provenance.resolution_outcome.value,
        "resolution_reason_code": provenance.resolution_reason_code,
        "matched_aliases": list(provenance.matched_aliases),
        "argument_provenance": {
            "mobius_dtype_rule": provenance.argument_provenance.mobius_dtype_rule,
            "mobius_dtype_confidence": provenance.argument_provenance.mobius_dtype_confidence.value,
            "olive_precision_rule": provenance.argument_provenance.olive_precision_rule,
            "olive_precision_confidence": provenance.argument_provenance.olive_precision_confidence.value,
            "contains_unverified_arguments": provenance.argument_provenance.contains_unverified_arguments,
        },
        "evidence": list(provenance.evidence),
        "toolchain": {
            "mobius_version": provenance.toolchain.mobius_version,
            "olive_version": provenance.toolchain.olive_version,
            "onnx_version": provenance.toolchain.onnx_version,
            "ort_version": provenance.toolchain.ort_version,
            "oga_version": provenance.toolchain.oga_version,
            "foundry_sdk_version": provenance.toolchain.foundry_sdk_version,
            "foundry_cli_version": provenance.toolchain.foundry_cli_version,
        },
        "input_metadata": {
            "model_id": provenance.input_metadata.model_id,
            "revision_sha": provenance.input_metadata.revision_sha,
            "model_type": provenance.input_metadata.model_type,
            "architectures": list(provenance.input_metadata.architectures),
            "task": provenance.input_metadata.task,
            "requested_device": provenance.input_metadata.requested_device,
            "requested_precision": provenance.input_metadata.requested_precision,
            "is_gated": provenance.input_metadata.is_gated,
            "requires_remote_code": provenance.input_metadata.requires_remote_code,
            "config_files": list(provenance.input_metadata.config_files),
            "tokenizer_files": list(provenance.input_metadata.tokenizer_files),
            "available_files": list(provenance.input_metadata.available_files),
        },
        "promotion": _promotion_to_payload(provenance.promotion),
    }


def _promotion_to_payload(promotion: RecipePromotionRecord | None) -> dict[str, object] | None:
    if promotion is None:
        return None
    return {
        "promoted_from_fingerprint": promotion.promoted_from_fingerprint,
        "new_version": promotion.new_version,
        "status_reason": promotion.status_reason,
        "gate_evidence": {
            "mobius_build": _gate_to_payload(promotion.gate_evidence.mobius_build),
            "olive_optimize": _gate_to_payload(promotion.gate_evidence.olive_optimize),
            "onnx_validation": _gate_to_payload(promotion.gate_evidence.onnx_validation),
            "ort_validation": _gate_to_payload(promotion.gate_evidence.ort_validation),
            "oga_validation": _gate_to_payload(promotion.gate_evidence.oga_validation),
            "fl_sdk_inference": _gate_to_payload(promotion.gate_evidence.fl_sdk_inference),
            "quality_validation": _gate_to_payload(promotion.gate_evidence.quality_validation),
        },
    }


def _gate_to_payload(gate: PromotionGateCheck) -> dict[str, object]:
    return {
        "passed": gate.passed,
        "evidence": gate.evidence,
    }


def _validate_promotion_gate_evidence(gates: PromotionGateEvidence) -> None:
    for name, gate in (
        ("mobius_build", gates.mobius_build),
        ("olive_optimize", gates.olive_optimize),
        ("onnx_validation", gates.onnx_validation),
        ("ort_validation", gates.ort_validation),
        ("oga_validation", gates.oga_validation),
        ("fl_sdk_inference", gates.fl_sdk_inference),
        ("quality_validation", gates.quality_validation),
    ):
        if not gate.passed:
            raise GeneratedRecipePromotionError(
                f"Gate '{name}' must pass before promotion."
            )
        if not gate.evidence.strip():
            raise GeneratedRecipePromotionError(
                f"Gate '{name}' is missing explicit evidence."
            )


def _compile_optimization_choices(capability: ArchitectureCapability) -> tuple[OptimizationChoice, ...]:
    if not capability.optimization_choices:
        raise GeneratedRecipeCompileError("Capability has no optimization choices.")
    choices: list[OptimizationChoice] = []
    seen: dict[tuple[str, bool], str] = {}
    default_count = 0
    for row in capability.optimization_choices:
        strategy = _coerce_str(row.strategy, "optimization_choices.strategy")
        precision = _normalize_effective_precision(row.precision, "optimization_choices.precision")
        task_profile = _coerce_str(row.task_profile, "optimization_choices.task_profile")
        key = (task_profile.lower(), bool(row.skip_olive))
        if key in seen:
            raise GeneratedRecipeCompileError(
                f"Ambiguous optimization choices for task_profile '{task_profile}' "
                f"and skip_olive={bool(row.skip_olive)}."
            )
        seen[key] = strategy
        choice = OptimizationChoice(
            strategy=strategy,
            precision=precision,
            task_profile=task_profile,
            skip_olive=bool(row.skip_olive),
            default=bool(row.default),
        )
        if choice.default:
            default_count += 1
        choices.append(choice)
    if default_count != 1:
        raise GeneratedRecipeCompileError(
            "Capability optimization choices must contain exactly one default entry."
        )
    return tuple(
        sorted(
            choices,
            key=lambda item: (
                item.task_profile.lower(),
                item.precision,
                item.strategy.lower(),
                item.skip_olive,
            ),
        )
    )


def _select_choice_for_precision(
    choices: tuple[OptimizationChoice, ...],
    precision: str,
) -> OptimizationChoice:
    matches = [item for item in choices if item.precision == precision]
    if not matches:
        raise GeneratedRecipeCompileError(
            f"No optimization choice supports effective precision '{precision}'."
        )
    defaults = [item for item in matches if item.default]
    if len(defaults) > 1:
        raise GeneratedRecipeCompileError(
            f"Ambiguous default optimization for precision '{precision}'."
        )
    if defaults:
        return defaults[0]
    return matches[0]


def _resolve_effective_precision(
    *,
    requested_precision: str,
    capability_olive_precision: str | None,
    optimization_choices: tuple[OptimizationChoice, ...],
) -> str:
    if requested_precision not in _ALLOWED_REQUEST_PRECISIONS:
        raise GeneratedRecipeCompileError(
            f"Requested precision '{requested_precision}' is not supported."
        )
    if requested_precision == "int4":
        return "int4"
    if capability_olive_precision is not None:
        normalized = _normalize_effective_precision(
            capability_olive_precision,
            "capability.olive_rules.precision",
        )
        if normalized not in _ALLOWED_EFFECTIVE_PRECISIONS:
            raise GeneratedRecipeCompileError(
                f"AUTO precision resolved to unsupported effective precision '{normalized}'."
            )
        return normalized
    candidate_precisions = {row.precision for row in optimization_choices}
    if len(candidate_precisions) != 1:
        raise GeneratedRecipeCompileError(
            "AUTO precision is ambiguous because optimization choices expose multiple precisions."
        )
    precision = next(iter(candidate_precisions))
    if precision not in _ALLOWED_EFFECTIVE_PRECISIONS:
        raise GeneratedRecipeCompileError(
            f"AUTO precision resolved to unsupported effective precision '{precision}'."
        )
    return precision


def _compile_ancillary_file_rules(capability: ArchitectureCapability) -> tuple[AncillaryFileRule, ...]:
    if not capability.required_artifacts:
        raise GeneratedRecipeCompileError("Capability required_artifacts is empty.")
    normalized: dict[str, AncillaryFileRule] = {}
    for index, row in enumerate(capability.required_artifacts, start=1):
        relative_path = _normalize_relative_path(
            _coerce_str(row.relative_path, f"required_artifacts[{index}].relative_path"),
            path=f"required_artifacts[{index}].relative_path",
        )
        source = _coerce_str(row.source, f"required_artifacts[{index}].source")
        key = relative_path.lower()
        if key in normalized:
            raise GeneratedRecipeCompileError(
                f"Ambiguous ancillary file rule for '{relative_path}'."
            )
        normalized[key] = AncillaryFileRule(
            relative_path=relative_path,
            required=bool(row.required),
            source=source,
        )
    rules = tuple(sorted(normalized.values(), key=lambda item: item.relative_path.lower()))
    required_paths = {row.relative_path.lower() for row in rules if row.required}
    if not required_paths:
        raise GeneratedRecipeCompileError("At least one ancillary file rule must be required.")

    contract_required_paths: list[str] = []
    seen_contract_paths: set[str] = set()
    for index, row in enumerate(capability.oga_runtime_contract.required_files, start=1):
        required_path = _normalize_relative_path(
            _coerce_str(row, f"oga_runtime_contract.required_files[{index}]"),
            path=f"oga_runtime_contract.required_files[{index}]",
        )
        lowered = required_path.lower()
        if lowered in seen_contract_paths:
            raise GeneratedRecipeCompileError(
                f"Ambiguous runtime contract required file '{required_path}'."
            )
        seen_contract_paths.add(lowered)
        contract_required_paths.append(required_path)

    for required_path in contract_required_paths:
        if required_path.lower() not in required_paths:
            raise GeneratedRecipeCompileError(
                "Runtime contract required file is missing from required ancillary rules: "
                f"'{required_path}'."
            )
    return rules


def _normalize_toolchain(toolchain: RecipeCompilerToolchain) -> RecipeCompilerToolchain:
    return RecipeCompilerToolchain(
        mobius_version=_coerce_str(toolchain.mobius_version, "toolchain.mobius_version"),
        olive_version=_coerce_str(toolchain.olive_version, "toolchain.olive_version"),
        onnx_version=_coerce_str(toolchain.onnx_version, "toolchain.onnx_version"),
        ort_version=_coerce_str(toolchain.ort_version, "toolchain.ort_version"),
        oga_version=_coerce_str(toolchain.oga_version, "toolchain.oga_version"),
        foundry_sdk_version=_coerce_str(toolchain.foundry_sdk_version, "toolchain.foundry_sdk_version"),
        foundry_cli_version=_coerce_str(toolchain.foundry_cli_version, "toolchain.foundry_cli_version"),
    )


def _candidate_status_reason(*, capability_id: str, capability_status: CapabilityStatus) -> str:
    if capability_status == CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED:
        return (
            f"Compiled from tool-supported-unverified capability '{capability_id}'. "
            "Mobius/Olive argument provenance remains candidate-unverified until explicit "
            "Mobius, Olive, ONNX, ORT, OGA, FL SDK inference, and quality gates promote this recipe."
        )
    return (
        f"Compiled from verified architecture capability '{capability_id}', but this exact "
        "model revision remains an experimental candidate until explicit Mobius, Olive, ONNX, ORT, "
        "OGA, FL SDK inference, and quality gates promote it."
    )


def _build_recipe_id(*, model_id: str, effective_precision: str) -> str:
    slug = _slugify(model_id)
    recipe_id = f"{slug}-llm-cpu-{effective_precision}"
    if len(recipe_id) <= 128:
        return recipe_id
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:10]
    trimmed = recipe_id[:117].rstrip("-")
    return f"{trimmed}-{digest}"


def _build_cache_prefix(model_id: str) -> str:
    trailing = model_id.split("/")[-1] if "/" in model_id else model_id
    slug = _slugify(trailing)
    if len(slug) <= 48:
        return slug
    digest = hashlib.sha256(trailing.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:37].rstrip('-')}-{digest}"


def _slugify(value: str) -> str:
    normalized = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return normalized or "model"


def _normalize_revision_sha(value: str) -> str:
    revision = _coerce_str(value, "revision_sha").lower()
    if not _REVISION_SHA_RE.fullmatch(revision):
        raise GeneratedRecipeCompileError(
            "revision_sha must be a full 40-character lowercase hexadecimal commit SHA."
        )
    return revision


def _normalize_task(value: str) -> str:
    normalized = _normalize_alias(_coerce_str(value, "task"))
    if normalized not in _ALLOWED_TASK_ALIASES:
        raise GeneratedRecipeCompileError(
            f"Unsupported task '{value}'. Only text-generation/LLM recipes are allowed."
        )
    return "llm"


def _normalize_device(value: str) -> str:
    normalized = _normalize_alias(_coerce_str(value, "requested_device"))
    if normalized not in _ALLOWED_DEVICE_ALIASES:
        raise GeneratedRecipeCompileError(
            f"Unsupported device '{value}'. Only CPU target is allowed."
        )
    return "cpu"


def _normalize_requested_precision(value: str) -> str:
    normalized = _normalize_alias(_coerce_str(value, "requested_precision"))
    if normalized in {"auto", "default"}:
        return "auto"
    if normalized == "int4":
        return "int4"
    raise GeneratedRecipeCompileError(
        f"Unsupported requested precision '{value}'. Only auto or int4 are allowed."
    )


def _normalize_effective_precision(value: str, path: str) -> str:
    normalized = _normalize_alias(_coerce_str(value, path))
    if normalized in {"int4", "fp32"}:
        return normalized
    raise GeneratedRecipeCompileError(f"{path} has unsupported precision value '{value}'.")


def _normalize_strings(
    values: tuple[str, ...],
    *,
    path: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise GeneratedRecipeCompileError(f"{path} must be a tuple of strings.")
    normalized: dict[str, str] = {}
    for index, row in enumerate(values, start=1):
        item = _coerce_str(row, f"{path}[{index}]")
        normalized.setdefault(item.lower(), item)
    ordered = tuple(normalized[key] for key in sorted(normalized))
    if not allow_empty and not ordered:
        raise GeneratedRecipeCompileError(f"{path} cannot be empty.")
    return ordered


def _normalize_paths(
    values: tuple[str, ...],
    *,
    path: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    normalized_strings = _normalize_strings(values, path=path, allow_empty=allow_empty)
    normalized: dict[str, str] = {}
    for index, row in enumerate(normalized_strings, start=1):
        candidate = _normalize_relative_path(row, path=f"{path}[{index}]")
        normalized.setdefault(candidate.lower(), candidate)
    ordered = tuple(normalized[key] for key in sorted(normalized))
    if not allow_empty and not ordered:
        raise GeneratedRecipeCompileError(f"{path} cannot be empty.")
    return ordered


def _normalize_relative_path(value: str, *, path: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        raise GeneratedRecipeCompileError(f"{path} cannot be empty.")
    parsed = PurePosixPath(normalized)
    if parsed.is_absolute():
        raise GeneratedRecipeCompileError(f"{path} must be relative, got '{value}'.")
    parts = parsed.parts
    if not parts:
        raise GeneratedRecipeCompileError(f"{path} is invalid.")
    safe_parts: list[str] = []
    for segment in parts:
        if segment in {".", ".."}:
            raise GeneratedRecipeCompileError(
                f"{path} contains unsafe path component '{segment}'."
            )
        if ":" in segment:
            raise GeneratedRecipeCompileError(
                f"{path} contains unsafe drive-qualified component '{segment}'."
            )
        if any(ord(ch) < 32 for ch in segment):
            raise GeneratedRecipeCompileError(
                f"{path} contains control characters."
            )
        safe_parts.append(segment)
    return "/".join(safe_parts)


def _require_subset(
    subset: tuple[str, ...],
    superset: tuple[str, ...],
    *,
    path: str,
) -> None:
    available = {item.lower() for item in superset}
    missing = [item for item in subset if item.lower() not in available]
    if missing:
        raise GeneratedRecipeCompileError(
            f"{path} contains files not present in available_files: {', '.join(missing)}."
        )


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_generated_recipe_payload(
    payload: dict[str, object],
    *,
    schema_path: Path | None,
) -> None:
    schema = _load_schema(schema_path)
    _validate_json_value(payload, schema=schema, schema_root=schema, path="$")


def _load_schema(schema_path: Path | None) -> dict[str, object]:
    effective_path = schema_path or generated_recipe_schema_path()
    loaded = json.loads(effective_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise GeneratedRecipeCompileError(
            f"Generated recipe schema at '{effective_path}' must be a JSON object."
        )
    return loaded


def _validate_json_value(
    value: object,
    *,
    schema: dict[str, object],
    schema_root: dict[str, object],
    path: str,
) -> None:
    if "$ref" in schema:
        ref = _coerce_str(schema["$ref"], f"{path}.$ref")
        resolved = _resolve_schema_ref(schema_root, ref)
        _validate_json_value(value, schema=resolved, schema_root=schema_root, path=path)
        return

    if "allOf" in schema:
        rows = _coerce_schema_array(schema["allOf"], f"{path}.allOf")
        for index, row in enumerate(rows, start=1):
            child = _coerce_schema_object(row, f"{path}.allOf[{index}]")
            _validate_json_value(value, schema=child, schema_root=schema_root, path=path)

    if "type" in schema:
        _validate_type(value, schema["type"], path=path)

    if "const" in schema and value != schema["const"]:
        raise GeneratedRecipeCompileError(
            f"{path} must equal constant value '{schema['const']}'."
        )

    if "enum" in schema:
        enum_values = _coerce_schema_array(schema["enum"], f"{path}.enum")
        if value not in enum_values:
            raise GeneratedRecipeCompileError(
                f"{path} must be one of {enum_values}; got '{value}'."
            )
    if "pattern" in schema and isinstance(value, str):
        pattern = _coerce_str(schema["pattern"], f"{path}.pattern")
        if re.fullmatch(pattern, value) is None:
            raise GeneratedRecipeCompileError(
                f"{path} must match pattern '{pattern}'."
            )

    if isinstance(value, dict):
        required = _coerce_optional_schema_array(schema.get("required"), f"{path}.required")
        for key in required:
            if not isinstance(key, str):
                raise GeneratedRecipeCompileError(f"{path}.required must contain only strings.")
            if key not in value:
                raise GeneratedRecipeCompileError(f"{path} is missing required key '{key}'.")

        properties = _coerce_optional_schema_object(schema.get("properties"), f"{path}.properties")
        additional_properties = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                child_schema = _coerce_schema_object(properties[key], f"{path}.properties.{key}")
                _validate_json_value(
                    item,
                    schema=child_schema,
                    schema_root=schema_root,
                    path=f"{path}.{key}",
                )
                continue
            if additional_properties is False:
                raise GeneratedRecipeCompileError(f"{path} does not allow additional key '{key}'.")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise GeneratedRecipeCompileError(
                f"{path} must have at least {min_items} items."
            )
        if "items" in schema:
            item_schema = _coerce_schema_object(schema["items"], f"{path}.items")
            for index, item in enumerate(value, start=1):
                _validate_json_value(
                    item,
                    schema=item_schema,
                    schema_root=schema_root,
                    path=f"{path}[{index}]",
                )


def _validate_type(value: object, schema_type: object, *, path: str) -> None:
    if isinstance(schema_type, str):
        if not _value_matches_type(value, schema_type):
            raise GeneratedRecipeCompileError(f"{path} must be of type '{schema_type}'.")
        return
    if isinstance(schema_type, list):
        if any(_value_matches_type(value, item) for item in schema_type if isinstance(item, str)):
            return
        raise GeneratedRecipeCompileError(f"{path} does not match any allowed type {schema_type}.")
    raise GeneratedRecipeCompileError(f"{path}.type must be a string or an array of strings.")


def _value_matches_type(value: object, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def _resolve_schema_ref(schema_root: dict[str, object], ref: str) -> dict[str, object]:
    if not ref.startswith("#/"):
        raise GeneratedRecipeCompileError(f"Only local schema refs are supported; got '{ref}'.")
    node: object = schema_root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            raise GeneratedRecipeCompileError(f"Unable to resolve schema ref '{ref}'.")
        node = node[part]
    if not isinstance(node, dict):
        raise GeneratedRecipeCompileError(f"Schema ref '{ref}' does not resolve to an object.")
    return node


def _coerce_optional_schema_object(value: object, path: str) -> dict[str, object]:
    if value is None:
        return {}
    return _coerce_schema_object(value, path)


def _coerce_schema_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GeneratedRecipeCompileError(f"{path} must be a schema object.")
    return value


def _coerce_optional_schema_array(value: object, path: str) -> list[object]:
    if value is None:
        return []
    return _coerce_schema_array(value, path)


def _coerce_schema_array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise GeneratedRecipeCompileError(f"{path} must be an array.")
    return value


def _coerce_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeneratedRecipeCompileError(f"{path} must be a non-empty string.")
    return value.strip()


def _coerce_optional_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise GeneratedRecipeCompileError(f"{path} must be a string or null.")
    stripped = value.strip()
    if not stripped:
        raise GeneratedRecipeCompileError(f"{path} cannot be empty when provided.")
    return stripped


def _normalize_alias(value: str) -> str:
    return _SLUG_RE.sub("", value.strip().lower())


__all__ = [
    "GENERATED_RECIPE_SCHEMA_VERSION",
    "GeneratedRecipe",
    "GeneratedRecipeCompileError",
    "GeneratedRecipePromotionError",
    "PromotionGateCheck",
    "PromotionGateEvidence",
    "RECIPE_AGENT_COMPILER_VERSION",
    "RecipeArgumentProvenance",
    "RecipeCompilerInput",
    "RecipeCompilerToolchain",
    "RecipeGenerationProvenance",
    "RecipeInputMetadata",
    "RecipePromotionRecord",
    "compile_generated_recipe",
    "generated_recipe_schema_path",
    "promote_generated_recipe",
]
