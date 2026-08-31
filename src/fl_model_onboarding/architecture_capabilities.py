from __future__ import annotations

import json
import re

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .hf_policy import config_requires_remote_code

_ALIAS_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


class CapabilityStatus(StrEnum):
    VERIFIED = "verified"
    TOOL_SUPPORTED_UNVERIFIED = "tool-supported-unverified"
    SOURCE_CHANGE_REQUIRED = "source-change-required"


class CapabilityTask(StrEnum):
    TEXT_GENERATION = "text-generation"
    SPEECH_TO_TEXT = "speech-to-text"


class CapabilityModality(StrEnum):
    LLM = "llm"
    ASR = "asr"


class CapabilityDevice(StrEnum):
    CPU = "cpu"


class CapabilityPrecision(StrEnum):
    INT4 = "int4"
    FP32 = "fp32"


class RequestTask(StrEnum):
    LLM = "llm"
    ASR = "asr"


class RequestPrecision(StrEnum):
    AUTO = "auto"
    INT4 = "int4"
    FP32 = "fp32"


class ResolutionOutcome(StrEnum):
    EXACT = "exact"
    NOT_ELIGIBLE = "not-eligible"
    AMBIGUOUS = "ambiguous"


class ResolutionReasonCode(StrEnum):
    RESOLVED = "resolved"
    SOURCE_CHANGE_REQUIRED = "source-change-required"
    GATED_MODEL = "gated-model"
    REMOTE_CODE_REQUIRED = "remote-code-required"
    MISSING_ARCHITECTURE_METADATA = "missing-architecture-metadata"
    UNSUPPORTED_TASK = "unsupported-task"
    UNSUPPORTED_DEVICE = "unsupported-device"
    UNSUPPORTED_PRECISION = "unsupported-precision"
    NON_TEXT_ARCHITECTURE = "non-text-architecture"
    UNKNOWN_ARCHITECTURE = "unknown-architecture"
    AMBIGUOUS_ARCHITECTURE_MATCH = "ambiguous-architecture-match"


class BlockerOwner(StrEnum):
    FL_ONBOARDING = "fl-onboarding"
    MOBIUS_PRODUCER = "mobius-producer"
    OGA_RUNTIME = "onnxruntime-genai"
    OLIVE = "olive"
    FOUNDRY_LOCAL_SDK = "foundry-local-sdk"
    UPSTREAM_MODEL = "upstream-model"


class BlockerClassification(StrEnum):
    MISSING_FL_VERIFICATION = "missing-fl-verification"
    SOURCE_RUNTIME_CONTRACT_INCOMPATIBLE = "source-runtime-contract-incompatible"
    UNSUPPORTED_TASK = "unsupported-task"
    UNSUPPORTED_DEVICE = "unsupported-device"
    UNSUPPORTED_PRECISION = "unsupported-precision"
    NON_TEXT_ARCHITECTURE = "non-text-architecture"


@dataclass(frozen=True)
class CapabilityEvidence:
    evidence_id: str
    source_type: str
    location: str
    summary: str


@dataclass(frozen=True)
class MobiusArgumentRules:
    task: str
    ep: str
    runtime: str
    dtype: str | None
    dtype_rule: str


@dataclass(frozen=True)
class OliveArgumentRules:
    task: str
    device: str
    provider: str
    precision: str | None
    precision_rule: str


@dataclass(frozen=True)
class RequiredArtifact:
    relative_path: str
    required: bool
    source: str


@dataclass(frozen=True)
class OgaRuntimeContract:
    runtime: str
    validator_profile: str
    required_files: tuple[str, ...]
    loader: str
    notes: str


@dataclass(frozen=True)
class AllowedOptimizationChoice:
    strategy: str
    precision: str
    task_profile: str
    skip_olive: bool
    default: bool


@dataclass(frozen=True)
class StaticEligibilityConstraints:
    allow_gated: bool
    allow_remote_code: bool
    allow_private: bool
    supported_tasks: tuple[RequestTask, ...]
    supported_devices: tuple[CapabilityDevice, ...]
    supported_request_precisions: tuple[RequestPrecision, ...]


@dataclass(frozen=True)
class KnownBlocker:
    blocker_id: str
    owner: BlockerOwner
    classification: BlockerClassification
    message: str


@dataclass(frozen=True)
class ArchitectureCapability:
    capability_id: str
    version: str
    status: CapabilityStatus
    status_reason: str
    allowed_status_transitions: tuple[CapabilityStatus, ...]
    family: str
    model_type_aliases: tuple[str, ...]
    architecture_aliases: tuple[str, ...]
    task: CapabilityTask
    modality: CapabilityModality
    device: CapabilityDevice
    precision: CapabilityPrecision
    mobius_rules: MobiusArgumentRules
    olive_rules: OliveArgumentRules
    required_artifacts: tuple[RequiredArtifact, ...]
    oga_runtime_contract: OgaRuntimeContract
    optimization_choices: tuple[AllowedOptimizationChoice, ...]
    static_eligibility_constraints: StaticEligibilityConstraints
    known_blockers: tuple[KnownBlocker, ...]
    evidence: tuple[CapabilityEvidence, ...]

    def normalized_aliases(self) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in (*self.model_type_aliases, *self.architecture_aliases):
            alias = _normalize_alias(value)
            if alias and alias not in normalized:
                normalized.append(alias)
        return tuple(normalized)


@dataclass(frozen=True)
class NormalizedHuggingFaceMetadata:
    model_id: str
    model_type: str | None
    architecture_aliases: tuple[str, ...]
    is_gated: bool
    is_private: bool
    requires_remote_code: bool

    def normalized_aliases(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.model_type:
            values.append(self.model_type)
        values.extend(self.architecture_aliases)
        normalized: list[str] = []
        for value in values:
            alias = _normalize_alias(value)
            if alias and alias not in normalized:
                normalized.append(alias)
        return tuple(normalized)


@dataclass(frozen=True)
class CapabilityResolution:
    outcome: ResolutionOutcome
    reason_code: ResolutionReasonCode
    reason: str
    capability: ArchitectureCapability | None
    matched_aliases: tuple[str, ...]
    evidence: tuple[CapabilityEvidence, ...]

    @property
    def eligible_for_candidate_recipe(self) -> bool:
        if self.outcome != ResolutionOutcome.EXACT or self.capability is None:
            return False
        return self.capability.status != CapabilityStatus.SOURCE_CHANGE_REQUIRED


_ALLOWED_STATUS_TRANSITIONS: dict[CapabilityStatus, frozenset[CapabilityStatus]] = {
    CapabilityStatus.VERIFIED: frozenset(
        {
            CapabilityStatus.VERIFIED,
            CapabilityStatus.SOURCE_CHANGE_REQUIRED,
        }
    ),
    CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED: frozenset(
        {
            CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED,
            CapabilityStatus.VERIFIED,
            CapabilityStatus.SOURCE_CHANGE_REQUIRED,
        }
    ),
    CapabilityStatus.SOURCE_CHANGE_REQUIRED: frozenset(
        {
            CapabilityStatus.SOURCE_CHANGE_REQUIRED,
            CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED,
        }
    ),
}


class ArchitectureCapabilityRegistry:
    def __init__(self, *, schema_version: str, capabilities: tuple[ArchitectureCapability, ...]) -> None:
        if not capabilities:
            raise ValueError("Architecture capability registry is empty.")
        self.schema_version = schema_version
        self._capabilities = capabilities
        self._capabilities_by_id: dict[str, ArchitectureCapability] = {}
        self._lookup: dict[tuple[str, RequestTask, CapabilityDevice, RequestPrecision], str] = {}
        self._lookup_task_device: dict[tuple[str, RequestTask, CapabilityDevice], set[str]] = {}
        self._lookup_alias_only: dict[str, set[str]] = {}

        for capability in capabilities:
            if capability.capability_id in self._capabilities_by_id:
                raise ValueError(f"Duplicate capability_id '{capability.capability_id}'.")
            self._capabilities_by_id[capability.capability_id] = capability
            aliases = capability.normalized_aliases()
            if not aliases:
                raise ValueError(f"Capability '{capability.capability_id}' has no normalized aliases.")
            for alias in aliases:
                self._lookup_alias_only.setdefault(alias, set()).add(capability.capability_id)
                for task in capability.static_eligibility_constraints.supported_tasks:
                    for device in capability.static_eligibility_constraints.supported_devices:
                        self._lookup_task_device.setdefault((alias, task, device), set()).add(
                            capability.capability_id
                        )
                        for request_precision in (
                            capability.static_eligibility_constraints.supported_request_precisions
                        ):
                            key = (alias, task, device, request_precision)
                            existing = self._lookup.get(key)
                            if existing and existing != capability.capability_id:
                                raise ValueError(
                                    "Ambiguous capability mapping for alias "
                                    f"'{alias}' and key ({task.value}, {device.value}, {request_precision.value}): "
                                    f"{existing}, {capability.capability_id}"
                                )
                            self._lookup[key] = capability.capability_id

    def all(self) -> tuple[ArchitectureCapability, ...]:
        return self._capabilities

    def resolve(
        self,
        *,
        metadata: NormalizedHuggingFaceMetadata,
        task: str,
        device: str,
        requested_precision: str,
    ) -> CapabilityResolution:
        normalized_task = _normalize_request_task(task)
        if normalized_task is None:
            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.UNSUPPORTED_TASK,
                reason=f"Unsupported task '{task}'. Only 'llm' task routing is eligible in this registry.",
                capability=None,
                matched_aliases=(),
                evidence=(),
            )
        if normalized_task != RequestTask.LLM:
            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.UNSUPPORTED_TASK,
                reason=(
                    f"Task '{normalized_task.value}' is outside the CPU text-generation capability boundary."
                ),
                capability=None,
                matched_aliases=(),
                evidence=(),
            )

        normalized_device = _normalize_request_device(device)
        if normalized_device is None:
            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.UNSUPPORTED_DEVICE,
                reason=f"Unsupported device '{device}'. Only CPU is eligible.",
                capability=None,
                matched_aliases=(),
                evidence=(),
            )

        normalized_precision = _normalize_request_precision(requested_precision)
        if normalized_precision is None:
            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.UNSUPPORTED_PRECISION,
                reason=f"Unsupported precision request '{requested_precision}'.",
                capability=None,
                matched_aliases=(),
                evidence=(),
            )

        if metadata.is_gated:
            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.GATED_MODEL,
                reason="Gated Hugging Face models are blocked at architecture capability resolution.",
                capability=None,
                matched_aliases=(),
                evidence=(),
            )
        if metadata.requires_remote_code:
            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.REMOTE_CODE_REQUIRED,
                reason="Models requiring remote code are blocked at architecture capability resolution.",
                capability=None,
                matched_aliases=(),
                evidence=(),
            )

        aliases = metadata.normalized_aliases()
        if not aliases:
            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.MISSING_ARCHITECTURE_METADATA,
                reason="Metadata is missing model_type/architecture aliases required for deterministic matching.",
                capability=None,
                matched_aliases=(),
                evidence=(),
            )

        candidate_ids: dict[str, set[str]] = {}
        for alias in aliases:
            key = (alias, normalized_task, normalized_device, normalized_precision)
            capability_id = self._lookup.get(key)
            if capability_id:
                candidate_ids.setdefault(capability_id, set()).add(alias)

        if not candidate_ids:
            task_device_ids: set[str] = set()
            for alias in aliases:
                task_device_ids.update(self._lookup_task_device.get((alias, normalized_task, normalized_device), set()))

            if task_device_ids:
                reason = (
                    f"Architecture matched, but precision '{normalized_precision.value}' is not eligible for "
                    f"task '{normalized_task.value}' on device '{normalized_device.value}'."
                )
                evidence = self._collect_evidence(tuple(task_device_ids))
                return CapabilityResolution(
                    outcome=ResolutionOutcome.NOT_ELIGIBLE,
                    reason_code=ResolutionReasonCode.UNSUPPORTED_PRECISION,
                    reason=reason,
                    capability=None,
                    matched_aliases=tuple(sorted(aliases)),
                    evidence=evidence,
                )

            alias_only_ids: set[str] = set()
            for alias in aliases:
                alias_only_ids.update(self._lookup_alias_only.get(alias, set()))

            if alias_only_ids:
                capabilities = [self._capabilities_by_id[capability_id] for capability_id in sorted(alias_only_ids)]
                non_text = [cap for cap in capabilities if cap.modality != CapabilityModality.LLM]
                if non_text:
                    evidence = self._collect_evidence(tuple(cap.capability_id for cap in non_text))
                    families = ", ".join(sorted({cap.family for cap in non_text}))
                    return CapabilityResolution(
                        outcome=ResolutionOutcome.NOT_ELIGIBLE,
                        reason_code=ResolutionReasonCode.NON_TEXT_ARCHITECTURE,
                        reason=(
                            f"Matched non-text architecture family ({families}). "
                            "CPU text-generation capability resolution rejects non-text families."
                        ),
                        capability=None,
                        matched_aliases=tuple(sorted(aliases)),
                        evidence=evidence,
                    )

            return CapabilityResolution(
                outcome=ResolutionOutcome.NOT_ELIGIBLE,
                reason_code=ResolutionReasonCode.UNKNOWN_ARCHITECTURE,
                reason=(
                    "No architecture capability matched this model_type/architecture alias set for "
                    "task=llm, device=cpu."
                ),
                capability=None,
                matched_aliases=tuple(sorted(aliases)),
                evidence=(),
            )

        if len(candidate_ids) > 1:
            capability_ids = tuple(sorted(candidate_ids.keys()))
            evidence = self._collect_evidence(capability_ids)
            return CapabilityResolution(
                outcome=ResolutionOutcome.AMBIGUOUS,
                reason_code=ResolutionReasonCode.AMBIGUOUS_ARCHITECTURE_MATCH,
                reason=(
                    "Ambiguous architecture aliases matched multiple capabilities: "
                    f"{', '.join(capability_ids)}."
                ),
                capability=None,
                matched_aliases=tuple(sorted(aliases)),
                evidence=evidence,
            )

        resolved_capability_id = next(iter(candidate_ids))
        capability = self._capabilities_by_id[resolved_capability_id]
        matched_aliases = tuple(sorted(candidate_ids[resolved_capability_id]))
        reason_code = (
            ResolutionReasonCode.SOURCE_CHANGE_REQUIRED
            if capability.status == CapabilityStatus.SOURCE_CHANGE_REQUIRED
            else ResolutionReasonCode.RESOLVED
        )
        reason = (
            f"Resolved to capability '{capability.capability_id}' "
            f"({capability.status.value}) via alias(es): {', '.join(matched_aliases)}."
        )
        return CapabilityResolution(
            outcome=ResolutionOutcome.EXACT,
            reason_code=reason_code,
            reason=reason,
            capability=capability,
            matched_aliases=matched_aliases,
            evidence=capability.evidence,
        )

    def _collect_evidence(self, capability_ids: tuple[str, ...]) -> tuple[CapabilityEvidence, ...]:
        collected: dict[str, CapabilityEvidence] = {}
        for capability_id in capability_ids:
            for row in self._capabilities_by_id[capability_id].evidence:
                collected.setdefault(row.evidence_id, row)
        return tuple(collected[key] for key in sorted(collected))


def architecture_capability_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts" / "architecture-capabilities.schema.json"


def architecture_capability_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "architecture-capabilities.json"


def normalize_huggingface_metadata(
    *,
    model_id: str,
    config: Mapping[str, object] | None,
    is_gated: bool | None,
    is_private: bool | None,
) -> NormalizedHuggingFaceMetadata:
    config_dict = _coerce_optional_mapping(config, "config")
    model_type = _coerce_optional_str(config_dict.get("model_type"), "config.model_type")
    architectures = _coerce_optional_str_tuple(config_dict.get("architectures"), "config.architectures")
    return NormalizedHuggingFaceMetadata(
        model_id=model_id,
        model_type=model_type,
        architecture_aliases=architectures,
        is_gated=bool(is_gated),
        is_private=bool(is_private),
        requires_remote_code=config_requires_remote_code(config_dict if config_dict else None),
    )


def load_architecture_capability_registry(
    data_path: Path | None = None,
    schema_path: Path | None = None,
) -> ArchitectureCapabilityRegistry:
    effective_schema_path = schema_path or architecture_capability_schema_path()
    effective_data_path = data_path or architecture_capability_data_path()
    schema_raw = _load_json_file(effective_schema_path)
    payload = _load_json_file(effective_data_path)
    _validate_payload_against_schema_header(payload, schema_raw)
    schema_version = _coerce_str(payload.get("schema_version"), "schema_version")
    capabilities = _parse_capabilities(payload)
    return ArchitectureCapabilityRegistry(schema_version=schema_version, capabilities=capabilities)


def _load_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file '{path}' must contain an object at top level.")
    return payload


def _validate_payload_against_schema_header(payload: dict[str, object], schema: dict[str, object]) -> None:
    required = _coerce_optional_str_tuple(schema.get("required"), "schema.required")
    for key in required:
        if key not in payload:
            raise ValueError(f"Capability data is missing required key '{key}'.")

    properties = _coerce_mapping(schema.get("properties"), "schema.properties")
    schema_version_property = _coerce_mapping(
        properties.get("schema_version"), "schema.properties.schema_version"
    )
    expected_version = _coerce_optional_str(schema_version_property.get("const"), "schema.properties.schema_version.const")
    actual_version = _coerce_optional_str(payload.get("schema_version"), "schema_version")
    if expected_version and actual_version != expected_version:
        raise ValueError(
            "Capability data schema_version mismatch: "
            f"expected '{expected_version}', got '{actual_version}'."
        )


def _parse_capabilities(payload: dict[str, object]) -> tuple[ArchitectureCapability, ...]:
    capabilities_raw = _coerce_sequence(payload.get("capabilities"), "capabilities")
    if not capabilities_raw:
        raise ValueError("Capability data has no capabilities.")
    parsed: list[ArchitectureCapability] = []
    for index, row in enumerate(capabilities_raw, start=1):
        parsed.append(_parse_capability(row, index))
    return tuple(parsed)


def _parse_capability(row: object, index: int) -> ArchitectureCapability:
    path = f"capabilities[{index}]"
    value = _coerce_mapping(row, path)
    status = CapabilityStatus(_coerce_str(value.get("status"), f"{path}.status"))
    transitions = tuple(
        CapabilityStatus(item)
        for item in _coerce_optional_str_tuple(value.get("allowed_status_transitions"), f"{path}.allowed_status_transitions")
    )
    _validate_status_transitions(status=status, transitions=transitions, path=path)
    evidence = _parse_evidence(
        _coerce_sequence(value.get("evidence"), f"{path}.evidence"),
        f"{path}.evidence",
    )
    if not evidence:
        raise ValueError(f"{path}.evidence must contain at least one evidence item.")

    model_type_aliases = _coerce_optional_str_tuple(value.get("model_type_aliases"), f"{path}.model_type_aliases")
    if not model_type_aliases:
        raise ValueError(f"{path}.model_type_aliases cannot be empty.")

    architecture_aliases = _coerce_optional_str_tuple(value.get("architecture_aliases"), f"{path}.architecture_aliases")
    required_artifacts = _parse_required_artifacts(
        _coerce_sequence(value.get("required_artifacts"), f"{path}.required_artifacts"),
        f"{path}.required_artifacts",
    )
    optimization_choices = _parse_optimization_choices(
        _coerce_sequence(value.get("optimization_choices"), f"{path}.optimization_choices"),
        f"{path}.optimization_choices",
    )
    known_blockers = _parse_known_blockers(
        _coerce_sequence(value.get("known_blockers"), f"{path}.known_blockers"),
        f"{path}.known_blockers",
    )
    static_constraints = _parse_static_eligibility_constraints(
        _coerce_mapping(value.get("static_eligibility_constraints"), f"{path}.static_eligibility_constraints"),
        f"{path}.static_eligibility_constraints",
    )
    mobius_rules = _parse_mobius_rules(
        _coerce_mapping(value.get("mobius_rules"), f"{path}.mobius_rules"),
        f"{path}.mobius_rules",
    )
    olive_rules = _parse_olive_rules(
        _coerce_mapping(value.get("olive_rules"), f"{path}.olive_rules"),
        f"{path}.olive_rules",
    )
    oga_contract = _parse_oga_runtime_contract(
        _coerce_mapping(value.get("oga_runtime_contract"), f"{path}.oga_runtime_contract"),
        f"{path}.oga_runtime_contract",
    )

    capability = ArchitectureCapability(
        capability_id=_coerce_str(value.get("capability_id"), f"{path}.capability_id"),
        version=_coerce_str(value.get("version"), f"{path}.version"),
        status=status,
        status_reason=_coerce_str(value.get("status_reason"), f"{path}.status_reason"),
        allowed_status_transitions=transitions,
        family=_coerce_str(value.get("family"), f"{path}.family"),
        model_type_aliases=model_type_aliases,
        architecture_aliases=architecture_aliases,
        task=CapabilityTask(_coerce_str(value.get("task"), f"{path}.task")),
        modality=CapabilityModality(_coerce_str(value.get("modality"), f"{path}.modality")),
        device=CapabilityDevice(_coerce_str(value.get("device"), f"{path}.device")),
        precision=CapabilityPrecision(_coerce_str(value.get("precision"), f"{path}.precision")),
        mobius_rules=mobius_rules,
        olive_rules=olive_rules,
        required_artifacts=required_artifacts,
        oga_runtime_contract=oga_contract,
        optimization_choices=optimization_choices,
        static_eligibility_constraints=static_constraints,
        known_blockers=known_blockers,
        evidence=evidence,
    )
    if not capability.normalized_aliases():
        raise ValueError(f"{path} has no usable aliases after normalization.")
    return capability


def _parse_evidence(rows: tuple[object, ...], path: str) -> tuple[CapabilityEvidence, ...]:
    evidence: list[CapabilityEvidence] = []
    for index, row in enumerate(rows, start=1):
        value = _coerce_mapping(row, f"{path}[{index}]")
        evidence.append(
            CapabilityEvidence(
                evidence_id=_coerce_str(value.get("evidence_id"), f"{path}[{index}].evidence_id"),
                source_type=_coerce_str(value.get("source_type"), f"{path}[{index}].source_type"),
                location=_coerce_str(value.get("location"), f"{path}[{index}].location"),
                summary=_coerce_str(value.get("summary"), f"{path}[{index}].summary"),
            )
        )
    return tuple(evidence)


def _parse_mobius_rules(row: dict[str, object], path: str) -> MobiusArgumentRules:
    return MobiusArgumentRules(
        task=_coerce_str(row.get("task"), f"{path}.task"),
        ep=_coerce_str(row.get("ep"), f"{path}.ep"),
        runtime=_coerce_str(row.get("runtime"), f"{path}.runtime"),
        dtype=_coerce_optional_str(row.get("dtype"), f"{path}.dtype"),
        dtype_rule=_coerce_str(row.get("dtype_rule"), f"{path}.dtype_rule"),
    )


def _parse_olive_rules(row: dict[str, object], path: str) -> OliveArgumentRules:
    return OliveArgumentRules(
        task=_coerce_str(row.get("task"), f"{path}.task"),
        device=_coerce_str(row.get("device"), f"{path}.device"),
        provider=_coerce_str(row.get("provider"), f"{path}.provider"),
        precision=_coerce_optional_str(row.get("precision"), f"{path}.precision"),
        precision_rule=_coerce_str(row.get("precision_rule"), f"{path}.precision_rule"),
    )


def _parse_required_artifacts(rows: tuple[object, ...], path: str) -> tuple[RequiredArtifact, ...]:
    artifacts: list[RequiredArtifact] = []
    for index, row in enumerate(rows, start=1):
        value = _coerce_mapping(row, f"{path}[{index}]")
        artifacts.append(
            RequiredArtifact(
                relative_path=_coerce_str(value.get("relative_path"), f"{path}[{index}].relative_path"),
                required=_coerce_bool(value.get("required"), f"{path}[{index}].required"),
                source=_coerce_str(value.get("source"), f"{path}[{index}].source"),
            )
        )
    return tuple(artifacts)


def _parse_oga_runtime_contract(row: dict[str, object], path: str) -> OgaRuntimeContract:
    required_files = _coerce_optional_str_tuple(row.get("required_files"), f"{path}.required_files")
    if not required_files:
        raise ValueError(f"{path}.required_files cannot be empty.")
    return OgaRuntimeContract(
        runtime=_coerce_str(row.get("runtime"), f"{path}.runtime"),
        validator_profile=_coerce_str(row.get("validator_profile"), f"{path}.validator_profile"),
        required_files=required_files,
        loader=_coerce_str(row.get("loader"), f"{path}.loader"),
        notes=_coerce_str(row.get("notes"), f"{path}.notes"),
    )


def _parse_optimization_choices(rows: tuple[object, ...], path: str) -> tuple[AllowedOptimizationChoice, ...]:
    choices: list[AllowedOptimizationChoice] = []
    for index, row in enumerate(rows, start=1):
        value = _coerce_mapping(row, f"{path}[{index}]")
        choices.append(
            AllowedOptimizationChoice(
                strategy=_coerce_str(value.get("strategy"), f"{path}[{index}].strategy"),
                precision=_coerce_str(value.get("precision"), f"{path}[{index}].precision"),
                task_profile=_coerce_str(value.get("task_profile"), f"{path}[{index}].task_profile"),
                skip_olive=_coerce_bool(value.get("skip_olive"), f"{path}[{index}].skip_olive"),
                default=_coerce_bool(value.get("default"), f"{path}[{index}].default"),
            )
        )
    return tuple(choices)


def _parse_static_eligibility_constraints(
    row: dict[str, object],
    path: str,
) -> StaticEligibilityConstraints:
    supported_tasks_raw = _coerce_optional_str_tuple(row.get("supported_tasks"), f"{path}.supported_tasks")
    supported_devices_raw = _coerce_optional_str_tuple(row.get("supported_devices"), f"{path}.supported_devices")
    supported_precisions_raw = _coerce_optional_str_tuple(
        row.get("supported_request_precisions"),
        f"{path}.supported_request_precisions",
    )
    if not supported_tasks_raw:
        raise ValueError(f"{path}.supported_tasks cannot be empty.")
    if not supported_devices_raw:
        raise ValueError(f"{path}.supported_devices cannot be empty.")
    if not supported_precisions_raw:
        raise ValueError(f"{path}.supported_request_precisions cannot be empty.")
    return StaticEligibilityConstraints(
        allow_gated=_coerce_bool(row.get("allow_gated"), f"{path}.allow_gated"),
        allow_remote_code=_coerce_bool(row.get("allow_remote_code"), f"{path}.allow_remote_code"),
        allow_private=_coerce_bool(row.get("allow_private"), f"{path}.allow_private"),
        supported_tasks=tuple(RequestTask(item) for item in supported_tasks_raw),
        supported_devices=tuple(CapabilityDevice(item) for item in supported_devices_raw),
        supported_request_precisions=tuple(RequestPrecision(item) for item in supported_precisions_raw),
    )


def _parse_known_blockers(rows: tuple[object, ...], path: str) -> tuple[KnownBlocker, ...]:
    blockers: list[KnownBlocker] = []
    for index, row in enumerate(rows, start=1):
        value = _coerce_mapping(row, f"{path}[{index}]")
        blockers.append(
            KnownBlocker(
                blocker_id=_coerce_str(value.get("blocker_id"), f"{path}[{index}].blocker_id"),
                owner=BlockerOwner(_coerce_str(value.get("owner"), f"{path}[{index}].owner")),
                classification=BlockerClassification(
                    _coerce_str(value.get("classification"), f"{path}[{index}].classification")
                ),
                message=_coerce_str(value.get("message"), f"{path}[{index}].message"),
            )
        )
    return tuple(blockers)


def _validate_status_transitions(
    *,
    status: CapabilityStatus,
    transitions: tuple[CapabilityStatus, ...],
    path: str,
) -> None:
    if not transitions:
        raise ValueError(f"{path}.allowed_status_transitions cannot be empty.")
    allowed = _ALLOWED_STATUS_TRANSITIONS[status]
    if status not in transitions:
        raise ValueError(
            f"{path}.allowed_status_transitions must contain current status '{status.value}'."
        )
    invalid = [value.value for value in transitions if value not in allowed]
    if invalid:
        raise ValueError(
            f"{path}.allowed_status_transitions contains invalid transition(s): {', '.join(invalid)}."
        )


def _normalize_alias(value: str) -> str:
    normalized = _ALIAS_NORMALIZE_RE.sub("", value.strip().lower())
    return normalized


def _normalize_request_task(value: str) -> RequestTask | None:
    normalized = _normalize_alias(value)
    if normalized in {"llm", "textgeneration"}:
        return RequestTask.LLM
    if normalized in {"asr", "speechtotext", "automaticspeechrecognition"}:
        return RequestTask.ASR
    return None


def _normalize_request_device(value: str) -> CapabilityDevice | None:
    normalized = _normalize_alias(value)
    if normalized == "cpu":
        return CapabilityDevice.CPU
    return None


def _normalize_request_precision(value: str) -> RequestPrecision | None:
    normalized = _normalize_alias(value)
    if normalized in {"auto", "default"}:
        return RequestPrecision.AUTO
    if normalized == "int4":
        return RequestPrecision.INT4
    if normalized == "fp32":
        return RequestPrecision.FP32
    return None


def _coerce_mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    return value


def _coerce_optional_mapping(value: object, path: str) -> dict[str, object]:
    if value is None:
        return {}
    return _coerce_mapping(value, path)


def _coerce_sequence(value: object, path: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array.")
    return tuple(value)


def _coerce_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value.strip()


def _coerce_optional_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string or null.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{path} cannot be empty when provided.")
    return stripped


def _coerce_optional_str_tuple(value: object, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array of strings.")
    coerced: list[str] = []
    for index, row in enumerate(value, start=1):
        item = _coerce_str(row, f"{path}[{index}]")
        coerced.append(item)
    return tuple(coerced)


def _coerce_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean.")
    return value


DEFAULT_ARCHITECTURE_CAPABILITY_REGISTRY = load_architecture_capability_registry()


__all__ = [
    "AllowedOptimizationChoice",
    "ArchitectureCapability",
    "ArchitectureCapabilityRegistry",
    "BlockerClassification",
    "BlockerOwner",
    "CapabilityDevice",
    "CapabilityEvidence",
    "CapabilityModality",
    "CapabilityPrecision",
    "CapabilityResolution",
    "CapabilityStatus",
    "CapabilityTask",
    "KnownBlocker",
    "MobiusArgumentRules",
    "NormalizedHuggingFaceMetadata",
    "OgaRuntimeContract",
    "OliveArgumentRules",
    "RequestPrecision",
    "RequestTask",
    "RequiredArtifact",
    "ResolutionOutcome",
    "ResolutionReasonCode",
    "StaticEligibilityConstraints",
    "DEFAULT_ARCHITECTURE_CAPABILITY_REGISTRY",
    "architecture_capability_data_path",
    "architecture_capability_schema_path",
    "load_architecture_capability_registry",
    "normalize_huggingface_metadata",
]
