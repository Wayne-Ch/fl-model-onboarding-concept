from __future__ import annotations

import json

from pathlib import Path

import pytest

from fl_model_onboarding.architecture_capabilities import (
    CapabilityModality,
    CapabilityStatus,
    ResolutionOutcome,
    ResolutionReasonCode,
    architecture_capability_data_path,
    architecture_capability_schema_path,
    load_architecture_capability_registry,
    normalize_huggingface_metadata,
)


def _metadata(
    *,
    model_id: str,
    model_type: str | None,
    architectures: tuple[str, ...] = (),
    is_gated: bool = False,
    config_extra: dict[str, object] | None = None,
):
    config: dict[str, object] = {}
    if model_type is not None:
        config["model_type"] = model_type
    if architectures:
        config["architectures"] = list(architectures)
    if config_extra:
        config.update(config_extra)
    return normalize_huggingface_metadata(
        model_id=model_id,
        config=config,
        is_gated=is_gated,
        is_private=False,
    )


def test_registry_contains_verified_unverified_and_source_change_required_statuses() -> None:
    registry = load_architecture_capability_registry()
    statuses = {cap.status for cap in registry.all()}
    assert CapabilityStatus.VERIFIED in statuses
    assert CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED in statuses
    assert CapabilityStatus.SOURCE_CHANGE_REQUIRED in statuses
    llm_families = {
        cap.family
        for cap in registry.all()
        if cap.modality == CapabilityModality.LLM
        and cap.status in {CapabilityStatus.VERIFIED, CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED}
    }
    assert len(llm_families) >= 3


def test_llama_resolution_for_unseen_model_by_model_type() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="new-owner/new-llama-model",
        model_type="llama",
        architectures=("LlamaForCausalLM",),
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="int4",
    )
    assert result.outcome == ResolutionOutcome.EXACT
    assert result.capability is not None
    assert result.capability.family == "llama"
    assert result.capability.status == CapabilityStatus.VERIFIED


def test_granite_resolution_for_auto_precision() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="org/granite-derived-model",
        model_type="granite",
        architectures=("GraniteForCausalLM",),
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="auto",
    )
    assert result.outcome == ResolutionOutcome.EXACT
    assert result.capability is not None
    assert result.capability.family == "granite"
    assert result.capability.status == CapabilityStatus.VERIFIED


def test_qwen_family_resolves_as_tool_supported_unverified() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="org/unseen-qwen",
        model_type="qwen2",
        architectures=("Qwen2ForCausalLM",),
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="int4",
    )
    assert result.outcome == ResolutionOutcome.EXACT
    assert result.capability is not None
    assert result.capability.status == CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED
    assert result.capability.family == "qwen"


def test_unknown_model_type_is_not_eligible() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="owner/something-new",
        model_type="totally-unknown",
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="auto",
    )
    assert result.outcome == ResolutionOutcome.NOT_ELIGIBLE
    assert result.reason_code == ResolutionReasonCode.UNKNOWN_ARCHITECTURE


def test_architecture_alias_resolution_works_without_model_type() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="owner/alias-only",
        model_type=None,
        architectures=("GraniteForCausalLM",),
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="int4",
    )
    assert result.outcome == ResolutionOutcome.EXACT
    assert result.capability is not None
    assert result.capability.family == "granite"


def test_gated_model_is_rejected_at_capability_boundary() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="owner/gated-llama",
        model_type="llama",
        is_gated=True,
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="int4",
    )
    assert result.outcome == ResolutionOutcome.NOT_ELIGIBLE
    assert result.reason_code == ResolutionReasonCode.GATED_MODEL


def test_remote_code_model_is_rejected_at_capability_boundary() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="owner/remote-code",
        model_type="llama",
        config_extra={"auto_map": {"AutoModelForCausalLM": "remote.module.Model"}},
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="int4",
    )
    assert result.outcome == ResolutionOutcome.NOT_ELIGIBLE
    assert result.reason_code == ResolutionReasonCode.REMOTE_CODE_REQUIRED


def test_non_text_architecture_is_rejected_for_llm_task() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="owner/whisper-like",
        model_type="whisper",
        architectures=("WhisperForConditionalGeneration",),
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="auto",
    )
    assert result.outcome == ResolutionOutcome.NOT_ELIGIBLE
    assert result.reason_code == ResolutionReasonCode.NON_TEXT_ARCHITECTURE
    assert len(result.evidence) >= 1


@pytest.mark.parametrize(
    ("task", "device", "precision", "reason_code"),
    [
        ("asr", "cpu", "auto", ResolutionReasonCode.UNSUPPORTED_TASK),
        ("llm", "cuda", "auto", ResolutionReasonCode.UNSUPPORTED_DEVICE),
        ("llm", "cpu", "bf16", ResolutionReasonCode.UNSUPPORTED_PRECISION),
        ("llm", "cpu", "fp32", ResolutionReasonCode.UNSUPPORTED_PRECISION),
    ],
)
def test_wrong_task_device_or_precision_is_rejected(
    task: str,
    device: str,
    precision: str,
    reason_code: ResolutionReasonCode,
) -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="owner/llama-candidate",
        model_type="llama",
        architectures=("LlamaForCausalLM",),
    )
    result = registry.resolve(
        metadata=metadata,
        task=task,
        device=device,
        requested_precision=precision,
    )
    assert result.outcome == ResolutionOutcome.NOT_ELIGIBLE
    assert result.reason_code == reason_code


def test_ambiguous_aliases_fail_closed() -> None:
    registry = load_architecture_capability_registry()
    metadata = _metadata(
        model_id="owner/ambiguous",
        model_type="llama",
        architectures=("GraniteForCausalLM",),
    )
    result = registry.resolve(
        metadata=metadata,
        task="llm",
        device="cpu",
        requested_precision="int4",
    )
    assert result.outcome == ResolutionOutcome.AMBIGUOUS
    assert result.reason_code == ResolutionReasonCode.AMBIGUOUS_ARCHITECTURE_MATCH


def test_model_id_is_not_used_for_exact_matching() -> None:
    registry = load_architecture_capability_registry()
    resolved = registry.resolve(
        metadata=_metadata(
            model_id="owner/some-brand-new-llama-fork",
            model_type="llama",
        ),
        task="llm",
        device="cpu",
        requested_precision="auto",
    )
    assert resolved.outcome == ResolutionOutcome.EXACT
    assert resolved.capability is not None
    assert resolved.capability.family == "llama"

    blocked = registry.resolve(
        metadata=_metadata(
            model_id="owner/this-name-contains-llama-but-type-is-unknown-llama",
            model_type="unknown-type",
        ),
        task="llm",
        device="cpu",
        requested_precision="auto",
    )
    assert blocked.outcome == ResolutionOutcome.NOT_ELIGIBLE
    assert blocked.reason_code == ResolutionReasonCode.UNKNOWN_ARCHITECTURE


def test_schema_and_data_paths_are_loadable() -> None:
    schema = json.loads(architecture_capability_schema_path().read_text(encoding="utf-8"))
    data = json.loads(architecture_capability_data_path().read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == data["schema_version"]
    assert len(data["capabilities"]) >= 3
    registry = load_architecture_capability_registry()
    assert len(registry.all()) == len(data["capabilities"])


def test_loader_rejects_duplicate_alias_precision_mapping(tmp_path: Path) -> None:
    schema_path = architecture_capability_schema_path()
    payload = json.loads(architecture_capability_data_path().read_text(encoding="utf-8"))
    duplicate = dict(payload["capabilities"][1])
    duplicate["capability_id"] = "granite-duplicate-for-test"
    duplicate["model_type_aliases"] = ["llama"]
    duplicate["architecture_aliases"] = ["LlamaForCausalLM"]
    payload["capabilities"].append(duplicate)
    data_path = tmp_path / "duplicate-alias.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Ambiguous capability mapping"):
        load_architecture_capability_registry(data_path=data_path, schema_path=schema_path)


def test_loader_rejects_missing_evidence(tmp_path: Path) -> None:
    schema_path = architecture_capability_schema_path()
    payload = json.loads(architecture_capability_data_path().read_text(encoding="utf-8"))
    payload["capabilities"][0]["evidence"] = []
    data_path = tmp_path / "missing-evidence.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence must contain at least one evidence item"):
        load_architecture_capability_registry(data_path=data_path, schema_path=schema_path)


def test_loader_rejects_invalid_status_transition(tmp_path: Path) -> None:
    schema_path = architecture_capability_schema_path()
    payload = json.loads(architecture_capability_data_path().read_text(encoding="utf-8"))
    payload["capabilities"][0]["allowed_status_transitions"] = [
        "verified",
        "tool-supported-unverified"
    ]
    data_path = tmp_path / "invalid-transitions.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid transition"):
        load_architecture_capability_registry(data_path=data_path, schema_path=schema_path)
