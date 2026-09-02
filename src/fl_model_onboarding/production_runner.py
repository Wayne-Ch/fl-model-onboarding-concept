from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping, Sequence

from .adapters.interfaces import CommandResult, CommandSpec, ProcessRunner
from .adapters.huggingface_acquisition import HuggingFaceAcquisitionAdapter
from .adapters.interfaces import HuggingFaceAcquisitionClient
from .architecture_capabilities import CapabilityStatus, ResolutionOutcome
from .contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    BuildRequest,
    CandidateModality,
    FailureClassification,
    FailureInfo,
    GeneratedRecipeAttemptBinding,
    JobState,
    ValidationResult,
    ValidationStatus,
)
from .recipe_attempt_store import AttemptState, RecipeAttempt, RecipeAttemptStore
from .recipe_compiler import GeneratedRecipeCompileError, validate_generated_recipe_payload
from .state_machine import fail_job, transition
from .recipes import (
    AncillaryFileRule,
    DEFAULT_RECIPE_REGISTRY,
    MobiusRecipeArgs,
    OliveRecipeArgs,
    OptimizationChoice,
    SMOLLM2_MODEL_ID as VERIFIED_SMOLLM2_MODEL_ID,
    SMOLLM2_VERIFIED_REVISION,
    ModelRecipe,
    RecipeRegistry,
    RecipeStatus,
)

SMOLLM2_MODEL_ID = VERIFIED_SMOLLM2_MODEL_ID
SMOLLM2_REVISION = SMOLLM2_VERIFIED_REVISION
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_QUANTIZED_OUTPUT_RE = re.compile(r"^(?P<base>.+)_Q(?P<bits>\d+)$", re.IGNORECASE)
_INDEXED_DECODER_OUTPUT_RE = re.compile(r"%(?:0?\d*)d")
_MAX_COMMAND_FAILURE_DETAIL_CHARS = 1200


def _result_payload(result: CommandResult) -> dict[str, object]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {"ok": False, "error": result.stderr or "Command returned no JSON result."}


def _compact_failure_detail(value: str) -> str:
    compact = " ".join(value.split())
    if not compact:
        return "process returned no diagnostic output."
    traceback_index = compact.lower().rfind("traceback")
    if traceback_index >= 0:
        compact = compact[traceback_index:]
    if len(compact) <= _MAX_COMMAND_FAILURE_DETAIL_CHARS:
        return compact
    return "..." + compact[-(_MAX_COMMAND_FAILURE_DETAIL_CHARS - 3) :].lstrip()


def _resolve_staging_relative_path(
    *,
    staging_dir: Path,
    relative_path: str,
    field_name: str,
) -> Path:
    normalized = relative_path.strip().replace("\\", "/")
    if not normalized:
        raise RuntimeError(f"{field_name} must be a non-empty relative path.")
    if normalized.startswith("/") or Path(normalized).is_absolute():
        raise RuntimeError(f"{field_name} must remain relative to the staging package root.")
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise RuntimeError(f"{field_name} contains unsafe path components.")
    staging_root = staging_dir.resolve()
    candidate = (staging_root / Path(*parts)).resolve()
    try:
        candidate.relative_to(staging_root)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} escapes the staging package root.") from exc
    return candidate


def _decoder_output_map_from_config_payload(payload: Mapping[str, object]) -> dict[str, str] | None:
    model = payload.get("model")
    if model is None:
        return None
    if not isinstance(model, dict):
        raise RuntimeError("genai_config.json field 'model' must be an object when present.")
    decoder = model.get("decoder")
    if decoder is None:
        return None
    if not isinstance(decoder, dict):
        raise RuntimeError("genai_config.json field 'model.decoder' must be an object when present.")
    outputs = decoder.get("outputs")
    if outputs is None:
        return None
    if not isinstance(outputs, dict):
        raise RuntimeError("genai_config.json field 'model.decoder.outputs' must be an object.")
    mapped: dict[str, str] = {}
    for key, value in outputs.items():
        if not isinstance(value, str):
            raise RuntimeError(
                "genai_config.json field 'model.decoder.outputs' must map to string output names.",
            )
        mapped[str(key)] = value
    return mapped


def _load_onnx_graph_output_names(model_path: Path) -> tuple[str, ...]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX Python package is required for decoder output reconciliation.",
        ) from exc
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to read ONNX graph outputs from '{model_path.name}'.",
        ) from exc
    outputs = tuple(dict.fromkeys(str(row.name) for row in model.graph.output if str(row.name)))
    if not outputs:
        raise RuntimeError(f"ONNX model '{model_path.name}' has no graph outputs to validate.")
    return outputs


def _decoder_indexed_output_pattern(name: str) -> re.Pattern[str] | None:
    if _INDEXED_DECODER_OUTPUT_RE.search(name) is None:
        return None
    parts: list[str] = []
    cursor = 0
    for match in _INDEXED_DECODER_OUTPUT_RE.finditer(name):
        parts.append(re.escape(name[cursor : match.start()]))
        parts.append(r"\d+")
        cursor = match.end()
    parts.append(re.escape(name[cursor:]))
    return re.compile(r"^" + "".join(parts) + r"$")


def _build_decoder_output_reconciliation(
    *,
    graph_outputs: Sequence[str],
    decoder_outputs: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, object]]]:
    available = tuple(dict.fromkeys(str(name) for name in graph_outputs if str(name)))
    present: dict[str, str] = {}
    remapped: dict[str, str] = {}
    unresolved: dict[str, dict[str, object]] = {}
    for logical_name, physical_name in decoder_outputs.items():
        mapped_name = str(physical_name)
        if mapped_name in available:
            present[str(logical_name)] = mapped_name
            continue
        indexed_pattern = _decoder_indexed_output_pattern(mapped_name)
        if indexed_pattern is not None:
            indexed_matches = [output_name for output_name in available if indexed_pattern.fullmatch(output_name)]
            if indexed_matches:
                present[str(logical_name)] = mapped_name
                continue
        candidates = [
            output_name
            for output_name in available
            if (
                (match := _QUANTIZED_OUTPUT_RE.fullmatch(output_name)) is not None
                and match.group("base") == mapped_name
            )
        ]
        if len(candidates) == 1:
            remapped[str(logical_name)] = candidates[0]
            continue
        unresolved[str(logical_name)] = {
            "requested_output": mapped_name,
            "candidates": candidates,
            "reason": "no_match" if len(candidates) == 0 else "ambiguous_match",
        }
    return present, remapped, unresolved


def _reconcile_decoder_outputs_in_staging_package(
    *,
    staging_dir: Path,
    model_relative_path: str = "model.onnx",
    config_relative_path: str = "genai_config.json",
) -> dict[str, object]:
    config_path = _resolve_staging_relative_path(
        staging_dir=staging_dir,
        relative_path=config_relative_path,
        field_name="config_relative_path",
    )
    if not config_path.is_file():
        raise RuntimeError(f"Staging package is missing required file '{config_relative_path}'.")

    payload_raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise RuntimeError("genai_config.json must contain a JSON object.")

    decoder_outputs_before = _decoder_output_map_from_config_payload(payload_raw)
    if decoder_outputs_before is None:
        return {
            "status": "skipped",
            "reason": "decoder-output-mapping-missing",
            "decoder_outputs_before": {},
            "decoder_outputs_after": {},
            "remapped_outputs": {},
            "graph_outputs": [],
            "present_outputs": {},
        }
    if not decoder_outputs_before:
        return {
            "status": "skipped",
            "reason": "decoder-output-mapping-empty",
            "decoder_outputs_before": {},
            "decoder_outputs_after": {},
            "remapped_outputs": {},
            "graph_outputs": [],
            "present_outputs": {},
        }

    model_path = _resolve_staging_relative_path(
        staging_dir=staging_dir,
        relative_path=model_relative_path,
        field_name="model_relative_path",
    )
    if not model_path.is_file():
        raise RuntimeError(f"Staging package is missing required file '{model_relative_path}'.")

    graph_outputs = _load_onnx_graph_output_names(model_path)
    present_outputs, remapped_outputs, unresolved = _build_decoder_output_reconciliation(
        graph_outputs=graph_outputs,
        decoder_outputs=decoder_outputs_before,
    )
    decoder_outputs_after = dict(decoder_outputs_before)
    decoder_outputs_after.update(remapped_outputs)

    if unresolved:
        details = ", ".join(
            f"{key}=>{row['requested_output']} (candidates={row['candidates']})"
            for key, row in sorted(unresolved.items())
        )
        raise RuntimeError(
            "Decoder output reconciliation failed for staging package due to unresolved mappings: "
            + details,
        )

    if remapped_outputs:
        model_payload = payload_raw.get("model")
        if not isinstance(model_payload, dict):
            raise RuntimeError("genai_config.json is missing required object 'model'.")
        decoder_payload = model_payload.get("decoder")
        if not isinstance(decoder_payload, dict):
            raise RuntimeError("genai_config.json is missing required object 'model.decoder'.")
        outputs_payload = decoder_payload.get("outputs")
        if not isinstance(outputs_payload, dict):
            raise RuntimeError("genai_config.json is missing required object 'model.decoder.outputs'.")
        for logical_name, remapped_name in remapped_outputs.items():
            outputs_payload[logical_name] = remapped_name
        config_path.write_text(json.dumps(payload_raw, indent=2), encoding="utf-8")

    return {
        "status": "applied" if remapped_outputs else "verified",
        "reason": "ok",
        "graph_outputs": list(graph_outputs),
        "present_outputs": present_outputs,
        "decoder_outputs_before": decoder_outputs_before,
        "decoder_outputs_after": decoder_outputs_after,
        "remapped_outputs": remapped_outputs,
    }


@dataclass(frozen=True)
class RecipeExecutionPlan:
    recipe: ModelRecipe
    pinned_revision: str
    source: str


class RecipeExecutionResolutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        classification: FailureClassification = FailureClassification.INVALID_REQUEST,
    ) -> None:
        super().__init__(message)
        self.classification = classification


@dataclass(frozen=True)
class PinnedModelSource:
    model_id: str
    revision_sha: str
    snapshot_dir: Path


class RecipeExecutionResolver:
    def __init__(
        self,
        *,
        recipe_registry: RecipeRegistry = DEFAULT_RECIPE_REGISTRY,
        recipe_attempt_store: RecipeAttemptStore | None = None,
    ) -> None:
        self._recipe_registry = recipe_registry
        self._recipe_attempt_store = recipe_attempt_store

    def resolve(self, request: BuildRequest) -> RecipeExecutionPlan:
        generated_binding = request.generated_recipe_attempt
        if generated_binding is None:
            return self._resolve_static_recipe(request)
        return self._resolve_generated_recipe(request, generated_binding)

    def _resolve_static_recipe(self, request: BuildRequest) -> RecipeExecutionPlan:
        recipe_match = self._recipe_registry.resolve(
            model_id=request.candidate.huggingface_model_id,
            modality=request.candidate.modality,
            task_profile=request.task_profile,
            allow_experimental=True,
        )
        recipe = recipe_match.recipe
        if recipe is None:
            raise RecipeExecutionResolutionError(
                recipe_match.reason,
                classification=FailureClassification.NOT_VERIFIED,
            )
        if recipe.status != RecipeStatus.VERIFIED:
            raise RecipeExecutionResolutionError(
                f"Production execution is verified only for recipe status 'verified'; "
                f"received '{recipe.id}' ({recipe.status.value}).",
                classification=FailureClassification.NOT_VERIFIED,
            )
        if recipe.verified_revision is None:
            raise RecipeExecutionResolutionError(
                f"Verified recipe '{recipe.id}' is missing a pinned verified revision.",
                classification=FailureClassification.NOT_VERIFIED,
            )
        if request.hf_revision != recipe.verified_revision:
            raise RecipeExecutionResolutionError(
                f"Production execution requires pinned revision {recipe.verified_revision}; "
                f"received {request.hf_revision or 'none'}.",
                classification=FailureClassification.NOT_VERIFIED,
            )
        return RecipeExecutionPlan(
            recipe=recipe,
            pinned_revision=recipe.verified_revision,
            source="static_verified_recipe_registry",
        )

    def _resolve_generated_recipe(
        self,
        request: BuildRequest,
        binding: GeneratedRecipeAttemptBinding,
    ) -> RecipeExecutionPlan:
        if self._recipe_attempt_store is None:
            raise RecipeExecutionResolutionError(
                "Generated recipe execution requires a recipe-attempt store.",
            )
        attempt_id = binding.attempt_id.strip()
        fingerprint = binding.recipe_fingerprint.strip().lower()
        if not attempt_id:
            raise RecipeExecutionResolutionError("Generated recipe attempt id is missing.")
        if not _HEX64_RE.fullmatch(fingerprint):
            raise RecipeExecutionResolutionError(
                "Generated recipe fingerprint must be a lowercase 64-character hex value.",
            )
        if not binding.confirmed:
            raise RecipeExecutionResolutionError(
                "Automatic generated recipe attempts require explicit confirmation.",
            )
        if not binding.confirmation_provenance.strip():
            raise RecipeExecutionResolutionError(
                "Automatic generated recipe attempts require explicit confirmation provenance.",
            )
        if not request.allow_experimental:
            raise RecipeExecutionResolutionError(
                "Generated recipe execution requires allow_experimental=true.",
            )
        try:
            attempt = self._recipe_attempt_store.get_attempt(attempt_id)
        except KeyError as exc:
            raise RecipeExecutionResolutionError(
                f"Recipe attempt '{attempt_id}' was not found for generated execution.",
            ) from exc
        if attempt.recipe_fingerprint != fingerprint:
            raise RecipeExecutionResolutionError(
                f"Recipe attempt '{attempt_id}' fingerprint mismatch: expected {fingerprint}, "
                f"store has {attempt.recipe_fingerprint}.",
            )
        if attempt.state != AttemptState.RUNNING:
            raise RecipeExecutionResolutionError(
                f"Recipe attempt '{attempt_id}' is in state '{attempt.state.value}' and cannot execute.",
            )
        generated_record = self._recipe_attempt_store.get_generated_recipe(fingerprint)
        if generated_record is None:
            raise RecipeExecutionResolutionError(
                f"Generated recipe record '{fingerprint}' was not found.",
            )
        if generated_record.recipe_status != RecipeStatus.EXPERIMENTAL:
            raise RecipeExecutionResolutionError(
                f"Generated recipe '{fingerprint}' must remain experimental before promotion; "
                f"found '{generated_record.recipe_status.value}'.",
            )
        self._assert_attempt_identity_matches_generated(attempt=attempt, generated=generated_record)
        recipe, pinned_revision, resolution_outcome, capability_status = _load_generated_recipe_execution_plan(
            generated_record.payload()
        )
        if recipe.status != RecipeStatus.EXPERIMENTAL:
            raise RecipeExecutionResolutionError(
                f"Generated execution requires experimental recipe status; got '{recipe.status.value}'.",
            )
        if resolution_outcome != ResolutionOutcome.EXACT.value:
            raise RecipeExecutionResolutionError(
                f"Generated recipe capability resolution must be '{ResolutionOutcome.EXACT.value}', "
                f"got '{resolution_outcome}'.",
            )
        if capability_status == CapabilityStatus.SOURCE_CHANGE_REQUIRED.value:
            raise RecipeExecutionResolutionError(
                "Generated recipe capability is source-change-required and cannot run tooling.",
            )
        if capability_status not in {
            CapabilityStatus.VERIFIED.value,
            CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED.value,
        }:
            raise RecipeExecutionResolutionError(
                f"Unsupported generated capability status '{capability_status}'.",
            )

        request_revision = _normalize_revision_sha(
            request.hf_revision,
            field_name="request.hf_revision",
        )
        if request_revision != pinned_revision:
            raise RecipeExecutionResolutionError(
                "Generated recipe request revision mismatch against persisted pinned revision.",
            )
        if request.candidate.huggingface_model_id != attempt.model_id:
            raise RecipeExecutionResolutionError(
                "Generated recipe request model id does not match persisted attempt identity.",
            )
        if request.candidate.huggingface_model_id != recipe.huggingface_model_id:
            raise RecipeExecutionResolutionError(
                "Generated recipe payload model id does not match request candidate.",
            )
        if request.candidate.modality != recipe.modality:
            raise RecipeExecutionResolutionError(
                "Generated recipe request modality does not match persisted recipe modality.",
            )
        if request.task_profile != recipe.task_profile:
            raise RecipeExecutionResolutionError(
                "Generated recipe request task profile does not match persisted recipe profile.",
            )
        if request.recipe_id is not None and request.recipe_id != recipe.id:
            raise RecipeExecutionResolutionError(
                "Generated recipe request recipe_id does not match persisted recipe id.",
            )
        if request.recipe_version is not None and request.recipe_version != recipe.version:
            raise RecipeExecutionResolutionError(
                "Generated recipe request recipe_version does not match persisted recipe version.",
            )
        if request.recipe_status is not None and request.recipe_status.strip().lower() != recipe.status.value:
            raise RecipeExecutionResolutionError(
                "Generated recipe request recipe_status does not match persisted recipe status.",
            )
        if (
            request.recipe_artifact_cache_prefix is not None
            and request.recipe_artifact_cache_prefix != recipe.artifact_cache_prefix
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request artifact cache prefix does not match persisted recipe.",
            )
        if (
            request.recipe_model_name_prefix is not None
            and request.recipe_model_name_prefix != recipe.model_name_prefix
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request model name prefix does not match persisted recipe.",
            )
        selected = recipe.choice_for_profile(request.task_profile, request.skip_olive)
        if selected is None:
            supported = ", ".join(
                f"{choice.task_profile}/skip_olive={choice.skip_olive}"
                for choice in recipe.optimization_choices
            )
            raise RecipeExecutionResolutionError(
                f"Generated recipe '{recipe.id}' does not support task_profile={request.task_profile} "
                f"with skip_olive={request.skip_olive}. Supported: {supported or 'none'}.",
            )
        if (
            request.optimization_strategy is not None
            and request.optimization_strategy.lower() != selected.strategy.lower()
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request optimization strategy does not match persisted recipe choice.",
            )
        if (
            request.optimization_precision is not None
            and request.optimization_precision.lower() != selected.precision.lower()
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request optimization precision does not match persisted recipe choice.",
            )
        expected_mobius_dtype = recipe.mobius.dtype
        if request.candidate.recommended_mobius_dtype != expected_mobius_dtype:
            raise RecipeExecutionResolutionError(
                "Generated recipe request candidate Mobius dtype does not match persisted recipe.",
            )
        expected_olive_precision = None if selected.skip_olive else selected.precision
        if request.candidate.recommended_olive_precision != expected_olive_precision:
            raise RecipeExecutionResolutionError(
                "Generated recipe request candidate Olive precision does not match persisted recipe.",
            )
        return RecipeExecutionPlan(
            recipe=recipe,
            pinned_revision=pinned_revision,
            source="generated_recipe_attempt_store",
        )

    @staticmethod
    def _assert_attempt_identity_matches_generated(
        *,
        attempt: RecipeAttempt,
        generated: Any,
    ) -> None:
        mismatches: list[str] = []
        for field_name in (
            "model_id",
            "revision_sha",
            "requested_device",
            "requested_precision",
            "compiler_version",
            "capability_fingerprint",
            "toolchain_fingerprint",
            "profile_fingerprint",
        ):
            if getattr(attempt, field_name) != getattr(generated, field_name):
                mismatches.append(field_name)
        if mismatches:
            raise RecipeExecutionResolutionError(
                "Recipe attempt identity mismatch against generated record for field(s): "
                + ", ".join(mismatches)
                + ".",
            )


def _load_generated_recipe_execution_plan(
    payload: dict[str, object],
) -> tuple[ModelRecipe, str, str, str]:
    try:
        validate_generated_recipe_payload(payload)
    except GeneratedRecipeCompileError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe payload failed schema validation: {exc}",
        ) from exc

    recipe_payload = _require_mapping(payload.get("recipe"), field_name="generated.recipe")
    recipe = _recipe_from_payload(recipe_payload)
    pinned_revision = _normalize_revision_sha(
        payload.get("pinned_revision"),
        field_name="generated.pinned_revision",
    )
    provenance_payload = _require_mapping(payload.get("provenance"), field_name="generated.provenance")
    resolution_outcome = _require_string(
        provenance_payload.get("resolution_outcome"),
        field_name="generated.provenance.resolution_outcome",
    ).lower()
    capability_status = _require_string(
        provenance_payload.get("capability_status"),
        field_name="generated.provenance.capability_status",
    ).lower()
    return recipe, pinned_revision, resolution_outcome, capability_status


def _recipe_from_payload(payload: dict[str, object]) -> ModelRecipe:
    mobius_payload = _require_mapping(payload.get("mobius"), field_name="generated.recipe.mobius")
    olive_raw = payload.get("olive")
    olive_payload = _require_mapping(olive_raw, field_name="generated.recipe.olive") if olive_raw is not None else None

    ancillary_rows = _require_array(payload.get("ancillary_files"), field_name="generated.recipe.ancillary_files")
    ancillary_files: list[AncillaryFileRule] = []
    for index, row in enumerate(ancillary_rows, start=1):
        item = _require_mapping(row, field_name=f"generated.recipe.ancillary_files[{index}]")
        ancillary_files.append(
            AncillaryFileRule(
                relative_path=_require_string(
                    item.get("relative_path"),
                    field_name=f"generated.recipe.ancillary_files[{index}].relative_path",
                ),
                required=_require_bool(
                    item.get("required"),
                    field_name=f"generated.recipe.ancillary_files[{index}].required",
                ),
                source=_require_string(
                    item.get("source"),
                    field_name=f"generated.recipe.ancillary_files[{index}].source",
                ),
            )
        )

    optimization_rows = _require_array(
        payload.get("optimization_choices"),
        field_name="generated.recipe.optimization_choices",
    )
    optimization_choices: list[OptimizationChoice] = []
    for index, row in enumerate(optimization_rows, start=1):
        item = _require_mapping(row, field_name=f"generated.recipe.optimization_choices[{index}]")
        optimization_choices.append(
            OptimizationChoice(
                strategy=_require_string(
                    item.get("strategy"),
                    field_name=f"generated.recipe.optimization_choices[{index}].strategy",
                ),
                precision=_require_string(
                    item.get("precision"),
                    field_name=f"generated.recipe.optimization_choices[{index}].precision",
                ),
                task_profile=_require_string(
                    item.get("task_profile"),
                    field_name=f"generated.recipe.optimization_choices[{index}].task_profile",
                ),
                skip_olive=_require_bool(
                    item.get("skip_olive"),
                    field_name=f"generated.recipe.optimization_choices[{index}].skip_olive",
                ),
                default=_require_bool(
                    item.get("default"),
                    field_name=f"generated.recipe.optimization_choices[{index}].default",
                ),
            )
        )
    try:
        modality = CandidateModality(
            _require_string(payload.get("modality"), field_name="generated.recipe.modality")
        )
    except ValueError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe modality is unsupported: {payload.get('modality')!r}.",
        ) from exc
    try:
        inference_modality = CandidateModality(
            _require_string(
                payload.get("inference_modality"),
                field_name="generated.recipe.inference_modality",
            )
        )
    except ValueError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe inference_modality is unsupported: {payload.get('inference_modality')!r}.",
        ) from exc
    try:
        status = RecipeStatus(_require_string(payload.get("status"), field_name="generated.recipe.status"))
    except ValueError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe status is unsupported: {payload.get('status')!r}.",
        ) from exc
    return ModelRecipe(
        id=_require_string(payload.get("id"), field_name="generated.recipe.id"),
        version=_require_string(payload.get("version"), field_name="generated.recipe.version"),
        status=status,
        status_reason=_require_string(
            payload.get("status_reason"),
            field_name="generated.recipe.status_reason",
        ),
        huggingface_model_id=_require_string(
            payload.get("huggingface_model_id"),
            field_name="generated.recipe.huggingface_model_id",
        ),
        modality=modality,
        task_profile=_require_string(payload.get("task_profile"), field_name="generated.recipe.task_profile"),
        verified_revision=_optional_string(payload.get("verified_revision")),
        preferred_revision=_optional_string(payload.get("preferred_revision")),
        mobius=MobiusRecipeArgs(
            ep=_require_string(mobius_payload.get("ep"), field_name="generated.recipe.mobius.ep"),
            runtime=_require_string(mobius_payload.get("runtime"), field_name="generated.recipe.mobius.runtime"),
            dtype=_optional_string(mobius_payload.get("dtype")),
            task=_optional_string(mobius_payload.get("task")),
        ),
        olive=(
            OliveRecipeArgs(
                input_source=_require_string(
                    olive_payload.get("input_source"),
                    field_name="generated.recipe.olive.input_source",
                ),
                task=_require_string(olive_payload.get("task"), field_name="generated.recipe.olive.task"),
                precision=_optional_string(olive_payload.get("precision")),
                device=_require_string(
                    olive_payload.get("device"),
                    field_name="generated.recipe.olive.device",
                ),
                provider=_require_string(
                    olive_payload.get("provider"),
                    field_name="generated.recipe.olive.provider",
                ),
                log_level=_require_string(
                    olive_payload.get("log_level"),
                    field_name="generated.recipe.olive.log_level",
                ),
            )
            if olive_payload is not None
            else None
        ),
        ancillary_files=tuple(ancillary_files),
        runtime_validation=_require_string(
            payload.get("runtime_validation"),
            field_name="generated.recipe.runtime_validation",
        ),
        inference_modality=inference_modality,
        optimization_choices=tuple(optimization_choices),
        artifact_cache_prefix=_require_string(
            payload.get("artifact_cache_prefix"),
            field_name="generated.recipe.artifact_cache_prefix",
        ),
        model_name_prefix=_require_string(
            payload.get("model_name_prefix"),
            field_name="generated.recipe.model_name_prefix",
        ),
        success_message=_require_string(
            payload.get("success_message"),
            field_name="generated.recipe.success_message",
        ),
    )


def _require_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RecipeExecutionResolutionError(f"{field_name} must be an object.")
    return value


def _require_array(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise RecipeExecutionResolutionError(f"{field_name} must be an array.")
    return value


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeExecutionResolutionError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RecipeExecutionResolutionError("Optional string field must be null or a string.")
    stripped = value.strip()
    return stripped or None


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeExecutionResolutionError(f"{field_name} must be a boolean.")
    return value


def _normalize_revision_sha(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeExecutionResolutionError(f"{field_name} must be a non-empty string.")
    normalized = value.strip().lower()
    if _HEX40_RE.fullmatch(normalized) is None:
        raise RecipeExecutionResolutionError(
            f"{field_name} must be a full 40-character lowercase hex revision SHA.",
        )
    return normalized


def _resolve_python_executable(value: Path | str | None) -> Path:
    if value is None:
        return Path(sys.executable).resolve()
    if isinstance(value, Path):
        candidate = value
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError("runtime python executable must be non-empty when provided.")
        candidate = Path(value.strip())
    else:
        raise TypeError("runtime python executable must be a path-like string or Path.")
    return candidate.resolve()


class FoundrySdkTextInferenceBackend:
    def __init__(
        self,
        process_runner: ProcessRunner,
        *,
        timeout_seconds: int = 900,
        cancellation_event: Event | None = None,
        runtime_python_executable: Path | str | None = None,
    ) -> None:
        self._process_runner = process_runner
        self._timeout_seconds = timeout_seconds
        self._cancellation_event = cancellation_event
        self._runtime_python_executable = _resolve_python_executable(runtime_python_executable)

    def infer(
        self,
        *,
        artifact: BuildArtifact,
        job: BuildJob,
        prompt: str,
        max_tokens: int,
    ) -> str:
        if len(prompt) > 8192:
            raise ValueError("Inference prompt exceeds the 8192 character limit.")
        model_dir = artifact.path.resolve()
        descriptor = json.loads((model_dir / "inference_model.json").read_text(encoding="utf-8"))
        model_name = descriptor["Name"]
        request_file = job.request.workspace_root / f"inference-{uuid.uuid4().hex}.json"
        request_file.write_text(
            json.dumps({"prompt": prompt, "max_tokens": max_tokens}),
            encoding="utf-8",
        )
        try:
            result = self._process_runner.run(
                CommandSpec(
                    argv=(
                        str(self._runtime_python_executable),
                        "-m",
                        "fl_model_onboarding.runtime_worker",
                        "foundry-infer",
                        "--model-dir",
                        str(model_dir),
                        "--model-name",
                        str(model_name),
                        "--request-file",
                        str(request_file),
                    ),
                    cwd=job.request.workspace_root,
                    timeout_seconds=self._timeout_seconds,
                ),
                cancel_event=self._cancellation_event,
            )
        finally:
            request_file.unlink(missing_ok=True)
        payload = _result_payload(result)
        if not result.ok or payload.get("ok") is not True:
            raise RuntimeError(str(payload.get("error") or "Foundry Local inference failed."))
        return str(payload["output"])


class ProductionBuildStageRunner:
    def __init__(
        self,
        process_runner: ProcessRunner,
        *,
        build_timeout_seconds: int = 7200,
        olive_timeout_seconds: int = 5400,
        runtime_timeout_seconds: int = 900,
        model_acquisition: HuggingFaceAcquisitionClient | None = None,
        recipe_registry: RecipeRegistry | None = None,
        recipe_attempt_store: RecipeAttemptStore | None = None,
        recipe_execution_resolver: RecipeExecutionResolver | None = None,
        runtime_python_executable: Path | str | None = None,
    ) -> None:
        self._process_runner = process_runner
        self._build_timeout_seconds = build_timeout_seconds
        self._olive_timeout_seconds = olive_timeout_seconds
        self._runtime_timeout_seconds = runtime_timeout_seconds
        self._model_acquisition = model_acquisition or HuggingFaceAcquisitionAdapter()
        self._runtime_python_executable = _resolve_python_executable(runtime_python_executable)
        self._recipe_registry = recipe_registry or DEFAULT_RECIPE_REGISTRY
        self._execution_resolver = recipe_execution_resolver or RecipeExecutionResolver(
            recipe_registry=self._recipe_registry,
            recipe_attempt_store=recipe_attempt_store,
        )

    def run(
        self,
        job: BuildJob,
        *,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        staging_dir: Path | None = None
        package_dir: Path | None = None
        staging_preexisting = False
        package_preexisting = False
        try:
            execution = self._execution_resolver.resolve(job.request)
            staging_dir, package_dir = production_package_paths(
                job,
                recipe_registry=self._recipe_registry,
                resolved_recipe=execution.recipe,
            )
            staging_preexisting = staging_dir.exists()
            package_preexisting = package_dir.exists()
            self._run(
                job,
                recipe=execution.recipe,
                pinned_revision=execution.pinned_revision,
                persist=persist,
                cancellation_event=cancellation_event,
            )
        except Exception as exc:
            if staging_dir is not None and not staging_preexisting and staging_dir.exists():
                shutil.rmtree(staging_dir)
            if package_dir is not None and not package_preexisting and package_dir.exists():
                shutil.rmtree(package_dir)
            if job.state == JobState.CANCELLED:
                return
            classification = FailureClassification.PROCESS_FAILED
            if isinstance(exc, FileNotFoundError):
                classification = FailureClassification.MISSING_DEPENDENCY
            elif isinstance(exc, RecipeExecutionResolutionError):
                classification = exc.classification
            fail_job(
                job,
                FailureInfo(
                    stage=job.state,
                    classification=classification,
                    message=str(exc),
                ),
            )
            job.finished_utc = datetime.now(timezone.utc)
            persist()

    def _run(
        self,
        job: BuildJob,
        *,
        recipe: ModelRecipe,
        pinned_revision: str,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        request = job.request
        if recipe.choice_for_profile(request.task_profile, request.skip_olive) is None:
            supported = ", ".join(
                f"{choice.task_profile}/skip_olive={choice.skip_olive}"
                for choice in recipe.optimization_choices
            )
            raise RuntimeError(
                f"Recipe '{recipe.id}' does not support task_profile={request.task_profile} "
                f"with skip_olive={request.skip_olive}. Supported: {supported or 'none'}."
            )
        if request.candidate.modality != CandidateModality.LLM:
            raise RuntimeError("Production execution currently supports LLM runtime validation only.")
        if recipe.olive is None:
            raise RuntimeError(f"Recipe '{recipe.id}' requires Olive settings for production packaging.")

        pinned_source = self._resolve_pinned_source(recipe=recipe, request=request, pinned_revision=pinned_revision)
        mobius_dir = request.workspace_root / "mobius"
        olive_dir = request.workspace_root / "olive"
        mobius_dir.mkdir(parents=True, exist_ok=False)
        olive_dir.mkdir(parents=True, exist_ok=False)

        transition(job, JobState.DOWNLOADING, f"Pinned Hugging Face revision {pinned_source.revision_sha}.")
        persist()
        mobius_dtype = recipe.mobius.dtype or "default"
        transition(
            job,
            JobState.MOBIUS_BUILDING,
            (
                f"Running recipe Mobius {recipe.mobius.ep} {recipe.mobius.runtime} "
                f"{mobius_dtype} build."
            ),
        )
        persist()
        mobius_argv: list[str] = [
            "mobius",
            "build",
            "--config",
            str(pinned_source.snapshot_dir),
            "--ep",
            recipe.mobius.ep,
            "--runtime",
            recipe.mobius.runtime,
        ]
        if recipe.mobius.task:
            mobius_argv.extend(["--task", recipe.mobius.task])
        if recipe.mobius.dtype:
            mobius_argv.extend(["--dtype", recipe.mobius.dtype])
        mobius_argv.append(str(mobius_dir))
        self._run_command(
            CommandSpec(
                argv=tuple(mobius_argv),
                cwd=request.workspace_root,
                timeout_seconds=self._build_timeout_seconds,
            ),
            cancellation_event,
            "Mobius build",
        )
        baseline_model_name = f"{recipe.model_name_prefix}-{job.job_id[:12]}-mobius-baseline:1"
        (mobius_dir / "inference_model.json").write_text(
            json.dumps({"Name": baseline_model_name}, indent=2),
            encoding="utf-8",
        )

        transition(job, JobState.MOBIUS_VALIDATING, "Mobius output created; ONNX validation follows Olive.")
        persist()
        transition(
            job,
            JobState.OLIVE_OPTIMIZING,
            (
                f"Running recipe Olive {recipe.olive.input_source} "
                f"{recipe.olive.precision or 'default'} optimization."
            ),
        )
        persist()
        olive_argv: list[str] = [
            "olive",
            "optimize",
            "--model_name_or_path",
            str(mobius_dir),
            "--task",
            recipe.olive.task,
            "--device",
            recipe.olive.device,
            "--provider",
            recipe.olive.provider,
        ]
        if recipe.olive.precision:
            olive_argv.extend(["--precision", recipe.olive.precision])
        olive_argv.extend(
            [
                "--output_path",
                str(olive_dir),
                "--log_level",
                recipe.olive.log_level,
            ]
        )
        self._run_command(
            CommandSpec(
                argv=tuple(olive_argv),
                cwd=request.workspace_root,
                timeout_seconds=self._olive_timeout_seconds,
            ),
            cancellation_event,
            "Olive optimize",
        )
        source_dir = olive_dir
        self._ensure_required_ancillary_files(source_dir=source_dir, recipe=recipe)

        transition(job, JobState.PACKAGING, "Creating immutable Foundry Local BYOM package.")
        persist()
        artifact_id = self._artifact_id(job)
        model_name = f"{recipe.model_name_prefix}-{artifact_id[:12]}:1"
        staging_dir, package_dir = production_package_paths(
            job,
            recipe_registry=self._recipe_registry,
            resolved_recipe=recipe,
        )
        if package_dir.exists():
            raise FileExistsError(f"Immutable artifact path already exists: {package_dir}")
        if staging_dir.exists():
            raise FileExistsError(f"Partial artifact path already exists: {staging_dir}")
        shutil.copytree(source_dir, staging_dir)
        (staging_dir / "inference_model.json").write_text(
            json.dumps({"Name": model_name}, indent=2),
            encoding="utf-8",
        )
        reconciliation = _reconcile_decoder_outputs_in_staging_package(staging_dir=staging_dir)
        if reconciliation["status"] == "applied":
            before = reconciliation["decoder_outputs_before"]
            after = reconciliation["decoder_outputs_after"]
            remapped = reconciliation["remapped_outputs"]
            if isinstance(before, dict) and isinstance(after, dict) and isinstance(remapped, dict):
                delta = {
                    key: {"before": before.get(key), "after": after.get(key)}
                    for key in sorted(remapped.keys())
                }
            else:
                delta = {}
            job.add_event(
                "Staging decoder output reconciliation applied before runtime validation: "
                + json.dumps(delta, sort_keys=True),
            )
            persist()
        elif reconciliation["status"] == "verified":
            job.add_event("Staging decoder output reconciliation verified existing decoder mappings.")
            persist()
        else:
            job.add_event(
                "Staging decoder output reconciliation skipped: "
                + str(reconciliation.get("reason") or "no decoder outputs mapping"),
            )
            persist()

        transition(job, JobState.RUNTIME_VALIDATING, "Validating ONNX, ORT CPU, and OGA generation.")
        persist()
        runtime_result = self._run_command(
            CommandSpec(
                argv=(
                    str(self._runtime_python_executable),
                    "-m",
                    "fl_model_onboarding.runtime_worker",
                    "validate-runtime",
                    "--model-dir",
                    str(staging_dir),
                ),
                cwd=request.workspace_root,
                timeout_seconds=self._runtime_timeout_seconds,
            ),
            cancellation_event,
            "Runtime validation",
        )
        runtime_payload = _result_payload(runtime_result)
        checks = tuple(str(item) for item in runtime_payload.get("checks", []))
        job.validations.append(
            ValidationResult(
                stage=JobState.RUNTIME_VALIDATING,
                status=ValidationStatus.PASSED,
                checks=checks,
            )
        )

        staging_dir.rename(package_dir)
        transition(job, JobState.FL_LOADING, "Foundry Local SDK discovered and loaded the BYOM package.")
        persist()
        transition(job, JobState.INFERENCING, "Running bounded Foundry Local SDK chat inference.")
        persist()
        inference_backend = FoundrySdkTextInferenceBackend(
            self._process_runner,
            timeout_seconds=self._runtime_timeout_seconds,
            cancellation_event=cancellation_event,
            runtime_python_executable=self._runtime_python_executable,
        )
        output = inference_backend.infer(
            artifact=BuildArtifact(
                artifact_id=artifact_id,
                kind=ArtifactKind.MODEL,
                path=package_dir,
                description="Immutable Foundry Local BYOM model package",
            ),
            job=job,
            prompt="Reply with: OK",
            max_tokens=64,
        )
        if not output.strip():
            raise RuntimeError("Foundry Local SDK inference returned empty output.")
        job.validations.append(
            ValidationResult(
                stage=JobState.INFERENCING,
                status=ValidationStatus.PASSED,
                checks=("foundry_local_sdk_chat=passed",),
            )
        )
        job.register_artifact(
            BuildArtifact(
                artifact_id=artifact_id,
                kind=ArtifactKind.MODEL,
                path=package_dir,
                description="Immutable Foundry Local BYOM model package",
            )
        )
        transition(job, JobState.SUCCEEDED, recipe.success_message)
        job.finished_utc = datetime.now(timezone.utc)
        persist()

    def _resolve_pinned_source(
        self,
        *,
        recipe: ModelRecipe,
        request: BuildRequest,
        pinned_revision: str,
    ) -> PinnedModelSource:
        snapshot_dir = pinned_snapshot_cache_path(
            request.model_cache_dir,
            model_id=recipe.huggingface_model_id,
            revision_sha=pinned_revision,
        )
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._model_acquisition.acquire_snapshot(
            recipe.huggingface_model_id,
            snapshot_dir,
            revision=pinned_revision,
        )
        if not snapshot_path.is_dir():
            raise RuntimeError(
                f"Pinned snapshot path is not a directory: {snapshot_path}."
            )
        return PinnedModelSource(
            model_id=recipe.huggingface_model_id,
            revision_sha=pinned_revision,
            snapshot_dir=snapshot_path,
        )

    def _run_command(
        self,
        spec: CommandSpec,
        cancellation_event: Event,
        label: str,
    ) -> CommandResult:
        result = self._process_runner.run(spec, cancel_event=cancellation_event)
        if not result.ok:
            detail = _compact_failure_detail(
                result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
            )
            raise RuntimeError(f"{label} failed: {detail}")
        return result

    @staticmethod
    def _artifact_id(job: BuildJob) -> str:
        request = job.request
        return hashlib.sha256(
            f"{request.candidate.huggingface_model_id}:{request.hf_revision}:{request.task_profile}:{job.job_id}".encode()
        ).hexdigest()

    @staticmethod
    def _ensure_required_ancillary_files(*, source_dir: Path, recipe: ModelRecipe) -> None:
        missing = [
            rule.relative_path
            for rule in recipe.ancillary_files
            if rule.required and not (source_dir / rule.relative_path).exists()
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise RuntimeError(
                f"Recipe '{recipe.id}' packaging is missing required ancillary files: {joined}."
            )


def production_package_paths(
    job: BuildJob,
    *,
    recipe_registry: RecipeRegistry = DEFAULT_RECIPE_REGISTRY,
    resolved_recipe: ModelRecipe | None = None,
) -> tuple[Path, Path]:
    artifact_id = ProductionBuildStageRunner._artifact_id(job)
    cache = job.request.model_cache_dir
    if resolved_recipe is not None:
        prefix = resolved_recipe.artifact_cache_prefix
    elif job.request.recipe_artifact_cache_prefix:
        prefix = job.request.recipe_artifact_cache_prefix
    else:
        recipe = recipe_registry.resolve(
            model_id=job.request.candidate.huggingface_model_id,
            modality=job.request.candidate.modality,
            task_profile=job.request.task_profile,
            allow_experimental=True,
        ).recipe
        prefix = (
            recipe.artifact_cache_prefix
            if recipe
            else _fallback_cache_prefix(job.request.candidate.huggingface_model_id)
        )
    normalized_prefix = _normalize_cache_prefix(prefix)
    return (
        cache / f".partial-{normalized_prefix}-{artifact_id[:12]}",
        cache / f"{normalized_prefix}-{artifact_id[:12]}",
    )


def _normalize_cache_prefix(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def _fallback_cache_prefix(model_id: str) -> str:
    return _normalize_cache_prefix(model_id.strip().split("/")[-1])


def pinned_snapshot_cache_path(model_cache_dir: Path, *, model_id: str, revision_sha: str) -> Path:
    normalized_revision = _normalize_revision_sha(
        revision_sha,
        field_name="snapshot.revision_sha",
    )
    model_slug = _normalize_cache_prefix(model_id.replace("/", "-"))
    return model_cache_dir / f"snapshot-{model_slug}-{normalized_revision}"
