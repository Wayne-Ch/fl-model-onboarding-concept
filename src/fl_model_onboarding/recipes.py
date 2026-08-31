from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .contracts import CandidateModality, ModelCandidate

SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SMOLLM2_VERIFIED_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
DISTIL_WHISPER_MODEL_ID = "distil-whisper/distil-medium.en"
DISTIL_WHISPER_BLOCKED_REVISION = "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7"
GRANITE_MODEL_ID = "ibm-granite/granite-3.3-2b-instruct"
GRANITE_DISCOVERY_REVISION = "707f574c62054322f6b5b04b6d075f0a8f05e0f0"


class RecipeStatus(StrEnum):
    VERIFIED = "verified"
    EXPERIMENTAL = "experimental"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class MobiusRecipeArgs:
    ep: str
    runtime: str
    dtype: str | None = None
    task: str | None = None


@dataclass(frozen=True)
class OliveRecipeArgs:
    input_source: str
    task: str
    precision: str | None
    device: str = "cpu"
    provider: str = "CPUExecutionProvider"
    log_level: str = "1"


@dataclass(frozen=True)
class AncillaryFileRule:
    relative_path: str
    required: bool
    source: str


@dataclass(frozen=True)
class OptimizationChoice:
    strategy: str
    precision: str
    task_profile: str
    skip_olive: bool
    default: bool = False


@dataclass(frozen=True)
class ModelRecipe:
    id: str
    version: str
    status: RecipeStatus
    status_reason: str
    huggingface_model_id: str
    modality: CandidateModality
    task_profile: str
    verified_revision: str | None
    preferred_revision: str | None
    mobius: MobiusRecipeArgs
    olive: OliveRecipeArgs | None
    ancillary_files: tuple[AncillaryFileRule, ...]
    runtime_validation: str
    inference_modality: CandidateModality
    optimization_choices: tuple[OptimizationChoice, ...]
    artifact_cache_prefix: str
    model_name_prefix: str
    success_message: str

    def default_optimization(self) -> OptimizationChoice | None:
        for choice in self.optimization_choices:
            if choice.default:
                return choice
        return self.optimization_choices[0] if self.optimization_choices else None

    def choice_for_profile(self, task_profile: str, skip_olive: bool) -> OptimizationChoice | None:
        normalized = task_profile.strip().lower()
        for choice in self.optimization_choices:
            if choice.task_profile.lower() == normalized and choice.skip_olive == skip_olive:
                return choice
        return None

    def to_candidate(self, *, task_profile: str, skip_olive: bool) -> ModelCandidate:
        selected = self.choice_for_profile(task_profile, skip_olive)
        fallback = self.default_optimization()
        olive_precision = None
        if selected is not None and not selected.skip_olive:
            olive_precision = selected.precision
        elif selected is None and fallback is not None and not fallback.skip_olive:
            olive_precision = fallback.precision
        return ModelCandidate(
            key=self.id,
            huggingface_model_id=self.huggingface_model_id,
            modality=self.modality,
            recommended_mobius_dtype=self.mobius.dtype,
            recommended_olive_precision=olive_precision,
            notes=f"Recipe {self.id}@{self.version} ({self.status.value})",
        )


@dataclass(frozen=True)
class RecipeResolution:
    recipe: ModelRecipe | None
    status: str
    reason: str
    buildable: bool
    requires_experimental_opt_in: bool


class RecipeRegistry:
    def __init__(self, recipes: tuple[ModelRecipe, ...]) -> None:
        self._recipes = recipes

    def all(self) -> tuple[ModelRecipe, ...]:
        return self._recipes

    def resolve(
        self,
        *,
        model_id: str,
        modality: CandidateModality,
        task_profile: str,
        allow_experimental: bool,
    ) -> RecipeResolution:
        matches = self._recipes_for_model(model_id=model_id, modality=modality)
        if not matches:
            return RecipeResolution(
                recipe=None,
                status="unregistered",
                reason=(
                    f"No recipe is registered for model '{model_id}' ({modality.value}). "
                    "Build remains blocked until a recipe is added."
                ),
                buildable=False,
                requires_experimental_opt_in=False,
            )
        normalized_profile = task_profile.strip().lower()
        default_aliases = {"", "default", f"{modality.value}-cpu-default"}
        if normalized_profile in default_aliases:
            recipe = self._preferred_recipe(matches)
        else:
            recipe = next((item for item in matches if item.task_profile.lower() == normalized_profile), None)
            if recipe is None:
                supported = ", ".join(sorted(item.task_profile for item in matches))
                return RecipeResolution(
                    recipe=None,
                    status="unregistered",
                    reason=(
                        f"Task profile '{task_profile}' is not registered for model '{model_id}'. "
                        f"Supported profiles: {supported}."
                    ),
                    buildable=False,
                    requires_experimental_opt_in=False,
                )
        if recipe.status == RecipeStatus.BLOCKED:
            return RecipeResolution(
                recipe=recipe,
                status=recipe.status.value,
                reason=recipe.status_reason,
                buildable=False,
                requires_experimental_opt_in=False,
            )
        if recipe.status == RecipeStatus.EXPERIMENTAL and not allow_experimental:
            return RecipeResolution(
                recipe=recipe,
                status=recipe.status.value,
                reason=(
                    f"Recipe '{recipe.id}' is experimental. "
                    "Pass explicit experimental opt-in to preflight/build this profile."
                ),
                buildable=False,
                requires_experimental_opt_in=True,
            )
        return RecipeResolution(
            recipe=recipe,
            status=recipe.status.value,
            reason=recipe.status_reason,
            buildable=True,
            requires_experimental_opt_in=False,
        )

    def describe_recipe(self, recipe: ModelRecipe) -> dict[str, object]:
        return {
            "id": recipe.id,
            "version": recipe.version,
            "status": recipe.status.value,
            "reason": recipe.status_reason,
            "model_id": recipe.huggingface_model_id,
            "task_profile": recipe.task_profile,
            "modality": recipe.modality.value,
            "verified_revision": recipe.verified_revision,
            "preferred_revision": recipe.preferred_revision,
            "runtime_validation": recipe.runtime_validation,
            "inference_modality": recipe.inference_modality.value,
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
                    "relative_path": rule.relative_path,
                    "required": rule.required,
                    "source": rule.source,
                }
                for rule in recipe.ancillary_files
            ],
            "supported_optimizations": [
                {
                    "strategy": choice.strategy,
                    "precision": choice.precision,
                    "task_profile": choice.task_profile,
                    "skip_olive": choice.skip_olive,
                    "default": choice.default,
                }
                for choice in recipe.optimization_choices
            ],
        }

    def _recipes_for_model(self, *, model_id: str, modality: CandidateModality) -> tuple[ModelRecipe, ...]:
        normalized = model_id.strip().lower()
        return tuple(
            recipe
            for recipe in self._recipes
            if recipe.huggingface_model_id.lower() == normalized and recipe.modality == modality
        )

    @staticmethod
    def _preferred_recipe(recipes: tuple[ModelRecipe, ...]) -> ModelRecipe:
        rank = {
            RecipeStatus.VERIFIED: 0,
            RecipeStatus.EXPERIMENTAL: 1,
            RecipeStatus.BLOCKED: 2,
        }
        return sorted(
            recipes,
            key=lambda item: (
                rank[item.status],
                item.version,
            ),
        )[0]


DEFAULT_MODEL_RECIPES: tuple[ModelRecipe, ...] = (
    ModelRecipe(
        id="smollm2-1.7b-cpu-int4",
        version="1.0.0",
        status=RecipeStatus.VERIFIED,
        status_reason=(
            "Verified Mobius->Olive->runtime->Foundry Local SDK chat path for the pinned SmolLM2 revision."
        ),
        huggingface_model_id=SMOLLM2_MODEL_ID,
        modality=CandidateModality.LLM,
        task_profile="llm-cpu-int4",
        verified_revision=SMOLLM2_VERIFIED_REVISION,
        preferred_revision=SMOLLM2_VERIFIED_REVISION,
        mobius=MobiusRecipeArgs(ep="cpu", runtime="ort-genai", dtype="f32"),
        olive=OliveRecipeArgs(
            input_source="mobius-output-dir",
            task="text-generation-with-past",
            precision="int4",
        ),
        ancillary_files=(
            AncillaryFileRule(relative_path="model.onnx", required=True, source="olive-output-dir"),
            AncillaryFileRule(relative_path="genai_config.json", required=True, source="olive-output-dir"),
            AncillaryFileRule(relative_path="tokenizer.json", required=True, source="olive-output-dir"),
            AncillaryFileRule(relative_path="model.onnx.data", required=False, source="olive-output-dir"),
        ),
        runtime_validation="onnx-checker + onnxruntime-cpu-load + onnxruntime-genai-generation",
        inference_modality=CandidateModality.LLM,
        optimization_choices=(
            OptimizationChoice(
                strategy="mobius-olive",
                precision="int4",
                task_profile="llm-cpu-int4",
                skip_olive=False,
                default=True,
            ),
        ),
        artifact_cache_prefix="smollm2",
        model_name_prefix="smollm2-onboarding",
        success_message="Verified SmolLM2 Foundry Local build and inference succeeded.",
    ),
    ModelRecipe(
        id="distil-whisper-cpu-fp16",
        version="1.0.0",
        status=RecipeStatus.BLOCKED,
        status_reason=(
            "Blocked: deterministic config adaptation reaches OGA parser/model-load, but OGA and Foundry "
            "transcription still fail with Missing Input: position_ids because WhisperDecoderState does not "
            "bind/update position_ids."
        ),
        huggingface_model_id=DISTIL_WHISPER_MODEL_ID,
        modality=CandidateModality.ASR,
        task_profile="asr-cpu-fp16",
        verified_revision=None,
        preferred_revision=DISTIL_WHISPER_BLOCKED_REVISION,
        mobius=MobiusRecipeArgs(ep="cpu", runtime="ort-genai", dtype="f32"),
        olive=OliveRecipeArgs(
            input_source="mobius-decoder-onnx",
            task="automatic-speech-recognition",
            precision="fp32",
        ),
        ancillary_files=(
            AncillaryFileRule(relative_path="decoder/model.onnx", required=True, source="mobius-output-dir"),
            AncillaryFileRule(relative_path="genai_config.json", required=True, source="mobius-output-dir"),
        ),
        runtime_validation=(
            "onnx-checker + onnxruntime-cpu-load + deterministic-parser/model-load-adaptation + "
            "oga/fl-transcription (blocked: Missing Input: position_ids)"
        ),
        inference_modality=CandidateModality.ASR,
        optimization_choices=(),
        artifact_cache_prefix="distil-whisper",
        model_name_prefix="distil-whisper-onboarding",
        success_message="",
    ),
    ModelRecipe(
        id="granite-3.3-2b-cpu-int4",
        version="1.0.0",
        status=RecipeStatus.VERIFIED,
        status_reason=(
            "Verified direct Mobius->Olive->runtime->Foundry Local SDK chat inference path "
            "for granite-3.3-2b pinned revision 707f574c62054322f6b5b04b6d075f0a8f05e0f0."
        ),
        huggingface_model_id=GRANITE_MODEL_ID,
        modality=CandidateModality.LLM,
        task_profile="llm-cpu-int4",
        verified_revision=GRANITE_DISCOVERY_REVISION,
        preferred_revision=GRANITE_DISCOVERY_REVISION,
        mobius=MobiusRecipeArgs(ep="cpu", runtime="ort-genai", dtype="f32"),
        olive=OliveRecipeArgs(
            input_source="mobius-output-dir",
            task="text-generation-with-past",
            precision="int4",
        ),
        ancillary_files=(
            AncillaryFileRule(relative_path="model.onnx", required=True, source="olive-output-dir"),
            AncillaryFileRule(relative_path="genai_config.json", required=True, source="olive-output-dir"),
            AncillaryFileRule(relative_path="tokenizer.json", required=True, source="olive-output-dir"),
        ),
        runtime_validation="onnx-checker + onnxruntime-cpu-load + onnxruntime-genai-generation",
        inference_modality=CandidateModality.LLM,
        optimization_choices=(
            OptimizationChoice(
                strategy="mobius-olive",
                precision="int4",
                task_profile="llm-cpu-int4",
                skip_olive=False,
                default=True,
            ),
        ),
        artifact_cache_prefix="granite-3.3-2b",
        model_name_prefix="granite-3.3-2b-onboarding",
        success_message="Verified Granite Foundry Local build and inference succeeded.",
    ),
)

DEFAULT_RECIPE_REGISTRY = RecipeRegistry(DEFAULT_MODEL_RECIPES)
