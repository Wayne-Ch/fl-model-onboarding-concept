from __future__ import annotations

from dataclasses import replace

from fl_model_onboarding.contracts import CandidateModality
from fl_model_onboarding.recipes import (
    DEFAULT_RECIPE_REGISTRY,
    DEFAULT_MODEL_RECIPES,
    GRANITE_MODEL_ID,
    SMOLLM2_MODEL_ID,
    RecipeRegistry,
    RecipeStatus,
)


def test_smollm2_recipe_is_verified_and_buildable() -> None:
    resolution = DEFAULT_RECIPE_REGISTRY.resolve(
        model_id=SMOLLM2_MODEL_ID,
        modality=CandidateModality.LLM,
        task_profile="llm-cpu-int4",
        allow_experimental=False,
    )
    assert resolution.recipe is not None
    assert resolution.status == RecipeStatus.VERIFIED.value
    assert resolution.buildable is True


def test_granite_recipe_is_verified_with_pinned_revision() -> None:
    resolved = DEFAULT_RECIPE_REGISTRY.resolve(
        model_id=GRANITE_MODEL_ID,
        modality=CandidateModality.LLM,
        task_profile="llm-cpu-int4",
        allow_experimental=False,
    )
    assert resolved.recipe is not None
    assert resolved.status == RecipeStatus.VERIFIED.value
    assert resolved.buildable is True
    assert resolved.recipe.verified_revision is not None


def test_registry_opt_in_logic_blocks_experimental_without_flag() -> None:
    granite = next(recipe for recipe in DEFAULT_MODEL_RECIPES if recipe.id == "granite-3.3-2b-cpu-int4")
    experimental = replace(
        granite,
        status=RecipeStatus.EXPERIMENTAL,
        status_reason="Recipe requires explicit experimental opt-in.",
        verified_revision=None,
    )
    experimental_registry = RecipeRegistry(
        tuple(
            experimental if recipe.id == experimental.id else recipe
            for recipe in DEFAULT_MODEL_RECIPES
        )
    )
    blocked = experimental_registry.resolve(
        model_id=GRANITE_MODEL_ID,
        modality=CandidateModality.LLM,
        task_profile="llm-cpu-int4",
        allow_experimental=False,
    )
    assert blocked.buildable is False
    assert blocked.requires_experimental_opt_in is True


def test_unknown_model_has_no_recipe_and_is_blocked() -> None:
    resolution = DEFAULT_RECIPE_REGISTRY.resolve(
        model_id="owner/unregistered-model",
        modality=CandidateModality.LLM,
        task_profile="llm-cpu-int4",
        allow_experimental=False,
    )
    assert resolution.recipe is None
    assert resolution.status == "unregistered"
    assert resolution.buildable is False
