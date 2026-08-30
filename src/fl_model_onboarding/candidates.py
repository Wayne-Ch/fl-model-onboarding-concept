from __future__ import annotations

from .contracts import CandidateModality, ModelCandidate


PHASE0_CANDIDATES: dict[str, ModelCandidate] = {
    "smollm2-1.7b-instruct": ModelCandidate(
        key="smollm2-1.7b-instruct",
        huggingface_model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        modality=CandidateModality.LLM,
        recommended_mobius_dtype="f16",
        recommended_olive_precision="int4",
        notes="Phase-0 LLM candidate.",
    ),
    "distil-medium-en": ModelCandidate(
        key="distil-medium-en",
        huggingface_model_id="distil-whisper/distil-medium.en",
        modality=CandidateModality.ASR,
        recommended_mobius_dtype=None,
        recommended_olive_precision="fp16",
        notes="Phase-0 ASR candidate.",
    ),
}


def resolve_candidate(key_or_model_id: str) -> ModelCandidate:
    normalized = key_or_model_id.strip().lower()
    if normalized in PHASE0_CANDIDATES:
        return PHASE0_CANDIDATES[normalized]
    for candidate in PHASE0_CANDIDATES.values():
        if candidate.huggingface_model_id.lower() == normalized:
            return candidate
    known = ", ".join(sorted(PHASE0_CANDIDATES.keys()))
    raise KeyError(f"Unknown candidate '{key_or_model_id}'. Known keys: {known}")
