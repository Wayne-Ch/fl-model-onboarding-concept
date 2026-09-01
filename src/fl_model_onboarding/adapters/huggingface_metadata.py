from __future__ import annotations

from datetime import datetime
from typing import Any

from .interfaces import HuggingFaceMetadata, HuggingFaceSearchResult
from ..hf_safetensors import coerce_total_bytes, derive_parameter_count, to_dict


class HuggingFaceMetadataAdapter:
    def search_models(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
    ) -> tuple[HuggingFaceSearchResult, ...]:
        from huggingface_hub import HfApi

        api = HfApi()
        sparse_models = api.list_models(search=query, sort=sort, limit=limit)
        results: list[HuggingFaceSearchResult] = []
        for model in sparse_models:
            results.append(
                HuggingFaceSearchResult(
                    model_id=model.id,
                    downloads=getattr(model, "downloads", None),
                    likes=getattr(model, "likes", None),
                    last_modified=_normalize_datetime(
                        getattr(model, "lastModified", None) or getattr(model, "last_modified", None)
                    ),
                )
            )
        return tuple(results)

    def get_metadata(
        self,
        model_id: str,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> HuggingFaceMetadata:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id=model_id, revision=revision, files_metadata=files_metadata)
        siblings = getattr(info, "siblings", None)
        safetensors = to_dict(getattr(info, "safetensors", None))
        return HuggingFaceMetadata(
            model_id=info.id,
            revision=revision or getattr(info, "sha", None),
            sha=getattr(info, "sha", None),
            is_private=getattr(info, "private", None),
            is_gated=getattr(info, "gated", None),
            last_modified=_normalize_datetime(
                getattr(info, "lastModified", None) or getattr(info, "last_modified", None)
            ),
            config=to_dict(getattr(info, "config", None)),
            safetensors_total_bytes=coerce_total_bytes(safetensors),
            safetensors_parameter_count=derive_parameter_count(safetensors),
            card_data=to_dict(getattr(info, "cardData", None) or getattr(info, "card_data", None)),
            sibling_count=len(siblings) if siblings is not None else None,
        )


def _normalize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
