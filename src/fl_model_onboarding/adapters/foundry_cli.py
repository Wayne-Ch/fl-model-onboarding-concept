from __future__ import annotations

import json
import re
from pathlib import Path

from .interfaces import CommandSpec
from ..contracts import CatalogMatchAssessment, MatchConfidence
from ..subprocess_runner import SafeSubprocessRunner


class FoundryCliCatalogAdapter:
    def __init__(self, runner: SafeSubprocessRunner | None = None) -> None:
        self._runner = runner or SafeSubprocessRunner()

    def list_matches(self, search_query: str) -> tuple[CatalogMatchAssessment, ...]:
        model_payload = self._run_list(search_query, include_variants=False)
        variant_payload = self._run_list(search_query, include_variants=True)
        matches: list[CatalogMatchAssessment] = []
        matches.extend(self._parse_models_payload(search_query, model_payload))
        matches.extend(self._parse_variants_payload(search_query, variant_payload))

        deduped: dict[str, CatalogMatchAssessment] = {}
        for match in matches:
            deduped.setdefault(f"{match.source_schema}:{match.model_or_variant_id}", match)
        return tuple(deduped.values())

    def cache_location(self) -> Path:
        spec = CommandSpec(argv=("foundry", "cache", "location", "-o", "json"), timeout_seconds=30)
        result = self._runner.run(spec)
        if not result.ok:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        payload = json.loads(result.stdout or "{}")
        path = payload.get("path")
        if not path:
            raise RuntimeError("Foundry cache location response did not include path.")
        return Path(path)

    def model_info(self, model_ref: str) -> dict[str, object]:
        spec = CommandSpec(argv=("foundry", "model", "info", model_ref, "-o", "json"), timeout_seconds=60)
        result = self._runner.run(spec)
        if not result.ok:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        payload = json.loads(result.stdout or "{}")
        model_payload = payload.get("model")
        if not isinstance(model_payload, dict):
            raise RuntimeError("Foundry model info response did not contain top-level 'model' object.")
        return model_payload

    def status(self) -> dict[str, object]:
        spec = CommandSpec(argv=("foundry", "status", "-o", "json"), timeout_seconds=60)
        result = self._runner.run(spec)
        if not result.ok:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return json.loads(result.stdout or "{}")

    def _run_list(self, search_query: str, include_variants: bool) -> dict[str, object]:
        argv = ["foundry", "model", "list", "--search", search_query, "-o", "json"]
        if include_variants:
            argv.append("--variants")
        result = self._runner.run(CommandSpec(argv=tuple(argv), timeout_seconds=120))
        if not result.ok:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return json.loads(result.stdout or "{}")

    def _parse_models_payload(
        self,
        search_query: str,
        payload: dict[str, object],
    ) -> list[CatalogMatchAssessment]:
        rows = payload.get("models", [])
        if not isinstance(rows, list):
            return []
        out: list[CatalogMatchAssessment] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            alias = str(row.get("alias", ""))
            model_id = str(row.get("id", ""))
            confidence, reason = _assess_name_match(search_query, alias, model_id)
            out.append(
                CatalogMatchAssessment(
                    alias=alias,
                    model_or_variant_id=model_id,
                    source_schema="models",
                    confidence=confidence,
                    reason=reason,
                    cached=_coerce_bool(row.get("cached")),
                    model_type=_optional_str(row.get("type")),
                )
            )
        return out

    def _parse_variants_payload(
        self,
        search_query: str,
        payload: dict[str, object],
    ) -> list[CatalogMatchAssessment]:
        rows = payload.get("variants", [])
        if not isinstance(rows, list):
            return []
        out: list[CatalogMatchAssessment] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            alias = str(row.get("alias", ""))
            variant_id = str(row.get("variantId", ""))
            confidence, reason = _assess_name_match(search_query, alias, variant_id)
            out.append(
                CatalogMatchAssessment(
                    alias=alias,
                    model_or_variant_id=variant_id,
                    source_schema="variants",
                    confidence=confidence,
                    reason=reason,
                    cached=_coerce_bool(row.get("cached")),
                    model_type=_optional_str(row.get("type")),
                )
            )
        return out


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _assess_name_match(search_query: str, alias: str, identifier: str) -> tuple[MatchConfidence, str]:
    normalized_query = _normalize_name(search_query)
    normalized_alias = _normalize_name(alias)
    normalized_identifier = _normalize_name(identifier)
    has_direct = normalized_query != "" and (
        normalized_query in normalized_alias or normalized_query in normalized_identifier
    )
    if has_direct:
        return (
            MatchConfidence.MEDIUM,
            "Likely alias match by name similarity only; Foundry catalog does not provide authoritative Hugging Face mapping.",
        )
    return (
        MatchConfidence.LOW,
        "Weak alias match (search hit only); compatibility requires Mobius/Olive/runtime validation.",
    )


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
