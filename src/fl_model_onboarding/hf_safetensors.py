from __future__ import annotations

from typing import Any


def derive_parameter_count(safetensors_payload: dict[str, object] | None) -> int | None:
    """Prefer explicit dtype parameter map over ambiguous `total` metrics."""
    if not isinstance(safetensors_payload, dict):
        return None
    parameters = safetensors_payload.get("parameters")
    if not isinstance(parameters, dict):
        return None
    total = 0
    for value in parameters.values():
        if isinstance(value, int):
            total += value
        elif isinstance(value, str) and value.isdigit():
            total += int(value)
    return total or None


def coerce_total_bytes(safetensors_payload: dict[str, object] | None) -> int | None:
    if not isinstance(safetensors_payload, dict):
        return None
    total = safetensors_payload.get("total")
    if isinstance(total, int):
        return total
    if isinstance(total, str) and total.isdigit():
        return int(total)
    return None


def to_dict(value: Any) -> dict[str, object] | None:
    if isinstance(value, dict):
        return dict(value)
    return None
