from __future__ import annotations

from fl_model_onboarding.hf_safetensors import coerce_total_bytes, derive_parameter_count


def test_parameter_count_prefers_dtype_map() -> None:
    payload = {
        "total": 999999,
        "parameters": {
            "F16": 10,
            "I8": 5,
            "extra": "3",
        },
    }
    assert derive_parameter_count(payload) == 18
    assert coerce_total_bytes(payload) == 999999


def test_parameter_count_none_without_map() -> None:
    assert derive_parameter_count({"total": 123}) is None
