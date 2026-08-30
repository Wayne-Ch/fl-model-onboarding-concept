from __future__ import annotations

from fl_model_onboarding.hf_policy import config_requires_remote_code


def test_config_requires_remote_code_on_auto_map() -> None:
    assert config_requires_remote_code({"auto_map": {"AutoModel": "x.y.Model"}})


def test_config_requires_remote_code_on_trust_remote_code_flag() -> None:
    assert config_requires_remote_code({"trust_remote_code": True})


def test_config_without_remote_code_is_allowed() -> None:
    assert not config_requires_remote_code({"model_type": "llama"})
