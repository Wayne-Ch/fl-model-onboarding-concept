from __future__ import annotations


def config_requires_remote_code(config: dict[str, object] | None) -> bool:
    if not isinstance(config, dict):
        return False
    auto_map = config.get("auto_map")
    if isinstance(auto_map, dict) and len(auto_map) > 0:
        return True
    if isinstance(auto_map, list) and len(auto_map) > 0:
        return True
    if isinstance(auto_map, str) and auto_map.strip():
        return True
    trust_remote_code = config.get("trust_remote_code")
    if isinstance(trust_remote_code, bool) and trust_remote_code:
        return True
    return False
