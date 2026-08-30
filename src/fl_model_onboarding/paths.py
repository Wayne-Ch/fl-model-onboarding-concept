from __future__ import annotations

from pathlib import Path


class PathContainmentError(ValueError):
    pass


def ensure_within(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PathContainmentError(
            f"Path '{candidate_resolved}' is outside allowed root '{root_resolved}'"
        ) from exc
    return candidate_resolved


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
