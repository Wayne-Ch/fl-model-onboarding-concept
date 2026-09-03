from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import uuid

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping, Sequence

from .adapters.interfaces import CommandResult, CommandSpec, ProcessRunner
from .adapters.huggingface_acquisition import HuggingFaceAcquisitionAdapter
from .adapters.interfaces import HuggingFaceAcquisitionClient
from .architecture_capabilities import CapabilityStatus, ResolutionOutcome
from .contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    BuildRequest,
    CandidateModality,
    FailureClassification,
    FailureInfo,
    GeneratedRecipeAttemptBinding,
    JobState,
    ProductionInvocationEvidence,
    ToolInvocationEvidence,
    ToolInvocationTerminalStage,
    ValidationResult,
    ValidationStatus,
)
from .recipe_attempt_store import (
    AttemptState,
    CandidateInvocationCounters,
    GeneratedRecipeRecord,
    RecipeAttempt,
    RecipeAttemptStore,
)
from .recipe_compiler import GeneratedRecipeCompileError, validate_generated_recipe_payload
from .state_machine import fail_job, transition
from .recipes import (
    AncillaryFileRule,
    DEFAULT_RECIPE_REGISTRY,
    MobiusRecipeArgs,
    OliveRecipeArgs,
    OptimizationChoice,
    SMOLLM2_MODEL_ID as VERIFIED_SMOLLM2_MODEL_ID,
    SMOLLM2_VERIFIED_REVISION,
    ModelRecipe,
    RecipeRegistry,
    RecipeStatus,
)

SMOLLM2_MODEL_ID = VERIFIED_SMOLLM2_MODEL_ID
SMOLLM2_REVISION = SMOLLM2_VERIFIED_REVISION
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_QUANTIZED_OUTPUT_RE = re.compile(r"^(?P<base>.+)_Q(?P<bits>\d+)$", re.IGNORECASE)
_INDEXED_DECODER_OUTPUT_RE = re.compile(r"%(?:0?\d*)d")
_MAX_COMMAND_FAILURE_DETAIL_CHARS = 1200
_BATCH_INFERENCE_TIMEOUT_GRACE_SECONDS = 15


def _result_payload(result: CommandResult) -> dict[str, object]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {"ok": False, "error": result.stderr or "Command returned no JSON result."}


def _compact_failure_detail(value: str) -> str:
    compact = " ".join(value.split())
    if not compact:
        return "process returned no diagnostic output."
    traceback_index = compact.lower().rfind("traceback")
    if traceback_index >= 0:
        compact = compact[traceback_index:]
    if len(compact) <= _MAX_COMMAND_FAILURE_DETAIL_CHARS:
        return compact
    return "..." + compact[-(_MAX_COMMAND_FAILURE_DETAIL_CHARS - 3) :].lstrip()


def production_invocation_evidence_to_candidate_counters(
    evidence: ProductionInvocationEvidence,
) -> CandidateInvocationCounters:
    """Convert real per-job Mobius/Olive invocation evidence into the store's
    nullable :class:`~fl_model_onboarding.recipe_attempt_store.CandidateInvocationCounters`
    shape.

    A tool's count is only ever reported once that tool's launch was actually
    attempted for this job; an untouched (not-run) tool stays ``None``, never
    ``0``, matching ``CandidateInvocationCounters``'s documented null
    semantics -- a validation-time rejection that prevented a launch must
    never be reported as if a run of zero invocations occurred. This function
    only builds the value; persisting it via
    ``RecipeAttemptStore.finalize_candidate_attempt_evidence`` is orchestration
    wiring left to a later slice.
    """
    mobius_count = evidence.mobius.invocation_count if evidence.mobius.invocation_count > 0 else None
    olive_count = evidence.olive.invocation_count if evidence.olive.invocation_count > 0 else None
    total_count = None
    if mobius_count is not None or olive_count is not None:
        total_count = (mobius_count or 0) + (olive_count or 0)
    wall_parts = [
        value
        for value in (evidence.mobius.wall_seconds, evidence.olive.wall_seconds)
        if value is not None
    ]
    wall_clock_seconds = sum(wall_parts) if wall_parts else None
    return CandidateInvocationCounters(
        mobius_build_invocation_count=mobius_count,
        olive_optimize_invocation_count=olive_count,
        total_invocation_count=total_count,
        wall_clock_seconds=wall_clock_seconds,
        estimated_cost_usd=None,
    )


# --- Slice 3A2: safe immutable reuse of a successful pre-Olive Mobius -------
#
# The primitives below let an approved trusted-candidate ("fallback") build
# skip re-running Mobius by reusing an already-captured, revalidated copy of
# a successful default candidate's pre-Olive Mobius output. Nothing here
# wires this into `local_service.py`, the recipe-attempt store, or any
# API/route: `capture_pre_olive_artifact`/`validate_pre_olive_reuse`/
# `materialize_pre_olive_copy` are standalone, runner-level functions a later
# slice (3B) can call directly, and `ProductionBuildStageRunner
# .run_fallback_with_pre_olive_reuse` is an additive entry point that never
# runs unless a caller explicitly invokes it. The existing `run()`/`_run()`
# one-shot path is untouched in behavior: it now delegates its post-Mobius
# half to the shared `_run_from_olive` helper, but that is a pure refactor
# with no observable difference for legacy callers.

_MANIFEST_HASH_CHUNK_BYTES = 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class PreOliveReuseError(RuntimeError):
    """Raised when Slice 3A2 pre-Olive artifact reuse must fail closed:
    missing/tampered content, an out-of-authorized-root path, a symlink or
    Windows reparse point, a manifest mismatch, source/destination overlap,
    a generation-identity mismatch, or a copy failure. Every raise site sets
    ``classification`` to the closest matching
    :class:`~fl_model_onboarding.contracts.FailureClassification` so callers
    (including ``ProductionBuildStageRunner``) can classify the failure the
    same way any other pre-launch validation failure is classified."""

    def __init__(
        self,
        message: str,
        *,
        classification: FailureClassification = FailureClassification.PATH_CONTAINMENT,
    ) -> None:
        super().__init__(message)
        self.classification = classification


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PreOliveGenerationIdentity:
    """The trusted model/revision/device/precision/compiler/capability/
    toolchain/profile identity that must match byte-for-byte between a
    default candidate's already-captured pre-Olive artifact and any fallback
    candidate attempting to reuse it -- proving both were compiled for the
    exact same generation. Field names deliberately mirror
    :class:`~fl_model_onboarding.recipe_attempt_store.RecipeReuseQuery` /
    ``GeneratedRecipeRecord``, but this is a separate, narrower type so this
    runtime-only runner primitive has no coupling to recipe-attempt store
    persistence.
    """

    model_id: str
    revision_sha: str
    requested_device: str
    requested_precision: str
    compiler_version: str
    capability_fingerprint: str
    toolchain_fingerprint: str
    profile_fingerprint: str


def pre_olive_generation_identity_from_generated_record(
    record: GeneratedRecipeRecord,
) -> PreOliveGenerationIdentity:
    """Convenience constructor for the common Slice 3B shape: both the
    default and fallback candidate are persisted ``GeneratedRecipeRecord``
    rows, which already carry every field this identity needs."""
    return PreOliveGenerationIdentity(
        model_id=record.model_id,
        revision_sha=record.revision_sha,
        requested_device=record.requested_device,
        requested_precision=record.requested_precision,
        compiler_version=record.compiler_version,
        capability_fingerprint=record.capability_fingerprint,
        toolchain_fingerprint=record.toolchain_fingerprint,
        profile_fingerprint=record.profile_fingerprint,
    )


def compute_mobius_args_fingerprint(mobius: MobiusRecipeArgs) -> str:
    """A deterministic fingerprint of exactly the Mobius arguments
    (`ep`/`runtime`/`dtype`/`task`) that `ProductionBuildStageRunner._run`
    threads into the real Mobius `build` command line -- the narrow, explicit
    "Mobius args are identical" check called for by Slice 3A2, independent of
    (and in addition to) the broader `PreOliveGenerationIdentity` match."""
    payload = {
        "ep": mobius.ep,
        "runtime": mobius.runtime,
        "dtype": mobius.dtype,
        "task": mobius.task,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreOliveManifestEntry:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PreOliveManifest:
    """A deterministic relative-path/file-size/content-hash manifest of every
    required file under a captured pre-Olive Mobius artifact directory.
    ``entries`` is always sorted by ``relative_path`` so two independently
    built manifests of byte-identical content always compare equal and
    produce the same ``manifest_hash``."""

    entries: tuple[PreOliveManifestEntry, ...]
    total_bytes: int
    file_count: int
    manifest_hash: str


@dataclass(frozen=True)
class PreOliveArtifactDescriptor:
    """Immutable evidence of one successful default candidate's pre-Olive
    Mobius output, captured once (see `capture_pre_olive_artifact`) and
    safe to reuse for an approved fallback candidate's Olive run.

    ``mobius_source_dir`` is runtime-only: it is a real filesystem path on
    this machine and must never be copied into persisted/sanitized evidence.
    ``logical_ref`` (the manifest's content hash) is the stable, path-free
    reference a caller should persist/log instead; `sanitized_payload()`
    only ever exposes that shape.
    """

    generation_identity: PreOliveGenerationIdentity
    mobius_args_fingerprint: str
    manifest: PreOliveManifest
    captured_utc: datetime
    mobius_source_dir: Path
    source_attempt_id: str | None = None
    source_candidate_id: str | None = None

    @property
    def logical_ref(self) -> str:
        return self.manifest.manifest_hash

    def sanitized_payload(self) -> dict[str, object]:
        """A persistable/loggable shape: identity fingerprints, the
        Mobius-args fingerprint, manifest hash/size/count, and the logical
        ref -- deliberately excluding `mobius_source_dir` (a raw absolute
        filesystem path)."""
        return {
            "generation_identity": {
                "model_id": self.generation_identity.model_id,
                "revision_sha": self.generation_identity.revision_sha,
                "requested_device": self.generation_identity.requested_device,
                "requested_precision": self.generation_identity.requested_precision,
                "compiler_version": self.generation_identity.compiler_version,
                "capability_fingerprint": self.generation_identity.capability_fingerprint,
                "toolchain_fingerprint": self.generation_identity.toolchain_fingerprint,
                "profile_fingerprint": self.generation_identity.profile_fingerprint,
            },
            "mobius_args_fingerprint": self.mobius_args_fingerprint,
            "manifest_hash": self.manifest.manifest_hash,
            "manifest_total_bytes": self.manifest.total_bytes,
            "manifest_file_count": self.manifest.file_count,
            "logical_ref": self.logical_ref,
            "captured_utc": self.captured_utc.isoformat(),
            "source_attempt_id": self.source_attempt_id,
            "source_candidate_id": self.source_candidate_id,
        }


def _path_is_link_or_reparse_point(path: Path) -> bool:
    """True for a symlink (any platform) or a Windows reparse point
    (junction) at `path`. Fails closed (returns True) if the path cannot be
    safely `lstat`-ed at all, since we would otherwise have to guess whether
    it is safe to traverse/copy."""
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    if sys.platform == "win32":
        try:
            attrs = path.lstat().st_file_attributes  # type: ignore[attr-defined]
        except (OSError, AttributeError):
            return True
        if attrs & _FILE_ATTRIBUTE_REPARSE_POINT:
            return True
    return False


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_MANIFEST_HASH_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _assert_no_case_insensitive_relative_path_collisions(
    entries: Sequence[PreOliveManifestEntry],
) -> None:
    """3A2 hardening follow-up: reject a manifest that contains two distinct
    relative paths differing only by case (for example ``Model.onnx`` vs
    ``model.onnx``). Such a manifest can be built faithfully from a
    case-sensitive source filesystem, but silently collapses -- one file
    silently overwriting the other -- the moment it is materialized onto a
    case-insensitive destination filesystem (Windows/NTFS by default), which
    is exactly what every real ``materialize_pre_olive_copy`` destination is.
    Fails closed before any copy is attempted rather than allowing a silent
    data-loss reuse."""
    seen: dict[str, str] = {}
    for entry in entries:
        lowered = entry.relative_path.lower()
        collision = seen.get(lowered)
        if collision is not None and collision != entry.relative_path:
            raise PreOliveReuseError(
                "Refusing to reuse a pre-Olive artifact whose manifest contains "
                f"case-insensitive relative-path collision: '{collision}' vs "
                f"'{entry.relative_path}'; this would silently collide on a "
                "case-insensitive destination filesystem.",
            )
        seen[lowered] = entry.relative_path


def _assert_no_link_in_relative_path_chain(base: Path, relative_path: str) -> None:
    """3A2 hardening follow-up: verify every path component from `base` down
    to and including the leaf named by `relative_path` -- not merely the
    fully-composed leaf path itself -- is not a symlink/Windows reparse
    point, checked fresh immediately before this specific file is copied.

    `_path_is_link_or_reparse_point` on the fully-composed leaf path alone
    only inspects the final path component's own attributes; on Windows,
    `lstat` still transparently follows an *intermediate* directory reparse
    point (a junction) while resolving the rest of the path. Without this
    check, an intermediate ancestor directory swapped for a junction in the
    window between the initial manifest revalidation and this file's copy
    (a Windows junction race) would go undetected even though the leaf name
    itself is not a reparse point. Checking every ancestor component here,
    fresh per file, closes that race."""
    current = base
    for part in Path(relative_path).parts:
        current = current / part
        if _path_is_link_or_reparse_point(current):
            raise PreOliveReuseError(
                f"Refusing to use a path with a symlink/reparse point ancestor: {current}",
            )


def _build_directory_manifest(root: Path) -> PreOliveManifest:
    """Walk `root` and build a deterministic manifest of every regular file
    under it, hashing each file with a bounded-memory streaming read.
    Fails closed (raises `PreOliveReuseError`) on a missing/non-directory
    root, any symlink/reparse point encountered anywhere in the tree
    (rejected rather than followed), any non-file/non-directory entry, or a
    case-insensitive relative-path collision between two distinct entries."""
    if _path_is_link_or_reparse_point(root):
        raise PreOliveReuseError(
            f"Refusing to traverse a symlink/reparse point manifest root: {root}",
        )
    if not root.is_dir():
        raise PreOliveReuseError(f"Manifest root is missing or not a directory: {root}")
    entries: list[PreOliveManifestEntry] = []
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        for child in current.iterdir():
            if _path_is_link_or_reparse_point(child):
                raise PreOliveReuseError(
                    f"Refusing to traverse a symlink/reparse point: {child}",
                )
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                size, digest = _hash_file(child)
                entries.append(
                    PreOliveManifestEntry(
                        relative_path=child.relative_to(root).as_posix(),
                        size_bytes=size,
                        sha256=digest,
                    )
                )
            else:
                raise PreOliveReuseError(f"Unsupported filesystem entry (not file/directory): {child}")
    _assert_no_case_insensitive_relative_path_collisions(entries)
    entries.sort(key=lambda entry: entry.relative_path)
    total_bytes = sum(entry.size_bytes for entry in entries)
    manifest_payload = [
        {
            "relative_path": entry.relative_path,
            "size_bytes": entry.size_bytes,
            "sha256": entry.sha256,
        }
        for entry in entries
    ]
    manifest_hash = hashlib.sha256(
        _canonical_json({"entries": manifest_payload}).encode("utf-8")
    ).hexdigest()
    return PreOliveManifest(
        entries=tuple(entries),
        total_bytes=total_bytes,
        file_count=len(entries),
        manifest_hash=manifest_hash,
    )


def _assert_within_authorized_roots(path: Path, *, authorized_roots: Sequence[Path]) -> None:
    resolved = path.resolve()
    for root in authorized_roots:
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise PreOliveReuseError(
        f"Path '{path}' escapes every authorized root; refusing to reuse a pre-Olive artifact "
        "outside its owning workspace.",
    )


def _assert_no_path_overlap(source_dir: Path, destination_dir: Path) -> None:
    source_resolved = source_dir.resolve()
    destination_resolved = destination_dir.resolve()
    if source_resolved == destination_resolved:
        raise PreOliveReuseError(
            "Pre-Olive reuse destination is identical to the captured source; refusing to reuse in place.",
        )
    try:
        destination_resolved.relative_to(source_resolved)
        overlap = True
    except ValueError:
        overlap = False
    if not overlap:
        try:
            source_resolved.relative_to(destination_resolved)
            overlap = True
        except ValueError:
            overlap = False
    if overlap:
        raise PreOliveReuseError(
            "Pre-Olive reuse destination overlaps the captured source directory; refusing to reuse "
            "in an overlapping path.",
        )


def capture_pre_olive_artifact(
    *,
    mobius_source_dir: Path,
    authorized_root: Path,
    generation_identity: PreOliveGenerationIdentity,
    mobius_args: MobiusRecipeArgs,
    source_attempt_id: str | None = None,
    source_candidate_id: str | None = None,
) -> PreOliveArtifactDescriptor:
    """Capture an immutable `PreOliveArtifactDescriptor` for one successful
    default candidate's pre-Olive Mobius output directory.

    `mobius_source_dir` must already exist, contain at least one file, and
    resolve inside `authorized_root` (the owning job's workspace root);
    anything else -- missing directory, empty directory, symlink/reparse
    point anywhere in the tree, or an escape from the authorized root --
    fails closed. Nothing here mutates `mobius_source_dir`.
    """
    _assert_within_authorized_roots(mobius_source_dir, authorized_roots=(authorized_root,))
    manifest = _build_directory_manifest(mobius_source_dir)
    if manifest.file_count == 0:
        raise PreOliveReuseError(
            f"Refusing to capture an empty pre-Olive Mobius artifact directory: {mobius_source_dir}",
        )
    return PreOliveArtifactDescriptor(
        generation_identity=generation_identity,
        mobius_args_fingerprint=compute_mobius_args_fingerprint(mobius_args),
        manifest=manifest,
        captured_utc=datetime.now(timezone.utc),
        mobius_source_dir=mobius_source_dir,
        source_attempt_id=source_attempt_id,
        source_candidate_id=source_candidate_id,
    )


def validate_pre_olive_reuse(
    descriptor: PreOliveArtifactDescriptor,
    *,
    candidate_identity: PreOliveGenerationIdentity,
    candidate_mobius_args: MobiusRecipeArgs,
) -> None:
    """Fail closed unless the fallback candidate's own generation identity
    and Mobius arguments match the captured descriptor byte-for-byte. This
    only compares in-memory identity/arguments; it does not touch the
    filesystem (see `revalidate_pre_olive_source` for the immediately-before-
    reuse content check)."""
    if candidate_identity != descriptor.generation_identity:
        raise PreOliveReuseError(
            "Fallback candidate generation identity does not match the captured pre-Olive "
            "artifact; refusing to reuse Mobius output across a different model/revision/device/"
            "precision/compiler/capability/toolchain/profile generation.",
            classification=FailureClassification.COMPATIBILITY,
        )
    candidate_fingerprint = compute_mobius_args_fingerprint(candidate_mobius_args)
    if candidate_fingerprint != descriptor.mobius_args_fingerprint:
        raise PreOliveReuseError(
            "Fallback candidate Mobius arguments do not match the captured pre-Olive artifact's "
            "Mobius arguments; only an approved Olive block_size may differ for a trusted reuse.",
            classification=FailureClassification.COMPATIBILITY,
        )


def revalidate_pre_olive_source(descriptor: PreOliveArtifactDescriptor) -> None:
    """Re-walk and re-hash the descriptor's captured source directory and
    fail closed if anything has changed since capture: missing directory,
    missing/added/renamed files, changed content, or a symlink/reparse point
    now present anywhere in the tree. Callers must call this immediately
    before reuse (see `materialize_pre_olive_copy`, which always calls it
    first)."""
    source_dir = descriptor.mobius_source_dir
    if not source_dir.exists():
        raise PreOliveReuseError(
            f"Captured pre-Olive source no longer exists: {source_dir}",
        )
    manifest = _build_directory_manifest(source_dir)
    if manifest.manifest_hash != descriptor.manifest.manifest_hash:
        raise PreOliveReuseError(
            "Captured pre-Olive source content changed since capture (manifest hash mismatch); "
            "refusing reuse of a tampered or drifted artifact.",
        )


def _stream_copy_file(source: Path, destination: Path) -> None:
    with source.open("rb") as read_handle, destination.open("wb") as write_handle:
        while True:
            chunk = read_handle.read(_MANIFEST_HASH_CHUNK_BYTES)
            if not chunk:
                break
            write_handle.write(chunk)


def materialize_pre_olive_copy(
    descriptor: PreOliveArtifactDescriptor,
    *,
    destination_dir: Path,
    authorized_roots: Sequence[Path],
    cancellation_event: Event | None = None,
) -> Path:
    """Revalidate the descriptor's captured source, then materialize a full,
    independent byte-for-byte copy of it into `destination_dir`.

    `destination_dir` must not already exist and must resolve inside one of
    `authorized_roots`; it must not equal or overlap the descriptor's source
    directory. Every copied file is streamed (never hardlinked, symlinked,
    or junctioned) and the destination is re-manifested and compared against
    the descriptor's manifest hash after every file is copied, so a silent
    partial or corrupted copy can never be mistaken for a faithful one. Any
    failure -- including an observed cancellation request -- removes exactly
    the `destination_dir` this call created (never a broader/unresolved
    path) before re-raising, leaving the source untouched and the failure
    auditable.
    """
    revalidate_pre_olive_source(descriptor)
    source_dir = descriptor.mobius_source_dir
    _assert_within_authorized_roots(destination_dir, authorized_roots=authorized_roots)
    _assert_no_path_overlap(source_dir, destination_dir)
    if destination_dir.exists():
        raise PreOliveReuseError(f"Pre-Olive reuse destination already exists: {destination_dir}")

    created_root = False
    try:
        destination_dir.mkdir(parents=True, exist_ok=False)
        created_root = True
        for entry in descriptor.manifest.entries:
            if cancellation_event is not None and cancellation_event.is_set():
                raise PreOliveReuseError(
                    "Pre-Olive artifact copy cancelled before completion.",
                    classification=FailureClassification.CANCELLED,
                )
            # 3A2 hardening follow-up: check every ancestor directory component of
            # this entry (not just the fully-composed leaf path) for a symlink/
            # reparse point, fresh right before this specific file is copied. See
            # `_assert_no_link_in_relative_path_chain` for why the leaf-only check
            # this replaced could miss an intermediate junction race.
            _assert_no_link_in_relative_path_chain(source_dir, entry.relative_path)
            source_file = source_dir / entry.relative_path
            destination_file = destination_dir / entry.relative_path
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            _stream_copy_file(source_file, destination_file)
        destination_manifest = _build_directory_manifest(destination_dir)
        if destination_manifest.manifest_hash != descriptor.manifest.manifest_hash:
            raise PreOliveReuseError(
                "Destination manifest hash mismatch after copy; refusing to reuse a corrupted copy.",
            )
    except Exception:
        if created_root and destination_dir.exists():
            shutil.rmtree(destination_dir)
        raise
    return destination_dir


def _resolve_staging_relative_path(
    *,
    staging_dir: Path,
    relative_path: str,
    field_name: str,
) -> Path:
    normalized = relative_path.strip().replace("\\", "/")
    if not normalized:
        raise RuntimeError(f"{field_name} must be a non-empty relative path.")
    if normalized.startswith("/") or Path(normalized).is_absolute():
        raise RuntimeError(f"{field_name} must remain relative to the staging package root.")
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise RuntimeError(f"{field_name} contains unsafe path components.")
    staging_root = staging_dir.resolve()
    candidate = (staging_root / Path(*parts)).resolve()
    try:
        candidate.relative_to(staging_root)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} escapes the staging package root.") from exc
    return candidate


def _decoder_output_map_from_config_payload(payload: Mapping[str, object]) -> dict[str, str] | None:
    model = payload.get("model")
    if model is None:
        return None
    if not isinstance(model, dict):
        raise RuntimeError("genai_config.json field 'model' must be an object when present.")
    decoder = model.get("decoder")
    if decoder is None:
        return None
    if not isinstance(decoder, dict):
        raise RuntimeError("genai_config.json field 'model.decoder' must be an object when present.")
    outputs = decoder.get("outputs")
    if outputs is None:
        return None
    if not isinstance(outputs, dict):
        raise RuntimeError("genai_config.json field 'model.decoder.outputs' must be an object.")
    mapped: dict[str, str] = {}
    for key, value in outputs.items():
        if not isinstance(value, str):
            raise RuntimeError(
                "genai_config.json field 'model.decoder.outputs' must map to string output names.",
            )
        mapped[str(key)] = value
    return mapped


def _load_onnx_graph_output_names(model_path: Path) -> tuple[str, ...]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX Python package is required for decoder output reconciliation.",
        ) from exc
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to read ONNX graph outputs from '{model_path.name}'.",
        ) from exc
    outputs = tuple(dict.fromkeys(str(row.name) for row in model.graph.output if str(row.name)))
    if not outputs:
        raise RuntimeError(f"ONNX model '{model_path.name}' has no graph outputs to validate.")
    return outputs


def _decoder_indexed_output_pattern(name: str) -> re.Pattern[str] | None:
    if _INDEXED_DECODER_OUTPUT_RE.search(name) is None:
        return None
    parts: list[str] = []
    cursor = 0
    for match in _INDEXED_DECODER_OUTPUT_RE.finditer(name):
        parts.append(re.escape(name[cursor : match.start()]))
        parts.append(r"\d+")
        cursor = match.end()
    parts.append(re.escape(name[cursor:]))
    return re.compile(r"^" + "".join(parts) + r"$")


def _build_decoder_output_reconciliation(
    *,
    graph_outputs: Sequence[str],
    decoder_outputs: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, object]]]:
    available = tuple(dict.fromkeys(str(name) for name in graph_outputs if str(name)))
    present: dict[str, str] = {}
    remapped: dict[str, str] = {}
    unresolved: dict[str, dict[str, object]] = {}
    for logical_name, physical_name in decoder_outputs.items():
        mapped_name = str(physical_name)
        if mapped_name in available:
            present[str(logical_name)] = mapped_name
            continue
        indexed_pattern = _decoder_indexed_output_pattern(mapped_name)
        if indexed_pattern is not None:
            indexed_matches = [output_name for output_name in available if indexed_pattern.fullmatch(output_name)]
            if indexed_matches:
                present[str(logical_name)] = mapped_name
                continue
        candidates = [
            output_name
            for output_name in available
            if (
                (match := _QUANTIZED_OUTPUT_RE.fullmatch(output_name)) is not None
                and match.group("base") == mapped_name
            )
        ]
        if len(candidates) == 1:
            remapped[str(logical_name)] = candidates[0]
            continue
        unresolved[str(logical_name)] = {
            "requested_output": mapped_name,
            "candidates": candidates,
            "reason": "no_match" if len(candidates) == 0 else "ambiguous_match",
        }
    return present, remapped, unresolved


def _reconcile_decoder_outputs_in_staging_package(
    *,
    staging_dir: Path,
    model_relative_path: str = "model.onnx",
    config_relative_path: str = "genai_config.json",
) -> dict[str, object]:
    config_path = _resolve_staging_relative_path(
        staging_dir=staging_dir,
        relative_path=config_relative_path,
        field_name="config_relative_path",
    )
    if not config_path.is_file():
        raise RuntimeError(f"Staging package is missing required file '{config_relative_path}'.")

    payload_raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise RuntimeError("genai_config.json must contain a JSON object.")

    decoder_outputs_before = _decoder_output_map_from_config_payload(payload_raw)
    if decoder_outputs_before is None:
        return {
            "status": "skipped",
            "reason": "decoder-output-mapping-missing",
            "decoder_outputs_before": {},
            "decoder_outputs_after": {},
            "remapped_outputs": {},
            "graph_outputs": [],
            "present_outputs": {},
        }
    if not decoder_outputs_before:
        return {
            "status": "skipped",
            "reason": "decoder-output-mapping-empty",
            "decoder_outputs_before": {},
            "decoder_outputs_after": {},
            "remapped_outputs": {},
            "graph_outputs": [],
            "present_outputs": {},
        }

    model_path = _resolve_staging_relative_path(
        staging_dir=staging_dir,
        relative_path=model_relative_path,
        field_name="model_relative_path",
    )
    if not model_path.is_file():
        raise RuntimeError(f"Staging package is missing required file '{model_relative_path}'.")

    graph_outputs = _load_onnx_graph_output_names(model_path)
    present_outputs, remapped_outputs, unresolved = _build_decoder_output_reconciliation(
        graph_outputs=graph_outputs,
        decoder_outputs=decoder_outputs_before,
    )
    decoder_outputs_after = dict(decoder_outputs_before)
    decoder_outputs_after.update(remapped_outputs)

    if unresolved:
        details = ", ".join(
            f"{key}=>{row['requested_output']} (candidates={row['candidates']})"
            for key, row in sorted(unresolved.items())
        )
        raise RuntimeError(
            "Decoder output reconciliation failed for staging package due to unresolved mappings: "
            + details,
        )

    if remapped_outputs:
        model_payload = payload_raw.get("model")
        if not isinstance(model_payload, dict):
            raise RuntimeError("genai_config.json is missing required object 'model'.")
        decoder_payload = model_payload.get("decoder")
        if not isinstance(decoder_payload, dict):
            raise RuntimeError("genai_config.json is missing required object 'model.decoder'.")
        outputs_payload = decoder_payload.get("outputs")
        if not isinstance(outputs_payload, dict):
            raise RuntimeError("genai_config.json is missing required object 'model.decoder.outputs'.")
        for logical_name, remapped_name in remapped_outputs.items():
            outputs_payload[logical_name] = remapped_name
        config_path.write_text(json.dumps(payload_raw, indent=2), encoding="utf-8")

    return {
        "status": "applied" if remapped_outputs else "verified",
        "reason": "ok",
        "graph_outputs": list(graph_outputs),
        "present_outputs": present_outputs,
        "decoder_outputs_before": decoder_outputs_before,
        "decoder_outputs_after": decoder_outputs_after,
        "remapped_outputs": remapped_outputs,
    }


@dataclass(frozen=True)
class RecipeExecutionPlan:
    recipe: ModelRecipe
    pinned_revision: str
    source: str


class RecipeExecutionResolutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        classification: FailureClassification = FailureClassification.INVALID_REQUEST,
    ) -> None:
        super().__init__(message)
        self.classification = classification


@dataclass(frozen=True)
class PinnedModelSource:
    model_id: str
    revision_sha: str
    snapshot_dir: Path


class RecipeExecutionResolver:
    def __init__(
        self,
        *,
        recipe_registry: RecipeRegistry = DEFAULT_RECIPE_REGISTRY,
        recipe_attempt_store: RecipeAttemptStore | None = None,
    ) -> None:
        self._recipe_registry = recipe_registry
        self._recipe_attempt_store = recipe_attempt_store

    def resolve(self, request: BuildRequest) -> RecipeExecutionPlan:
        generated_binding = request.generated_recipe_attempt
        if generated_binding is None:
            return self._resolve_static_recipe(request)
        return self._resolve_generated_recipe(request, generated_binding)

    def _resolve_static_recipe(self, request: BuildRequest) -> RecipeExecutionPlan:
        recipe_match = self._recipe_registry.resolve(
            model_id=request.candidate.huggingface_model_id,
            modality=request.candidate.modality,
            task_profile=request.task_profile,
            allow_experimental=True,
        )
        recipe = recipe_match.recipe
        if recipe is None:
            raise RecipeExecutionResolutionError(
                recipe_match.reason,
                classification=FailureClassification.NOT_VERIFIED,
            )
        if recipe.status != RecipeStatus.VERIFIED:
            raise RecipeExecutionResolutionError(
                f"Production execution is verified only for recipe status 'verified'; "
                f"received '{recipe.id}' ({recipe.status.value}).",
                classification=FailureClassification.NOT_VERIFIED,
            )
        if recipe.verified_revision is None:
            raise RecipeExecutionResolutionError(
                f"Verified recipe '{recipe.id}' is missing a pinned verified revision.",
                classification=FailureClassification.NOT_VERIFIED,
            )
        if request.hf_revision != recipe.verified_revision:
            raise RecipeExecutionResolutionError(
                f"Production execution requires pinned revision {recipe.verified_revision}; "
                f"received {request.hf_revision or 'none'}.",
                classification=FailureClassification.NOT_VERIFIED,
            )
        return RecipeExecutionPlan(
            recipe=recipe,
            pinned_revision=recipe.verified_revision,
            source="static_verified_recipe_registry",
        )

    def _resolve_generated_recipe(
        self,
        request: BuildRequest,
        binding: GeneratedRecipeAttemptBinding,
    ) -> RecipeExecutionPlan:
        if self._recipe_attempt_store is None:
            raise RecipeExecutionResolutionError(
                "Generated recipe execution requires a recipe-attempt store.",
            )
        attempt_id = binding.attempt_id.strip()
        fingerprint = binding.recipe_fingerprint.strip().lower()
        if not attempt_id:
            raise RecipeExecutionResolutionError("Generated recipe attempt id is missing.")
        if not _HEX64_RE.fullmatch(fingerprint):
            raise RecipeExecutionResolutionError(
                "Generated recipe fingerprint must be a lowercase 64-character hex value.",
            )
        if not binding.confirmed:
            raise RecipeExecutionResolutionError(
                "Automatic generated recipe attempts require explicit confirmation.",
            )
        if not binding.confirmation_provenance.strip():
            raise RecipeExecutionResolutionError(
                "Automatic generated recipe attempts require explicit confirmation provenance.",
            )
        if not request.allow_experimental:
            raise RecipeExecutionResolutionError(
                "Generated recipe execution requires allow_experimental=true.",
            )
        try:
            attempt = self._recipe_attempt_store.get_attempt(attempt_id)
        except KeyError as exc:
            raise RecipeExecutionResolutionError(
                f"Recipe attempt '{attempt_id}' was not found for generated execution.",
            ) from exc
        if attempt.recipe_fingerprint != fingerprint:
            raise RecipeExecutionResolutionError(
                f"Recipe attempt '{attempt_id}' fingerprint mismatch: expected {fingerprint}, "
                f"store has {attempt.recipe_fingerprint}.",
            )
        if attempt.state != AttemptState.RUNNING:
            raise RecipeExecutionResolutionError(
                f"Recipe attempt '{attempt_id}' is in state '{attempt.state.value}' and cannot execute.",
            )
        generated_record = self._recipe_attempt_store.get_generated_recipe(fingerprint)
        if generated_record is None:
            raise RecipeExecutionResolutionError(
                f"Generated recipe record '{fingerprint}' was not found.",
            )
        if generated_record.recipe_status != RecipeStatus.EXPERIMENTAL:
            raise RecipeExecutionResolutionError(
                f"Generated recipe '{fingerprint}' must remain experimental before promotion; "
                f"found '{generated_record.recipe_status.value}'.",
            )
        self._assert_attempt_identity_matches_generated(attempt=attempt, generated=generated_record)
        recipe, pinned_revision, resolution_outcome, capability_status = _load_generated_recipe_execution_plan(
            generated_record.payload()
        )
        if recipe.status != RecipeStatus.EXPERIMENTAL:
            raise RecipeExecutionResolutionError(
                f"Generated execution requires experimental recipe status; got '{recipe.status.value}'.",
            )
        if resolution_outcome != ResolutionOutcome.EXACT.value:
            raise RecipeExecutionResolutionError(
                f"Generated recipe capability resolution must be '{ResolutionOutcome.EXACT.value}', "
                f"got '{resolution_outcome}'.",
            )
        if capability_status == CapabilityStatus.SOURCE_CHANGE_REQUIRED.value:
            raise RecipeExecutionResolutionError(
                "Generated recipe capability is source-change-required and cannot run tooling.",
            )
        if capability_status not in {
            CapabilityStatus.VERIFIED.value,
            CapabilityStatus.TOOL_SUPPORTED_UNVERIFIED.value,
        }:
            raise RecipeExecutionResolutionError(
                f"Unsupported generated capability status '{capability_status}'.",
            )

        request_revision = _normalize_revision_sha(
            request.hf_revision,
            field_name="request.hf_revision",
        )
        if request_revision != pinned_revision:
            raise RecipeExecutionResolutionError(
                "Generated recipe request revision mismatch against persisted pinned revision.",
            )
        if request.candidate.huggingface_model_id != attempt.model_id:
            raise RecipeExecutionResolutionError(
                "Generated recipe request model id does not match persisted attempt identity.",
            )
        if request.candidate.huggingface_model_id != recipe.huggingface_model_id:
            raise RecipeExecutionResolutionError(
                "Generated recipe payload model id does not match request candidate.",
            )
        if request.candidate.modality != recipe.modality:
            raise RecipeExecutionResolutionError(
                "Generated recipe request modality does not match persisted recipe modality.",
            )
        if request.task_profile != recipe.task_profile:
            raise RecipeExecutionResolutionError(
                "Generated recipe request task profile does not match persisted recipe profile.",
            )
        if request.recipe_id is not None and request.recipe_id != recipe.id:
            raise RecipeExecutionResolutionError(
                "Generated recipe request recipe_id does not match persisted recipe id.",
            )
        if request.recipe_version is not None and request.recipe_version != recipe.version:
            raise RecipeExecutionResolutionError(
                "Generated recipe request recipe_version does not match persisted recipe version.",
            )
        if request.recipe_status is not None and request.recipe_status.strip().lower() != recipe.status.value:
            raise RecipeExecutionResolutionError(
                "Generated recipe request recipe_status does not match persisted recipe status.",
            )
        if (
            request.recipe_artifact_cache_prefix is not None
            and request.recipe_artifact_cache_prefix != recipe.artifact_cache_prefix
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request artifact cache prefix does not match persisted recipe.",
            )
        if (
            request.recipe_model_name_prefix is not None
            and request.recipe_model_name_prefix != recipe.model_name_prefix
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request model name prefix does not match persisted recipe.",
            )
        selected = recipe.choice_for_profile(request.task_profile, request.skip_olive)
        if selected is None:
            supported = ", ".join(
                f"{choice.task_profile}/skip_olive={choice.skip_olive}"
                for choice in recipe.optimization_choices
            )
            raise RecipeExecutionResolutionError(
                f"Generated recipe '{recipe.id}' does not support task_profile={request.task_profile} "
                f"with skip_olive={request.skip_olive}. Supported: {supported or 'none'}.",
            )
        if (
            request.optimization_strategy is not None
            and request.optimization_strategy.lower() != selected.strategy.lower()
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request optimization strategy does not match persisted recipe choice.",
            )
        if (
            request.optimization_precision is not None
            and request.optimization_precision.lower() != selected.precision.lower()
        ):
            raise RecipeExecutionResolutionError(
                "Generated recipe request optimization precision does not match persisted recipe choice.",
            )
        expected_mobius_dtype = recipe.mobius.dtype
        if request.candidate.recommended_mobius_dtype != expected_mobius_dtype:
            raise RecipeExecutionResolutionError(
                "Generated recipe request candidate Mobius dtype does not match persisted recipe.",
            )
        expected_olive_precision = None if selected.skip_olive else selected.precision
        if request.candidate.recommended_olive_precision != expected_olive_precision:
            raise RecipeExecutionResolutionError(
                "Generated recipe request candidate Olive precision does not match persisted recipe.",
            )
        return RecipeExecutionPlan(
            recipe=recipe,
            pinned_revision=pinned_revision,
            source="generated_recipe_attempt_store",
        )

    @staticmethod
    def _assert_attempt_identity_matches_generated(
        *,
        attempt: RecipeAttempt,
        generated: Any,
    ) -> None:
        mismatches: list[str] = []
        for field_name in (
            "model_id",
            "revision_sha",
            "requested_device",
            "requested_precision",
            "compiler_version",
            "capability_fingerprint",
            "toolchain_fingerprint",
            "profile_fingerprint",
        ):
            if getattr(attempt, field_name) != getattr(generated, field_name):
                mismatches.append(field_name)
        if mismatches:
            raise RecipeExecutionResolutionError(
                "Recipe attempt identity mismatch against generated record for field(s): "
                + ", ".join(mismatches)
                + ".",
            )


def _load_generated_recipe_execution_plan(
    payload: dict[str, object],
) -> tuple[ModelRecipe, str, str, str]:
    try:
        validate_generated_recipe_payload(payload)
    except GeneratedRecipeCompileError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe payload failed schema validation: {exc}",
        ) from exc

    recipe_payload = _require_mapping(payload.get("recipe"), field_name="generated.recipe")
    recipe = _recipe_from_payload(recipe_payload)
    pinned_revision = _normalize_revision_sha(
        payload.get("pinned_revision"),
        field_name="generated.pinned_revision",
    )
    provenance_payload = _require_mapping(payload.get("provenance"), field_name="generated.provenance")
    resolution_outcome = _require_string(
        provenance_payload.get("resolution_outcome"),
        field_name="generated.provenance.resolution_outcome",
    ).lower()
    capability_status = _require_string(
        provenance_payload.get("capability_status"),
        field_name="generated.provenance.capability_status",
    ).lower()
    return recipe, pinned_revision, resolution_outcome, capability_status


def _recipe_from_payload(payload: dict[str, object]) -> ModelRecipe:
    mobius_payload = _require_mapping(payload.get("mobius"), field_name="generated.recipe.mobius")
    olive_raw = payload.get("olive")
    olive_payload = _require_mapping(olive_raw, field_name="generated.recipe.olive") if olive_raw is not None else None

    ancillary_rows = _require_array(payload.get("ancillary_files"), field_name="generated.recipe.ancillary_files")
    ancillary_files: list[AncillaryFileRule] = []
    for index, row in enumerate(ancillary_rows, start=1):
        item = _require_mapping(row, field_name=f"generated.recipe.ancillary_files[{index}]")
        ancillary_files.append(
            AncillaryFileRule(
                relative_path=_require_string(
                    item.get("relative_path"),
                    field_name=f"generated.recipe.ancillary_files[{index}].relative_path",
                ),
                required=_require_bool(
                    item.get("required"),
                    field_name=f"generated.recipe.ancillary_files[{index}].required",
                ),
                source=_require_string(
                    item.get("source"),
                    field_name=f"generated.recipe.ancillary_files[{index}].source",
                ),
            )
        )

    optimization_rows = _require_array(
        payload.get("optimization_choices"),
        field_name="generated.recipe.optimization_choices",
    )
    optimization_choices: list[OptimizationChoice] = []
    for index, row in enumerate(optimization_rows, start=1):
        item = _require_mapping(row, field_name=f"generated.recipe.optimization_choices[{index}]")
        optimization_choices.append(
            OptimizationChoice(
                strategy=_require_string(
                    item.get("strategy"),
                    field_name=f"generated.recipe.optimization_choices[{index}].strategy",
                ),
                precision=_require_string(
                    item.get("precision"),
                    field_name=f"generated.recipe.optimization_choices[{index}].precision",
                ),
                task_profile=_require_string(
                    item.get("task_profile"),
                    field_name=f"generated.recipe.optimization_choices[{index}].task_profile",
                ),
                skip_olive=_require_bool(
                    item.get("skip_olive"),
                    field_name=f"generated.recipe.optimization_choices[{index}].skip_olive",
                ),
                default=_require_bool(
                    item.get("default"),
                    field_name=f"generated.recipe.optimization_choices[{index}].default",
                ),
            )
        )
    try:
        modality = CandidateModality(
            _require_string(payload.get("modality"), field_name="generated.recipe.modality")
        )
    except ValueError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe modality is unsupported: {payload.get('modality')!r}.",
        ) from exc
    try:
        inference_modality = CandidateModality(
            _require_string(
                payload.get("inference_modality"),
                field_name="generated.recipe.inference_modality",
            )
        )
    except ValueError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe inference_modality is unsupported: {payload.get('inference_modality')!r}.",
        ) from exc
    try:
        status = RecipeStatus(_require_string(payload.get("status"), field_name="generated.recipe.status"))
    except ValueError as exc:
        raise RecipeExecutionResolutionError(
            f"Generated recipe status is unsupported: {payload.get('status')!r}.",
        ) from exc
    return ModelRecipe(
        id=_require_string(payload.get("id"), field_name="generated.recipe.id"),
        version=_require_string(payload.get("version"), field_name="generated.recipe.version"),
        status=status,
        status_reason=_require_string(
            payload.get("status_reason"),
            field_name="generated.recipe.status_reason",
        ),
        huggingface_model_id=_require_string(
            payload.get("huggingface_model_id"),
            field_name="generated.recipe.huggingface_model_id",
        ),
        modality=modality,
        task_profile=_require_string(payload.get("task_profile"), field_name="generated.recipe.task_profile"),
        verified_revision=_optional_string(payload.get("verified_revision")),
        preferred_revision=_optional_string(payload.get("preferred_revision")),
        mobius=MobiusRecipeArgs(
            ep=_require_string(mobius_payload.get("ep"), field_name="generated.recipe.mobius.ep"),
            runtime=_require_string(mobius_payload.get("runtime"), field_name="generated.recipe.mobius.runtime"),
            dtype=_optional_string(mobius_payload.get("dtype")),
            task=_optional_string(mobius_payload.get("task")),
        ),
        olive=(
            OliveRecipeArgs(
                input_source=_require_string(
                    olive_payload.get("input_source"),
                    field_name="generated.recipe.olive.input_source",
                ),
                task=_require_string(olive_payload.get("task"), field_name="generated.recipe.olive.task"),
                precision=_optional_string(olive_payload.get("precision")),
                device=_require_string(
                    olive_payload.get("device"),
                    field_name="generated.recipe.olive.device",
                ),
                provider=_require_string(
                    olive_payload.get("provider"),
                    field_name="generated.recipe.olive.provider",
                ),
                log_level=_require_string(
                    olive_payload.get("log_level"),
                    field_name="generated.recipe.olive.log_level",
                ),
                block_size=_optional_int(
                    olive_payload.get("block_size"),
                    field_name="generated.recipe.olive.block_size",
                ),
            )
            if olive_payload is not None
            else None
        ),
        ancillary_files=tuple(ancillary_files),
        runtime_validation=_require_string(
            payload.get("runtime_validation"),
            field_name="generated.recipe.runtime_validation",
        ),
        inference_modality=inference_modality,
        optimization_choices=tuple(optimization_choices),
        artifact_cache_prefix=_require_string(
            payload.get("artifact_cache_prefix"),
            field_name="generated.recipe.artifact_cache_prefix",
        ),
        model_name_prefix=_require_string(
            payload.get("model_name_prefix"),
            field_name="generated.recipe.model_name_prefix",
        ),
        success_message=_require_string(
            payload.get("success_message"),
            field_name="generated.recipe.success_message",
        ),
    )


def _require_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RecipeExecutionResolutionError(f"{field_name} must be an object.")
    return value


def _require_array(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise RecipeExecutionResolutionError(f"{field_name} must be an array.")
    return value


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeExecutionResolutionError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RecipeExecutionResolutionError("Optional string field must be null or a string.")
    stripped = value.strip()
    return stripped or None


def _optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeExecutionResolutionError(f"{field_name} must be null or an integer.")
    return value


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeExecutionResolutionError(f"{field_name} must be a boolean.")
    return value


def _normalize_revision_sha(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeExecutionResolutionError(f"{field_name} must be a non-empty string.")
    normalized = value.strip().lower()
    if _HEX40_RE.fullmatch(normalized) is None:
        raise RecipeExecutionResolutionError(
            f"{field_name} must be a full 40-character lowercase hex revision SHA.",
        )
    return normalized


def _resolve_python_executable(value: Path | str | None) -> Path:
    if value is None:
        return Path(sys.executable).resolve()
    if isinstance(value, Path):
        candidate = value
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError("runtime python executable must be non-empty when provided.")
        candidate = Path(value.strip())
    else:
        raise TypeError("runtime python executable must be a path-like string or Path.")
    return candidate.resolve()


class FoundrySdkTextInferenceBackend:
    def __init__(
        self,
        process_runner: ProcessRunner,
        *,
        timeout_seconds: int = 900,
        cancellation_event: Event | None = None,
        runtime_python_executable: Path | str | None = None,
    ) -> None:
        self._process_runner = process_runner
        self._timeout_seconds = timeout_seconds
        self._cancellation_event = cancellation_event
        self._runtime_python_executable = _resolve_python_executable(runtime_python_executable)
        self._last_batch_diagnostics: dict[str, object] | None = None

    def consume_last_batch_diagnostics(self) -> dict[str, object] | None:
        diagnostics = self._last_batch_diagnostics
        self._last_batch_diagnostics = None
        return diagnostics

    def infer(
        self,
        *,
        artifact: BuildArtifact,
        job: BuildJob,
        prompt: str,
        max_tokens: int,
    ) -> str:
        if len(prompt) > 8192:
            raise ValueError("Inference prompt exceeds the 8192 character limit.")
        model_dir = artifact.path.resolve()
        model_name = self._read_model_name(model_dir)
        request_file = job.request.workspace_root / f"inference-{uuid.uuid4().hex}.json"
        request_file.write_text(
            json.dumps({"prompt": prompt, "max_tokens": max_tokens}),
            encoding="utf-8",
        )
        try:
            result = self._process_runner.run(
                CommandSpec(
                    argv=(
                        str(self._runtime_python_executable),
                        "-m",
                        "fl_model_onboarding.runtime_worker",
                        "foundry-infer",
                        "--model-dir",
                        str(model_dir),
                        "--model-name",
                        str(model_name),
                        "--request-file",
                        str(request_file),
                    ),
                    cwd=job.request.workspace_root,
                    timeout_seconds=self._timeout_seconds,
                ),
                cancel_event=self._cancellation_event,
            )
        finally:
            request_file.unlink(missing_ok=True)
        payload = _result_payload(result)
        if not result.ok or payload.get("ok") is not True:
            raise RuntimeError(str(payload.get("error") or "Foundry Local inference failed."))
        return str(payload["output"])

    def infer_batch(
        self,
        *,
        artifact: BuildArtifact,
        job: BuildJob,
        prompts: Sequence[tuple[str, str]],
        max_tokens: int,
    ) -> tuple[str, ...]:
        if isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("Inference max_tokens must be a positive integer.")
        self._last_batch_diagnostics = None
        prompt_payload: list[dict[str, object]] = []
        for prompt_id, prompt_text in prompts:
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                raise ValueError("Inference prompt_id must be a non-empty string.")
            if len(prompt_text) > 8192:
                raise ValueError("Inference prompt exceeds the 8192 character limit.")
            prompt_payload.append(
                {
                    "prompt_id": prompt_id.strip(),
                    "prompt": prompt_text,
                    "max_tokens": int(max_tokens),
                }
            )
        if not prompt_payload:
            return ()

        model_dir = artifact.path.resolve()
        model_name = self._read_model_name(model_dir)
        batch_timeout = self._timeout_seconds * len(prompt_payload)
        command_timeout = batch_timeout + _BATCH_INFERENCE_TIMEOUT_GRACE_SECONDS
        request_summary: dict[str, object] = {
            "mode": "single-worker-batch",
            "prompt_ids": [str(row["prompt_id"]) for row in prompt_payload],
            "prompt_count": len(prompt_payload),
            "max_tokens": int(max_tokens),
            "model_name": model_name,
            "expected_model_load_count": 1,
            "per_prompt_timeout_seconds": self._timeout_seconds,
            "batch_timeout_seconds": batch_timeout,
            "outer_command_timeout_seconds": command_timeout,
            "outer_timeout_grace_seconds": _BATCH_INFERENCE_TIMEOUT_GRACE_SECONDS,
        }
        request_file = job.request.workspace_root / f"inference-{uuid.uuid4().hex}.json"
        request_file.write_text(
            json.dumps(
                {
                    "prompts": prompt_payload,
                    "per_prompt_timeout_seconds": self._timeout_seconds,
                    "batch_timeout_seconds": batch_timeout,
                }
            ),
            encoding="utf-8",
        )
        try:
            try:
                result = self._process_runner.run(
                    CommandSpec(
                        argv=(
                            str(self._runtime_python_executable),
                            "-m",
                            "fl_model_onboarding.runtime_worker",
                            "foundry-infer-batch",
                            "--model-dir",
                            str(model_dir),
                            "--model-name",
                            str(model_name),
                            "--request-file",
                            str(request_file),
                        ),
                        cwd=job.request.workspace_root,
                        timeout_seconds=command_timeout,
                    ),
                    cancel_event=self._cancellation_event,
                )
            except TimeoutError as exc:
                self._last_batch_diagnostics = {
                    "request": request_summary,
                    "response": {
                        "ok": False,
                        "failure_stage": "outer_command_timeout",
                        "error": "Foundry Local batch inference timed out before completing the prompt suite.",
                        "completed_prompt_ids": [],
                        "results": [],
                    },
                }
                raise RuntimeError(
                    "Foundry Local batch inference timed out before completing the prompt suite."
                ) from exc
        finally:
            request_file.unlink(missing_ok=True)
        payload = _result_payload(result)
        response_results: list[dict[str, object]] = []
        payload_results = payload.get("results")
        if isinstance(payload_results, list):
            for row in payload_results:
                if not isinstance(row, dict):
                    continue
                response_results.append(
                    {
                        "prompt_id": row.get("prompt_id"),
                        "duration_seconds": row.get("duration_seconds"),
                        "timed_out": row.get("timed_out"),
                    }
                )
        response_summary: dict[str, object] = {
            "ok": payload.get("ok"),
            "failure_stage": payload.get("failure_stage"),
            "failed_prompt_id": payload.get("failed_prompt_id"),
            "completed_prompt_ids": payload.get("completed_prompt_ids"),
            "duration_seconds": payload.get("duration_seconds"),
            "results": response_results,
        }
        self._last_batch_diagnostics = {
            "request": request_summary,
            "response": response_summary,
        }
        if not result.ok or payload.get("ok") is not True:
            detail = str(payload.get("error") or "Foundry Local batch inference failed.")
            failure_stage = payload.get("failure_stage")
            failed_prompt_id = payload.get("failed_prompt_id")
            if isinstance(failure_stage, str) and failure_stage.strip():
                detail += f" stage={failure_stage.strip()}"
            if isinstance(failed_prompt_id, str) and failed_prompt_id.strip():
                detail += f" prompt_id={failed_prompt_id.strip()}"
            raise RuntimeError(detail)
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Foundry Local batch inference response is missing prompt results.")
        if len(results) != len(prompt_payload):
            raise RuntimeError(
                "Foundry Local batch inference response did not return all requested prompt results."
            )
        outputs: list[str] = []
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                raise RuntimeError("Foundry Local batch inference result rows must be objects.")
            expected_prompt_id = str(prompt_payload[index]["prompt_id"])
            actual_prompt_id = row.get("prompt_id")
            if actual_prompt_id != expected_prompt_id:
                response_summary["prompt_order_preserved"] = False
                raise RuntimeError(
                    "Foundry Local batch inference response prompt order mismatch "
                    f"(expected '{expected_prompt_id}', got '{actual_prompt_id}')."
                )
            outputs.append(str(row.get("output") or ""))
        response_summary["prompt_order_preserved"] = True
        return tuple(outputs)

    @staticmethod
    def _read_model_name(model_dir: Path) -> str:
        descriptor = json.loads((model_dir / "inference_model.json").read_text(encoding="utf-8"))
        model_name = descriptor.get("Name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise RuntimeError(f"Model descriptor is missing non-empty Name in '{model_dir}'.")
        return model_name.strip()


class ProductionBuildStageRunner:
    def __init__(
        self,
        process_runner: ProcessRunner,
        *,
        build_timeout_seconds: int = 7200,
        olive_timeout_seconds: int = 5400,
        runtime_timeout_seconds: int = 900,
        model_acquisition: HuggingFaceAcquisitionClient | None = None,
        recipe_registry: RecipeRegistry | None = None,
        recipe_attempt_store: RecipeAttemptStore | None = None,
        recipe_execution_resolver: RecipeExecutionResolver | None = None,
        runtime_python_executable: Path | str | None = None,
        on_mobius_ready: Callable[[BuildJob, Path], None] | None = None,
    ) -> None:
        self._process_runner = process_runner
        self._build_timeout_seconds = build_timeout_seconds
        self._olive_timeout_seconds = olive_timeout_seconds
        self._runtime_timeout_seconds = runtime_timeout_seconds
        self._model_acquisition = model_acquisition or HuggingFaceAcquisitionAdapter()
        self._runtime_python_executable = _resolve_python_executable(runtime_python_executable)
        self._recipe_registry = recipe_registry or DEFAULT_RECIPE_REGISTRY
        self._execution_resolver = recipe_execution_resolver or RecipeExecutionResolver(
            recipe_registry=self._recipe_registry,
            recipe_attempt_store=recipe_attempt_store,
        )
        # Slice 3B1: an optional, typed, additive result hook invoked exactly once
        # right after a real Mobius build succeeds -- while `mobius_dir` is
        # guaranteed to still exist on disk and before any future retention
        # cleanup could remove it -- so a caller (`local_service.py`) can capture
        # a `PreOliveArtifactDescriptor` for later trusted-candidate reuse without
        # this runner knowing anything about candidates/lineages/policies itself.
        # Never invoked on the `run_fallback_with_pre_olive_reuse` path, which
        # never runs Mobius. Defaults to a no-op for every existing caller.
        self._on_mobius_ready = on_mobius_ready

    def run(
        self,
        job: BuildJob,
        *,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        staging_dir: Path | None = None
        package_dir: Path | None = None
        staging_preexisting = False
        package_preexisting = False
        try:
            execution = self._execution_resolver.resolve(job.request)
            staging_dir, package_dir = production_package_paths(
                job,
                recipe_registry=self._recipe_registry,
                resolved_recipe=execution.recipe,
            )
            staging_preexisting = staging_dir.exists()
            package_preexisting = package_dir.exists()
            self._run(
                job,
                recipe=execution.recipe,
                pinned_revision=execution.pinned_revision,
                persist=persist,
                cancellation_event=cancellation_event,
            )
        except Exception as exc:
            if staging_dir is not None and not staging_preexisting and staging_dir.exists():
                shutil.rmtree(staging_dir)
            if package_dir is not None and not package_preexisting and package_dir.exists():
                shutil.rmtree(package_dir)
            if job.state == JobState.CANCELLED:
                return
            classification = FailureClassification.PROCESS_FAILED
            if isinstance(exc, FileNotFoundError):
                classification = FailureClassification.MISSING_DEPENDENCY
            elif isinstance(exc, RecipeExecutionResolutionError):
                classification = exc.classification
            fail_job(
                job,
                FailureInfo(
                    stage=job.state,
                    classification=classification,
                    message=str(exc),
                ),
            )
            job.finished_utc = datetime.now(timezone.utc)
            persist()

    def _run(
        self,
        job: BuildJob,
        *,
        recipe: ModelRecipe,
        pinned_revision: str,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        request = job.request
        job.production_invocation_evidence = ProductionInvocationEvidence()
        if recipe.choice_for_profile(request.task_profile, request.skip_olive) is None:
            supported = ", ".join(
                f"{choice.task_profile}/skip_olive={choice.skip_olive}"
                for choice in recipe.optimization_choices
            )
            raise RuntimeError(
                f"Recipe '{recipe.id}' does not support task_profile={request.task_profile} "
                f"with skip_olive={request.skip_olive}. Supported: {supported or 'none'}."
            )
        if request.candidate.modality != CandidateModality.LLM:
            raise RuntimeError("Production execution currently supports LLM runtime validation only.")
        if recipe.olive is None:
            raise RuntimeError(f"Recipe '{recipe.id}' requires Olive settings for production packaging.")

        pinned_source = self._resolve_pinned_source(recipe=recipe, request=request, pinned_revision=pinned_revision)
        mobius_dir = request.workspace_root / "mobius"
        olive_dir = request.workspace_root / "olive"
        mobius_dir.mkdir(parents=True, exist_ok=False)
        olive_dir.mkdir(parents=True, exist_ok=False)

        transition(job, JobState.DOWNLOADING, f"Pinned Hugging Face revision {pinned_source.revision_sha}.")
        persist()
        mobius_dtype = recipe.mobius.dtype or "default"
        transition(
            job,
            JobState.MOBIUS_BUILDING,
            (
                f"Running recipe Mobius {recipe.mobius.ep} {recipe.mobius.runtime} "
                f"{mobius_dtype} build."
            ),
        )
        persist()
        mobius_argv: list[str] = [
            "mobius",
            "build",
            "--config",
            str(pinned_source.snapshot_dir),
            "--ep",
            recipe.mobius.ep,
            "--runtime",
            recipe.mobius.runtime,
        ]
        if recipe.mobius.task:
            mobius_argv.extend(["--task", recipe.mobius.task])
        if recipe.mobius.dtype:
            mobius_argv.extend(["--dtype", recipe.mobius.dtype])
        mobius_argv.append(str(mobius_dir))
        self._run_instrumented_tool_command(
            job,
            tool="mobius",
            spec=CommandSpec(
                argv=tuple(mobius_argv),
                cwd=request.workspace_root,
                timeout_seconds=self._build_timeout_seconds,
            ),
            cancellation_event=cancellation_event,
            label="Mobius build",
        )
        baseline_model_name = f"{recipe.model_name_prefix}-{job.job_id[:12]}-mobius-baseline:1"
        (mobius_dir / "inference_model.json").write_text(
            json.dumps({"Name": baseline_model_name}, indent=2),
            encoding="utf-8",
        )
        if self._on_mobius_ready is not None:
            self._on_mobius_ready(job, mobius_dir)

        transition(job, JobState.MOBIUS_VALIDATING, "Mobius output created; ONNX validation follows Olive.")
        persist()
        self._run_from_olive(
            job,
            recipe=recipe,
            request=request,
            mobius_dir=mobius_dir,
            olive_dir=olive_dir,
            persist=persist,
            cancellation_event=cancellation_event,
        )

    def _run_from_olive(
        self,
        job: BuildJob,
        *,
        recipe: ModelRecipe,
        request: BuildRequest,
        mobius_dir: Path,
        olive_dir: Path,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        """The Olive-optimize-through-inference half of a production build,
        shared by the legacy one-shot `_run` (Mobius just ran into
        `mobius_dir`) and Slice 3A2's `run_fallback_with_pre_olive_reuse`
        (`mobius_dir` is instead a freshly materialized, revalidated copy of
        an already-captured pre-Olive artifact). Callers are responsible for
        getting `mobius_dir`/`olive_dir` into the right state and for every
        state transition up through `JobState.MOBIUS_VALIDATING`; this method
        starts at `JobState.OLIVE_OPTIMIZING` and runs unchanged either way,
        which is what keeps the two paths' Olive+downstream behavior
        identical.
        """
        assert recipe.olive is not None
        transition(
            job,
            JobState.OLIVE_OPTIMIZING,
            (
                f"Running recipe Olive {recipe.olive.input_source} "
                f"{recipe.olive.precision or 'default'} optimization."
            ),
        )
        persist()
        olive_argv: list[str] = [
            "olive",
            "optimize",
            "--model_name_or_path",
            str(mobius_dir),
            "--task",
            recipe.olive.task,
            "--device",
            recipe.olive.device,
            "--provider",
            recipe.olive.provider,
        ]
        if recipe.olive.precision:
            olive_argv.extend(["--precision", recipe.olive.precision])
        if recipe.olive.block_size is not None:
            olive_argv.extend(["--block_size", str(recipe.olive.block_size)])
        olive_argv.extend(
            [
                "--output_path",
                str(olive_dir),
                "--log_level",
                recipe.olive.log_level,
            ]
        )
        self._run_instrumented_tool_command(
            job,
            tool="olive",
            spec=CommandSpec(
                argv=tuple(olive_argv),
                cwd=request.workspace_root,
                timeout_seconds=self._olive_timeout_seconds,
            ),
            cancellation_event=cancellation_event,
            label="Olive optimize",
        )
        source_dir = olive_dir
        self._ensure_required_ancillary_files(source_dir=source_dir, recipe=recipe)

        transition(job, JobState.PACKAGING, "Creating immutable Foundry Local BYOM package.")
        persist()
        artifact_id = self._artifact_id(job)
        model_name = f"{recipe.model_name_prefix}-{artifact_id[:12]}:1"
        staging_dir, package_dir = production_package_paths(
            job,
            recipe_registry=self._recipe_registry,
            resolved_recipe=recipe,
        )
        if package_dir.exists():
            raise FileExistsError(f"Immutable artifact path already exists: {package_dir}")
        if staging_dir.exists():
            raise FileExistsError(f"Partial artifact path already exists: {staging_dir}")
        shutil.copytree(source_dir, staging_dir)
        (staging_dir / "inference_model.json").write_text(
            json.dumps({"Name": model_name}, indent=2),
            encoding="utf-8",
        )
        reconciliation = _reconcile_decoder_outputs_in_staging_package(staging_dir=staging_dir)
        if reconciliation["status"] == "applied":
            before = reconciliation["decoder_outputs_before"]
            after = reconciliation["decoder_outputs_after"]
            remapped = reconciliation["remapped_outputs"]
            if isinstance(before, dict) and isinstance(after, dict) and isinstance(remapped, dict):
                delta = {
                    key: {"before": before.get(key), "after": after.get(key)}
                    for key in sorted(remapped.keys())
                }
            else:
                delta = {}
            job.add_event(
                "Staging decoder output reconciliation applied before runtime validation: "
                + json.dumps(delta, sort_keys=True),
            )
            persist()
        elif reconciliation["status"] == "verified":
            job.add_event("Staging decoder output reconciliation verified existing decoder mappings.")
            persist()
        else:
            job.add_event(
                "Staging decoder output reconciliation skipped: "
                + str(reconciliation.get("reason") or "no decoder outputs mapping"),
            )
            persist()

        transition(job, JobState.RUNTIME_VALIDATING, "Validating ONNX, ORT CPU, and OGA generation.")
        persist()
        runtime_result = self._run_command(
            CommandSpec(
                argv=(
                    str(self._runtime_python_executable),
                    "-m",
                    "fl_model_onboarding.runtime_worker",
                    "validate-runtime",
                    "--model-dir",
                    str(staging_dir),
                ),
                cwd=request.workspace_root,
                timeout_seconds=self._runtime_timeout_seconds,
            ),
            cancellation_event,
            "Runtime validation",
        )
        runtime_payload = _result_payload(runtime_result)
        checks = tuple(str(item) for item in runtime_payload.get("checks", []))
        job.validations.append(
            ValidationResult(
                stage=JobState.RUNTIME_VALIDATING,
                status=ValidationStatus.PASSED,
                checks=checks,
            )
        )

        staging_dir.rename(package_dir)
        transition(job, JobState.FL_LOADING, "Foundry Local SDK discovered and loaded the BYOM package.")
        persist()
        transition(job, JobState.INFERENCING, "Running bounded Foundry Local SDK chat inference.")
        persist()
        inference_backend = FoundrySdkTextInferenceBackend(
            self._process_runner,
            timeout_seconds=self._runtime_timeout_seconds,
            cancellation_event=cancellation_event,
            runtime_python_executable=self._runtime_python_executable,
        )
        output = inference_backend.infer(
            artifact=BuildArtifact(
                artifact_id=artifact_id,
                kind=ArtifactKind.MODEL,
                path=package_dir,
                description="Immutable Foundry Local BYOM model package",
            ),
            job=job,
            prompt="Reply with: OK",
            max_tokens=64,
        )
        if not output.strip():
            raise RuntimeError("Foundry Local SDK inference returned empty output.")
        job.validations.append(
            ValidationResult(
                stage=JobState.INFERENCING,
                status=ValidationStatus.PASSED,
                checks=("foundry_local_sdk_chat=passed",),
            )
        )
        job.register_artifact(
            BuildArtifact(
                artifact_id=artifact_id,
                kind=ArtifactKind.MODEL,
                path=package_dir,
                description="Immutable Foundry Local BYOM model package",
            )
        )
        transition(job, JobState.SUCCEEDED, recipe.success_message)
        job.finished_utc = datetime.now(timezone.utc)
        persist()

    def run_fallback_with_pre_olive_reuse(
        self,
        job: BuildJob,
        *,
        descriptor: PreOliveArtifactDescriptor,
        fallback_generation_identity: PreOliveGenerationIdentity,
        persist: Callable[[], None],
        cancellation_event: Event,
    ) -> None:
        """Slice 3A2 runner-level primitive: execute one fallback candidate's
        Olive+downstream path by reusing an already-captured, revalidated
        `PreOliveArtifactDescriptor` instead of re-running Mobius.

        Resolves `job.request` through the same `RecipeExecutionResolver`
        every other execution path uses (so every existing generated-recipe/
        attempt-identity/experimental-status check still applies unchanged),
        then additionally requires the fallback candidate's own generation
        identity and Mobius arguments to match the descriptor byte-for-byte
        -- any mismatch, tamper, or containment violation fails closed before
        any filesystem copy or Olive launch is attempted. Mirrors `run()`'s
        existing staging/package failure cleanup exactly; the materialized
        Mobius copy cleans up its own partial state on failure (see
        `materialize_pre_olive_copy`) and is otherwise left in place on a
        later failure, matching the legacy path's existing asymmetric
        treatment of `mobius_dir`/`olive_dir` on failure.

        The Mobius tool itself is never invoked on this path:
        `job.production_invocation_evidence` keeps its default not-run
        Mobius evidence (`invocation_count=0`, `terminal_stage=NOT_RUN`),
        which `production_invocation_evidence_to_candidate_counters` already
        maps to a `None` (not `0`) Mobius count -- consistent with the
        existing null-count semantics for a tool that was never launched.
        Olive runs exactly once through the shared `_run_from_olive` helper.
        """
        staging_dir: Path | None = None
        package_dir: Path | None = None
        staging_preexisting = False
        package_preexisting = False
        try:
            execution = self._execution_resolver.resolve(job.request)
            recipe = execution.recipe
            pinned_revision = execution.pinned_revision
            request = job.request
            job.production_invocation_evidence = ProductionInvocationEvidence()
            if recipe.choice_for_profile(request.task_profile, request.skip_olive) is None:
                supported = ", ".join(
                    f"{choice.task_profile}/skip_olive={choice.skip_olive}"
                    for choice in recipe.optimization_choices
                )
                raise RuntimeError(
                    f"Recipe '{recipe.id}' does not support task_profile={request.task_profile} "
                    f"with skip_olive={request.skip_olive}. Supported: {supported or 'none'}."
                )
            if request.candidate.modality != CandidateModality.LLM:
                raise RuntimeError("Production execution currently supports LLM runtime validation only.")
            if recipe.olive is None:
                raise RuntimeError(f"Recipe '{recipe.id}' requires Olive settings for production packaging.")
            if pinned_revision != descriptor.generation_identity.revision_sha:
                raise PreOliveReuseError(
                    "Fallback candidate pinned revision does not match the captured pre-Olive "
                    "artifact's revision; refusing reuse.",
                    classification=FailureClassification.COMPATIBILITY,
                )
            if recipe.huggingface_model_id != descriptor.generation_identity.model_id:
                raise PreOliveReuseError(
                    "Fallback candidate model id does not match the captured pre-Olive artifact's "
                    "model id; refusing reuse.",
                    classification=FailureClassification.COMPATIBILITY,
                )
            validate_pre_olive_reuse(
                descriptor,
                candidate_identity=fallback_generation_identity,
                candidate_mobius_args=recipe.mobius,
            )

            staging_dir, package_dir = production_package_paths(
                job,
                recipe_registry=self._recipe_registry,
                resolved_recipe=recipe,
            )
            staging_preexisting = staging_dir.exists()
            package_preexisting = package_dir.exists()

            mobius_dir = request.workspace_root / "mobius"
            olive_dir = request.workspace_root / "olive"
            if mobius_dir.exists():
                raise PreOliveReuseError(
                    f"Fallback workspace Mobius destination already exists: {mobius_dir}",
                )
            if olive_dir.exists():
                raise PreOliveReuseError(
                    f"Fallback workspace Olive destination already exists: {olive_dir}",
                )

            transition(
                job,
                JobState.DOWNLOADING,
                "No Hugging Face download required; reusing a captured pre-Olive Mobius artifact.",
            )
            persist()
            transition(
                job,
                JobState.MOBIUS_BUILDING,
                "Reusing validated pre-Olive Mobius artifact for trusted fallback candidate "
                "(no Mobius invocation).",
            )
            persist()
            materialize_pre_olive_copy(
                descriptor,
                destination_dir=mobius_dir,
                authorized_roots=(request.workspace_root,),
                cancellation_event=cancellation_event,
            )

            olive_dir.mkdir(parents=True, exist_ok=False)
            transition(
                job,
                JobState.MOBIUS_VALIDATING,
                "Reused Mobius artifact revalidated against its captured manifest; ONNX "
                "validation follows Olive.",
            )
            persist()

            self._run_from_olive(
                job,
                recipe=recipe,
                request=request,
                mobius_dir=mobius_dir,
                olive_dir=olive_dir,
                persist=persist,
                cancellation_event=cancellation_event,
            )
        except Exception as exc:
            if staging_dir is not None and not staging_preexisting and staging_dir.exists():
                shutil.rmtree(staging_dir)
            if package_dir is not None and not package_preexisting and package_dir.exists():
                shutil.rmtree(package_dir)
            if job.state == JobState.CANCELLED:
                return
            classification = FailureClassification.PROCESS_FAILED
            if isinstance(exc, FileNotFoundError):
                classification = FailureClassification.MISSING_DEPENDENCY
            elif isinstance(exc, (RecipeExecutionResolutionError, PreOliveReuseError)):
                classification = exc.classification
            fail_job(
                job,
                FailureInfo(
                    stage=job.state,
                    classification=classification,
                    message=str(exc),
                ),
            )
            job.finished_utc = datetime.now(timezone.utc)
            persist()

    def _resolve_pinned_source(
        self,
        *,
        recipe: ModelRecipe,
        request: BuildRequest,
        pinned_revision: str,
    ) -> PinnedModelSource:
        snapshot_dir = pinned_snapshot_cache_path(
            request.model_cache_dir,
            model_id=recipe.huggingface_model_id,
            revision_sha=pinned_revision,
        )
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._model_acquisition.acquire_snapshot(
            recipe.huggingface_model_id,
            snapshot_dir,
            revision=pinned_revision,
        )
        if not snapshot_path.is_dir():
            raise RuntimeError(
                f"Pinned snapshot path is not a directory: {snapshot_path}."
            )
        return PinnedModelSource(
            model_id=recipe.huggingface_model_id,
            revision_sha=pinned_revision,
            snapshot_dir=snapshot_path,
        )

    def _run_command(
        self,
        spec: CommandSpec,
        cancellation_event: Event,
        label: str,
    ) -> CommandResult:
        result = self._process_runner.run(spec, cancel_event=cancellation_event)
        if not result.ok:
            detail = _compact_failure_detail(
                result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
            )
            raise RuntimeError(f"{label} failed: {detail}")
        return result

    def _run_instrumented_tool_command(
        self,
        job: BuildJob,
        *,
        tool: str,
        spec: CommandSpec,
        cancellation_event: Event,
        label: str,
    ) -> CommandResult:
        """Run one real Mobius/Olive process launch with per-job invocation
        instrumentation.

        Reached only after every upstream validation check in ``_run`` has
        already passed, so every call here is a real launch *attempt*: the
        invocation counter increments immediately before the external process
        is actually started, before we know whether it will succeed, fail,
        time out, or be cancelled. That increment and the terminal
        stage/success/timing recorded afterward live on ``job`` (a fresh,
        job-specific ``ProductionInvocationEvidence``), never on ``self``, so
        concurrent jobs handled by the same runner instance never share or
        race on these counters.
        """
        if tool not in ("mobius", "olive"):
            raise ValueError(f"Unknown instrumented tool '{tool}'.")
        started = datetime.now(timezone.utc)
        self._update_tool_evidence(
            job,
            tool=tool,
            update=lambda current: replace(
                current,
                invocation_count=current.invocation_count + 1,
                terminal_stage=ToolInvocationTerminalStage.NOT_RUN,
                success=None,
                started_utc=started,
                finished_utc=None,
                wall_seconds=None,
            ),
        )
        try:
            result = self._process_runner.run(spec, cancel_event=cancellation_event)
        except TimeoutError:
            self._finish_tool_evidence(
                job,
                tool=tool,
                started=started,
                terminal_stage=ToolInvocationTerminalStage.TIMED_OUT,
                success=False,
            )
            raise
        except Exception:
            terminal_stage = (
                ToolInvocationTerminalStage.CANCELLED
                if cancellation_event.is_set()
                else ToolInvocationTerminalStage.FAILED
            )
            self._finish_tool_evidence(
                job,
                tool=tool,
                started=started,
                terminal_stage=terminal_stage,
                success=False,
            )
            raise
        if not result.ok:
            self._finish_tool_evidence(
                job,
                tool=tool,
                started=started,
                terminal_stage=ToolInvocationTerminalStage.FAILED,
                success=False,
            )
            detail = _compact_failure_detail(
                result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
            )
            raise RuntimeError(f"{label} failed: {detail}")
        self._finish_tool_evidence(
            job,
            tool=tool,
            started=started,
            terminal_stage=ToolInvocationTerminalStage.COMPLETED,
            success=True,
        )
        return result

    @staticmethod
    def _update_tool_evidence(
        job: BuildJob,
        *,
        tool: str,
        update: Callable[[ToolInvocationEvidence], ToolInvocationEvidence],
    ) -> None:
        evidence = job.production_invocation_evidence or ProductionInvocationEvidence()
        current = evidence.mobius if tool == "mobius" else evidence.olive
        updated = update(current)
        job.production_invocation_evidence = (
            replace(evidence, mobius=updated) if tool == "mobius" else replace(evidence, olive=updated)
        )

    @classmethod
    def _finish_tool_evidence(
        cls,
        job: BuildJob,
        *,
        tool: str,
        started: datetime,
        terminal_stage: ToolInvocationTerminalStage,
        success: bool,
    ) -> None:
        finished = datetime.now(timezone.utc)
        wall_seconds = max((finished - started).total_seconds(), 0.0)
        cls._update_tool_evidence(
            job,
            tool=tool,
            update=lambda current: replace(
                current,
                terminal_stage=terminal_stage,
                success=success,
                finished_utc=finished,
                wall_seconds=wall_seconds,
            ),
        )

    @staticmethod
    def _artifact_id(job: BuildJob) -> str:
        request = job.request
        return hashlib.sha256(
            f"{request.candidate.huggingface_model_id}:{request.hf_revision}:{request.task_profile}:{job.job_id}".encode()
        ).hexdigest()

    @staticmethod
    def _ensure_required_ancillary_files(*, source_dir: Path, recipe: ModelRecipe) -> None:
        missing = [
            rule.relative_path
            for rule in recipe.ancillary_files
            if rule.required and not (source_dir / rule.relative_path).exists()
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise RuntimeError(
                f"Recipe '{recipe.id}' packaging is missing required ancillary files: {joined}."
            )


def production_package_paths(
    job: BuildJob,
    *,
    recipe_registry: RecipeRegistry = DEFAULT_RECIPE_REGISTRY,
    resolved_recipe: ModelRecipe | None = None,
) -> tuple[Path, Path]:
    artifact_id = ProductionBuildStageRunner._artifact_id(job)
    cache = job.request.model_cache_dir
    if resolved_recipe is not None:
        prefix = resolved_recipe.artifact_cache_prefix
    elif job.request.recipe_artifact_cache_prefix:
        prefix = job.request.recipe_artifact_cache_prefix
    else:
        recipe = recipe_registry.resolve(
            model_id=job.request.candidate.huggingface_model_id,
            modality=job.request.candidate.modality,
            task_profile=job.request.task_profile,
            allow_experimental=True,
        ).recipe
        prefix = (
            recipe.artifact_cache_prefix
            if recipe
            else _fallback_cache_prefix(job.request.candidate.huggingface_model_id)
        )
    normalized_prefix = _normalize_cache_prefix(prefix)
    return (
        cache / f".partial-{normalized_prefix}-{artifact_id[:12]}",
        cache / f"{normalized_prefix}-{artifact_id[:12]}",
    )


def _normalize_cache_prefix(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def _fallback_cache_prefix(model_id: str) -> str:
    return _normalize_cache_prefix(model_id.strip().split("/")[-1])


def pinned_snapshot_cache_path(model_cache_dir: Path, *, model_id: str, revision_sha: str) -> Path:
    normalized_revision = _normalize_revision_sha(
        revision_sha,
        field_name="snapshot.revision_sha",
    )
    model_slug = _normalize_cache_prefix(model_id.replace("/", "-"))
    return model_cache_dir / f"snapshot-{model_slug}-{normalized_revision}"
