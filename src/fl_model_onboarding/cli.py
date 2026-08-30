from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

from .adapters.foundry_cli import FoundryCliCatalogAdapter
from .adapters.huggingface_metadata import HuggingFaceMetadataAdapter
from .adapters.mobius_cli import MobiusCliAdapter
from .adapters.olive_cli import OliveCliAdapter
from .candidates import PHASE0_CANDIDATES, resolve_candidate
from .contracts import BuildRequest
from .job_runner import LocalJobRunner
from .preflight import PreflightInspector
from .serialization import to_jsonable
from .subprocess_runner import SafeSubprocessRunner
from .version import __version__


def _default_output_dir(workspace_root: Path, candidate_key: str) -> Path:
    return workspace_root / ".local-artifacts" / candidate_key


def _build_request(args: argparse.Namespace) -> BuildRequest:
    candidate = resolve_candidate(args.candidate)
    workspace_root = Path(args.workspace_root or Path.cwd()).resolve()
    foundry = FoundryCliCatalogAdapter()
    model_cache_dir = (
        Path(args.model_cache_dir).resolve()
        if args.model_cache_dir
        else foundry.cache_location().resolve()
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _default_output_dir(workspace_root, candidate.key).resolve()
    )
    return BuildRequest(
        candidate=candidate,
        workspace_root=workspace_root,
        model_cache_dir=model_cache_dir,
        output_dir=output_dir,
        task_profile=getattr(args, "task_profile", "default"),
        hf_revision=args.hf_revision,
        skip_olive=bool(args.skip_olive),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def _print_json(value: object) -> None:
    print(json.dumps(to_jsonable(value), indent=2))


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fl-onboarding")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print CLI version.")

    doctor = sub.add_parser("doctor", help="Probe toolchain and base connectivity.")
    doctor.add_argument("--candidate", default="smollm2-1.7b-instruct")
    doctor.add_argument("--workspace-root")
    doctor.add_argument("--model-cache-dir")
    doctor.add_argument("--output-dir")
    doctor.add_argument("--task-profile", default="default")
    doctor.add_argument("--hf-revision")
    doctor.add_argument("--skip-olive", action="store_true")

    preflight = sub.add_parser("preflight", help="Run candidate preflight checks.")
    preflight.add_argument("--candidate", required=True, choices=sorted(PHASE0_CANDIDATES.keys()))
    preflight.add_argument("--workspace-root")
    preflight.add_argument("--model-cache-dir")
    preflight.add_argument("--output-dir")
    preflight.add_argument("--task-profile", default="default")
    preflight.add_argument("--hf-revision")
    preflight.add_argument("--skip-olive", action="store_true")

    dry = sub.add_parser("dry-run", help="Run deterministic build-state dry run.")
    dry.add_argument("--candidate", required=True, choices=sorted(PHASE0_CANDIDATES.keys()))
    dry.add_argument("--workspace-root")
    dry.add_argument("--model-cache-dir")
    dry.add_argument("--output-dir")
    dry.add_argument("--task-profile", default="default")
    dry.add_argument("--hf-revision")
    dry.add_argument("--skip-olive", action="store_true")

    hf_search = sub.add_parser(
        "hf-search",
        help="Sparse Hugging Face discovery (list_models: search + downloads sort).",
    )
    hf_search.add_argument("--query", required=True)
    hf_search.add_argument("--limit", type=int, default=20)

    hf_info = sub.add_parser(
        "hf-info",
        help="Selected-model Hugging Face metadata (model_info).",
    )
    hf_info.add_argument("--model-id", required=True)
    hf_info.add_argument("--revision")
    hf_info.add_argument("--files-metadata", action="store_true")

    return parser


def _run_preflight(args: argparse.Namespace) -> int:
    request = _build_request(args)
    runner = SafeSubprocessRunner()
    inspector = PreflightInspector(
        runner=runner,
        foundry=FoundryCliCatalogAdapter(runner),
        hf_metadata=HuggingFaceMetadataAdapter(),
    )
    result = inspector.inspect(request)
    _print_json(result)
    return 0 if result.ok else 2


def _run_doctor(args: argparse.Namespace) -> int:
    request = _build_request(args)
    runner = SafeSubprocessRunner()
    foundry = FoundryCliCatalogAdapter(runner)
    inspector = PreflightInspector(
        runner=runner,
        foundry=foundry,
        hf_metadata=HuggingFaceMetadataAdapter(),
    )
    preflight = inspector.inspect(request)
    payload = {
        "cli_version": __version__,
        "foundry_status": foundry.status(),
        "foundry_cache_location": str(foundry.cache_location()),
        "preflight": preflight,
    }
    _print_json(payload)
    return 0 if preflight.ok else 2


def _run_dry(args: argparse.Namespace) -> int:
    request = _build_request(args)
    runner = SafeSubprocessRunner()
    inspector = PreflightInspector(
        runner=runner,
        foundry=FoundryCliCatalogAdapter(runner),
        hf_metadata=HuggingFaceMetadataAdapter(),
    )
    job_runner = LocalJobRunner(inspector)
    mobius = MobiusCliAdapter(runner)
    olive = OliveCliAdapter(runner)

    command_plan = [
        mobius.build_command(request, request.output_dir, no_weights=False),
    ]
    if not request.skip_olive:
        command_plan.append(
            olive.auto_opt_command(
                input_model_or_dir=request.output_dir,
                output_dir=request.output_dir / "olive-optimized",
                precision=request.candidate.recommended_olive_precision,
            )
        )

    dry_plan = job_runner.run_dry(request=request, commands=tuple(command_plan))
    _print_json(dry_plan)
    return 0 if dry_plan.preflight.ok else 2


def _run_hf_search(args: argparse.Namespace) -> int:
    adapter = HuggingFaceMetadataAdapter()
    results = adapter.search_models(query=args.query, limit=args.limit, sort="downloads")
    _print_json({"query": args.query, "results": results})
    return 0


def _run_hf_info(args: argparse.Namespace) -> int:
    adapter = HuggingFaceMetadataAdapter()
    metadata = adapter.get_metadata(
        model_id=args.model_id,
        revision=args.revision,
        files_metadata=bool(args.files_metadata),
    )
    _print_json(metadata)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "doctor":
        return _run_doctor(args)
    if args.command == "preflight":
        return _run_preflight(args)
    if args.command == "dry-run":
        return _run_dry(args)
    if args.command == "hf-search":
        return _run_hf_search(args)
    if args.command == "hf-info":
        return _run_hf_info(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive top-level CLI guard
        print(f"fl-onboarding fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1)
