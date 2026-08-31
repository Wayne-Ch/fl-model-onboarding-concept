from __future__ import annotations

import argparse
import json
import sys
import webbrowser

from pathlib import Path
from typing import Callable

from .contracts import CandidateModality
from .local_api import create_app
from .local_service import BuildSubmission, LocalOnboardingService, ServiceError, enforce_loopback_host
from .serialization import to_jsonable
from .version import __version__


def _print_json(value: object) -> None:
    print(json.dumps(to_jsonable(value), indent=2))


def _print_error(exc: ServiceError) -> None:
    payload: dict[str, object] = {"code": exc.code, "message": str(exc)}
    if exc.detail:
        payload["detail"] = exc.detail
    print(json.dumps(payload), file=sys.stderr)


def _exit_code_for_service_error(exc: ServiceError) -> int:
    return 3 if 400 <= exc.status_code < 500 else 4


def _add_storage_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-path", help="SQLite state path.")
    parser.add_argument("--workspace-base", help="Workspace root for per-job output directories.")
    parser.add_argument("--model-cache-dir", help="Foundry model cache root.")


def _service_from_args(args: argparse.Namespace) -> LocalOnboardingService:
    return LocalOnboardingService(
        db_path=Path(args.db_path).resolve() if getattr(args, "db_path", None) else None,
        workspace_base=Path(args.workspace_base).resolve()
        if getattr(args, "workspace_base", None)
        else None,
        model_cache_dir=Path(args.model_cache_dir).resolve()
        if getattr(args, "model_cache_dir", None)
        else None,
    )


def _submission_from_args(args: argparse.Namespace) -> BuildSubmission:
    return BuildSubmission(
        model_id=args.model_id,
        task=CandidateModality(args.task),
        task_profile=args.task_profile,
        hf_revision=args.hf_revision,
        skip_olive=bool(args.skip_olive),
    )


def _run_with_service(
    args: argparse.Namespace,
    fn: Callable[[LocalOnboardingService], int],
) -> int:
    service = _service_from_args(args)
    try:
        return fn(service)
    except ServiceError as exc:
        _print_error(exc)
        return _exit_code_for_service_error(exc)
    finally:
        service.close()


def _run_doctor(args: argparse.Namespace) -> int:
    def run(service: LocalOnboardingService) -> int:
        submission = _submission_from_args(args)
        preflight = service.preflight(submission)
        payload = {
            "cli_version": __version__,
            "health": service.health(),
            "preflight": preflight,
        }
        _print_json(payload)
        return 0 if preflight.get("ok") else 2

    return _run_with_service(args, run)


def _run_model_search(args: argparse.Namespace) -> int:
    def run(service: LocalOnboardingService) -> int:
        _print_json(service.search_models(query=args.query, limit=args.limit))
        return 0

    return _run_with_service(args, run)


def _run_model_detail(args: argparse.Namespace) -> int:
    def run(service: LocalOnboardingService) -> int:
        _print_json(service.model_detail(model_id=args.model_id))
        return 0

    return _run_with_service(args, run)


def _run_model_preflight(args: argparse.Namespace) -> int:
    def run(service: LocalOnboardingService) -> int:
        result = service.preflight(_submission_from_args(args))
        _print_json(result)
        return 0 if result.get("ok") else 2

    return _run_with_service(args, run)


def _run_build_create(args: argparse.Namespace) -> int:
    def run(service: LocalOnboardingService) -> int:
        job, replay = service.create_build(
            _submission_from_args(args),
            idempotency_key=args.idempotency_key,
        )
        _print_json({"idempotent_replay": replay, "job": job})
        return 0

    return _run_with_service(args, run)


def _run_build_status(args: argparse.Namespace) -> int:
    def run(service: LocalOnboardingService) -> int:
        _print_json(service.get_build(job_id=args.job_id))
        return 0

    return _run_with_service(args, run)


def _run_build_cancel(args: argparse.Namespace) -> int:
    def run(service: LocalOnboardingService) -> int:
        job, quarantine = service.cancel_build(job_id=args.job_id, reason=args.reason)
        payload = {"job": job}
        if quarantine is not None:
            payload["quarantine_path"] = str(quarantine)
        _print_json(payload)
        return 0

    return _run_with_service(args, run)


def _run_service_serve(args: argparse.Namespace) -> int:
    try:
        warning = enforce_loopback_host(host=args.host, allow_non_loopback=bool(args.allow_non_loopback))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    service = _service_from_args(args)
    try:
        if warning:
            print(warning, file=sys.stderr)
        try:
            import uvicorn
        except ModuleNotFoundError as exc:
            print(f"Missing runtime dependency: {exc}", file=sys.stderr)
            return 4

        app = create_app(service=service)
        if args.open_browser:
            webbrowser.open(f"http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return 0
    finally:
        service.close()


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fl-onboarding")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print CLI version.")

    doctor = sub.add_parser("doctor", help="Run health and preflight diagnostics.")
    _add_storage_options(doctor)
    doctor.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    doctor.add_argument("--task", choices=("llm", "asr"), default="llm")
    doctor.add_argument("--task-profile", default="default")
    doctor.add_argument("--hf-revision")
    doctor.add_argument("--skip-olive", action="store_true")

    model = sub.add_parser("model", help="Model operations.")
    model_sub = model.add_subparsers(dest="model_command", required=True)

    model_search = model_sub.add_parser("search", help="Search Hugging Face models.")
    _add_storage_options(model_search)
    model_search.add_argument("--query", required=True)
    model_search.add_argument("--limit", type=int, default=20)

    model_detail = model_sub.add_parser("detail", help="Get model detail and buildability.")
    _add_storage_options(model_detail)
    model_detail.add_argument("--model-id", required=True)

    model_preflight = model_sub.add_parser("preflight", help="Run idempotent preflight.")
    _add_storage_options(model_preflight)
    model_preflight.add_argument("--model-id", required=True)
    model_preflight.add_argument("--task", choices=("llm", "asr"), required=True)
    model_preflight.add_argument("--task-profile", default="default")
    model_preflight.add_argument("--hf-revision")
    model_preflight.add_argument("--skip-olive", action="store_true")

    build = sub.add_parser("build", help="Build job operations.")
    build_sub = build.add_subparsers(dest="build_command", required=True)

    build_create = build_sub.add_parser("create", help="Create a build job.")
    _add_storage_options(build_create)
    build_create.add_argument("--model-id", required=True)
    build_create.add_argument("--task", choices=("llm", "asr"), required=True)
    build_create.add_argument("--task-profile", default="default")
    build_create.add_argument("--hf-revision")
    build_create.add_argument("--skip-olive", action="store_true")
    build_create.add_argument("--idempotency-key", required=True)

    build_status = build_sub.add_parser("status", help="Get build status.")
    _add_storage_options(build_status)
    build_status.add_argument("--job-id", required=True)

    build_cancel = build_sub.add_parser("cancel", help="Cancel a build.")
    _add_storage_options(build_cancel)
    build_cancel.add_argument("--job-id", required=True)
    build_cancel.add_argument("--reason", default="Cancelled by client request.")

    service = sub.add_parser("service", help="Local service operations.")
    service_sub = service.add_subparsers(dest="service_command", required=True)
    serve = service_sub.add_parser("serve", help="Run local API server.")
    _add_storage_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8777)
    serve.add_argument("--allow-non-loopback", action="store_true")
    serve.add_argument("--open-browser", action="store_true")
    serve.add_argument("--log-level", default="info")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "doctor":
        return _run_doctor(args)
    if args.command == "model" and args.model_command == "search":
        return _run_model_search(args)
    if args.command == "model" and args.model_command == "detail":
        return _run_model_detail(args)
    if args.command == "model" and args.model_command == "preflight":
        return _run_model_preflight(args)
    if args.command == "build" and args.build_command == "create":
        return _run_build_create(args)
    if args.command == "build" and args.build_command == "status":
        return _run_build_status(args)
    if args.command == "build" and args.build_command == "cancel":
        return _run_build_cancel(args)
    if args.command == "service" and args.service_command == "serve":
        return _run_service_serve(args)
    parser.error("Unsupported command.")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level fatal guard
        print(f"fl-onboarding fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1)

