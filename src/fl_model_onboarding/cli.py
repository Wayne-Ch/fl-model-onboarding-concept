from __future__ import annotations

import argparse
import json
import sys
import webbrowser

from pathlib import Path
from typing import Callable

from .architecture_capabilities import ResolutionOutcome, load_architecture_capability_registry, normalize_huggingface_metadata
from .contracts import CandidateModality
from .local_api import create_app
from .local_service import BuildSubmission, LocalOnboardingService, ServiceError, enforce_loopback_host
from .recipe_compiler import (
    GeneratedRecipeCompileError,
    RecipeCompilerInput,
    RecipeCompilerToolchain,
    compile_generated_recipe,
)
from .serialization import to_jsonable
from .version import __version__

_DEFAULT_FROZEN_SET_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation" / "recipe-agent-v1" / "models.json"
)
_DRY_RUN_TOOLCHAIN = RecipeCompilerToolchain(
    mobius_version="0.1.0",
    olive_version="0.13.0",
    onnx_version="1.22.0",
    ort_version="1.29.0",
    oga_version="0.15.2",
    foundry_sdk_version="1.2.4",
    foundry_cli_version="0.11.0",
)


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
        enable_production_runner=bool(getattr(args, "enable_production_runner", False)),
    )


def _submission_from_args(args: argparse.Namespace) -> BuildSubmission:
    return BuildSubmission(
        model_id=args.model_id,
        task=CandidateModality(args.task),
        task_profile=args.task_profile,
        hf_revision=args.hf_revision,
        skip_olive=bool(args.skip_olive),
        allow_experimental=bool(getattr(args, "allow_experimental", False)),
        optimization_strategy=getattr(args, "optimization_strategy", None),
        optimization_precision=getattr(args, "optimization_precision", None),
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


def _load_frozen_set(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen set manifest must be a JSON object.")
    return payload


def _validate_frozen_manifest(payload: dict[str, object]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    models = payload.get("models")
    if not isinstance(models, list):
        errors.append("models must be an array.")
        return False, errors
    if len(models) != 5:
        errors.append(f"models must contain exactly five entries; found {len(models)}.")
    for index, row in enumerate(models, start=1):
        if not isinstance(row, dict):
            errors.append(f"models[{index}] must be an object.")
            continue
        model_id = row.get("model_id")
        sha = row.get("sha")
        model_type = row.get("model_type")
        architectures = row.get("architectures")
        if not isinstance(model_id, str) or not model_id.strip():
            errors.append(f"models[{index}].model_id must be a non-empty string.")
        if not isinstance(sha, str) or len(sha) != 40:
            errors.append(f"models[{index}].sha must be a full 40-character commit SHA.")
        if not isinstance(model_type, str) or not model_type.strip():
            errors.append(f"models[{index}].model_type must be a non-empty string.")
        if not isinstance(architectures, list) or not all(isinstance(item, str) for item in architectures):
            errors.append(f"models[{index}].architectures must be an array of strings.")
    return not errors, errors


def _run_recipe_agent_frozen_list(args: argparse.Namespace) -> int:
    try:
        path = Path(args.path).resolve() if args.path else _DEFAULT_FROZEN_SET_PATH
        payload = _load_frozen_set(path)
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError("Frozen set models array is missing.")
        _print_json(
            {
                "manifest_path": str(path),
                "schema_version": payload.get("schema_version"),
                "selection_timestamp": payload.get("selection_timestamp"),
                "count": len(models),
                "models": [
                    {
                        "index": row.get("index"),
                        "model_id": row.get("model_id"),
                        "sha": row.get("sha"),
                        "model_type": row.get("model_type"),
                        "architectures": row.get("architectures"),
                        "catalog_match": row.get("catalog_match"),
                    }
                    for row in models
                    if isinstance(row, dict)
                ],
            }
        )
        return 0
    except Exception as exc:
        print(json.dumps({"code": "FROZEN_SET_LOAD_FAILED", "message": str(exc)}), file=sys.stderr)
        return 2


def _run_recipe_agent_frozen_validate(args: argparse.Namespace) -> int:
    try:
        path = Path(args.path).resolve() if args.path else _DEFAULT_FROZEN_SET_PATH
        payload = _load_frozen_set(path)
        valid, errors = _validate_frozen_manifest(payload)
        _print_json(
            {
                "manifest_path": str(path),
                "valid": valid,
                "errors": errors,
                "model_count": len(payload.get("models", [])) if isinstance(payload.get("models"), list) else 0,
            }
        )
        return 0 if valid else 2
    except Exception as exc:
        print(json.dumps({"code": "FROZEN_SET_VALIDATE_FAILED", "message": str(exc)}), file=sys.stderr)
        return 2


def _run_recipe_agent_frozen_dry_run(args: argparse.Namespace) -> int:
    try:
        path = Path(args.path).resolve() if args.path else _DEFAULT_FROZEN_SET_PATH
        payload = _load_frozen_set(path)
        valid, errors = _validate_frozen_manifest(payload)
        if not valid:
            _print_json({"manifest_path": str(path), "valid": False, "errors": errors})
            return 2
        models = payload["models"]
        assert isinstance(models, list)
        capability_registry = load_architecture_capability_registry()

        outcomes: list[dict[str, object]] = []
        for row in models:
            assert isinstance(row, dict)
            model_id = str(row["model_id"])
            revision_sha = str(row["sha"]).lower()
            model_type = str(row["model_type"])
            architectures_raw = row.get("architectures")
            architectures = (
                tuple(str(item) for item in architectures_raw)
                if isinstance(architectures_raw, list)
                else ()
            )
            catalog_match = row.get("catalog_match")
            catalog_present = (
                isinstance(catalog_match, dict)
                and bool(catalog_match.get("matched"))
            )
            normalized_metadata = normalize_huggingface_metadata(
                model_id=model_id,
                config={"model_type": model_type, "architectures": list(architectures)},
                is_gated=False,
                is_private=False,
            )
            capability_resolution = capability_registry.resolve(
                metadata=normalized_metadata,
                task=CandidateModality.LLM.value,
                device="cpu",
                requested_precision="auto",
            )
            outcome: dict[str, object] = {
                "model_id": model_id,
                "sha": revision_sha,
                "model_type": model_type,
                "architectures": list(architectures),
                "catalog_present": catalog_present,
                "capability": {
                    "outcome": capability_resolution.outcome.value,
                    "reason_code": capability_resolution.reason_code.value,
                    "reason": capability_resolution.reason,
                    "matched_aliases": list(capability_resolution.matched_aliases),
                    "capability_id": (
                        capability_resolution.capability.capability_id
                        if capability_resolution.capability is not None
                        else None
                    ),
                    "status": (
                        capability_resolution.capability.status.value
                        if capability_resolution.capability is not None
                        else None
                    ),
                },
                "tool_execution": "not-started",
            }
            if catalog_present:
                outcome["status"] = "blocked"
                outcome["reason"] = "catalog-present"
                outcomes.append(outcome)
                continue

            try:
                candidate = compile_generated_recipe(
                    RecipeCompilerInput(
                        model_id=model_id,
                        revision_sha=revision_sha,
                        model_type=model_type,
                        architectures=architectures,
                        task=CandidateModality.LLM.value,
                        requested_device="cpu",
                        requested_precision="auto",
                        is_gated=False,
                        requires_remote_code=False,
                        config_files=("config.json",),
                        tokenizer_files=("tokenizer.json",),
                        available_files=("config.json", "tokenizer.json"),
                        capability_resolution=capability_resolution,
                        toolchain=_DRY_RUN_TOOLCHAIN,
                    )
                )
            except GeneratedRecipeCompileError as exc:
                outcome["status"] = (
                    "capability-gap"
                    if capability_resolution.outcome != ResolutionOutcome.EXACT
                    else "compile-failed"
                )
                outcome["reason"] = str(exc)
                outcomes.append(outcome)
                continue

            outcome["status"] = "compiled"
            outcome["fingerprint"] = candidate.fingerprint
            outcome["recipe_id"] = candidate.recipe.id
            outcome["recipe_version"] = candidate.recipe.version
            outcome["recipe_status"] = candidate.recipe.status.value
            outcomes.append(outcome)

        compiled = sum(1 for row in outcomes if row.get("status") == "compiled")
        _print_json(
            {
                "manifest_path": str(path),
                "valid": True,
                "tool_execution": "disabled",
                "summary": {
                    "total_models": len(outcomes),
                    "compiled_models": compiled,
                    "non_compiled_models": len(outcomes) - compiled,
                },
                "outcomes": outcomes,
            }
        )
        return 0
    except Exception as exc:
        print(json.dumps({"code": "FROZEN_SET_DRY_RUN_FAILED", "message": str(exc)}), file=sys.stderr)
        return 2


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
    doctor.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Opt in to preflight/build for recipes marked experimental.",
    )

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
    model_preflight.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Opt in to preflight/build for recipes marked experimental.",
    )

    build = sub.add_parser("build", help="Build job operations.")
    build_sub = build.add_subparsers(dest="build_command", required=True)

    build_create = build_sub.add_parser("create", help="Create a build job.")
    _add_storage_options(build_create)
    build_create.add_argument("--model-id", required=True)
    build_create.add_argument("--task", choices=("llm", "asr"), required=True)
    build_create.add_argument("--task-profile", default="default")
    build_create.add_argument("--hf-revision")
    build_create.add_argument("--skip-olive", action="store_true")
    build_create.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Opt in to build profiles marked experimental.",
    )
    build_create.add_argument("--optimization-strategy")
    build_create.add_argument("--optimization-precision")
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
    serve.add_argument(
        "--enable-production-runner",
        action="store_true",
        help="Enable the verified SmolLM2 Mobius/Olive/Foundry Local execution pipeline.",
    )

    recipe_agent = sub.add_parser("recipe-agent", help="Recipe Agent v1 utilities.")
    recipe_agent_sub = recipe_agent.add_subparsers(dest="recipe_agent_command", required=True)

    frozen_list = recipe_agent_sub.add_parser(
        "frozen-list",
        help="List frozen Recipe Agent v1 evaluation models.",
    )
    frozen_list.add_argument(
        "--path",
        help="Path to evaluation/recipe-agent-v1/models.json (defaults to repository frozen set).",
    )

    frozen_validate = recipe_agent_sub.add_parser(
        "frozen-validate",
        help="Validate frozen Recipe Agent v1 evaluation manifest.",
    )
    frozen_validate.add_argument(
        "--path",
        help="Path to evaluation/recipe-agent-v1/models.json (defaults to repository frozen set).",
    )

    frozen_dry_run = recipe_agent_sub.add_parser(
        "frozen-dry-run",
        help="Dry-run deterministic candidate compilation for all frozen models (no build/download).",
    )
    frozen_dry_run.add_argument(
        "--path",
        help="Path to evaluation/recipe-agent-v1/models.json (defaults to repository frozen set).",
    )

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
    if args.command == "recipe-agent" and args.recipe_agent_command == "frozen-list":
        return _run_recipe_agent_frozen_list(args)
    if args.command == "recipe-agent" and args.recipe_agent_command == "frozen-validate":
        return _run_recipe_agent_frozen_validate(args)
    if args.command == "recipe-agent" and args.recipe_agent_command == "frozen-dry-run":
        return _run_recipe_agent_frozen_dry_run(args)
    parser.error("Unsupported command.")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level fatal guard
        print(f"fl-onboarding fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1)
