from __future__ import annotations

import argparse
import json
import time

from pathlib import Path


def _validate_runtime(model_dir: Path) -> dict[str, object]:
    import onnx
    import onnxruntime as ort
    import onnxruntime_genai as og

    onnx_files = sorted(model_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found under {model_dir}")
    for model_path in onnx_files:
        onnx.checker.check_model(str(model_path))
    _ = ort.InferenceSession(
        str(onnx_files[0]),
        providers=["CPUExecutionProvider"],
    )
    model = og.Model(str(model_dir))
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=64)
    generator = og.Generator(model, params)
    generator.append_tokens(tokenizer.encode("Reply with: OK"))
    output = ""
    while not generator.is_done():
        generator.generate_next_token()
        output += tokenizer_stream.decode(generator.get_next_tokens()[0])
    return {
        "ok": True,
        "checks": [
            f"onnx_checker={len(onnx_files)}",
            "ort_cpu_load=passed",
            "oga_generation=passed",
        ],
        "output": output,
    }


def _load_foundry_chat_client(model_dir: Path, model_name: str) -> tuple[object, object]:
    from foundry_local_sdk import Configuration, FoundryLocalManager

    FoundryLocalManager.initialize(
        Configuration(
            app_name="fl-model-onboarding",
            model_cache_dir=str(model_dir.parent),
        )
    )
    manager = FoundryLocalManager.instance
    model = next(
        (candidate for candidate in manager.catalog.get_cached_models() if model_name in candidate.id),
        None,
    )
    if model is None:
        raise RuntimeError(f"Model '{model_name}' was not discovered in the configured cache.")
    model.load()
    client = model.get_chat_client()
    return model, client


def _coerce_output(response: object) -> str:
    choices = getattr(response, "choices", [])
    if not choices:
        raise RuntimeError("Foundry Local SDK returned no chat choices.")
    message = getattr(choices[0], "message", None)
    output = getattr(message, "content", None)
    if isinstance(output, str):
        return output
    return str(response)


def _run_foundry_chat(client: object, *, prompt: str, max_tokens: int) -> str:
    if hasattr(client, "settings"):
        client.settings.max_tokens = max_tokens
        client.settings.temperature = 0.0
    response = client.complete_chat([{"role": "user", "content": prompt}])
    return _coerce_output(response)


def _coerce_batch_timeout(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive number when provided.")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be greater than zero when provided.")
    return numeric


def _coerce_batch_request_prompts(request: object) -> tuple[dict[str, object], ...]:
    if not isinstance(request, dict):
        raise ValueError("Batch request payload must be a JSON object.")
    prompts_raw = request.get("prompts")
    if not isinstance(prompts_raw, list) or not prompts_raw:
        raise ValueError("Batch request payload must include a non-empty 'prompts' array.")
    prompts: list[dict[str, object]] = []
    for index, row in enumerate(prompts_raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Batch prompt #{index} must be an object.")
        prompt_id = row.get("prompt_id")
        prompt_text = row.get("prompt")
        max_tokens = row.get("max_tokens")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"Batch prompt #{index} is missing required non-empty string 'prompt_id'.")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(
                f"Batch prompt '{prompt_id}' is missing required non-empty string 'prompt'."
            )
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(
                f"Batch prompt '{prompt_id}' must include a positive integer 'max_tokens'."
            )
        prompts.append(
            {
                "prompt_id": prompt_id.strip(),
                "prompt": prompt_text,
                "max_tokens": max_tokens,
            }
        )
    return tuple(prompts)


def _foundry_infer(model_dir: Path, model_name: str, prompt: str, max_tokens: int) -> dict[str, object]:
    model, client = _load_foundry_chat_client(model_dir, model_name)
    try:
        output = _run_foundry_chat(client, prompt=prompt, max_tokens=max_tokens)
        return {"ok": True, "output": output}
    finally:
        model.unload()


def _foundry_infer_batch(model_dir: Path, model_name: str, request: object) -> dict[str, object]:
    if not isinstance(request, dict):
        raise ValueError("Batch request payload must be a JSON object.")
    prompts = _coerce_batch_request_prompts(request)
    per_prompt_timeout_seconds = _coerce_batch_timeout(
        request.get("per_prompt_timeout_seconds"),
        field_name="per_prompt_timeout_seconds",
    )
    batch_timeout_seconds = _coerce_batch_timeout(
        request.get("batch_timeout_seconds"),
        field_name="batch_timeout_seconds",
    )
    started = time.monotonic()
    results: list[dict[str, object]] = []
    completed_prompt_ids: list[str] = []
    model, client = _load_foundry_chat_client(model_dir, model_name)
    try:
        for prompt in prompts:
            prompt_id = str(prompt["prompt_id"])
            prompt_text = str(prompt["prompt"])
            max_tokens = int(prompt["max_tokens"])
            elapsed = time.monotonic() - started
            if batch_timeout_seconds is not None and elapsed >= batch_timeout_seconds:
                return {
                    "ok": False,
                    "error": (
                        f"Batch quality inference exceeded deadline before prompt '{prompt_id}' "
                        f"after {elapsed:.3f}s (limit {batch_timeout_seconds:.3f}s)."
                    ),
                    "failure_stage": "batch_timeout",
                    "failed_prompt_id": prompt_id,
                    "completed_prompt_ids": completed_prompt_ids,
                    "results": results,
                    "duration_seconds": round(elapsed, 3),
                }
            prompt_started = time.monotonic()
            try:
                output = _run_foundry_chat(client, prompt=prompt_text, max_tokens=max_tokens)
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"Batch quality inference prompt '{prompt_id}' failed: {exc}",
                    "failure_stage": "prompt_execution",
                    "failed_prompt_id": prompt_id,
                    "completed_prompt_ids": completed_prompt_ids,
                    "results": results,
                    "duration_seconds": round(time.monotonic() - started, 3),
                }
            prompt_elapsed = time.monotonic() - prompt_started
            row: dict[str, object] = {
                "prompt_id": prompt_id,
                "output": output,
                "duration_seconds": round(prompt_elapsed, 3),
                "timed_out": False,
            }
            results.append(row)
            completed_prompt_ids.append(prompt_id)
            if per_prompt_timeout_seconds is not None and prompt_elapsed > per_prompt_timeout_seconds:
                row["timed_out"] = True
                return {
                    "ok": False,
                    "error": (
                        f"Batch quality inference prompt '{prompt_id}' exceeded timeout "
                        f"after {prompt_elapsed:.3f}s (limit {per_prompt_timeout_seconds:.3f}s)."
                    ),
                    "failure_stage": "prompt_timeout",
                    "failed_prompt_id": prompt_id,
                    "completed_prompt_ids": completed_prompt_ids,
                    "results": results,
                    "duration_seconds": round(time.monotonic() - started, 3),
                }
        return {
            "ok": True,
            "results": results,
            "completed_prompt_ids": completed_prompt_ids,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        model.unload()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-runtime")
    validate.add_argument("--model-dir", required=True)
    infer = sub.add_parser("foundry-infer")
    infer.add_argument("--model-dir", required=True)
    infer.add_argument("--model-name", required=True)
    infer.add_argument("--request-file", required=True)
    infer_batch = sub.add_parser("foundry-infer-batch")
    infer_batch.add_argument("--model-dir", required=True)
    infer_batch.add_argument("--model-name", required=True)
    infer_batch.add_argument("--request-file", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "validate-runtime":
            result = _validate_runtime(Path(args.model_dir).resolve())
        elif args.command == "foundry-infer":
            request = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
            result = _foundry_infer(
                Path(args.model_dir).resolve(),
                args.model_name,
                str(request["prompt"]),
                int(request["max_tokens"]),
            )
        else:
            request = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
            result = _foundry_infer_batch(
                Path(args.model_dir).resolve(),
                args.model_name,
                request,
            )
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
