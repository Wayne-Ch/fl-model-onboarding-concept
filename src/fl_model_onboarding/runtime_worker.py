from __future__ import annotations

import argparse
import json

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


def _foundry_infer(model_dir: Path, model_name: str, prompt: str, max_tokens: int) -> dict[str, object]:
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
    try:
        client = model.get_chat_client()
        if hasattr(client, "settings"):
            client.settings.max_tokens = max_tokens
            client.settings.temperature = 0.0
        response = client.complete_chat([{"role": "user", "content": prompt}])
        choices = getattr(response, "choices", [])
        if not choices:
            raise RuntimeError("Foundry Local SDK returned no chat choices.")
        message = getattr(choices[0], "message", None)
        output = getattr(message, "content", None)
        if not isinstance(output, str):
            output = str(response)
        return {"ok": True, "output": output}
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
    args = parser.parse_args(argv)

    try:
        if args.command == "validate-runtime":
            result = _validate_runtime(Path(args.model_dir).resolve())
        else:
            request = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
            result = _foundry_infer(
                Path(args.model_dir).resolve(),
                args.model_name,
                request["prompt"],
                int(request["max_tokens"]),
            )
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
