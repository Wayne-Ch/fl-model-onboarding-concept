from __future__ import annotations

import argparse
import copy
import json
import re

from pathlib import Path
from typing import Mapping, Sequence

_QUANTIZED_OUTPUT_RE = re.compile(r"^(?P<base>.+)_Q(?P<bits>\d+)$", re.IGNORECASE)


def build_decoder_output_reconciliation(
    graph_outputs: Sequence[str],
    decoder_outputs: Mapping[str, str],
) -> dict[str, object]:
    available = tuple(dict.fromkeys(str(name) for name in graph_outputs))
    present: dict[str, str] = {}
    proposed_remap: dict[str, str] = {}
    unresolved: dict[str, dict[str, object]] = {}

    for logical_name, physical_name_raw in decoder_outputs.items():
        physical_name = str(physical_name_raw)
        if physical_name in available:
            present[str(logical_name)] = physical_name
            continue

        candidates = []
        for output_name in available:
            if output_name == physical_name:
                candidates.append(output_name)
                continue
            match = _QUANTIZED_OUTPUT_RE.fullmatch(output_name)
            if match and match.group("base") == physical_name:
                candidates.append(output_name)

        if len(candidates) == 1:
            proposed_remap[str(logical_name)] = candidates[0]
        else:
            unresolved[str(logical_name)] = {
                "requested_output": physical_name,
                "candidates": candidates,
            }

    resolved_outputs = {str(key): str(value) for key, value in decoder_outputs.items()}
    resolved_outputs.update(proposed_remap)

    return {
        "graph_output_count": len(available),
        "present": present,
        "proposed_remap": proposed_remap,
        "unresolved": unresolved,
        "is_safe_to_apply": len(unresolved) == 0,
        "resolved_decoder_outputs": resolved_outputs,
    }


def reconcile_genai_payload(
    payload: Mapping[str, object],
    graph_outputs: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    updated_payload = copy.deepcopy(dict(payload))
    model = updated_payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("genai_config.json is missing 'model' object.")

    decoder = model.get("decoder")
    if not isinstance(decoder, dict):
        raise ValueError("genai_config.json is missing 'model.decoder' object.")

    outputs = decoder.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("genai_config.json is missing 'model.decoder.outputs' object.")

    decoder_output_map: dict[str, str] = {}
    for key, value in outputs.items():
        if not isinstance(value, str):
            continue
        decoder_output_map[str(key)] = value

    report = build_decoder_output_reconciliation(graph_outputs=graph_outputs, decoder_outputs=decoder_output_map)
    if not report["is_safe_to_apply"]:
        return updated_payload, report

    decoder["outputs"] = report["resolved_decoder_outputs"]  # type: ignore[index]
    return updated_payload, report


def load_onnx_graph_outputs(model_path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return [str(item.name) for item in model.graph.output]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile genai_config decoder output mapping against ONNX graph outputs."
    )
    parser.add_argument("--model-onnx", required=True, help="Path to ONNX model file.")
    parser.add_argument("--genai-config", required=True, help="Path to genai_config.json.")
    parser.add_argument(
        "--write-updated-config",
        default=None,
        help="Optional output path for reconciled genai_config.json. Fails closed when unresolved mappings remain.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = Path(args.model_onnx).resolve()
    config_path = Path(args.genai_config).resolve()
    output_path = Path(args.write_updated_config).resolve() if args.write_updated_config else None

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    graph_outputs = load_onnx_graph_outputs(model_path)
    updated_payload, report = reconcile_genai_payload(payload=payload, graph_outputs=graph_outputs)
    report_with_paths = {
        "model_onnx": str(model_path),
        "genai_config": str(config_path),
        **report,
    }

    if output_path is not None:
        if not report["is_safe_to_apply"]:
            print(json.dumps(report_with_paths, indent=2))
            raise SystemExit(
                "Refusing to write updated config because unresolved output mappings remain. "
                "Review candidates and rerun with an unambiguous mapping."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(updated_payload, indent=2), encoding="utf-8")
        report_with_paths["updated_config"] = str(output_path)

    print(json.dumps(report_with_paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
