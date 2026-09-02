from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any


def _dtype_name(dtype: int) -> str:
    from onnx import TensorProto

    try:
        return str(TensorProto.DataType.Name(dtype))
    except Exception:
        return str(dtype)


def _collect_value_dtypes(model: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for tensor in model.graph.initializer:
        result[str(tensor.name)] = int(tensor.data_type)

    value_infos = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for value in value_infos:
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("elem_type"):
            result[str(value.name)] = int(tensor_type.elem_type)
    return result


def summarize_quantized_ops(model_path: Path, *, sample_limit: int = 5) -> dict[str, object]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    dtypes = _collect_value_dtypes(model)

    gather_nodes: list[dict[str, object]] = []
    matmul_nbits_nodes: list[dict[str, object]] = []
    dequantize_nodes: list[dict[str, object]] = []
    all_graph_outputs = [str(item.name) for item in model.graph.output]

    for node in model.graph.node:
        node_name = str(node.name)
        node_inputs = [str(item) for item in node.input]
        if node.op_type == "GatherBlockQuantized":
            summary: dict[str, object] = {"name": node_name, "inputs": node_inputs}
            if len(node_inputs) >= 3 and node_inputs[2] in dtypes:
                summary["scale_dtype"] = _dtype_name(dtypes[node_inputs[2]])
            gather_nodes.append(summary)
        elif node.op_type == "MatMulNBits":
            summary = {"name": node_name, "inputs": node_inputs}
            if len(node_inputs) >= 3 and node_inputs[2] in dtypes:
                summary["scale_dtype"] = _dtype_name(dtypes[node_inputs[2]])
            matmul_nbits_nodes.append(summary)
        elif node.op_type == "DequantizeLinear":
            dequantize_nodes.append({"name": node_name, "inputs": node_inputs})

    observed_scale_dtypes = {
        item["scale_dtype"]
        for item in gather_nodes + matmul_nbits_nodes
        if isinstance(item.get("scale_dtype"), str)
    }
    sorted_scale_dtypes = sorted(observed_scale_dtypes)
    has_bfloat16_scales = "BFLOAT16" in observed_scale_dtypes

    return {
        "model_onnx": str(model_path),
        "graph_output_count": len(all_graph_outputs),
        "graph_outputs_head": all_graph_outputs[: min(sample_limit, len(all_graph_outputs))],
        "op_counts": {
            "GatherBlockQuantized": len(gather_nodes),
            "MatMulNBits": len(matmul_nbits_nodes),
            "DequantizeLinear": len(dequantize_nodes),
        },
        "observed_scale_dtypes": sorted_scale_dtypes,
        "has_bfloat16_scales": has_bfloat16_scales,
        "sample_nodes": {
            "GatherBlockQuantized": gather_nodes[:sample_limit],
            "MatMulNBits": matmul_nbits_nodes[:sample_limit],
            "DequantizeLinear": dequantize_nodes[:sample_limit],
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ONNX quantized contrib-ops and scale dtypes for ORT/OGA runtime compatibility triage."
    )
    parser.add_argument("--model-onnx", required=True, help="Path to model.onnx.")
    parser.add_argument("--sample-limit", type=int, default=5, help="Sample node count per op type in JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = Path(args.model_onnx).resolve()
    payload = summarize_quantized_ops(model_path=model_path, sample_limit=max(1, int(args.sample_limit)))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
