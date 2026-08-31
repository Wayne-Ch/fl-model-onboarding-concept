from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

OFFICIAL_OGA_VERSION = "0.15.2"
OFFICIAL_OGA_TAG = "v0.15.2"
OFFICIAL_WHISPER_REFERENCE_FILES = {
    "added_tokens.json",
    "audio_processor_config.json",
    "genai_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}

# From onnxruntime-genai/src/config.cpp (v0.15.2), DecoderInputs_Element::OnValue.
OGA_DECODER_INPUT_KEYS = {
    "input_ids",
    "inputs_embeds",
    "attention_mask",
    "position_ids",
    "past_key_names",
    "past_value_names",
    "past_names",
    "cross_past_key_names",
    "cross_past_value_names",
    "past_sequence_length",
    "current_sequence_length",
    "total_sequence_length",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "rnn_states_prev",
    "past_key_values_length",
    "cache_indirection",
    "cumulative_sequence_lengths",
    "past_sequence_lengths",
    "block_table",
    "past_conv_names",
    "targets",
    "lstm_hidden_state",
    "lstm_cell_state",
    "per_layer_inputs",
    "targets_length",
}

# From onnxruntime-genai/src/config.cpp (v0.15.2), DecoderOutputs_Element::OnValue.
OGA_DECODER_OUTPUT_KEYS = {
    "logits",
    "present_key_names",
    "present_value_names",
    "present_names",
    "output_cross_qk_names",
    "rnn_states",
    "present_conv_names",
    "outputs",
    "lstm_hidden_state",
    "lstm_cell_state",
    "outputs_length",
}

# From onnxruntime-genai/src/config.cpp (v0.15.2), EncoderInputs_Element::OnValue.
OGA_ENCODER_INPUT_KEYS = {
    "input_ids",
    "inputs_embeds",
    "attention_mask",
    "position_ids",
    "audio_features",
    "input_lengths",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_len",
    "lang_id",
}

# From onnxruntime-genai/src/config.cpp (v0.15.2), EncoderOutputs_Element::OnValue.
OGA_ENCODER_OUTPUT_KEYS = {
    "encoder_hidden_states",
    "encoder_outputs",
    "output_lengths",
    "cache_last_channel_next",
    "cache_last_time_next",
    "cache_last_channel_len_next",
    "cross_present_key_names",
    "cross_present_value_names",
}

PAST_KEY_REGEX = re.compile(r"past_key_values\.(\d+)\.key$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hardlink_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for node in sorted(source.rglob("*")):
        relative = node.relative_to(source)
        target = destination / relative
        if node.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(node, target)
        except OSError:
            shutil.copy2(node, target)


def collect_file_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                }
            )
    return rows


def _tensor_shape(value_info: Any) -> list[str | int]:
    tensor = value_info.type.tensor_type
    dims: list[str | int] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append("?")
    return dims


def _extract_position_limit(initializers: list[Any]) -> int | None:
    for initializer in initializers:
        if "embed_positions.weight" in initializer.name and len(initializer.dims) >= 1:
            try:
                return int(initializer.dims[0])
            except (TypeError, ValueError):
                return None
    return None


def inspect_onnx_graph(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [item for item in model.graph.input if item.name not in initializer_names]
    graph_outputs = list(model.graph.output)
    external_initializers = [
        initializer for initializer in model.graph.initializer if initializer.data_location == onnx.TensorProto.EXTERNAL
    ]
    external_files = sorted(
        {
            entry.value
            for initializer in external_initializers
            for entry in initializer.external_data
            if entry.key == "location"
        }
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    return {
        "path": str(path),
        "inputs": [
            {
                "name": item.name,
                "elem_type": int(item.type.tensor_type.elem_type),
                "shape": _tensor_shape(item),
            }
            for item in graph_inputs
        ],
        "outputs": [
            {
                "name": item.name,
                "elem_type": int(item.type.tensor_type.elem_type),
                "shape": _tensor_shape(item),
            }
            for item in graph_outputs
        ],
        "input_names": [item.name for item in session.get_inputs()],
        "output_names": [item.name for item in session.get_outputs()],
        "initializer_count": len(model.graph.initializer),
        "external_initializer_count": len(external_initializers),
        "external_data_files": external_files,
        "position_limit": _extract_position_limit(list(model.graph.initializer)),
    }


def infer_decoder_layer_count(decoder_graph: dict[str, Any], fallback: int = 0) -> int:
    seen_indices: set[int] = set()
    for name in decoder_graph.get("input_names", []):
        match = PAST_KEY_REGEX.match(name)
        if match:
            seen_indices.add(int(match.group(1)))
    if seen_indices:
        return max(seen_indices) + 1
    return fallback


def _expand_template(name_template: str, count: int) -> list[str]:
    names: list[str] = []
    if not name_template or count <= 0:
        return names
    for index in range(count):
        names.append(name_template % index)
    return names


def _diff(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        rows: list[dict[str, Any]] = []
        keys = sorted(set(before.keys()) | set(after.keys()))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                rows.append({"path": path, "before": None, "after": after[key]})
            elif key not in after:
                rows.append({"path": path, "before": before[key], "after": None})
            else:
                rows.extend(_diff(before[key], after[key], path))
        return rows
    if before != after:
        return [{"path": prefix, "before": before, "after": after}]
    return []


def apply_minimal_parser_fix(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapted = copy.deepcopy(config)
    before = copy.deepcopy(adapted)

    decoder_inputs = (
        adapted.setdefault("model", {})
        .setdefault("decoder", {})
        .setdefault("inputs", {})
    )
    if "decoder_input_ids" in decoder_inputs and "input_ids" not in decoder_inputs:
        decoder_inputs["input_ids"] = decoder_inputs.pop("decoder_input_ids")

    return adapted, _diff(before, adapted)


def apply_full_adapter(
    config: dict[str, Any],
    encoder_graph: dict[str, Any],
    decoder_graph: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    minimally_adapted, _ = apply_minimal_parser_fix(config)
    adapted = copy.deepcopy(minimally_adapted)
    before = copy.deepcopy(adapted)

    model = adapted.setdefault("model", {})
    decoder = model.setdefault("decoder", {})
    decoder_inputs = decoder.setdefault("inputs", {})
    decoder.setdefault("filename", "decoder/model.onnx")

    decoder_input_names = set(decoder_graph.get("input_names", []))
    if "input_ids" not in decoder_inputs:
        decoder_inputs["input_ids"] = "input_ids"
    if decoder_inputs["input_ids"] not in decoder_input_names and "decoder_input_ids" in decoder_input_names:
        decoder_inputs["input_ids"] = "decoder_input_ids"

    decoder_layers = int(decoder.get("num_hidden_layers") or 0)
    if decoder_layers <= 0:
        decoder_layers = infer_decoder_layer_count(decoder_graph, fallback=2)
        decoder["num_hidden_layers"] = decoder_layers

    encoder_input_name = encoder_graph.get("input_names", ["input_features"])[0]
    encoder_output_name = encoder_graph.get("output_names", ["encoder_hidden_states"])[0]
    hidden_size = 1024
    encoder_outputs = encoder_graph.get("outputs", [])
    if encoder_outputs and len(encoder_outputs[0].get("shape", [])) >= 3:
        maybe_hidden = encoder_outputs[0]["shape"][2]
        if isinstance(maybe_hidden, int) and maybe_hidden > 0:
            hidden_size = maybe_hidden

    decoder_inputs_by_name = {item["name"]: item for item in decoder_graph.get("inputs", [])}
    past_key = decoder_inputs_by_name.get("past_key_values.0.key")
    inferred_heads = int(decoder.get("num_attention_heads") or 0) or 16
    inferred_head_size = int(decoder.get("head_size") or 0) or 64
    inferred_kv_heads = int(decoder.get("num_key_value_heads") or 0) or inferred_heads
    if past_key:
        shape = past_key.get("shape", [])
        if len(shape) >= 4:
            if isinstance(shape[1], int) and shape[1] > 0:
                inferred_kv_heads = shape[1]
                inferred_heads = shape[1]
            if isinstance(shape[3], int) and shape[3] > 0:
                inferred_head_size = shape[3]

    model["encoder"] = {
        "session_options": {
            "log_id": "onnxruntime-genai",
            "provider_options": [],
        },
        "filename": "encoder/model.onnx",
        "head_size": inferred_head_size,
        "hidden_size": hidden_size,
        "inputs": {
            "audio_features": encoder_input_name,
        },
        "outputs": {
            "encoder_hidden_states": encoder_output_name,
        },
        "num_attention_heads": inferred_heads,
        "num_hidden_layers": decoder_layers,
        "num_key_value_heads": inferred_kv_heads,
    }

    position_limit = decoder_graph.get("position_limit")
    if isinstance(position_limit, int) and position_limit > 0:
        context_length = model.get("context_length")
        if not isinstance(context_length, int) or context_length <= 0 or context_length > position_limit:
            model["context_length"] = position_limit
        search = adapted.setdefault("search", {})
        max_length = search.get("max_length")
        if not isinstance(max_length, int) or max_length <= 0 or max_length > position_limit:
            search["max_length"] = position_limit

    return adapted, _diff(before, adapted)


def build_audio_processor_config(n_mel: int) -> dict[str, Any]:
    return {
        "feature_extraction": {
            "sequence": [
                {"operation": {"name": "audio_decoder", "type": "AudioDecoder"}},
                {
                    "operation": {
                        "name": "STFT",
                        "type": "STFTNorm",
                        "attrs": {
                            "n_fft": 400,
                            "frame_length": 400,
                            "hop_length": 160,
                        },
                    }
                },
                {
                    "operation": {
                        "name": "log_mel_spectrogram",
                        "type": "LogMelSpectrum",
                        "attrs": {
                            "chunk_size": 30,
                            "hop_length": 160,
                            "n_fft": 400,
                            "n_mel": n_mel,
                        },
                    }
                },
            ]
        }
    }


def ensure_audio_processor_config(package_dir: Path, encoder_graph: dict[str, Any]) -> bool:
    config_path = package_dir / "audio_processor_config.json"
    if config_path.exists():
        return False
    input_shape = encoder_graph.get("inputs", [{}])[0].get("shape", [])
    n_mel = 80
    if len(input_shape) >= 2 and isinstance(input_shape[1], int) and input_shape[1] > 0:
        n_mel = int(input_shape[1])
    write_json(config_path, build_audio_processor_config(n_mel))
    return True


def compare_config_graph_and_reference(
    package_dir: Path,
    config: dict[str, Any],
    encoder_graph: dict[str, Any],
    decoder_graph: dict[str, Any],
) -> dict[str, Any]:
    model = config.get("model", {})
    decoder = model.get("decoder", {})
    encoder = model.get("encoder", {})
    decoder_inputs = decoder.get("inputs", {})
    decoder_outputs = decoder.get("outputs", {})
    encoder_inputs = encoder.get("inputs", {})
    encoder_outputs = encoder.get("outputs", {})
    layer_count = int(decoder.get("num_hidden_layers") or 0)
    decoder_layer_count = infer_decoder_layer_count(decoder_graph, fallback=layer_count)

    decoder_input_names = set(decoder_graph.get("input_names", []))
    decoder_output_names = set(decoder_graph.get("output_names", []))
    encoder_input_names = set(encoder_graph.get("input_names", []))
    encoder_output_names = set(encoder_graph.get("output_names", []))
    inventory_paths = {entry["path"] for entry in collect_file_inventory(package_dir)}

    mismatches: list[dict[str, Any]] = []

    unknown_decoder_inputs = sorted(set(decoder_inputs.keys()) - OGA_DECODER_INPUT_KEYS)
    if unknown_decoder_inputs:
        mismatches.append(
            {
                "id": "decoder-input-parser-keys",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.decoder.inputs",
                "message": "Unsupported decoder input keys for OGA v0.15.2 parser.",
                "details": {"unknown_keys": unknown_decoder_inputs},
            }
        )

    unknown_decoder_outputs = sorted(set(decoder_outputs.keys()) - OGA_DECODER_OUTPUT_KEYS)
    if unknown_decoder_outputs:
        mismatches.append(
            {
                "id": "decoder-output-parser-keys",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.decoder.outputs",
                "message": "Unsupported decoder output keys for OGA v0.15.2 parser.",
                "details": {"unknown_keys": unknown_decoder_outputs},
            }
        )

    unknown_encoder_inputs = sorted(set(encoder_inputs.keys()) - OGA_ENCODER_INPUT_KEYS)
    if unknown_encoder_inputs:
        mismatches.append(
            {
                "id": "encoder-input-parser-keys",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.encoder.inputs",
                "message": "Unsupported encoder input keys for OGA v0.15.2 parser.",
                "details": {"unknown_keys": unknown_encoder_inputs},
            }
        )

    unknown_encoder_outputs = sorted(set(encoder_outputs.keys()) - OGA_ENCODER_OUTPUT_KEYS)
    if unknown_encoder_outputs:
        mismatches.append(
            {
                "id": "encoder-output-parser-keys",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.encoder.outputs",
                "message": "Unsupported encoder output keys for OGA v0.15.2 parser.",
                "details": {"unknown_keys": unknown_encoder_outputs},
            }
        )

    if "encoder" not in model:
        mismatches.append(
            {
                "id": "missing-encoder-section",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.encoder",
                "message": "Whisper package omits model.encoder section required by OGA Whisper runtime.",
            }
        )
    else:
        audio_name = encoder_inputs.get("audio_features", "audio_features")
        if audio_name not in encoder_input_names:
            mismatches.append(
                {
                    "id": "encoder-audio-map",
                    "severity": "blocking",
                    "owner": "Mobius producer",
                    "path": "model.encoder.inputs.audio_features",
                    "message": "Configured encoder audio input name does not exist in ONNX graph.",
                    "details": {"configured": audio_name, "graph_inputs": sorted(encoder_input_names)},
                }
            )
        hidden_name = encoder_outputs.get("encoder_hidden_states", "encoder_hidden_states")
        if hidden_name not in encoder_output_names:
            mismatches.append(
                {
                    "id": "encoder-hidden-map",
                    "severity": "blocking",
                    "owner": "Mobius producer",
                    "path": "model.encoder.outputs.encoder_hidden_states",
                    "message": "Configured encoder hidden-state output name does not exist in ONNX graph.",
                    "details": {"configured": hidden_name, "graph_outputs": sorted(encoder_output_names)},
                }
            )

    mapped_input_ids = decoder_inputs.get("input_ids", "input_ids")
    if mapped_input_ids not in decoder_input_names:
        mismatches.append(
            {
                "id": "decoder-input-ids-map",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.decoder.inputs.input_ids",
                "message": "Configured decoder input_ids name does not exist in ONNX graph.",
                "details": {"configured": mapped_input_ids, "graph_inputs": sorted(decoder_input_names)},
            }
        )

    mapped_encoder_hidden = decoder_inputs.get("encoder_hidden_states", "encoder_hidden_states")
    if mapped_encoder_hidden not in decoder_input_names:
        mismatches.append(
            {
                "id": "decoder-encoder-hidden-map",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.decoder.inputs.encoder_hidden_states",
                "message": "Configured decoder encoder_hidden_states input does not exist in ONNX graph.",
                "details": {"configured": mapped_encoder_hidden, "graph_inputs": sorted(decoder_input_names)},
            }
        )

    mapped_logits = decoder_outputs.get("logits", "logits")
    if mapped_logits not in decoder_output_names:
        mismatches.append(
            {
                "id": "decoder-logits-map",
                "severity": "blocking",
                "owner": "Mobius producer",
                "path": "model.decoder.outputs.logits",
                "message": "Configured decoder logits output does not exist in ONNX graph.",
                "details": {"configured": mapped_logits, "graph_outputs": sorted(decoder_output_names)},
            }
        )

    for key_name, defaults, container, graph_names, direction in [
        ("past_key_names", "past_key_values.%d.key", decoder_inputs, decoder_input_names, "inputs"),
        ("past_value_names", "past_key_values.%d.value", decoder_inputs, decoder_input_names, "inputs"),
        ("present_key_names", "present.%d.key", decoder_outputs, decoder_output_names, "outputs"),
        ("present_value_names", "present.%d.value", decoder_outputs, decoder_output_names, "outputs"),
    ]:
        name_template = str(container.get(key_name, defaults))
        expanded = _expand_template(name_template, decoder_layer_count)
        missing = [name for name in expanded if name not in graph_names]
        if missing:
            mismatches.append(
                {
                    "id": f"decoder-{key_name}-template",
                    "severity": "blocking",
                    "owner": "Mobius producer",
                    "path": f"model.decoder.{direction}.{key_name}",
                    "message": f"Expanded template names are missing from decoder graph {direction}.",
                    "details": {"template": name_template, "missing": missing},
                }
            )

    # OGA v0.15.2 whisper runtime does not bind a position_ids input in WhisperDecoderState.
    mapped_position_ids = decoder_inputs.get("position_ids", "position_ids")
    if mapped_position_ids in decoder_input_names:
        mismatches.append(
            {
                "id": "whisper-runtime-position-ids",
                "severity": "blocking",
                "owner": "OGA runtime",
                "path": "src/models/whisper.cpp",
                "message": "Decoder graph requires position_ids, but Whisper runtime state does not bind position_ids input.",
                "details": {"required_input_name": mapped_position_ids},
            }
        )

    position_limit = decoder_graph.get("position_limit")
    context_length = model.get("context_length")
    if isinstance(position_limit, int) and position_limit > 0 and isinstance(context_length, int):
        if context_length > position_limit:
            mismatches.append(
                {
                    "id": "context-length-overflow",
                    "severity": "warning",
                    "owner": "Mobius producer",
                    "path": "model.context_length",
                    "message": "Configured context_length exceeds decoder positional embedding table.",
                    "details": {"context_length": context_length, "position_limit": position_limit},
                }
            )
        max_length = config.get("search", {}).get("max_length")
        if isinstance(max_length, int) and max_length > position_limit:
            mismatches.append(
                {
                    "id": "search-max-length-overflow",
                    "severity": "warning",
                    "owner": "Mobius producer",
                    "path": "search.max_length",
                    "message": "Configured search.max_length exceeds decoder positional embedding table.",
                    "details": {"max_length": max_length, "position_limit": position_limit},
                }
            )

    missing_reference_files = sorted(
        [name for name in OFFICIAL_WHISPER_REFERENCE_FILES if name not in inventory_paths]
    )
    if missing_reference_files:
        mismatches.append(
            {
                "id": "official-reference-files-missing",
                "severity": "info",
                "owner": "Mobius producer",
                "path": "package-root",
                "message": "Files present in official OGA whisper package examples are missing.",
                "details": {"missing_files": missing_reference_files},
            }
        )

    return {
        "official_references": {
            "oga_version": OFFICIAL_OGA_VERSION,
            "oga_tag": OFFICIAL_OGA_TAG,
            "parser_source": [
                "src/config.cpp",
                "src/config.h",
                "src/models/whisper.cpp",
                "src/models/whisper_processor.cpp",
            ],
            "whisper_example": "test/models/whisper/genai_config.json",
        },
        "mismatches": mismatches,
    }

