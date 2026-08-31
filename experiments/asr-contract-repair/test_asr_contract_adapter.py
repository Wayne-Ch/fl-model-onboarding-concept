from __future__ import annotations

from pathlib import Path

from asr_contract_adapter import (
    apply_full_adapter,
    apply_minimal_parser_fix,
    compare_config_graph_and_reference,
    infer_decoder_layer_count,
)


def _mobius_like_config() -> dict:
    return {
        "model": {
            "type": "whisper",
            "context_length": 4096,
            "decoder": {
                "filename": "decoder/model.onnx",
                "head_size": 64,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_hidden_layers": 2,
                "num_key_value_heads": 16,
                "inputs": {
                    "decoder_input_ids": "decoder_input_ids",
                    "encoder_hidden_states": "encoder_hidden_states",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
            },
        },
        "search": {
            "max_length": 4096,
        },
    }


def _encoder_graph() -> dict:
    return {
        "input_names": ["input_features"],
        "output_names": ["encoder_hidden_states"],
        "inputs": [{"name": "input_features", "shape": ["batch", 80, "audio_seq_len"], "elem_type": 1}],
        "outputs": [{"name": "encoder_hidden_states", "shape": ["batch", 1500, 1024], "elem_type": 1}],
    }


def _decoder_graph() -> dict:
    return {
        "input_names": [
            "decoder_input_ids",
            "encoder_hidden_states",
            "position_ids",
            "past_key_values.0.key",
            "past_key_values.0.value",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ],
        "output_names": [
            "logits",
            "present.0.key",
            "present.0.value",
            "present.1.key",
            "present.1.value",
        ],
        "inputs": [
            {"name": "past_key_values.0.key", "shape": ["batch", 16, "past_sequence_len", 64], "elem_type": 1},
        ],
        "position_limit": 448,
    }


def test_apply_minimal_parser_fix_renames_decoder_input_key() -> None:
    adapted, changes = apply_minimal_parser_fix(_mobius_like_config())
    assert "decoder_input_ids" not in adapted["model"]["decoder"]["inputs"]
    assert adapted["model"]["decoder"]["inputs"]["input_ids"] == "decoder_input_ids"
    changed_paths = {item["path"] for item in changes}
    assert "model.decoder.inputs.decoder_input_ids" in changed_paths
    assert "model.decoder.inputs.input_ids" in changed_paths


def test_apply_full_adapter_adds_encoder_and_clamps_length() -> None:
    adapted, _ = apply_full_adapter(
        config=_mobius_like_config(),
        encoder_graph=_encoder_graph(),
        decoder_graph=_decoder_graph(),
    )
    assert adapted["model"]["encoder"]["filename"] == "encoder/model.onnx"
    assert adapted["model"]["encoder"]["inputs"]["audio_features"] == "input_features"
    assert adapted["model"]["context_length"] == 448
    assert adapted["search"]["max_length"] == 448


def test_compare_flags_source_contract_mismatches(tmp_path: Path) -> None:
    package_dir = tmp_path / "asr"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "genai_config.json").write_text("{}", encoding="utf-8")

    comparison = compare_config_graph_and_reference(
        package_dir=package_dir,
        config=_mobius_like_config(),
        encoder_graph=_encoder_graph(),
        decoder_graph=_decoder_graph(),
    )
    ids = {item["id"] for item in comparison["mismatches"]}
    assert "decoder-input-parser-keys" in ids
    assert "missing-encoder-section" in ids
    assert "whisper-runtime-position-ids" in ids
    assert "context-length-overflow" in ids


def test_infer_decoder_layer_count_from_graph_names() -> None:
    count = infer_decoder_layer_count(_decoder_graph(), fallback=0)
    assert count == 2

