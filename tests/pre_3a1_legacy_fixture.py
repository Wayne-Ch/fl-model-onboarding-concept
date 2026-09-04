"""Shared pre-3A1 legacy generated-recipe payload fixture.

Rejected revision `wayne-ch-linus-recipe-3a1-trusted-block64` added
`provenance.trusted_candidate` to `contracts/generated-recipe.schema.json`'s
`required` list without a migration. Every already-persisted pre-3A1
generated recipe payload has no `trusted_candidate` key at all (not even
`null`) and no `recipe.olive.block_size` key, so it failed schema validation
and became unexecutable by the production runner.

The canonical JSON/fingerprint below is a byte-for-byte snapshot of what the
actual pre-3A1 compiler (commit `51f6009`, immediately before the rejected
Slice 3A1 commit) produced for the deterministic `RecipeCompilerInput` also
reproduced here. It was captured once by checking out that commit and running
`compile_generated_recipe` on this exact input; it is a real legacy payload
shape, not a hand-authored approximation.

Both `tests/test_recipe_compiler.py` and `tests/test_production_runner.py`
import this module so the "legacy payload" used to prove schema-validation
compatibility and the one used to prove the production execution-plan loader
still resolves are guaranteed to be identical.
"""

from __future__ import annotations

import json

PRE_3A1_MODEL_ID = "owner/pre-3a1-legacy-fixture-model"
PRE_3A1_REVISION_SHA = "3234567890abcdef1234567890abcdef12345678"

PRE_3A1_FINGERPRINT = (
    "8550df0408f3529f1822a9e4c571b53edfe5e67a5a960da286dd3da695ec4290"
)

PRE_3A1_CANONICAL_JSON = (
    '{"fingerprint":"8550df0408f3529f1822a9e4c571b53edfe5e67a5a960da286dd3da695ec4290",'
    '"pinned_revision":"3234567890abcdef1234567890abcdef12345678",'
    '"provenance":{"argument_provenance":{"contains_unverified_arguments":false,'
    '"mobius_dtype_confidence":"evidence-pinned",'
    '"mobius_dtype_rule":"Pinned by verified SmolLM2 and Granite probes.",'
    '"olive_precision_confidence":"evidence-pinned",'
    '"olive_precision_rule":"Pinned by verified SmolLM2 and Granite probes."},'
    '"capability_id":"llama-text-generation-cpu-int4-v1","capability_status":"verified",'
    '"capability_version":"1.0.0","compiler_version":"1.0.0",'
    '"evidence":[{"evidence_id":"smollm2-llama-model-type",'
    '"location":"docs/contract-probe-results.md#verified","source_type":"repo-doc",'
    '"summary":"mobius info recognized SmolLM2 as model_type=llama with default task '
    'text-generation."},{"evidence_id":"smollm2-llama-runtime-happy-path",'
    '"location":"docs/contract-probe-results.md#verified","source_type":"repo-doc",'
    '"summary":"SmolLM2 path passed Olive INT4, OGA load/generation, and Foundry Local '
    'SDK chat inference."}],"generation_kind":"deterministic-recipe-agent-v1",'
    '"input_metadata":{"architectures":["LlamaForCausalLM"],'
    '"available_files":["config.json","tokenizer.json"],"config_files":["config.json"],'
    '"is_gated":false,"model_id":"owner/pre-3a1-legacy-fixture-model",'
    '"model_type":"llama","requested_device":"cpu","requested_precision":"auto",'
    '"requires_remote_code":false,'
    '"revision_sha":"3234567890abcdef1234567890abcdef12345678","task":"llm",'
    '"tokenizer_files":["tokenizer.json"]},'
    '"matched_aliases":["llama","llamaforcausallm"],"promotion":null,'
    '"resolution_outcome":"exact","resolution_reason_code":"resolved",'
    '"toolchain":{"foundry_cli_version":"0.11.0","foundry_sdk_version":"1.2.4",'
    '"mobius_version":"0.1.0","oga_version":"0.15.2","olive_version":"0.13.0",'
    '"onnx_version":"1.22.0","ort_version":"1.29.0"}},'
    '"recipe":{"ancillary_files":[{"relative_path":"genai_config.json","required":true,'
    '"source":"olive-output-dir"},{"relative_path":"model.onnx","required":true,'
    '"source":"olive-output-dir"},{"relative_path":"model.onnx.data","required":false,'
    '"source":"olive-output-dir"},{"relative_path":"tokenizer.json","required":true,'
    '"source":"olive-output-dir"}],'
    '"artifact_cache_prefix":"pre-3a1-legacy-fixture-model",'
    '"huggingface_model_id":"owner/pre-3a1-legacy-fixture-model",'
    '"id":"owner-pre-3a1-legacy-fixture-model-llm-cpu-int4","inference_modality":"llm",'
    '"mobius":{"dtype":"f32","ep":"cpu","runtime":"ort-genai","task":"text-generation"},'
    '"modality":"llm","model_name_prefix":"pre-3a1-legacy-fixture-model-onboarding",'
    '"olive":{"device":"cpu","input_source":"mobius-output-dir","log_level":"1",'
    '"precision":"int4","provider":"CPUExecutionProvider",'
    '"task":"text-generation-with-past"},'
    '"optimization_choices":[{"default":true,"precision":"int4","skip_olive":false,'
    '"strategy":"mobius-olive","task_profile":"llm-cpu-int4"}],'
    '"preferred_revision":"3234567890abcdef1234567890abcdef12345678",'
    '"runtime_validation":"llm-cpu-oga-load-v1 (onnxruntime-genai; '
    'loader=onnxruntime_genai.Model(model_dir))","status":"experimental",'
    '"status_reason":"Compiled from verified architecture capability '
    '\'llama-text-generation-cpu-int4-v1\', but this exact model revision remains an '
    'experimental candidate until explicit Mobius, Olive, ONNX, ORT, OGA, FL SDK '
    'inference, and quality gates promote it.",'
    '"success_message":"Candidate recipe compiled deterministically; promotion requires '
    'successful Mobius, Olive, ONNX, ORT, OGA, FL SDK inference, and quality gates.",'
    '"task_profile":"llm-cpu-int4","verified_revision":null,"version":"0.1.0"},'
    '"schema_version":"1.0.0"}'
)


def legacy_payload() -> dict[str, object]:
    """Parse and return a fresh copy of the pinned legacy payload dict."""
    payload = json.loads(PRE_3A1_CANONICAL_JSON)
    assert isinstance(payload, dict)
    return payload
