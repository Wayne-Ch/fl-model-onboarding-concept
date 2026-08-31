# Contract probe results

**Branch:** `wayne-ch-basher-probing-contracts`  
**Primary run:** `20260830-225442-66553c73`  
**Probe entrypoint:** `experiments/contract-probes/run_contract_probes.py`  
**Interpreter:** `C:\flprobe-venv\Scripts\python.exe`  
**Scratch root:** `C:\fmo-poc\work`

## VERIFIED

1. **Post-restart tool/version checks**
   - `foundry --version` => `0.11.0`
   - `mobius --help` exposes `build`, `build-gguf`, `list`, `info`, `convert-comfyui`
   - `olive --help` and `olive optimize --help` succeeded
   - Installed probe packages: `mobius-onnx=0.1.0`, `olive-ai=0.13.0`, `onnx=1.22.0`, `onnxruntime=1.29.0`, `onnxruntime-genai=0.15.2`, `foundry-local-sdk=1.2.4`
2. **Foundry catalog JSON contracts**
   - `foundry model list -o json` => top-level `models` (49 rows at run time)
   - `foundry model list --variants -o json` => top-level `variants` (98 rows at run time)
   - Both candidate IDs absent from live catalog, confirming BYOM path is required.
3. **HF revision capture**
   - `HuggingFaceTB/SmolLM2-1.7B-Instruct` => `31b70e2e869a7173562077fd711b654946d38674`
   - `distil-whisper/distil-medium.en` => `6e61418885eaf4d5cc9f64e508e80ac5b4c052b7`
   - Current `mobius build` CLI does not expose `--revision`; probe records exact SHA even when command-line pinning is unavailable.
4. **Direct Mobius build (CPU + ort-genai)**
   - LLM command:
     - `mobius build --model HuggingFaceTB/SmolLM2-1.7B-Instruct --ep cpu --runtime ort-genai --dtype f32 <scratch>\mobius\llm`
     - Success in `255.046s`
     - `mobius info` recognized model type `llama`, default task `text-generation`
   - ASR command:
     - `mobius build --model distil-whisper/distil-medium.en --ep cpu --runtime ort-genai --dtype f32 <scratch>\mobius\asr`
     - Success in `94.203s`
     - `mobius info` recognized model type `whisper`, default task `speech-to-text`
5. **Mobius output inventory + validation**
   - LLM outputs: `model.onnx` + `model.onnx.data` + tokenizer artifacts + `genai_config.json`
   - ASR outputs: `encoder/model.onnx(.data)` + `decoder/model.onnx(.data)` + tokenizer artifacts + `genai_config.json`
   - ONNX checker passed for all three Mobius ONNX graphs (LLM, ASR encoder, ASR decoder)
   - ORT CPU session load succeeded for all three graphs.
6. **Olive existing-ONNX optimization probes**
   - LLM INT4 path succeeded on Mobius ONNX directory:
     - `olive optimize --model_name_or_path <scratch>\mobius\llm --task text-generation-with-past --device cpu --provider CPUExecutionProvider --precision int4 --output_path <scratch>\olive\llm`
     - Passes included peephole optimizer + blockwise RTN quantization.
   - ASR path:
     - `olive optimize` against Mobius ASR directory failed (`ValueError: Unrecognized model ... should have a model_type key in config.json`) for both INT8 and FP32.
     - Fallback to existing ONNX component succeeded for decoder with modality-appropriate FP32:
       - `olive optimize --model_name_or_path <scratch>\mobius\asr\decoder\model.onnx --task automatic-speech-recognition --device cpu --provider CPUExecutionProvider --precision fp32 --output_path <scratch>\olive\asr`
     - INT8 decoder attempt failed due missing calibration dataset dependency (`ModuleNotFoundError: datasets`) after quantization preprocess warnings.
7. **Runtime validations**
   - LLM (optimized output):
     - ONNX checker ✅
     - ORT CPU load ✅
     - OGA generation ✅ (produced text)
     - Foundry Local SDK BYOM discovery/load/chat ✅ using `model_cache_dir=<scratch>\olive` and `inference_model.json` (`Name: smollm2-contract-probe:1`)
   - ASR:
     - ONNX checker ✅ on optimized decoder ONNX
     - ORT CPU load ✅ on optimized decoder ONNX
     - OGA Whisper package load ❌ on Mobius output (`Unknown value "decoder_input_ids"` while parsing `genai_config.json`)
     - Foundry Local SDK BYOM load ❌ with same underlying parser failure when loading `distil-whisper-contract-probe:1`
     - Decoder-only Olive output was not discoverable as a Foundry model entry (no valid package-level runtime metadata for SDK discovery).
8. **Probe script tests**
   - `python -m pytest experiments\contract-probes\test_run_contract_probes.py -q` => `5 passed`

## VERSION-SENSITIVE

1. `mobius build --help` can emit Unicode that fails under non-UTF8 console decoding; probe runner now forces UTF-8 subprocess decoding.
2. Olive cache location defaults under repository root (`.olive-cache`) unless overridden; cache contents and run reuse behavior are version-dependent.
3. Foundry catalog membership and row counts are live-service data and can drift between runs.

## NOT VERIFIED

1. ASR end-to-end successful OGA/Foundry execution is **not verified** because current generated Whisper package metadata fails OGA parser validation in this environment.
2. `inference_model.json` schema beyond `Name` is **not proven** by official runtime acceptance in this run; only observed behavior against generated artifacts is reported.

## BLOCKED

1. **ASR runtime contract blocker (current environment):** Mobius-generated Whisper `genai_config.json` rejected by OGA/Foundry runtime parser (`model:decoder:inputs: Unknown value "decoder_input_ids"`).  
   - Stage evidence: OGA load and Foundry SDK model load fail with the same parse error.
   - Impact: prevents ASR BYOM load/inference happy path even though ONNX graphs are valid and ORT-loadable.
2. **ASR INT8 optimization blocker:** Olive INT8 path requires calibration dataset plumbing (`datasets` module) and did not complete from the existing-ONNX decoder probe path in this run.

## Candidate outcome summary

| Candidate | Outcome | Last successful stage | Blocking stage |
| --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | **HAPPY PATH** | Foundry SDK BYOM chat inference | N/A |
| `distil-whisper/distil-medium.en` | **BLOCKED** | ORT load of Olive FP32 decoder ONNX | OGA/Foundry runtime parse of Whisper `genai_config.json` |

## Artifacts and cleanup

1. Scratch run retained for evidence: `C:\fmo-poc\work\20260830-225442-66553c73` (~`10037.43 MB`).
2. Largest artifacts are Mobius/Olive ONNX weight files in scratch only (not in git).
3. Repository cleanup performed:
   - removed `.olive-cache/`
   - removed `experiments/contract-probes/__pycache__/`
4. No model weights, ONNX artifacts, caches, venvs, or tokens were committed.

## Re-run command

```powershell
C:\flprobe-venv\Scripts\python.exe experiments\contract-probes\run_contract_probes.py --scratch-root C:\fmo-poc\work
```

## Next action

1. Keep LLM candidate as the reliable POC path.
2. For ASR, investigate/align Whisper `genai_config.json` input contract expected by current OGA/Foundry runtime before attempting another BYOM load probe.
