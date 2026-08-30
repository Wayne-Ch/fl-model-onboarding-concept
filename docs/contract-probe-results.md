# Contract probe results (WIP checkpoint)

**Checkpoint type:** Pause-before-shutdown handoff  
**Branch:** `wayne-ch-basher-probing-contracts`  
**Scope in this checkpoint:** toolchain contract verification + reusable scripts only (no model builds started after pause request)

## VERIFIED

1. **Foundry CLI present and current on this machine**
   - `foundry --version` => `0.11.0`
2. **Foundry catalog JSON contracts**
   - `foundry model list -o json` top-level key is `models` (49 entries at probe time)
   - `foundry model list --variants -o json` top-level key is `variants` (98 entries at probe time)
3. **Candidate presence in live catalog**
   - `HuggingFaceTB/SmolLM2-1.7B-Instruct` not present
   - `distil-whisper/distil-medium.en` not present
4. **Short-path environment workaround for WinError 206**
   - Full dependency graph installed successfully when using a short external venv root.
5. **Installed probe dependency versions (short-path venv)**
   - `mobius-onnx=0.1.0`
   - `olive-ai=0.13.0`
   - `onnx=1.22.0`
   - `onnxruntime=1.29.0`
   - `onnxruntime-genai=0.15.2`
   - `foundry-local-sdk=1.2.4`
   - `huggingface_hub=1.28.0`
   - `torch=2.13.0`
6. **Current Mobius CLI surface (from installed help)**
   - Entry point: `mobius`
   - Subcommands include: `build`, `build-gguf`, `list`, `info`
   - `mobius build` uses positional `output_dir` in this version (not `--output`)
   - `mobius list tasks` includes both `text-generation` and `speech-to-text`
   - `mobius list eps` includes `cpu` EP
7. **Current Olive CLI surface (from installed help)**
   - `olive optimize` supports `--model_name_or_path`, `--device`, `--provider`, `--precision`
   - `--precision` includes `int4`, `int8`, `fp32`
8. **Committed probe script unit tests**
   - `pytest experiments\contract-probes\test_run_contract_probes.py -q` => **5 passed**

## VERSION-SENSITIVE

1. Running `mobius build --help` under non-UTF8 console encoding can fail with `UnicodeEncodeError` (help text contains Unicode arrow characters).
2. Probe script now forces UTF-8 for subprocesses (`PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`) to keep help/capability probes stable.
3. Foundry catalog contents and model counts are live and may change between runs.

## NOT VERIFIED (deferred by pause request)

1. Direct Mobius build execution for both candidates.
2. Olive optimization of Mobius ONNX outputs (LLM INT4 and ASR modality path).
3. ONNX checker / ORT CPU load against generated outputs.
4. OGA load/execution against generated outputs.
5. Foundry Local SDK BYOM discovery/load/inference using generated model package.

## BLOCKED

1. **No active technical blocker remains** after short-path venv setup.
2. Remaining work is paused intentionally by shutdown checkpoint request.

## REUSABLE ARTIFACTS COMMITTED IN THIS CHECKPOINT

1. `experiments/contract-probes/run_contract_probes.py`
2. `experiments/contract-probes/test_run_contract_probes.py`
3. `experiments/contract-probes/README.md`
4. `docs/contract-probe-results.md` (this file)

## RESUME COMMAND

Use a short external root (example):

```powershell
C:\fmo-poc\venv\Scripts\python.exe experiments\contract-probes\run_contract_probes.py --scratch-root C:\fmo-poc\work
```

If the venv does not exist yet:

```powershell
python -m venv C:\fmo-poc\venv
C:\fmo-poc\venv\Scripts\python.exe -m pip install --upgrade pip
C:\fmo-poc\venv\Scripts\python.exe -m pip install mobius-onnx olive-ai onnx onnxruntime onnxruntime-genai huggingface_hub foundry-local-sdk pytest
```
