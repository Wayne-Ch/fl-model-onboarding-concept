# Contract probe scripts (Basher)

These scripts are intentionally **path-agnostic** and write probe artifacts outside the repo.

## Short-path prerequisite (Windows)

`mobius-onnx` pulls a large dependency graph (`torch` + transitive licenses). A long venv path can fail with:

`WinError 206: The filename or extension is too long`

Use a short external root. Verified working path on this machine:

```powershell
python -m venv C:\flprobe-venv
C:\flprobe-venv\Scripts\python.exe -m pip install --upgrade pip
C:\flprobe-venv\Scripts\python.exe -m pip install mobius-onnx olive-ai onnx onnxruntime onnxruntime-genai huggingface_hub foundry-local-sdk pytest
```

## Local checks for committed scripts

```powershell
C:\flprobe-venv\Scripts\python.exe -m pytest experiments\contract-probes\test_run_contract_probes.py -q
```

## Resume full probe run

```powershell
C:\flprobe-venv\Scripts\python.exe experiments\contract-probes\run_contract_probes.py --scratch-root C:\fmo-poc\work
```

The script records command logs and a `probe-summary.json` under the scratch root and never writes model artifacts into the repository.

## Notes

1. Mobius/Olive can emit Unicode in help output; the script forces UTF-8 decoding for subprocess probes.
2. Foundry SDK probing runs in isolated subprocesses to avoid singleton re-initialization conflicts across candidates.
