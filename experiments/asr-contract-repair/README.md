# ASR contract-repair experiment

This experiment evaluates whether a deterministic **package/config adapter** can reconcile a retained Mobius Whisper package with the pinned OGA + Foundry Local runtime contracts.

## Entrypoint

```powershell
python experiments\asr-contract-repair\run_asr_contract_repair.py --source-package <retained_asr_package_dir> --scratch-root <scratch_root>
```

The script:
1. Inspects retained encoder/decoder ONNX contracts.
2. Builds a machine-readable mismatch comparison against OGA `v0.15.2` parser/runtime expectations.
3. Generates adapter candidates in a fresh scratch directory (hardlink/copy for large assets, mutable JSON detached before edits).
4. Runs bounded gates per candidate:
   - JSON load
   - OGA parser load (`onnxruntime_genai.Config`)
   - OGA model load (`onnxruntime_genai.Model`)
   - OGA transcription on a known-transcript TTS WAV
   - Foundry Local SDK discovery/load/audio transcription
5. Writes `asr-contract-repair-report.json` under the run directory.

## Tests

```powershell
python -m pytest experiments\asr-contract-repair\test_asr_contract_adapter.py -q
```

