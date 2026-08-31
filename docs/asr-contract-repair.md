# ASR contract repair spike (Basher)

**Outcome:** **source change required**.  
A deterministic package/config adapter can get past parser load and model load, but cannot complete OGA/Foundry transcription with the retained Mobius Whisper decoder contract.

## Run metadata

- **Run ID:** `20260831-124030-fc016713`
- **Machine-readable report:** `<scratch-root>\20260831-124030-fc016713\asr-contract-repair-report.json`
- **Known transcript audio generation:** Windows `System.Speech` TTS
  - phrase: `hello world from contract test`
  - SHA-256: `49ca0074eb17471f8b5b9333418def9fa23042d0e967e42da90dec3d8868d5fd`
  - size: `125464` bytes
- **Versions observed in run:**
  - `onnxruntime-genai 0.15.2`
  - `onnxruntime 1.29.0`
  - `foundry-local-sdk 1.2.4`
  - `mobius-onnx 0.1.0`
  - `foundry CLI 0.11.0`

## Official OGA references used

- `microsoft/onnxruntime-genai@v0.15.2`
  - `src/config.cpp`
  - `src/config.h`
  - `src/models/whisper.cpp`
  - `src/models/whisper_processor.cpp`
  - `test/models/whisper/genai_config.json`

## Machine-readable mismatch inventory (source Mobius package)

| Mismatch ID | Severity | Owner | Evidence |
| --- | --- | --- | --- |
| `decoder-input-parser-keys` | blocking | Mobius producer | `model.decoder.inputs.decoder_input_ids` is not an allowed key in OGA `DecoderInputs_Element` (expects `input_ids` key with value mapping). |
| `missing-encoder-section` | blocking | Mobius producer | Whisper package omits `model.encoder` section; OGA Whisper runtime expects encoder + decoder sessions. |
| `decoder-input-ids-map` | blocking | Mobius producer | Effective `input_ids` mapping cannot resolve to graph input without adapter rewrite. |
| `whisper-runtime-position-ids` | blocking | OGA runtime | Decoder graph requires `position_ids`, but `WhisperDecoderState` does not bind a position-ids input path. |
| `context-length-overflow` | warning | Mobius producer | `model.context_length=4096` exceeds decoder positional embedding limit inferred from `decoder.embed_positions.weight` (`448`). |
| `search-max-length-overflow` | warning | Mobius producer | `search.max_length=4096` exceeds same positional limit (`448`). |
| `official-reference-files-missing` | info | Mobius producer | Deviates from official Whisper package file set (including `audio_processor_config.json`). |

## Candidate gates

| Candidate | Applied config transformation | JSON load | OGA parser load | OGA model load | OGA transcription | FL SDK transcription | First failed stage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `minimal-parser-fix` | Rename key only: `model.decoder.inputs.decoder_input_ids` -> `model.decoder.inputs.input_ids` (value unchanged) | ✅ | ✅ | ❌ | N/A | N/A | `oga_model_load` |
| `full-contract-adapter` | Adds `model.encoder` from graph contract, rewrites input key mapping, clamps `context_length/search.max_length` to `448`, emits audio processor config | ✅ | ✅ | ✅ | ❌ | ❌ | `oga_transcription` |

## Irreducible failure boundary

### 1) OGA transcription failure (first irreducible)

- Error signature: `Missing Input: position_ids`
- Stage: `oga_transcription`
- Reason this is irreducible by config-only mapping:
  - Decoder ONNX graph has a required `position_ids` input tensor.
  - OGA Whisper runtime path (`src/models/whisper.cpp`) binds input IDs + KV cache but does not bind a position-ids tensor for Whisper decode state.
  - Renaming config keys cannot synthesize a missing runtime-bound tensor.

### 2) Foundry Local SDK failure (mirrors same runtime contract)

- Stage: `fl_sdk_transcription`
- Error signature includes `Missing Input: position_ids` from OGA under `AudioClient.Transcribe`.
- This confirms FL SDK cannot complete the task contract with this adapted package under current pinned runtime behavior.

## Reproduction command

```powershell
python experiments\asr-contract-repair\run_asr_contract_repair.py --source-package <retained_asr_package_dir> --scratch-root <scratch_root>
```

## Required owner actions

1. **Mobius producer owner:** emit OGA-compatible Whisper decoder contract (no required `position_ids` input in decoder graph, or aligned with OGA Whisper runtime expectations).
2. **OGA runtime owner:** alternatively add Whisper decode-state support to bind/populate `position_ids` when decoder graphs require it.
3. **FL SDK owner:** once OGA/Mobius contract is aligned, re-validate `AudioClient.transcribe` end-to-end with the same adapted-package gate script.
