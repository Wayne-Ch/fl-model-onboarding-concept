# ASR upstream triage: `position_ids` runtime contract gap

**As of:** 2026-08-31  
**Scope:** upstream source review only (no Mobius/OGA/ORT source modifications in this repo)

## Verified upstream facts

1. **Mobius Whisper export requires and wires `position_ids`.**
   - Decoder forward signature requires `position_ids`, then adds positional embeddings from it:  
     https://github.com/onnxruntime/mobius/blob/main/src/mobius/models/whisper.py#L147-L157
   - Speech-to-text graph builder explicitly creates a `position_ids` input and passes it in decoder kwargs:  
     https://github.com/onnxruntime/mobius/blob/main/src/mobius/tasks/_speech_to_text.py#L128-L157

2. **OGA config structures include `position_ids` fields.**
   - Default input-name constant includes `PositionIdsName = "position_ids"`:  
     https://github.com/microsoft/onnxruntime-genai/blob/main/src/config.h#L23
   - Decoder/encoder input structs include `position_ids` members in config schema:  
     https://github.com/microsoft/onnxruntime-genai/blob/main/src/config.h#L214-L218  
     https://github.com/microsoft/onnxruntime-genai/blob/main/src/config.h#L421-L425

3. **OGA Whisper decoder runtime state does not bind/update `position_ids` today.**
   - `WhisperDecoderState` constructor adds input IDs, logits, KV cache, and optional past-length/cache-indirection tensors; no `position_ids` tensor bind is present:  
     https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/whisper.cpp#L52-L83
   - `WhisperDecoderState::UpdateInputsOutputs` updates `input_ids`, KV cache, logits, past-length, cache-indirection; no `position_ids` update path is present:  
     https://github.com/microsoft/onnxruntime-genai/blob/main/src/models/whisper.cpp#L115-L156

## Current upstream status

- The mismatch is still visible on upstream `main` as of 2026-08-31 (Mobius decoder contract requires `position_ids`, while Whisper runtime decode state path does not bind/update it).
- No known **dedicated** upstream issue/PR for this exact mismatch was found in targeted searches at that date:  
  https://github.com/microsoft/onnxruntime-genai/issues?q=is%3Aissue+WhisperDecoderState+position_ids  
  https://github.com/microsoft/onnxruntime-genai/pulls?q=is%3Apr+WhisperDecoderState+position_ids

## Recommended smallest fix (OGA side)

Implement optional Whisper-decoder `position_ids` binding/updates in `WhisperDecoderState`, derived from prompt length and past sequence length when the decoder graph requires that input.

Then run regression coverage with:

1. a Mobius-exported Whisper package (same contract shape),
2. direct OGA transcription path, and
3. Foundry Local SDK transcription path.
