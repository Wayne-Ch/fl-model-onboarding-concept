# Foundry Local Model Onboarding Concept

- [View the live concept mock](https://wayne-ch.github.io/fl-model-onboarding-concept/)
- [View the source](https://github.com/Wayne-Ch/fl-model-onboarding-concept/blob/main/index.html)
- [Open a feedback issue](https://github.com/Wayne-Ch/fl-model-onboarding-concept/issues/new)

Interactive concept prototype for a moonshot Foundry Local model-onboarding experience. All model statuses, capabilities, and benchmark values are illustrative and are not product commitments.
The GitHub Pages URL above is a concept-only mock and cannot run local Mobius/Olive/Foundry tooling.

## Local reproducible POC setup (Windows x64)

First check out the PR branch:

```powershell
git checkout wayne-ch-linus-ui-api-integration
```

Then run exactly two commands:

```powershell
.\scripts\bootstrap-local-poc.ps1
.\scripts\run-local-ui.ps1
```

`bootstrap-local-poc.ps1` defaults to `C:\fl-onboarding-venv`, verifies Windows x64 and Python 3.11+, installs
`.[dev,runtime]`, and builds `web/`. Use script parameters to override paths/interpreter.

`run-local-ui.ps1` prepends `<venv>\Scripts` to `PATH` and starts the real local API + packaged React UI with:

- loopback-only host (`127.0.0.1` by default)
- production runner enabled
- short defaults (`--workspace-base C:\fmo\w`, `--model-cache-dir C:\fmo\cache`)
- browser auto-open enabled

## Manual browser validation (real local UI/API)

1. Run `.\scripts\run-local-ui.ps1` and wait for the browser to open.
2. In the UI search box, search for and select **HuggingFaceTB/SmolLM2-1.7B-Instruct**.
3. Choose the **Mobius + Olive / INT4** path and start build.
4. Wait for build completion, then submit a simple chat prompt (for example, `Say hello in one sentence.`).
5. Confirm that the response is returned through the local runtime flow.

Optional retained-package E2E (only if the local retained artifact already exists):

```powershell
$env:FL_ONBOARDING_E2E_PYTHON = "C:\fl-onboarding-venv\Scripts\python.exe"
$env:FL_ONBOARDING_E2E_MODEL_DIR = "C:\fmo\w\<run>\olive\llm"
python -m pytest -m e2e tests\test_real_toolchain_e2e.py
```

Doctor + model commands:

```powershell
fl-onboarding doctor --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct --task llm
fl-onboarding model search --query smollm --limit 10
fl-onboarding model detail --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct
fl-onboarding model preflight --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct --task llm
```

Build lifecycle commands:

```powershell
fl-onboarding build create --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct --task llm --idempotency-key run-001
fl-onboarding build status --job-id <JOB_ID>
fl-onboarding build cancel --job-id <JOB_ID>
```

## Recipe registry status (phase 2)

Buildability now comes from the typed recipe registry (not model ID hardcoding and not Mobius model-type support alone).

| Recipe | Status | HF model | Pinned revision | Notes |
| --- | --- | --- | --- | --- |
| `smollm2-1.7b-cpu-int4` | verified | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | Preserved production happy path (Mobius f32 -> Olive int4 -> runtime -> FL SDK inference). |
| `granite-3.3-2b-cpu-int4` | verified | `ibm-granite/granite-3.3-2b-instruct` | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` | Promoted from candidate to verified after direct Mobius/Olive/runtime/FL SDK evidence. |
| `distil-whisper-cpu-fp16` | blocked | `distil-whisper/distil-medium.en` | `6e61418885eaf4d5cc9f64e508e80ac5b4c052b7` | Runtime contract blocker remains explicit and non-happy-path: decoder ONNX requires `position_ids`, but OGA Whisper decoder state does not bind/update it (`Missing Input: position_ids`). |

Notes:

- `/api/builds` requires `Idempotency-Key`.
- Unknown/unregistered models remain not buildable until a recipe exists.
- Experimental recipes (when present) require explicit opt-in (`--allow-experimental` / UI checkbox) and cannot be treated as tested-successful until artifact-scoped Foundry inference succeeds.
- Job state and event replay are persisted in SQLite.
- Inference is artifact-scoped and only available for succeeded builds with matching task modality.
- A model appears in the tested-model index only after an artifact-scoped Foundry Local inference succeeds.
- Production mode materializes the resolved HF SHA locally, runs the verified Mobius f32 + Olive INT4
  argument-array contracts with bounded timeouts, validates ONNX/ORT/OGA, creates an immutable
  BYOM cache package, and records tested status only after Foundry Local SDK chat succeeds.
- Missing build or inference adapters report structured `not_verified`/`INFERENCE_NOT_IMPLEMENTED`
  results; the service never substitutes fixture success.
- ASR remains visible for discovery, but preflight reports the final verified runtime blocker as a
  structured non-success candidate outcome: deterministic adaptation clears parser/model-load gates,
  then OGA/Foundry transcription fails with `Missing Input: position_ids` because Whisper decoder
  state does not bind/update `position_ids`.
- Upstream source-triage evidence is documented in `docs/asr-upstream-triage.md`.
- The original repository-root `index.html` remains the standalone concept mock. The service serves
  the separately built React UI from the Python package.