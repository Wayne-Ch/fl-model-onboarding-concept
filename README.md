# Foundry Local Model Onboarding Concept

- [View the live concept mock](https://wayne-ch.github.io/fl-model-onboarding-concept/)
- [View the source](https://github.com/Wayne-Ch/fl-model-onboarding-concept/blob/main/index.html)
- [Open a feedback issue](https://github.com/Wayne-Ch/fl-model-onboarding-concept/issues/new)

Interactive concept prototype for a moonshot Foundry Local model-onboarding experience. All model statuses, capabilities, and benchmark values are illustrative and are not product commitments.

## Local service + CLI (POC)

Build the web UI once when working from source:

```powershell
Set-Location web
npm ci
npm run build
Set-Location ..
```

Launch the packaged UI and API together with one command:

```powershell
fl-onboarding service serve --host 127.0.0.1 --port 8777
```

Non-loopback binding is blocked by default and requires explicit opt-in:

```powershell
fl-onboarding service serve --host 0.0.0.0 --allow-non-loopback
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

Notes:

- `/api/builds` requires `Idempotency-Key`.
- Job state and event replay are persisted in SQLite.
- Inference is artifact-scoped and only available for succeeded builds with matching task modality.
- A model appears in the tested-model index only after an artifact-scoped Foundry Local inference succeeds.
- Missing build or inference adapters report structured `not_verified`/`INFERENCE_NOT_IMPLEMENTED`
  results; the service never substitutes fixture success.
- ASR remains visible for discovery, but preflight reports the verified Whisper `genai_config`
  incompatibility as a structured blocker and exposes no working precision.
- The original repository-root `index.html` remains the standalone concept mock. The service serves
  the separately built React UI from the Python package.