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

Notes:

- `/api/builds` requires `Idempotency-Key`.
- Job state and event replay are persisted in SQLite.
- Inference is artifact-scoped and only available for succeeded builds with matching task modality.
- A model appears in the tested-model index only after an artifact-scoped Foundry Local inference succeeds.
- Production mode materializes the resolved HF SHA locally, runs the verified Mobius f32 + Olive INT4
  argument-array contracts with bounded timeouts, validates ONNX/ORT/OGA, creates an immutable
  BYOM cache package, and records tested status only after Foundry Local SDK chat succeeds.
- Missing build or inference adapters report structured `not_verified`/`INFERENCE_NOT_IMPLEMENTED`
  results; the service never substitutes fixture success.
- ASR remains visible for discovery, but preflight reports the verified Whisper `genai_config`
  incompatibility as a structured blocker and exposes no working precision.
- If you install or update Mobius/Olive/Foundry/Python packages while the local service is running,
  stop it (`Ctrl+C`) and rerun `.\scripts\run-local-ui.ps1` so preflight probes the updated toolchain.
- The original repository-root `index.html` remains the standalone concept mock. The service serves
  the separately built React UI from the Python package.