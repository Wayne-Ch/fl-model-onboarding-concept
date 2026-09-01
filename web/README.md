# Foundry Local onboarding web shell

This folder contains the real browser UI shell for model onboarding and CPU build orchestration. It intentionally keeps API wiring isolated from backend contract churn while Python integration lands.

## Quick start

```bash
cd web
npm install
npm run dev
```

## Scripts

- `npm run dev` - run Vite dev server
- `npm run test` - run Vitest suite
- `npm run build` - type-check and build production assets
- `npm run preview` - preview production build locally

## Provisional API contract implemented in frontend

The client is centralized in `src/api/client.ts`, and API-facing types are isolated in `src/api/types.ts` for easy regeneration/revision.

- `GET /api/health`
- `GET /api/models/search?q=&limit=`
- `GET /api/models/detail?id=<HF_ID>`
- `POST /api/models/preflight`
- `POST /api/builds` (requires `Idempotency-Key`)
- `GET /api/builds/{job_id}`
- `GET /api/builds/{job_id}/events?after=<sequence>`
- `POST /api/builds/{job_id}/cancel`
- `POST /api/artifacts/{artifact_id}/infer/text`
- `POST /api/artifacts/{artifact_id}/infer/asr` (multipart)

## Canonical contracts after merge

After branch integration, backend-owned canonical sources are:

- `contracts/openapi.yaml`
- `contracts/job-state-machine.json`

Frontend keeps adapters and runtime parsing separate so those canonical contracts can replace provisional parser assumptions with minimal churn.

## Development fixture mode

A typed fixture transport is available for development and tests.

- Enable with `VITE_USE_FIXTURE_API=true` when running `npm run dev`.
- Fixture mode is development-only and visibly announced in the UI.
- Production builds never auto-fallback to fixture success; backend outages surface as **Local service unavailable**.

## Security and behavior constraints

- No HF token capture/storage in browser storage.
- Gated models are marked not buildable in the UI and blocked from build submission.
- Backend logs are rendered as plain text only (no HTML injection).
- API base defaults to loopback; non-loopback must be explicitly opted in (`VITE_ALLOW_NON_LOOPBACK_API=true`) and is warned in UI.

## Current integration deltas to backend

1. Parsers currently accept both snake_case and camelCase fields where needed to absorb provisional backend shape variation.
2. Status rendering uses backend-reported `stage` and event stream directly; it does not authoritatively execute local transition logic.
3. Event polling persists a per-job `after` cursor in component state and resumes incrementally for the active job.
