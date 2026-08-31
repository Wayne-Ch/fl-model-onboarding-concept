from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Header, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .contracts import CandidateModality
from .local_service import BuildSubmission, LocalOnboardingService, ServiceError
from .serialization import to_jsonable


class ModelPreflightRequest(BaseModel):
    model_id: str = Field(min_length=1)
    task: Literal["llm", "asr"]
    task_profile: str = Field(default="default", min_length=1)
    hf_revision: str | None = None
    skip_olive: bool = False


class BuildCreateRequest(BaseModel):
    model_id: str = Field(min_length=1)
    task: Literal["llm", "asr"]
    task_profile: str = Field(default="default", min_length=1)
    hf_revision: str | None = None
    skip_olive: bool = False


class TextInferenceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=4096)


class CancelRequest(BaseModel):
    reason: str = Field(default="Cancelled by client request.", min_length=1)


def default_web_dist() -> Path:
    return Path(__file__).resolve().parent / "web_dist"


def create_app(
    service: LocalOnboardingService | None = None,
    *,
    web_dist: Path | None = None,
) -> FastAPI:
    live_service = service or LocalOnboardingService()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            live_service.close()

    app = FastAPI(title="Foundry Local Onboarding POC API", version="0.1.0", lifespan=lifespan)
    app.state.service = live_service
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=live_service.cors_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(ServiceError)
    async def _handle_service_error(_, exc: ServiceError) -> JSONResponse:
        payload = {"code": exc.code, "message": str(exc)}
        if exc.detail:
            payload["detail"] = exc.detail
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return live_service.health()

    @app.get("/api/models/search")
    async def model_search(
        q: str = Query(..., min_length=1),
        limit: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, object]:
        return live_service.search_models(query=q, limit=limit)

    @app.get("/api/models/detail")
    async def model_detail(id: str = Query(..., min_length=1)) -> dict[str, object]:
        return live_service.model_detail(model_id=id)

    @app.post("/api/models/preflight")
    async def model_preflight(body: ModelPreflightRequest) -> dict[str, object]:
        submission = BuildSubmission(
            model_id=body.model_id,
            task=CandidateModality(body.task),
            task_profile=body.task_profile,
            hf_revision=body.hf_revision,
            skip_olive=body.skip_olive,
        )
        return live_service.preflight(submission)

    @app.post("/api/builds")
    async def create_build(
        body: BuildCreateRequest,
        idempotency_key: str = Header(..., alias="Idempotency-Key"),
    ) -> dict[str, object]:
        submission = BuildSubmission(
            model_id=body.model_id,
            task=CandidateModality(body.task),
            task_profile=body.task_profile,
            hf_revision=body.hf_revision,
            skip_olive=body.skip_olive,
        )
        job, replay = live_service.create_build(submission, idempotency_key=idempotency_key)
        return {"idempotent_replay": replay, "job": to_jsonable(job)}

    @app.get("/api/builds/{job_id}")
    async def get_build(job_id: str) -> dict[str, object]:
        return to_jsonable(live_service.get_build(job_id))

    @app.get("/api/builds/{job_id}/events")
    async def get_events(job_id: str, after: int = Query(default=0, ge=0)) -> dict[str, object]:
        events = live_service.get_events(job_id, after=after)
        return {"job_id": job_id, "after": after, "events": to_jsonable(events)}

    @app.post("/api/builds/{job_id}/cancel")
    async def cancel_build(job_id: str, body: CancelRequest | None = None) -> dict[str, object]:
        reason = body.reason if body is not None else "Cancelled by client request."
        job, _ = live_service.cancel_build(job_id, reason=reason)
        return to_jsonable(job)

    @app.post("/api/artifacts/{artifact_id}/infer/text")
    async def infer_text(artifact_id: str, body: TextInferenceRequest) -> dict[str, object]:
        return live_service.infer_text(
            artifact_id=artifact_id,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
        )

    @app.post("/api/artifacts/{artifact_id}/infer/asr")
    async def infer_asr(artifact_id: str, audio: UploadFile = File(...)) -> dict[str, object]:
        payload = await audio.read()
        if len(payload) == 0:
            raise ServiceError(
                code="INVALID_AUDIO_PAYLOAD",
                message="Audio payload must not be empty.",
                status_code=400,
            )
        return live_service.infer_asr(
            artifact_id=artifact_id,
            audio_bytes=payload,
            filename=audio.filename or "audio.bin",
        )

    static_root = (web_dist or default_web_dist()).resolve()
    if static_root.is_dir():
        app.mount("/assets", StaticFiles(directory=static_root / "assets"), name="web-assets")

        @app.get("/", include_in_schema=False)
        async def web_index() -> FileResponse:
            return FileResponse(static_root / "index.html")

    return app
