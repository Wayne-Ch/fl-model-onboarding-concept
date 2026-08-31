from __future__ import annotations

import time

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from fl_model_onboarding.contracts import (
    ArtifactKind,
    BuildArtifact,
    BuildJob,
    BuildRequest,
    CandidateModality,
    CatalogMatchAssessment,
    JobState,
    MatchConfidence,
    PreflightResult,
    ToolAvailability,
)
from fl_model_onboarding.local_api import create_app
from fl_model_onboarding.local_service import BuildSubmission, LocalOnboardingService
from fl_model_onboarding.state_machine import transition

_TOOLS = (
    ToolAvailability("foundry", "command", True, "0.11.0"),
    ToolAvailability("mobius", "command", True, "0.1.0"),
    ToolAvailability("olive", "command", True, "0.13.0"),
    ToolAvailability("onnxruntime", "python-package", True, "1.26.0"),
    ToolAvailability("onnxruntime-genai", "python-package", True, "0.14.0"),
    ToolAvailability("foundry-local-sdk", "python-package", True, "1.2.0"),
    ToolAvailability("huggingface_hub", "python-package", True, "1.22.0"),
)


class FakeHFMetadata:
    def search_models(self, query: str, limit: int = 20, sort: str = "downloads"):  # noqa: ANN001
        from fl_model_onboarding.adapters.interfaces import HuggingFaceSearchResult

        return (
            HuggingFaceSearchResult(
                model_id=f"{query}-one",
                downloads=100,
                likes=20,
                last_modified="2026-01-01T00:00:00Z",
            ),
            HuggingFaceSearchResult(
                model_id=f"{query}-two",
                downloads=50,
                likes=10,
                last_modified="2026-01-02T00:00:00Z",
            ),
        )[:limit]

    def get_metadata(self, model_id: str, revision: str | None = None, files_metadata: bool = False):  # noqa: ANN001
        from fl_model_onboarding.adapters.interfaces import HuggingFaceMetadata

        config: dict[str, object] = {"model_type": "llama"}
        is_gated = False
        if "gated" in model_id:
            is_gated = True
        if "remote" in model_id:
            config = {"auto_map": {"AutoModelForCausalLM": "remote.module.Model"}}
        if "whisper" in model_id:
            config = {"model_type": "whisper"}
        return HuggingFaceMetadata(
            model_id=model_id,
            revision=revision or "rev-1",
            sha="1234567890abcdef1234567890abcdef12345678",
            is_private=False,
            is_gated=is_gated,
            last_modified="2026-01-01T00:00:00Z",
            config=config,
            safetensors_total_bytes=1024,
            safetensors_parameter_count=256,
            card_data={"license": "apache-2.0"},
            sibling_count=10,
        )


class FakeFoundryCatalog:
    def list_matches(self, search_query: str):
        return (
            CatalogMatchAssessment(
                alias=search_query,
                model_or_variant_id=f"{search_query}-generic-gpu:2",
                source_schema="models",
                confidence=MatchConfidence.MEDIUM,
                reason="Likely alias match by name similarity only",
                cached=False,
                model_type="Unknown",
            ),
            CatalogMatchAssessment(
                alias=search_query,
                model_or_variant_id=f"{search_query}-variant:1",
                source_schema="variants",
                confidence=MatchConfidence.LOW,
                reason="Weak alias match",
                cached=None,
                model_type="Unknown",
            ),
        )

    def model_info(self, model_ref: str) -> dict[str, object]:
        return {"alias": model_ref}

    def cache_location(self) -> Path:
        return Path("C:\\cache")

    def status(self) -> dict[str, object]:
        return {"service": {"state": "ready"}}


class PassingPreflightInspector:
    def inspect(self, request: BuildRequest) -> PreflightResult:
        return PreflightResult(
            candidate=request.candidate,
            workspace_root=request.workspace_root,
            model_cache_dir=request.model_cache_dir,
            output_dir=request.output_dir,
            disk_free_gb_workspace=100.0,
            disk_free_gb_cache=100.0,
            tools=_TOOLS,
            foundry_catalog_matches=(),
            huggingface_revision=request.hf_revision or "rev-1",
            huggingface_sha="1234567890abcdef1234567890abcdef12345678",
            huggingface_private=False,
            huggingface_gated=False,
            cache_key=f"cache::{request.candidate.huggingface_model_id}::{request.task_profile}",
            blockers=(),
            warnings=(),
        )


class DeterministicSuccessRunner:
    def run(self, job: BuildJob, *, persist, cancellation_event):  # noqa: ANN001
        if cancellation_event.is_set():
            return
        for stage in (
            JobState.DOWNLOADING,
            JobState.MOBIUS_BUILDING,
            JobState.MOBIUS_VALIDATING,
            JobState.OLIVE_OPTIMIZING,
            JobState.PACKAGING,
            JobState.RUNTIME_VALIDATING,
            JobState.FL_LOADING,
            JobState.INFERENCING,
        ):
            transition(job, stage, f"Reached '{stage.value}'")
            persist()
            if cancellation_event.is_set():
                return
        artifact_path = job.request.output_dir / "artifact.bin"
        artifact_path.write_bytes(b"artifact")
        job.register_artifact(
            BuildArtifact(
                artifact_id=f"artifact-{job.job_id}",
                kind=ArtifactKind.MODEL,
                path=artifact_path,
                description="runtime artifact",
                size_bytes=artifact_path.stat().st_size,
            )
        )
        transition(job, JobState.SUCCEEDED, "Build succeeded.")
        job.finished_utc = datetime.now(timezone.utc)
        persist()


class SlowCancellableRunner:
    def run(self, job: BuildJob, *, persist, cancellation_event):  # noqa: ANN001
        if cancellation_event.is_set():
            return
        transition(job, JobState.DOWNLOADING, "Downloading model data.")
        partial = job.request.output_dir / "partial.bin"
        partial.write_bytes(b"x")
        persist()
        for _ in range(200):
            if cancellation_event.is_set():
                return
            time.sleep(0.01)


class EchoTextBackend:
    def infer(self, *, artifact, job, prompt: str, max_tokens: int) -> str:  # noqa: ANN001
        return f"{artifact.artifact_id}:{job.job_id}:{prompt}:{max_tokens}"


class EchoAsrBackend:
    def infer(self, *, artifact, job, audio_bytes: bytes, filename: str) -> str:  # noqa: ANN001
        return f"{artifact.artifact_id}:{job.job_id}:{filename}:{len(audio_bytes)}"


def _service(
    tmp_path: Path,
    *,
    stage_runner=None,
    text_backend=None,
    asr_backend=None,
) -> LocalOnboardingService:
    return LocalOnboardingService(
        db_path=tmp_path / "state.sqlite3",
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        hf_metadata=FakeHFMetadata(),  # type: ignore[arg-type]
        foundry_catalog=FakeFoundryCatalog(),  # type: ignore[arg-type]
        preflight_inspector=PassingPreflightInspector(),  # type: ignore[arg-type]
        build_stage_runner=stage_runner,  # type: ignore[arg-type]
        text_inference_backend=text_backend,  # type: ignore[arg-type]
        asr_inference_backend=asr_backend,  # type: ignore[arg-type]
    )


def _wait_for_terminal(client: TestClient, job_id: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/builds/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["state"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for terminal state for job {job_id}")


def test_health_and_cors_allow_only_local_origins(tmp_path: Path) -> None:
    service = _service(tmp_path)
    app = create_app(service=service)
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "storage_path" in payload

        allow = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert allow.status_code in {200, 204}
        assert allow.headers.get("access-control-allow-origin") == "http://localhost:3000"

        deny = client.options(
            "/api/health",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert deny.headers.get("access-control-allow-origin") is None


def test_model_search_detail_and_policy_blockers(tmp_path: Path) -> None:
    app = create_app(service=_service(tmp_path))
    with TestClient(app) as client:
        search = client.get("/api/models/search", params={"q": "smol", "limit": 2})
        assert search.status_code == 200
        body = search.json()
        assert body["query"] == "smol"
        assert len(body["results"]) == 2

        gated = client.get("/api/models/detail", params={"id": "owner/gated-model"})
        assert gated.status_code == 200
        gated_body = gated.json()
        assert gated_body["buildable"] is False
        assert "gated_model_blocked" in gated_body["build_blockers"]

        remote = client.get("/api/models/detail", params={"id": "owner/remote-model"})
        assert remote.status_code == 200
        remote_body = remote.json()
        assert remote_body["buildable"] is False
        assert "remote_code_blocked" in remote_body["build_blockers"]


def test_preflight_build_idempotency_and_event_polling(tmp_path: Path) -> None:
    app = create_app(service=_service(tmp_path))
    with TestClient(app) as client:
        payload = {
            "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "task": "llm",
            "task_profile": "llm-cpu-int4",
        }
        first_preflight = client.post("/api/models/preflight", json=payload)
        assert first_preflight.status_code == 200
        assert first_preflight.json()["cached"] is False
        second_preflight = client.post("/api/models/preflight", json=payload)
        assert second_preflight.status_code == 200
        assert second_preflight.json()["cached"] is True

        first_build = client.post(
            "/api/builds",
            headers={"Idempotency-Key": "idem-1"},
            json=payload,
        )
        assert first_build.status_code == 200
        first_json = first_build.json()
        assert first_json["idempotent_replay"] is False
        job_id = first_json["job"]["job_id"]

        replay = client.post(
            "/api/builds",
            headers={"Idempotency-Key": "idem-1"},
            json=payload,
        )
        assert replay.status_code == 200
        replay_json = replay.json()
        assert replay_json["idempotent_replay"] is True
        assert replay_json["job"]["job_id"] == job_id

        conflict = client.post(
            "/api/builds",
            headers={"Idempotency-Key": "idem-1"},
            json={
                "model_id": payload["model_id"],
                "task": payload["task"],
                "task_profile": "different-profile",
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "IDEMPOTENCY_BODY_CONFLICT"

        completed = _wait_for_terminal(client, job_id)
        assert completed["state"] in {"failed", "cancelled", "succeeded"}

        events = client.get(f"/api/builds/{job_id}/events", params={"after": 0})
        assert events.status_code == 200
        event_payload = events.json()
        assert event_payload["job_id"] == job_id
        assert len(event_payload["events"]) >= 1
        last_sequence = event_payload["events"][-1]["sequence"]
        empty = client.get(f"/api/builds/{job_id}/events", params={"after": last_sequence})
        assert empty.status_code == 200
        assert empty.json()["events"] == []


def test_asr_preflight_is_a_structured_unsupported_blocker(tmp_path: Path) -> None:
    app = create_app(service=_service(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/models/preflight",
            json={
                "model_id": "distil-whisper/distil-medium.en",
                "task": "asr",
                "task_profile": "asr-cpu-fp16",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is False
        blocker = payload["result"]["blockers"][0]
        assert blocker["classification"] == "oga_runtime_contract_incompatible"
        assert blocker["detail"]["decoder_input_ids"] == "parse_error"
        outcome = payload["candidate_outcome"]
        assert outcome["model_id"] == "distil-whisper/distil-medium.en"
        assert outcome["revision"] == "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7"
        assert outcome["versions"]["mobius"] == "0.1.0"
        assert outcome["versions"]["onnxruntime_genai"] == "0.15.2"
        assert outcome["versions"]["foundry_local_sdk"] == "1.2.4"
        assert outcome["gate_outcomes"][-1]["status"] == "failed"
        assert outcome["tested_status"] == "not_verified"
        assert client.get("/api/health").json()["compatibility_index"] == []

        detail = client.get(
            "/api/models/detail",
            params={"id": "distil-whisper/distil-medium.en"},
        ).json()
        assert detail["candidate_outcome"]["classification"] == (
            "oga_runtime_contract_incompatible"
        )


def test_cancel_endpoint_and_conflict(tmp_path: Path) -> None:
    app = create_app(service=_service(tmp_path, stage_runner=SlowCancellableRunner()))
    with TestClient(app) as client:
        payload = {
            "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "task": "llm",
            "task_profile": "llm-cpu-int4",
        }
        created = client.post(
            "/api/builds",
            headers={"Idempotency-Key": "cancel-1"},
            json=payload,
        )
        assert created.status_code == 200
        job_id = created.json()["job"]["job_id"]

        cancelled = client.post(
            f"/api/builds/{job_id}/cancel",
            json={"reason": "Cancelled by API test"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "cancelled"

        second_cancel = client.post(f"/api/builds/{job_id}/cancel", json={"reason": "again"})
        assert second_cancel.status_code == 409
        assert second_cancel.json()["code"] == "JOB_NOT_CANCELLABLE"


def test_artifact_scoped_inference_routes(tmp_path: Path) -> None:
    app = create_app(
        service=_service(
            tmp_path,
            stage_runner=DeterministicSuccessRunner(),
            text_backend=EchoTextBackend(),
            asr_backend=EchoAsrBackend(),
        )
    )
    with TestClient(app) as client:
        llm_payload = {
            "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "task": "llm",
            "task_profile": "llm-cpu-int4",
        }
        llm_build = client.post("/api/builds", headers={"Idempotency-Key": "infer-llm"}, json=llm_payload)
        assert llm_build.status_code == 200
        llm_job_id = llm_build.json()["job"]["job_id"]
        llm_completed = _wait_for_terminal(client, llm_job_id)
        assert llm_completed["state"] == "succeeded"
        llm_artifact = llm_completed["result_artifact_id"]
        assert llm_artifact

        text = client.post(
            f"/api/artifacts/{llm_artifact}/infer/text",
            json={"prompt": "hello", "max_tokens": 32},
        )
        assert text.status_code == 200
        assert text.json()["artifact_id"] == llm_artifact
        assert "hello" in text.json()["output"]

        wrong_task = client.post(
            f"/api/artifacts/{llm_artifact}/infer/asr",
            files={"audio": ("sample.wav", b"\x00\x01", "audio/wav")},
        )
        assert wrong_task.status_code == 409
        assert wrong_task.json()["code"] == "ARTIFACT_TASK_MISMATCH"

        asr_payload = {
            "model_id": "distil-whisper/distil-medium.en",
            "task": "asr",
            "task_profile": "asr-cpu-fp16",
        }
        asr_build = client.post("/api/builds", headers={"Idempotency-Key": "infer-asr"}, json=asr_payload)
        assert asr_build.status_code == 200
        asr_job_id = asr_build.json()["job"]["job_id"]
        asr_completed = _wait_for_terminal(client, asr_job_id)
        assert asr_completed["state"] == "succeeded"
        asr_artifact = asr_completed["result_artifact_id"]
        assert asr_artifact

        asr = client.post(
            f"/api/artifacts/{asr_artifact}/infer/asr",
            files={"audio": ("speech.wav", b"\x01\x02\x03", "audio/wav")},
        )
        assert asr.status_code == 200
        assert asr.json()["artifact_id"] == asr_artifact
        assert "speech.wav:3" in asr.json()["transcript"]

        empty = client.post(
            f"/api/artifacts/{asr_artifact}/infer/asr",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert empty.status_code == 400
        assert empty.json()["code"] == "INVALID_AUDIO_PAYLOAD"


def test_tested_model_index_requires_successful_inference_and_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    service = _service(
        tmp_path,
        stage_runner=DeterministicSuccessRunner(),
        text_backend=EchoTextBackend(),
    )
    app = create_app(service=service)
    with TestClient(app) as client:
        unverified = client.get("/api/models/detail", params={"id": "owner/model"}).json()
        assert unverified["verification"] == {
            "status": "not_verified",
            "evidence": "none",
            "artifact_id": None,
            "verified_utc": None,
        }

        created = client.post(
            "/api/builds",
            headers={"Idempotency-Key": "tested-index"},
            json={"model_id": "owner/model", "task": "llm", "task_profile": "llm-cpu-int4"},
        )
        completed = _wait_for_terminal(client, created.json()["job"]["job_id"])
        assert completed["state"] == "succeeded"
        assert client.get("/api/health").json()["compatibility_index"] == []

        artifact_id = completed["result_artifact_id"]
        inferred = client.post(
            f"/api/artifacts/{artifact_id}/infer/text",
            json={"prompt": "verify"},
        )
        assert inferred.status_code == 200
        tested = client.get("/api/health").json()["compatibility_index"]
        assert len(tested) == 1
        assert tested[0]["model_id"] == "owner/model"
        assert tested[0]["evidence"] == "successful_fl_inference"

    restarted = LocalOnboardingService(
        db_path=db_path,
        workspace_base=tmp_path / "w",
        model_cache_dir=tmp_path / "cache",
        hf_metadata=FakeHFMetadata(),  # type: ignore[arg-type]
        foundry_catalog=FakeFoundryCatalog(),  # type: ignore[arg-type]
        preflight_inspector=PassingPreflightInspector(),  # type: ignore[arg-type]
    )
    try:
        assert restarted.health()["compatibility_index"][0]["model_id"] == "owner/model"  # type: ignore[index]
    finally:
        restarted.close()


def test_built_ui_is_served_without_replacing_repository_concept(tmp_path: Path) -> None:
    web_dist = tmp_path / "web-dist"
    assets = web_dist / "assets"
    assets.mkdir(parents=True)
    (web_dist / "index.html").write_text("<html>packaged-ui</html>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('ui')", encoding="utf-8")
    concept_before = Path("index.html").read_bytes()

    app = create_app(service=_service(tmp_path), web_dist=web_dist)
    with TestClient(app) as client:
        index = client.get("/")
        asset = client.get("/assets/app.js")
        assert index.status_code == 200
        assert "packaged-ui" in index.text
        assert asset.status_code == 200
        assert "console.log" in asset.text

    assert Path("index.html").read_bytes() == concept_before


def test_text_inference_prompt_is_bounded(tmp_path: Path) -> None:
    app = create_app(service=_service(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/artifacts/missing/infer/text",
            json={"prompt": "x" * 8193},
        )
        assert response.status_code == 422
