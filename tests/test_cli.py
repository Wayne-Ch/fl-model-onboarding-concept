from __future__ import annotations

import json

from pathlib import Path

import pytest

from fl_model_onboarding.cli import main
from fl_model_onboarding.contracts import BuildJob, BuildRequest, CandidateModality, JobState, ModelCandidate


class StubService:
    def __init__(self, tmp_path: Path, *, preflight_ok: bool = True) -> None:
        request = BuildRequest(
            candidate=ModelCandidate(
                key="llm-test-model",
                huggingface_model_id="owner/model",
                modality=CandidateModality.LLM,
                recommended_mobius_dtype="f16",
                recommended_olive_precision="int4",
                notes="stub",
            ),
            workspace_root=tmp_path / "w",
            model_cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
            task_profile="default",
            dry_run=False,
        )
        self.job = BuildJob(job_id="job-1", request=request, state=JobState.FAILED)
        self.job.add_event("stub event")
        self.closed = False
        self.preflight_ok = preflight_ok

    def close(self) -> None:
        self.closed = True

    def health(self) -> dict[str, object]:
        return {"status": "ok", "jobs_total": 1, "storage_path": "C:\\state.sqlite3"}

    def preflight(self, _submission) -> dict[str, object]:  # noqa: ANN001
        return {"cache_key": "cache-1", "ok": self.preflight_ok, "cached": False, "result": {}}

    def search_models(self, *, query: str, limit: int) -> dict[str, object]:
        return {"query": query, "limit": limit, "results": [{"model_id": "owner/model"}]}

    def model_detail(self, *, model_id: str) -> dict[str, object]:
        return {
            "model_id": model_id,
            "buildable": True,
            "build_blockers": [],
            "task_hints": ["llm"],
            "foundry_catalog_matches": [],
            "warnings": [],
        }

    def create_build(self, _submission, idempotency_key: str):  # noqa: ANN001
        assert idempotency_key == "idem-1"
        return self.job, False

    def get_build(self, *, job_id: str) -> BuildJob:
        assert job_id == self.job.job_id
        return self.job

    def cancel_build(self, *, job_id: str, reason: str):  # noqa: ARG002
        assert job_id == self.job.job_id
        self.job.state = JobState.CANCELLED
        return self.job, None


def test_version_command_outputs_version(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main(["version"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "0.1.0"


def test_doctor_returns_nonzero_when_preflight_not_ok(monkeypatch, tmp_path: Path) -> None:
    stub = StubService(tmp_path, preflight_ok=False)
    monkeypatch.setattr("fl_model_onboarding.cli._service_from_args", lambda _: stub)
    code = main(
        [
            "doctor",
            "--model-id",
            "owner/model",
            "--task",
            "llm",
        ]
    )
    assert code == 2
    assert stub.closed


def test_model_search_and_build_commands_use_service(monkeypatch, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    stub = StubService(tmp_path)
    monkeypatch.setattr("fl_model_onboarding.cli._service_from_args", lambda _: stub)

    search_code = main(["model", "search", "--query", "smol", "--limit", "5"])
    assert search_code == 0
    search_out = capsys.readouterr().out
    assert '"query": "smol"' in search_out

    create_code = main(
        [
            "build",
            "create",
            "--model-id",
            "owner/model",
            "--task",
            "llm",
            "--idempotency-key",
            "idem-1",
        ]
    )
    assert create_code == 0
    create_out = capsys.readouterr().out
    assert '"job_id": "job-1"' in create_out

    status_code = main(["build", "status", "--job-id", "job-1"])
    assert status_code == 0
    status_out = capsys.readouterr().out
    assert '"state": "failed"' in status_out

    cancel_code = main(["build", "cancel", "--job-id", "job-1"])
    assert cancel_code == 0
    cancel_out = capsys.readouterr().out
    assert '"state": "cancelled"' in cancel_out


def test_service_serve_rejects_non_loopback_by_default(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main(["service", "serve", "--host", "0.0.0.0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "allow-non-loopback" in captured.err


def test_build_create_requires_idempotency_key() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "build",
                "create",
                "--model-id",
                "owner/model",
                "--task",
                "llm",
            ]
        )
    assert exc.value.code == 2


def test_recipe_agent_frozen_commands(capsys) -> None:  # type: ignore[no-untyped-def]
    manifest = Path("evaluation") / "recipe-agent-v1" / "models.json"

    list_code = main(["recipe-agent", "frozen-list", "--path", str(manifest)])
    assert list_code == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 5
    assert len(listed["models"]) == 5

    validate_code = main(["recipe-agent", "frozen-validate", "--path", str(manifest)])
    assert validate_code == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True
    assert validated["model_count"] == 5

    dry_run_code = main(["recipe-agent", "frozen-dry-run", "--path", str(manifest)])
    assert dry_run_code == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["valid"] is True
    assert dry_run["tool_execution"] == "disabled"
    assert dry_run["summary"]["total_models"] == 5
    assert dry_run["summary"]["compiled_models"] == 5
    assert all(row["status"] == "compiled" for row in dry_run["outcomes"])


def test_recipe_agent_frozen_validate_rejects_invalid_manifest(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    invalid_manifest = tmp_path / "invalid-frozen.json"
    invalid_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "selection_timestamp": "2026-01-01T00:00:00Z",
                "models": [
                    {
                        "index": 1,
                        "model_id": "owner/only-one-model",
                        "sha": "1234567890abcdef1234567890abcdef12345678",
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "catalog_match": {"matched": False, "confidence": "none", "reason": "none", "matched_entry": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(["recipe-agent", "frozen-validate", "--path", str(invalid_manifest)])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert any("exactly five entries" in message for message in payload["errors"])
