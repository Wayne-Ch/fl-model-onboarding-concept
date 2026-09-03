from __future__ import annotations

import sys

from pathlib import Path
from threading import Event

from fl_model_onboarding.adapters.interfaces import CommandResult, CommandSpec, HuggingFaceMetadata
from fl_model_onboarding.candidates import PHASE0_CANDIDATES
from fl_model_onboarding.contracts import CatalogMatchAssessment, MatchConfidence
from fl_model_onboarding.contracts import BuildRequest, FailureClassification, JobState
from fl_model_onboarding.preflight import PreflightInspector


def _is_python_probe(spec: CommandSpec) -> bool:
    command = Path(spec.argv[0]).name.lower()
    return command.startswith("python") and len(spec.argv) >= 3 and spec.argv[1] == "-c"


def _package_probe_response(spec: CommandSpec) -> CommandResult:
    source = spec.argv[2]
    for pkg, version in (
        ("onnxruntime", "1.29.0"),
        ("onnxruntime-genai", "0.15.2"),
        ("foundry-local-sdk", "1.2.4"),
        ("huggingface_hub", "1.22.0"),
    ):
        if f"version('{pkg}')" in source:
            return CommandResult(spec=spec, exit_code=0, stdout=f"{version}\n", stderr="")
    raise RuntimeError(f"Unhandled fake python probe: {spec.argv}")


class FakeRunner:
    def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
        cmd = spec.argv[0]
        if cmd == "foundry":
            return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
        if cmd == "mobius":
            raise FileNotFoundError("mobius not found")
        if cmd == "olive":
            return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
        if _is_python_probe(spec):
            return _package_probe_response(spec)
        raise RuntimeError(f"Unhandled fake command: {spec.argv}")


class FakeFoundry:
    def list_matches(self, search_query: str) -> tuple[CatalogMatchAssessment, ...]:
        return ()

    def model_info(self, model_ref: str) -> dict[str, object]:
        return {"alias": model_ref}

    def cache_location(self) -> Path:
        raise NotImplementedError

    def status(self) -> dict[str, object]:
        return {}


class FakeHF:
    def search_models(self, query: str, limit: int = 20, sort: str = "downloads"):  # noqa: ANN001
        return ()

    def get_metadata(
        self,
        model_id: str,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> HuggingFaceMetadata:
        return HuggingFaceMetadata(
            model_id=model_id,
            revision=revision or "1234567890abcdef1234567890abcdef12345678",
            sha="1234567890abcdef1234567890abcdef12345678",
            is_private=False,
            is_gated=False,
            last_modified="2026-01-01T00:00:00Z",
            config={"model_type": "llama"},
            safetensors_total_bytes=1024,
            safetensors_parameter_count=256,
            card_data={"license": "apache-2.0"},
            sibling_count=12,
        )


def _request(workspace: Path, output: Path) -> BuildRequest:
    return BuildRequest(
        candidate=PHASE0_CANDIDATES["smollm2-1.7b-instruct"],
        workspace_root=workspace,
        model_cache_dir=workspace / "cache",
        output_dir=output,
    )


def test_preflight_reports_missing_mobius(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), FakeHF())
    result = inspector.inspect(request)
    assert not result.ok
    assert any(
        blocker.stage == JobState.MOBIUS_BUILDING
        and blocker.classification == FailureClassification.MISSING_DEPENDENCY
        for blocker in result.blockers
    )
    assert any("not found in Foundry catalog" in warning for warning in result.warnings)


def test_preflight_reports_path_containment_blocker(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()
    outside = tmp_path.parent / "outside-output"
    request = _request(tmp_path, outside)
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), FakeHF())
    result = inspector.inspect(request)
    assert not result.ok
    assert any(
        blocker.classification == FailureClassification.PATH_CONTAINMENT
        for blocker in result.blockers
    )


def test_preflight_keeps_likely_match_semantics(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class FoundryLikely(FakeFoundry):
        def list_matches(self, search_query: str) -> tuple[CatalogMatchAssessment, ...]:
            return (
                CatalogMatchAssessment(
                    alias="qwen3.5-0.8b",
                    model_or_variant_id="qwen3.5-0.8b-generic-gpu:2",
                    source_schema="models",
                    confidence=MatchConfidence.MEDIUM,
                    reason="Likely alias match by name similarity only",
                ),
            )

    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FoundryLikely(), FakeHF())
    result = inspector.inspect(request)
    assert result.foundry_catalog_matches
    assert result.foundry_catalog_matches[0].confidence == MatchConfidence.MEDIUM
    assert result.cache_key is not None


def test_preflight_rejects_gated_models(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class GatedHF(FakeHF):
        def get_metadata(  # type: ignore[override]
            self, model_id: str, revision: str | None = None, files_metadata: bool = False
        ) -> HuggingFaceMetadata:
            base = super().get_metadata(model_id, revision=revision, files_metadata=files_metadata)
            return HuggingFaceMetadata(
                model_id=base.model_id,
                revision=base.revision,
                sha=base.sha,
                is_private=base.is_private,
                is_gated=True,
                last_modified=base.last_modified,
                config=base.config,
                safetensors_total_bytes=base.safetensors_total_bytes,
                safetensors_parameter_count=base.safetensors_parameter_count,
                card_data=base.card_data,
                sibling_count=base.sibling_count,
            )

    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), GatedHF())
    result = inspector.inspect(request)
    assert any("rejects gated" in blocker.message for blocker in result.blockers)


def test_preflight_rejects_remote_code_models(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class AutoMapHF(FakeHF):
        def get_metadata(  # type: ignore[override]
            self, model_id: str, revision: str | None = None, files_metadata: bool = False
        ) -> HuggingFaceMetadata:
            base = super().get_metadata(model_id, revision=revision, files_metadata=files_metadata)
            return HuggingFaceMetadata(
                model_id=base.model_id,
                revision=base.revision,
                sha=base.sha,
                is_private=base.is_private,
                is_gated=base.is_gated,
                last_modified=base.last_modified,
                config={"auto_map": {"AutoModelForCausalLM": "remote.module.Model"}},
                safetensors_total_bytes=base.safetensors_total_bytes,
                safetensors_parameter_count=base.safetensors_parameter_count,
                card_data=base.card_data,
                sibling_count=base.sibling_count,
            )

    request = _request(tmp_path, tmp_path / "out")
    inspector = PreflightInspector(FakeRunner(), FakeFoundry(), AutoMapHF())
    result = inspector.inspect(request)
    assert any("requires remote code" in blocker.message for blocker in result.blockers)


def test_preflight_mobius_help_success_with_unsupported_version_still_available(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            argv = spec.argv
            if argv[:2] == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
            if argv[:2] == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius help\n", stderr="")
            if argv[:2] == ("mobius", "--version"):
                return CommandResult(spec=spec, exit_code=2, stdout="", stderr="unknown option --version\n")
            if argv[:2] == ("olive", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
            if argv[:2] == ("olive", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0\n", stderr="")
            if _is_python_probe(spec):
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    request = _request(tmp_path, tmp_path / "out")
    result = PreflightInspector(Runner(), FakeFoundry(), FakeHF()).inspect(request)
    mobius = next(tool for tool in result.tools if tool.name == "mobius")
    assert mobius.available is True
    assert not any(
        blocker.stage == JobState.MOBIUS_BUILDING and blocker.classification == FailureClassification.MISSING_DEPENDENCY
        for blocker in result.blockers
    )
    assert result.ok


def test_preflight_olive_help_success_with_unsupported_version_still_available(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            argv = spec.argv
            if argv[:2] == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
            if argv[:2] == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0 help\n", stderr="")
            if argv[:2] == ("mobius", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0\n", stderr="")
            if argv[:2] == ("olive", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
            if argv[:2] == ("olive", "--version"):
                return CommandResult(spec=spec, exit_code=2, stdout="", stderr="unrecognized arguments: --version\n")
            if _is_python_probe(spec):
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    request = _request(tmp_path, tmp_path / "out")
    result = PreflightInspector(Runner(), FakeFoundry(), FakeHF()).inspect(request)
    olive = next(tool for tool in result.tools if tool.name == "olive")
    assert olive.available is True
    assert not any(
        blocker.stage == JobState.OLIVE_OPTIMIZING
        and blocker.classification == FailureClassification.MISSING_DEPENDENCY
        for blocker in result.blockers
    )
    assert result.ok


def test_preflight_olive_probe_uses_extended_timeout(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def __init__(self) -> None:
            self.timeouts: dict[tuple[str, ...], int] = {}

        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            argv = tuple(spec.argv)
            self.timeouts[argv] = spec.timeout_seconds
            if argv == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
            if argv == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0 help\n", stderr="")
            if argv == ("mobius", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0\n", stderr="")
            if argv == ("olive", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
            if argv == ("olive", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0\n", stderr="")
            if _is_python_probe(spec):
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    runner = Runner()
    request = _request(tmp_path, tmp_path / "out")
    result = PreflightInspector(runner, FakeFoundry(), FakeHF()).inspect(request)
    assert result.ok
    assert runner.timeouts[("mobius", "--help")] == 30
    assert runner.timeouts[("mobius", "--version")] == 30
    assert runner.timeouts[("olive", "--help")] == 90
    assert runner.timeouts[("olive", "--version")] == 90


def test_preflight_olive_probe_tolerates_slow_startup(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            argv = tuple(spec.argv)
            if argv == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
            if argv == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0 help\n", stderr="")
            if argv == ("mobius", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0\n", stderr="")
            if argv in {("olive", "--help"), ("olive", "--version")}:
                if spec.timeout_seconds <= 30:
                    raise TimeoutError(f"Command timed out after {spec.timeout_seconds}s: {spec.argv}")
                if argv == ("olive", "--help"):
                    return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0\n", stderr="")
            if _is_python_probe(spec):
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    request = _request(tmp_path, tmp_path / "out")
    result = PreflightInspector(Runner(), FakeFoundry(), FakeHF()).inspect(request)
    olive = next(tool for tool in result.tools if tool.name == "olive")
    assert olive.available is True
    assert result.ok


def test_preflight_package_probe_uses_current_sys_executable(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def __init__(self) -> None:
            self.python_commands: list[str] = []

        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            argv = spec.argv
            if argv[:2] == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
            if argv[:2] == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0 help\n", stderr="")
            if argv[:2] == ("mobius", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0\n", stderr="")
            if argv[:2] == ("olive", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0 help\n", stderr="")
            if argv[:2] == ("olive", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0\n", stderr="")
            if _is_python_probe(spec):
                self.python_commands.append(spec.argv[0])
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    runner = Runner()
    request = _request(tmp_path, tmp_path / "out")
    result = PreflightInspector(runner, FakeFoundry(), FakeHF()).inspect(request)
    assert result.ok
    assert runner.python_commands
    expected = Path(sys.executable).resolve()
    assert all(Path(command).resolve() == expected for command in runner.python_commands)


def test_preflight_probe_forwards_cancellation_event(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def __init__(self) -> None:
            self.cancel_events: list[object | None] = []

        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            self.cancel_events.append(cancel_event)
            argv = spec.argv
            if argv[:2] == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
            if argv[:2] == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0 help\n", stderr="")
            if argv[:2] == ("mobius", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0\n", stderr="")
            if argv[:2] == ("olive", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0 help\n", stderr="")
            if argv[:2] == ("olive", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0\n", stderr="")
            if _is_python_probe(spec):
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    runner = Runner()
    request = _request(tmp_path, tmp_path / "out")
    cancellation = Event()
    result = PreflightInspector(runner, FakeFoundry(), FakeHF()).inspect(
        request,
        cancellation_event=cancellation,
    )
    assert result.ok
    assert runner.cancel_events
    assert all(event is cancellation for event in runner.cancel_events)


def test_preflight_probe_failure_is_not_reported_as_missing_dependency(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            argv = spec.argv
            if argv[:2] == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="0.11.0\n", stderr="")
            if argv[:2] == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=2, stdout="", stderr="unable to load config\n")
            if argv[:2] == ("olive", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0 help\n", stderr="")
            if argv[:2] == ("olive", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0\n", stderr="")
            if _is_python_probe(spec):
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    request = _request(tmp_path, tmp_path / "out")
    result = PreflightInspector(Runner(), FakeFoundry(), FakeHF()).inspect(request)
    mobius_blocker = next(
        blocker for blocker in result.blockers if blocker.stage == JobState.MOBIUS_BUILDING
    )
    assert mobius_blocker.classification == FailureClassification.TOOL_UNAVAILABLE
    assert mobius_blocker.detail["tool"] == "mobius"
    assert "probe-failed" in mobius_blocker.detail["probe_detail"]


def test_preflight_smollm_installed_tool_fixture_is_buildable_with_expected_versions(tmp_path: Path) -> None:
    (tmp_path / "cache").mkdir()

    class Runner:
        def run(self, spec: CommandSpec, cancel_event=None) -> CommandResult:  # noqa: ANN001
            argv = spec.argv
            if argv[:2] == ("foundry", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="foundry 0.11.0\n", stderr="")
            if argv[:2] == ("mobius", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius help\n", stderr="")
            if argv[:2] == ("mobius", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="mobius 0.1.0\n", stderr="")
            if argv[:2] == ("olive", "--help"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive help\n", stderr="")
            if argv[:2] == ("olive", "--version"):
                return CommandResult(spec=spec, exit_code=0, stdout="olive 0.13.0\n", stderr="")
            if _is_python_probe(spec):
                return _package_probe_response(spec)
            raise RuntimeError(f"Unhandled fake command: {spec.argv}")

    request = _request(tmp_path, tmp_path / "out")
    result = PreflightInspector(Runner(), FakeFoundry(), FakeHF()).inspect(request)
    versions = {tool.name: tool.version for tool in result.tools}
    assert result.ok
    assert versions["mobius"] == "0.1.0"
    assert versions["olive"] == "0.13.0"
    assert versions["foundry"] == "0.11.0"
    assert versions["onnxruntime"] == "1.29.0"
    assert versions["onnxruntime-genai"] == "0.15.2"
    assert versions["foundry-local-sdk"] == "1.2.4"
