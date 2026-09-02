from __future__ import annotations

import importlib.util
import json
import sys

from pathlib import Path
from typing import Any

EXPECTED_MODELS = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "ibm-granite/granite-3.2-2b-instruct",
)


def _load_round4_runner_main():
    round4_script = Path(__file__).resolve().parents[1] / "round-4" / "run_round4.py"
    spec = importlib.util.spec_from_file_location("round4_runner", round4_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load round-4 runner module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def _has_arg(argv: list[str], arg_name: str) -> bool:
    return any(part == arg_name or part.startswith(f"{arg_name}=") for part in argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _recipe_status(row: dict[str, Any]) -> str:
    recipe_verification = _as_dict(row.get("recipe_verification"))
    status = recipe_verification.get("status")
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"verified", "blocked", "inconclusive"}:
            return normalized
    return "unknown"


def _model_capability_all_pass(row: dict[str, Any]) -> bool:
    capability = _as_dict(row.get("model_capability"))
    checks_passed = _as_int(capability.get("checks_passed"))
    total_checks = _as_int(capability.get("total_checks"))
    return checks_passed is not None and total_checks is not None and total_checks > 0 and checks_passed == total_checks


def _enrich_round6_artifacts(script_root: Path) -> None:
    summary_path = script_root / "round-6-summary.json"
    manifest_path = script_root / "round-manifest.json"
    report_path = script_root / "round-6-report.md"
    round5_summary_path = script_root.parent / "round-5" / "round-5-summary.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        return

    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    results = [row for row in _as_list(summary.get("results")) if isinstance(row, dict)]

    recipe_status_counts = {"verified": 0, "blocked": 0, "inconclusive": 0, "unknown": 0}
    capability_all_pass_count = 0
    failure_category_counts: dict[str, int] = {}
    per_model_recipe_status: list[dict[str, Any]] = []
    per_model_capability_status: list[dict[str, Any]] = []
    remaining_models: list[dict[str, Any]] = []
    lingering_total = 0

    for row in results:
        model_id = str(row.get("model_id") or "")
        recipe_status = _recipe_status(row)
        recipe_status_counts[recipe_status] = recipe_status_counts.get(recipe_status, 0) + 1
        recipe_verification = _as_dict(row.get("recipe_verification"))
        model_capability = _as_dict(row.get("model_capability"))
        row["recipe_verification_status"] = recipe_status
        row["model_capability_all_pass"] = _model_capability_all_pass(row)
        if row["model_capability_all_pass"]:
            capability_all_pass_count += 1

        classification = row.get("first_failed_classification")
        if isinstance(classification, str) and classification.strip():
            failure_category_counts[classification] = failure_category_counts.get(classification, 0) + 1

        lingering = _as_int(row.get("lingering_process_count"))
        if lingering is not None:
            lingering_total += lingering

        per_model_recipe_status.append(
            {
                "model_id": model_id,
                "status": recipe_status,
                "gate_status": recipe_verification.get("gate_status"),
                "can_promote": recipe_verification.get("can_promote"),
                "runtime_functional": recipe_verification.get("runtime_functional"),
                "baseline_available": recipe_verification.get("baseline_available"),
                "regression_free": recipe_verification.get("regression_free"),
                "integrity_failures": recipe_verification.get("integrity_failures", []),
            }
        )
        per_model_capability_status.append(
            {
                "model_id": model_id,
                "checks_passed": model_capability.get("checks_passed"),
                "total_checks": model_capability.get("total_checks"),
                "all_pass": row["model_capability_all_pass"],
                "warnings": model_capability.get("warnings", []),
                "confidence": _as_dict(model_capability.get("confidence")).get("level"),
            }
        )
        if recipe_status != "verified":
            remaining_models.append(
                {
                    "model_id": model_id,
                    "recipe_status": recipe_status,
                    "next_action": row.get("next_action"),
                }
            )

    models_total = len(results)
    baseline_valid = bool(summary.get("baseline_valid"))
    round_classification = str(summary.get("round_classification") or "")
    environment_blockers = _as_list(summary.get("environment_blockers"))
    invalid_reason = None
    if not baseline_valid:
        if environment_blockers:
            invalid_reason = f"environment_blockers:{','.join(str(item) for item in environment_blockers)}"
        elif round_classification:
            invalid_reason = round_classification
        else:
            invalid_reason = "unknown"

    round5_delta: dict[str, Any] | None = None
    if round5_summary_path.is_file():
        round5_summary = _read_json(round5_summary_path)
        round5_success_rate = round5_summary.get("success_rate")
        round5_attempt_success_rate = round5_summary.get("attempt_success_rate")
        round5_models_succeeded = round5_summary.get("models_succeeded")
        round5_models_total = round5_summary.get("models_total")
        round5_quality_gate_outcome = round5_summary.get("round_classification")
        round5_recipe_verified_count = round5_summary.get("recipe_verified_count")
        round5_semantics = "legacy quality-gate outcome"
        round5_recipe_verified_rate = (
            f"{round5_recipe_verified_count}/{round5_models_total}"
            if isinstance(round5_recipe_verified_count, int) and isinstance(round5_models_total, int) and round5_models_total > 0
            else None
        )
        round5_delta = {
            "round5_quality_gate_outcome": round5_quality_gate_outcome,
            "round5_success_rate": round5_success_rate,
            "round5_attempt_success_rate": round5_attempt_success_rate,
            "round5_models_succeeded": round5_models_succeeded,
            "round5_models_total": round5_models_total,
            "round5_recipe_verified_rate": round5_recipe_verified_rate,
            "round5_semantics": round5_semantics,
            "round6_recipe_verified_rate": f"{recipe_status_counts['verified']}/{models_total}",
            "round6_model_capability_all_pass_rate": f"{capability_all_pass_count}/{models_total}",
            "semantics_change_note": (
                "Round 6 reports split outcomes by product decision: "
                "Recipe Verification is blocking/promotion integrity, while Model Capability is non-blocking absolute task quality."
            ),
        }

    summary["valid_baseline"] = baseline_valid
    summary["invalid_baseline_reason"] = invalid_reason
    summary["recipe_verified_count"] = recipe_status_counts["verified"]
    summary["recipe_blocked_count"] = recipe_status_counts["blocked"]
    summary["recipe_inconclusive_count"] = recipe_status_counts["inconclusive"]
    summary["recipe_status_unknown_count"] = recipe_status_counts.get("unknown", 0)
    summary["recipe_verified_rate"] = f"{recipe_status_counts['verified']}/{models_total}"
    summary["model_capability_all_pass_count"] = capability_all_pass_count
    summary["model_capability_all_pass_rate"] = f"{capability_all_pass_count}/{models_total}"
    summary["recipe_verification_status_counts"] = {
        "VERIFIED": recipe_status_counts["verified"],
        "BLOCKED": recipe_status_counts["blocked"],
        "INCONCLUSIVE": recipe_status_counts["inconclusive"],
        "UNKNOWN": recipe_status_counts.get("unknown", 0),
    }
    summary["failure_category_counts"] = failure_category_counts
    summary["per_model_recipe_status"] = per_model_recipe_status
    summary["per_model_model_capability"] = per_model_capability_status
    summary["remaining_path_to_5of5_recipe_verification"] = remaining_models
    summary["lingering_process_total"] = lingering_total
    if round5_delta is not None:
        summary["round5_delta"] = round5_delta
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    cleanup = _as_dict(summary.get("current_run_workspace_cleanup"))
    final_snapshot = _as_dict(manifest.get("final_snapshot"))
    reuse_checks = [row for row in _as_list(summary.get("reuse_checks")) if isinstance(row, dict)]
    lines: list[str] = []
    lines.append("# Recipe Agent v1 Round 6 Report")
    lines.append("")
    lines.append(f"- **Run ID:** `{summary.get('run_id')}`")
    lines.append(f"- **Branch:** `{summary.get('branch')}`")
    lines.append(f"- **Commit:** `{summary.get('commit')}`")
    lines.append(f"- **Window (UTC):** `{summary.get('started_utc')}` -> `{summary.get('finished_utc')}`")
    lines.append(f"- **valid_baseline:** `{summary.get('valid_baseline')}`")
    if summary.get("valid_baseline") is not True:
        lines.append(f"- **invalid_baseline_reason:** `{summary.get('invalid_baseline_reason')}`")
    lines.append(f"- **recipe_verified_count/5:** `{summary.get('recipe_verified_rate')}`")
    lines.append(f"- **model_capability_all_pass_count/5:** `{summary.get('model_capability_all_pass_rate')}`")
    status_counts = _as_dict(summary.get("recipe_verification_status_counts"))
    lines.append(
        "- **Recipe Verification counts:** "
        f"VERIFIED={status_counts.get('VERIFIED')}, "
        f"BLOCKED={status_counts.get('BLOCKED')}, "
        f"INCONCLUSIVE={status_counts.get('INCONCLUSIVE')}, "
        f"UNKNOWN={status_counts.get('UNKNOWN')}"
    )
    lines.append("")
    lines.append("## Recipe Verification (blocking / promotion)")
    lines.append("")
    lines.append("| Model | Status | Gate | Can promote | Runtime functional | Baseline available | Regression free |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in per_model_recipe_status:
        lines.append(
            "| "
            + f"{row.get('model_id')} | {row.get('status')} | {row.get('gate_status')} | {row.get('can_promote')} | "
            + f"{row.get('runtime_functional')} | {row.get('baseline_available')} | {row.get('regression_free')} |"
        )
    lines.append("")
    lines.append("## Model Capability (non-blocking advisory)")
    lines.append("")
    lines.append("| Model | Checks passed | All pass | Confidence |")
    lines.append("| --- | --- | --- | --- |")
    for row in per_model_capability_status:
        checks_passed = row.get("checks_passed")
        total_checks = row.get("total_checks")
        checks_text = (
            f"{checks_passed}/{total_checks}"
            if isinstance(checks_passed, int) and isinstance(total_checks, int)
            else "-"
        )
        lines.append(
            "| "
            + f"{row.get('model_id')} | {checks_text} | {row.get('all_pass')} | {row.get('confidence')} |"
        )
    lines.append("")
    lines.append("## Reuse verification (post-promotion)")
    lines.append("")
    if not reuse_checks:
        lines.append("- No verified recipes were available for reuse verification in this run.")
    else:
        lines.append("| Model | Reuse identity match | Reuse attempt id | build_invocation_delta |")
        lines.append("| --- | --- | --- | --- |")
        for row in reuse_checks:
            lines.append(
                "| "
                + f"{row.get('model_id')} | {row.get('reuse_identity_match')} | {row.get('reuse_attempt_id')} | "
                + f"{row.get('build_invocation_delta')} |"
            )
    lines.append("")
    lines.append("## Setup, failures, and cleanup")
    lines.append("")
    toolchain = _as_dict(manifest.get("toolchain_probe"))
    lines.append(
        f"- Toolchain ready: `{toolchain.get('ready_for_round')}`; missing_required: `{toolchain.get('missing_required')}`"
    )
    lines.append(f"- Failure category counts: `{json.dumps(failure_category_counts, sort_keys=True)}`")
    if round5_delta is not None:
        lines.append("- Round 5 delta (semantics changed):")
        lines.append(
            f"  - Round 5 success_rate={round5_delta.get('round5_success_rate')} (legacy quality-gate semantics)"
        )
        lines.append(
            f"  - Round 6 recipe_verified_rate={round5_delta.get('round6_recipe_verified_rate')} "
            f"and model_capability_all_pass_rate={round5_delta.get('round6_model_capability_all_pass_rate')}"
        )
        lines.append(f"  - Note: {round5_delta.get('semantics_change_note')}")
    lines.append(
        "- Cleanup bytes: "
        + f"current_run_freed={cleanup.get('freed_bytes')}, "
        + f"runtime_bytes={final_snapshot.get('runtime_bytes')}, "
        + f"cache_bytes={final_snapshot.get('cache_bytes')}, "
        + f"state_bytes={final_snapshot.get('state_bytes')}, "
        + f"workspace_bytes={final_snapshot.get('workspace_bytes')}"
    )
    lines.append(f"- Lingering process total after per-model cleanup checks: `{lingering_total}`")
    lines.append("")
    if remaining_models:
        lines.append("## Remaining path to 5/5 Recipe Verification")
        lines.append("")
        for row in remaining_models:
            lines.append(
                f"- `{row.get('model_id')}` => status `{row.get('recipe_status')}`; next action: {row.get('next_action')}"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    argv = list(sys.argv[1:])
    script_root = Path(__file__).resolve().parent
    if not _has_arg(argv, "--output-dir"):
        argv.extend(["--output-dir", str(script_root)])
    if not _has_arg(argv, "--round-name"):
        argv.extend(["--round-name", script_root.name])
    if not _has_arg(argv, "--scratch-root"):
        argv.extend(["--scratch-root", r"C:\fmo-r6"])

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        round4_main = _load_round4_runner_main()
        exit_code = int(round4_main())
    finally:
        sys.argv = original_argv
    if exit_code == 0:
        _enrich_round6_artifacts(script_root)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
