from __future__ import annotations

import hashlib
import json
import re
import string

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_ALIAS_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_DETERMINISM_FIELDS = frozenset({"temperature", "seed", "max_tokens"})
_ALLOWED_ANSWER_EDGE_PUNCTUATION = frozenset(string.punctuation + "“”‘’")


class QualityValidationTask(StrEnum):
    TEXT_GENERATION = "text-generation"


class PromptCategory(StrEnum):
    ARITHMETIC = "arithmetic"
    FACTUAL_RECALL = "factual-recall"
    INSTRUCTION_FOLLOWING = "instruction-following"
    OUTPUT_FORMAT = "output-format"


class GateState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    MISSING = "missing"
    RECORDED = "recorded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DeterministicInferenceConfig:
    temperature: float
    seed: int | None
    max_tokens: int


@dataclass(frozen=True)
class PromptExpectedConstraints:
    exact_match: str | None = None
    allowed_answers: tuple[str, ...] = ()
    required_tokens: tuple[str, ...] = ()
    forbidden_tokens: tuple[str, ...] = ()
    relevance_keywords: tuple[str, ...] = ()
    required_json_keys: tuple[str, ...] = ()
    max_words: int | None = None
    min_chars: int = 1
    max_repetition_ratio: float = 0.45
    max_consecutive_token_repeats: int = 3


@dataclass(frozen=True)
class ValidationPrompt:
    prompt_id: str
    category: PromptCategory
    prompt: str
    expected: PromptExpectedConstraints


@dataclass(frozen=True)
class QualityValidationProfile:
    profile_id: str
    version: str
    task: QualityValidationTask
    deterministic_inference: DeterministicInferenceConfig
    prompts: tuple[ValidationPrompt, ...]

    @property
    def fingerprint(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "version": self.version,
            "task": self.task.value,
            "deterministic_inference": {
                "temperature": self.deterministic_inference.temperature,
                "seed": self.deterministic_inference.seed,
                "max_tokens": self.deterministic_inference.max_tokens,
            },
            "prompts": [
                {
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category.value,
                    "prompt": prompt.prompt,
                    "expected": {
                        "exact_match": prompt.expected.exact_match,
                        "allowed_answers": list(prompt.expected.allowed_answers),
                        "required_tokens": list(prompt.expected.required_tokens),
                        "forbidden_tokens": list(prompt.expected.forbidden_tokens),
                        "relevance_keywords": list(prompt.expected.relevance_keywords),
                        "required_json_keys": list(prompt.expected.required_json_keys),
                        "max_words": prompt.expected.max_words,
                        "min_chars": prompt.expected.min_chars,
                        "max_repetition_ratio": prompt.expected.max_repetition_ratio,
                        "max_consecutive_token_repeats": prompt.expected.max_consecutive_token_repeats,
                    },
                }
                for prompt in self.prompts
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> str:
        return f"{self.profile_id}:{self.version}:{self.fingerprint}"


@dataclass(frozen=True)
class PromptExecutionRecord:
    prompt_id: str
    output_text: str
    applied_determinism: DeterministicInferenceConfig | None
    unsupported_determinism_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityMetrics:
    latency_ms: float | None = None
    peak_memory_mb: float | None = None
    package_size_mb: float | None = None


@dataclass(frozen=True)
class DeterminismCheckResult:
    recorded: bool
    fully_enforced: bool
    unsupported_fields: tuple[str, ...]
    mismatched_fields: tuple[str, ...]


@dataclass(frozen=True)
class PromptCheckResult:
    prompt_id: str
    category: PromptCategory
    passed: bool
    failures: tuple[str, ...]
    determinism: DeterminismCheckResult


@dataclass(frozen=True)
class FunctionalValidationResult:
    profile_id: str
    profile_version: str
    profile_fingerprint: str
    passed: bool
    prompt_results: tuple[PromptCheckResult, ...]


@dataclass(frozen=True)
class QuantizationComparisonResult:
    passed: bool
    regressions: tuple[str, ...]


@dataclass(frozen=True)
class MetricsCaptureResult:
    optimized: QualityMetrics | None
    baseline: QualityMetrics | None
    gate_state: GateState


@dataclass(frozen=True)
class PromotionEvidence:
    profile_id: str
    profile_version: str
    profile_fingerprint: str
    functional_gate: GateState
    baseline_comparison_gate: GateState
    metrics_gate: GateState
    can_promote: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class QualityValidationResult:
    profile_identity: str
    optimized_functional: FunctionalValidationResult
    baseline_functional: FunctionalValidationResult | None
    baseline_comparison: QuantizationComparisonResult | None
    metrics: MetricsCaptureResult
    promotion_evidence: PromotionEvidence


class QualityValidationProfileRegistry:
    def __init__(self, *, schema_version: str, profiles: tuple[QualityValidationProfile, ...]) -> None:
        if not profiles:
            raise ValueError("Quality validation profile registry is empty.")
        self.schema_version = schema_version
        self._profiles = profiles
        self._profiles_by_id: dict[str, QualityValidationProfile] = {}
        for profile in profiles:
            if profile.profile_id in self._profiles_by_id:
                raise ValueError(f"Duplicate quality validation profile_id '{profile.profile_id}'.")
            self._profiles_by_id[profile.profile_id] = profile

    def all(self) -> tuple[QualityValidationProfile, ...]:
        return self._profiles

    def get(self, profile_id: str) -> QualityValidationProfile:
        if profile_id not in self._profiles_by_id:
            raise ValueError(f"Unknown quality validation profile_id '{profile_id}'.")
        return self._profiles_by_id[profile_id]


def quality_validation_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts" / "quality-validation.schema.json"


def quality_validation_profiles_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "quality-validation-profiles.json"


def load_quality_validation_profile_registry(
    data_path: Path | None = None,
    schema_path: Path | None = None,
) -> QualityValidationProfileRegistry:
    effective_schema = schema_path or quality_validation_schema_path()
    effective_data = data_path or quality_validation_profiles_path()
    schema_raw = _load_json_file(effective_schema)
    payload = _load_json_file(effective_data)
    _validate_payload_against_schema_header(payload, schema_raw)
    profiles = _parse_profiles(payload)
    schema_version = _coerce_str(payload.get("schema_version"), "schema_version")
    return QualityValidationProfileRegistry(schema_version=schema_version, profiles=profiles)


def evaluate_quality_validation(
    *,
    profile: QualityValidationProfile,
    model_task: str,
    optimized_outputs: tuple[PromptExecutionRecord, ...],
    baseline_outputs: tuple[PromptExecutionRecord, ...] | None = None,
    optimized_metrics: QualityMetrics | None = None,
    baseline_metrics: QualityMetrics | None = None,
    require_baseline_comparison: bool = True,
) -> QualityValidationResult:
    normalized_task = _normalize_task(model_task)
    if normalized_task is None:
        raise ValueError(f"Unsupported model task '{model_task}'.")
    if normalized_task != profile.task:
        raise ValueError(
            f"Profile task '{profile.task.value}' does not match requested task '{normalized_task.value}'."
        )
    optimized_functional = _evaluate_functional(
        profile=profile,
        outputs=optimized_outputs,
        run_label="optimized",
    )

    baseline_functional: FunctionalValidationResult | None = None
    comparison: QuantizationComparisonResult | None = None
    if baseline_outputs is not None:
        baseline_functional = _evaluate_functional(
            profile=profile,
            outputs=baseline_outputs,
            run_label="baseline",
        )
        comparison = _compare_quantization(
            profile=profile,
            baseline=baseline_functional,
            optimized=optimized_functional,
        )

    metrics_gate = (
        GateState.RECORDED if (optimized_metrics is not None or baseline_metrics is not None) else GateState.UNAVAILABLE
    )
    metrics = MetricsCaptureResult(
        optimized=optimized_metrics,
        baseline=baseline_metrics,
        gate_state=metrics_gate,
    )

    functional_gate = GateState.PASSED if optimized_functional.passed else GateState.FAILED
    if require_baseline_comparison:
        if comparison is None:
            baseline_gate = GateState.MISSING
        elif comparison.passed:
            baseline_gate = GateState.PASSED
        else:
            baseline_gate = GateState.FAILED
    else:
        baseline_gate = GateState.UNAVAILABLE

    required_gate_states = [functional_gate]
    if require_baseline_comparison:
        required_gate_states.append(baseline_gate)
    can_promote = all(state == GateState.PASSED for state in required_gate_states)

    notes: list[str] = []
    if metrics.gate_state == GateState.UNAVAILABLE:
        notes.append("Optional latency/memory/package metrics were not supplied.")
    if optimized_functional.passed and _has_partial_determinism(optimized_functional):
        notes.append("Runtime could not enforce all deterministic settings for at least one prompt.")

    evidence = PromotionEvidence(
        profile_id=profile.profile_id,
        profile_version=profile.version,
        profile_fingerprint=profile.fingerprint,
        functional_gate=functional_gate,
        baseline_comparison_gate=baseline_gate,
        metrics_gate=metrics.gate_state,
        can_promote=can_promote,
        notes=tuple(notes),
    )
    return QualityValidationResult(
        profile_identity=profile.identity,
        optimized_functional=optimized_functional,
        baseline_functional=baseline_functional,
        baseline_comparison=comparison,
        metrics=metrics,
        promotion_evidence=evidence,
    )


def _has_partial_determinism(result: FunctionalValidationResult) -> bool:
    return any(
        prompt_result.determinism.recorded and not prompt_result.determinism.fully_enforced
        for prompt_result in result.prompt_results
    )


def _evaluate_functional(
    *,
    profile: QualityValidationProfile,
    outputs: tuple[PromptExecutionRecord, ...],
    run_label: str,
) -> FunctionalValidationResult:
    expected_ids = {prompt.prompt_id for prompt in profile.prompts}
    if not expected_ids:
        raise ValueError(f"Profile '{profile.profile_id}' has no prompts.")
    outputs_by_prompt: dict[str, PromptExecutionRecord] = {}
    for row in outputs:
        if row.prompt_id in outputs_by_prompt:
            raise ValueError(f"{run_label} outputs contain duplicate prompt_id '{row.prompt_id}'.")
        outputs_by_prompt[row.prompt_id] = row
    actual_ids = set(outputs_by_prompt.keys())
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing:
        raise ValueError(f"{run_label} outputs missing prompt results for: {', '.join(missing)}.")
    if unexpected:
        raise ValueError(f"{run_label} outputs include unexpected prompt results: {', '.join(unexpected)}.")

    prompt_results: list[PromptCheckResult] = []
    for prompt in profile.prompts:
        row = outputs_by_prompt[prompt.prompt_id]
        determinism = _evaluate_determinism(
            expected=profile.deterministic_inference,
            actual=row.applied_determinism,
            unsupported_fields=row.unsupported_determinism_fields,
            prompt_id=prompt.prompt_id,
        )
        failures = list(_evaluate_output(prompt.expected, row.output_text))
        if not determinism.recorded:
            failures.append("determinism_not_recorded")
        for field in determinism.mismatched_fields:
            failures.append(f"determinism_mismatch:{field}")
        passed = len(failures) == 0
        prompt_results.append(
            PromptCheckResult(
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                passed=passed,
                failures=tuple(failures),
                determinism=determinism,
            )
        )
    return FunctionalValidationResult(
        profile_id=profile.profile_id,
        profile_version=profile.version,
        profile_fingerprint=profile.fingerprint,
        passed=all(row.passed for row in prompt_results),
        prompt_results=tuple(prompt_results),
    )


def _evaluate_determinism(
    *,
    expected: DeterministicInferenceConfig,
    actual: DeterministicInferenceConfig | None,
    unsupported_fields: tuple[str, ...],
    prompt_id: str,
) -> DeterminismCheckResult:
    for field in unsupported_fields:
        if field not in _DETERMINISM_FIELDS:
            raise ValueError(
                f"Prompt '{prompt_id}' has unsupported determinism field name '{field}'."
            )
    if actual is None:
        return DeterminismCheckResult(
            recorded=False,
            fully_enforced=False,
            unsupported_fields=unsupported_fields,
            mismatched_fields=(),
        )
    mismatched: list[str] = []
    if "temperature" not in unsupported_fields and actual.temperature != expected.temperature:
        mismatched.append("temperature")
    if "seed" not in unsupported_fields and actual.seed != expected.seed:
        mismatched.append("seed")
    if "max_tokens" not in unsupported_fields and actual.max_tokens != expected.max_tokens:
        mismatched.append("max_tokens")
    return DeterminismCheckResult(
        recorded=True,
        fully_enforced=(not mismatched and not unsupported_fields),
        unsupported_fields=unsupported_fields,
        mismatched_fields=tuple(mismatched),
    )


def _evaluate_output(expected: PromptExpectedConstraints, output_text: str) -> tuple[str, ...]:
    stripped = output_text.strip()
    failures: list[str] = []
    if len(stripped) < expected.min_chars:
        failures.append("output_too_short")
        return tuple(failures)
    if _looks_garbled(stripped):
        failures.append("output_garbled")
    if _looks_repetitive(
        stripped,
        max_ratio=expected.max_repetition_ratio,
        max_consecutive_repeats=expected.max_consecutive_token_repeats,
    ):
        failures.append("output_repetitive")

    normalized_output = _normalize_text(stripped)
    if expected.exact_match is not None and normalized_output != _normalize_text(expected.exact_match):
        failures.append("exact_match_failed")

    if expected.allowed_answers:
        canonical_output = _canonicalize_allowed_answer(stripped)
        allowed = {_canonicalize_allowed_answer(item) for item in expected.allowed_answers}
        if not canonical_output or canonical_output not in allowed:
            failures.append("allowed_answers_failed")

    if expected.required_tokens:
        for token in expected.required_tokens:
            if _normalize_text(token) not in normalized_output:
                failures.append(f"required_token_missing:{token}")

    if expected.forbidden_tokens:
        for token in expected.forbidden_tokens:
            if _normalize_text(token) in normalized_output:
                failures.append(f"forbidden_token_present:{token}")

    if expected.relevance_keywords:
        relevance_hits = [
            keyword
            for keyword in expected.relevance_keywords
            if _normalize_text(keyword) in normalized_output
        ]
        if not relevance_hits:
            failures.append("relevance_keyword_missing")

    if expected.max_words is not None:
        words = _WORD_RE.findall(normalized_output)
        if len(words) > expected.max_words:
            failures.append("max_words_exceeded")

    if expected.required_json_keys:
        parsed = _parse_json_object(output_text)
        if parsed is None:
            failures.append("json_format_invalid")
        else:
            for key in expected.required_json_keys:
                if key not in parsed:
                    failures.append(f"json_key_missing:{key}")

    return tuple(failures)


def _parse_json_object(value: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _compare_quantization(
    *,
    profile: QualityValidationProfile,
    baseline: FunctionalValidationResult,
    optimized: FunctionalValidationResult,
) -> QuantizationComparisonResult:
    baseline_map = {row.prompt_id: row for row in baseline.prompt_results}
    optimized_map = {row.prompt_id: row for row in optimized.prompt_results}
    if baseline_map.keys() != optimized_map.keys():
        raise ValueError(
            "Inconsistent baseline/optimized prompt coverage in quantization comparison."
        )
    regressions: list[str] = []
    if not baseline.passed:
        regressions.append("baseline_failed_functional_checks")
    for prompt in profile.prompts:
        baseline_row = baseline_map[prompt.prompt_id]
        optimized_row = optimized_map[prompt.prompt_id]
        if baseline_row.passed and not optimized_row.passed:
            regressions.append(
                f"optimized_failed_prompt:{prompt.prompt_id}"
            )
    return QuantizationComparisonResult(
        passed=(len(regressions) == 0),
        regressions=tuple(regressions),
    )


def _looks_repetitive(
    value: str,
    *,
    max_ratio: float,
    max_consecutive_repeats: int,
) -> bool:
    tokens = _WORD_RE.findall(_normalize_text(value))
    if len(tokens) < 4:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    repetition_ratio = max(counts.values()) / len(tokens)
    if repetition_ratio > max_ratio:
        return True
    run_length = 1
    for index in range(1, len(tokens)):
        if tokens[index] == tokens[index - 1]:
            run_length += 1
            if run_length > max_consecutive_repeats:
                return True
        else:
            run_length = 1
    return False


def _looks_garbled(value: str) -> bool:
    if "\ufffd" in value:
        return True
    visible_chars = [char for char in value if not char.isspace()]
    if not visible_chars:
        return True
    non_printable = sum(1 for char in visible_chars if not char.isprintable())
    if non_printable > 0:
        return True
    alnum = sum(1 for char in visible_chars if char.isalnum())
    punctuation = sum(1 for char in visible_chars if char in string.punctuation)
    if alnum == 0:
        return True
    punctuation_ratio = punctuation / len(visible_chars)
    alnum_ratio = alnum / len(visible_chars)
    return punctuation_ratio > 0.70 and alnum_ratio < 0.30


def _normalize_alias(value: str) -> str:
    return _ALIAS_NORMALIZE_RE.sub("", value.strip().lower())


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _canonicalize_allowed_answer(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    start = 0
    end = len(normalized)
    while start < end and normalized[start] in _ALLOWED_ANSWER_EDGE_PUNCTUATION:
        start += 1
    while end > start and normalized[end - 1] in _ALLOWED_ANSWER_EDGE_PUNCTUATION:
        end -= 1
    return normalized[start:end].strip()


def _normalize_task(value: str) -> QualityValidationTask | None:
    normalized = _normalize_alias(value)
    if normalized in {"llm", "textgeneration"}:
        return QualityValidationTask.TEXT_GENERATION
    return None


def _load_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file '{path}' must contain an object at top level.")
    return payload


def _validate_payload_against_schema_header(payload: dict[str, object], schema: dict[str, object]) -> None:
    required = _coerce_str_tuple(schema.get("required"), "schema.required")
    for key in required:
        if key not in payload:
            raise ValueError(f"Quality validation data is missing required key '{key}'.")
    properties = _coerce_mapping(schema.get("properties"), "schema.properties")
    schema_version_prop = _coerce_mapping(
        properties.get("schema_version"),
        "schema.properties.schema_version",
    )
    expected = _coerce_optional_str(
        schema_version_prop.get("const"),
        "schema.properties.schema_version.const",
    )
    actual = _coerce_optional_str(payload.get("schema_version"), "schema_version")
    if expected and actual != expected:
        raise ValueError(
            "Quality validation schema_version mismatch: "
            f"expected '{expected}', got '{actual}'."
        )


def _parse_profiles(payload: dict[str, object]) -> tuple[QualityValidationProfile, ...]:
    rows = _coerce_sequence(payload.get("profiles"), "profiles")
    if not rows:
        raise ValueError("Quality validation data has no profiles.")
    parsed = [_parse_profile(row, index) for index, row in enumerate(rows, start=1)]
    return tuple(parsed)


def _parse_profile(row: object, index: int) -> QualityValidationProfile:
    path = f"profiles[{index}]"
    value = _coerce_mapping(row, path)
    prompts_raw = _coerce_sequence(value.get("prompts"), f"{path}.prompts")
    prompts: list[ValidationPrompt] = []
    seen_prompt_ids: set[str] = set()
    for prompt_index, prompt_row in enumerate(prompts_raw, start=1):
        prompt = _parse_prompt(prompt_row, path=f"{path}.prompts[{prompt_index}]")
        if prompt.prompt_id in seen_prompt_ids:
            raise ValueError(
                f"{path}.prompts contains duplicate prompt_id '{prompt.prompt_id}'."
            )
        seen_prompt_ids.add(prompt.prompt_id)
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"{path}.prompts cannot be empty.")
    task = QualityValidationTask(_coerce_str(value.get("task"), f"{path}.task"))
    profile = QualityValidationProfile(
        profile_id=_coerce_str(value.get("profile_id"), f"{path}.profile_id"),
        version=_coerce_str(value.get("version"), f"{path}.version"),
        task=task,
        deterministic_inference=_parse_deterministic_inference(
            _coerce_mapping(value.get("deterministic_inference"), f"{path}.deterministic_inference"),
            path=f"{path}.deterministic_inference",
        ),
        prompts=tuple(prompts),
    )
    return profile


def _parse_prompt(row: object, *, path: str) -> ValidationPrompt:
    value = _coerce_mapping(row, path)
    expected = _parse_expected_constraints(
        _coerce_mapping(value.get("expected"), f"{path}.expected"),
        path=f"{path}.expected",
    )
    return ValidationPrompt(
        prompt_id=_coerce_str(value.get("prompt_id"), f"{path}.prompt_id"),
        category=PromptCategory(_coerce_str(value.get("category"), f"{path}.category")),
        prompt=_coerce_str(value.get("prompt"), f"{path}.prompt"),
        expected=expected,
    )


def _parse_expected_constraints(row: dict[str, object], *, path: str) -> PromptExpectedConstraints:
    expected = PromptExpectedConstraints(
        exact_match=_coerce_optional_str(row.get("exact_match"), f"{path}.exact_match"),
        allowed_answers=_coerce_unique_str_tuple(row.get("allowed_answers"), f"{path}.allowed_answers"),
        required_tokens=_coerce_unique_str_tuple(row.get("required_tokens"), f"{path}.required_tokens"),
        forbidden_tokens=_coerce_unique_str_tuple(row.get("forbidden_tokens"), f"{path}.forbidden_tokens"),
        relevance_keywords=_coerce_unique_str_tuple(row.get("relevance_keywords"), f"{path}.relevance_keywords"),
        required_json_keys=_coerce_unique_str_tuple(row.get("required_json_keys"), f"{path}.required_json_keys"),
        max_words=_coerce_optional_positive_int(row.get("max_words"), f"{path}.max_words"),
        min_chars=_coerce_positive_int_default(row.get("min_chars"), f"{path}.min_chars", default=1),
        max_repetition_ratio=_coerce_ratio_default(
            row.get("max_repetition_ratio"),
            f"{path}.max_repetition_ratio",
            default=0.45,
        ),
        max_consecutive_token_repeats=_coerce_positive_int_default(
            row.get("max_consecutive_token_repeats"),
            f"{path}.max_consecutive_token_repeats",
            default=3,
        ),
    )
    _validate_expected_constraints(expected, path=path)
    return expected


def _validate_expected_constraints(expected: PromptExpectedConstraints, *, path: str) -> None:
    assessable_constraints = (
        expected.exact_match is not None
        or bool(expected.allowed_answers)
        or bool(expected.required_tokens)
        or bool(expected.forbidden_tokens)
        or bool(expected.relevance_keywords)
        or bool(expected.required_json_keys)
        or expected.max_words is not None
    )
    if not assessable_constraints:
        raise ValueError(
            f"{path} must define at least one assessable constraint."
        )
    if expected.exact_match is not None and expected.allowed_answers:
        raise ValueError(
            f"{path} cannot define both exact_match and allowed_answers."
        )


def _parse_deterministic_inference(
    row: dict[str, object],
    *,
    path: str,
) -> DeterministicInferenceConfig:
    max_tokens = _coerce_positive_int(row.get("max_tokens"), f"{path}.max_tokens")
    seed = _coerce_optional_int(row.get("seed"), f"{path}.seed")
    temperature = _coerce_float(row.get("temperature"), f"{path}.temperature")
    return DeterministicInferenceConfig(
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )


def _coerce_mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    return value


def _coerce_sequence(value: object, path: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array.")
    return tuple(value)


def _coerce_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value.strip()


def _coerce_optional_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _coerce_str(value, path)


def _coerce_str_tuple(value: object, path: str) -> tuple[str, ...]:
    rows = _coerce_sequence(value, path)
    return tuple(_coerce_str(item, f"{path}[{index}]") for index, item in enumerate(rows, start=1))


def _coerce_unique_str_tuple(value: object, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    rows = _coerce_sequence(value, path)
    coerced: list[str] = []
    for index, item in enumerate(rows, start=1):
        text = _coerce_str(item, f"{path}[{index}]")
        if text in coerced:
            raise ValueError(f"{path} contains duplicate value '{text}'.")
        coerced.append(text)
    return tuple(coerced)


def _coerce_optional_positive_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _coerce_positive_int(value, path)


def _coerce_positive_int_default(value: object, path: str, *, default: int) -> int:
    if value is None:
        return default
    return _coerce_positive_int(value, path)


def _coerce_positive_int(value: object, path: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    if value <= 0:
        raise ValueError(f"{path} must be greater than zero.")
    return value


def _coerce_optional_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{path} must be an integer or null.")
    return value


def _coerce_float(value: object, path: str) -> float:
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    raise ValueError(f"{path} must be a number.")


def _coerce_ratio_default(value: object, path: str, *, default: float) -> float:
    if value is None:
        return default
    ratio = _coerce_float(value, path)
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"{path} must be in the range (0, 1].")
    return ratio


DEFAULT_QUALITY_VALIDATION_PROFILE_REGISTRY = load_quality_validation_profile_registry()
DEFAULT_TEXT_GENERATION_QUALITY_PROFILE = DEFAULT_QUALITY_VALIDATION_PROFILE_REGISTRY.get(
    "textgen-basic-quality-v1"
)


__all__ = [
    "DEFAULT_QUALITY_VALIDATION_PROFILE_REGISTRY",
    "DEFAULT_TEXT_GENERATION_QUALITY_PROFILE",
    "DeterminismCheckResult",
    "DeterministicInferenceConfig",
    "FunctionalValidationResult",
    "GateState",
    "MetricsCaptureResult",
    "PromptCategory",
    "PromptCheckResult",
    "PromptExecutionRecord",
    "PromptExpectedConstraints",
    "PromotionEvidence",
    "QualityMetrics",
    "QualityValidationProfile",
    "QualityValidationProfileRegistry",
    "QualityValidationResult",
    "QualityValidationTask",
    "QuantizationComparisonResult",
    "ValidationPrompt",
    "evaluate_quality_validation",
    "load_quality_validation_profile_registry",
    "quality_validation_profiles_path",
    "quality_validation_schema_path",
]
