# Quality validation (Recipe Agent v1)

`quality_validation.py` implements deterministic, local smoke validation for CPU text-generation recipes.

It is intentionally narrow:

- It evaluates fixed, machine-readable prompts from `config/quality-validation-profiles.json`.
- It checks objective constraints only (short exact/allowed answers, token requirements, token forbids, output format keys, non-empty, non-repetitive, and simple garble heuristics).
- Allowed-answer matching canonicalizes only case/whitespace and safe leading/trailing punctuation (for example `Mars.`, `Mars!`, `"Mars"`), while still rejecting explanatory expansions when brevity constraints apply.
- It records deterministic inference settings (`temperature`, `seed`, `max_tokens`) and explicitly records when a runtime cannot enforce one or more settings.
- It compares baseline vs optimized outputs conservatively for obvious regressions (for example, optimized fails a prompt that baseline passes).
- It records latency/memory/package metrics when supplied, but does not use them as hard pass thresholds in v1.

Promotion evidence is emitted with separate gate states:

- functional gate
- baseline comparison gate
- metrics gate (recorded/unavailable)

Promotion is blocked when required quality gates are missing or failed.

## Limitations

This is a deterministic smoke/basic-regression gate only. It is **not** a comprehensive model quality benchmark and does not claim broad semantic equivalence or end-user quality parity.
