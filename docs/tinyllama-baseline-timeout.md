# TinyLlama Round 5 baseline timeout diagnosis

## Scope

- Target failure: Round 5 TinyLlama baseline quality timeout on prompt `factual-red-planet` (`timed out after 900s`).
- Frozen model and revision: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea1ffedaf415f4da2f062534de366a451e6`.
- Source evidence: `evaluation/recipe-agent-v1/round-5/*` and retained cache `scratch://round-5/r5-0902d/cache`.

## What was reproduced

- Rebuilt only TinyLlama pre-Olive baseline (Mobius `f32`) from the retained frozen snapshot in short external scratch.
- Re-ran current quality-style per-prompt `runtime_worker foundry-infer` execution for all fixed profile prompts on:
  - baseline package (`mobius`)
  - optimized package (`tinyllama-1-1b-chat-v1-0-b1747dcf0660`)
- Result: **Round 5 timeout did not reproduce**. All prompts returned within bounded time.

## Timings (sanitized, bounded)

| Probe | Baseline | Optimized |
| --- | ---:| ---:|
| Current design total (4 prompt subprocesses) | 46.161s | 25.710s |
| Single-worker total (load once, 4 prompts) | 10.869s | 7.282s |
| FL SDK discovery + load + unload | 6.085s | 2.776s |
| OGA load | 3.706s | 0.787s |
| OGA 64-token generation | 6.990s | 2.334s |

Notes:
- FL SDK latency probes (max tokens 1/8/64) remained bounded for baseline and optimized.
- OGA direct load and generation remained bounded for baseline and optimized.

## Diagnosis

Primary diagnosis in this rerun: `round5_timeout_not_reproduced_with_retained_snapshot`.

Most likely interpretation:
1. The original Round 5 timeout was intermittent/transient rather than a deterministic malformed-baseline failure.
2. The current harness design is still vulnerable because it executes each prompt through a fresh subprocess and another model discovery/load/unload cycle.

## Smallest generic fix (not implemented here)

Use one bounded quality worker **per artifact** (baseline and optimized) that:
1. loads model once,
2. executes all fixed prompts,
3. records per-prompt timing and output snippets,
4. unloads once.

This keeps prompt set and pass criteria unchanged while reducing timeout surface area.

## Exact tests to add (not implemented here)

1. `quality-worker-load-once-call-count` (unit): assert one load/unload per artifact, not per prompt.
2. `quality-worker-timeout-attribution` (unit): force hang and assert bounded failure with prompt id + phase attribution.
3. `quality-worker-semantic-equivalence` (integration): given identical prompt outputs, legacy and load-once executors produce the same promotion decision/regression labels.

## Artifacts

- Diagnostic runner: `evaluation/recipe-agent-v1/diagnostics/tinyllama-timeout/diagnose_tinyllama_timeout.py`
- Diagnostic report: `evaluation/recipe-agent-v1/diagnostics/tinyllama-timeout/diagnostic-report.json`
- Artifact test: `evaluation/recipe-agent-v1/diagnostics/tinyllama-timeout/test_tinyllama_timeout_artifacts.py`
