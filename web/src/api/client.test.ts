import { describe, expect, it, vi } from "vitest";
import { ApiError, createApiClient } from "./client";
import type { Transport } from "./fixtureServer";

describe("api client", () => {
  it("parses tested models from health compatibility index", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            status: "ok",
            service: "local",
            compatibility_index: [
              { id: "HuggingFaceTB/SmolLM2-1.7B-Instruct", display_name: "SmolLM2", tested_status: "tested", task: "llm" }
            ]
          })
        )
      )
    };

    const client = createApiClient({ transport, baseUrl: "http://127.0.0.1:8080" });
    const health = await client.getHealth();

    expect(health.status).toBe("ok");
    expect(health.testedModels).toHaveLength(1);
    expect(health.testedModels[0].id).toBe("HuggingFaceTB/SmolLM2-1.7B-Instruct");
  });

  it("uses the event cursor for incremental polling", async () => {
    const request = vi.fn(async (path: string) => {
      if (path.includes("/events")) {
        return new Response(JSON.stringify({ events: [] }));
      }
      return new Response(
        JSON.stringify({
          job_id: "job-1",
          model_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
          task: "llm",
          stage: "queued",
          cancellable: true
        })
      );
    });

    const client = createApiClient({
      transport: { request },
      baseUrl: "http://127.0.0.1:8080"
    });

    await client.getBuildEvents("job-1", 7);
    expect(request).toHaveBeenCalledWith("/api/builds/job-1/events?after=7", undefined);
  });

  it("maps transport failures to Local service unavailable", async () => {
    const transport: Transport = {
      request: vi.fn(async () => {
        throw new TypeError("fetch failed");
      })
    };

    const client = createApiClient({ transport });
    await expect(client.getHealth()).rejects.toEqual(new ApiError(0, "Local service unavailable"));
  });

  it("uses same-origin requests by default so custom service ports work", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "ok",
          service: "local",
          jobs_total: 0,
          storage_path: "state.sqlite3",
          compatibility_index: []
        })
      )
    );

    await createApiClient().getHealth();

    expect(fetchSpy).toHaveBeenCalledWith("/api/health", undefined);
    fetchSpy.mockRestore();
  });

  it("throws parse error for malformed payloads", async () => {
    const transport: Transport = {
      request: vi.fn(async () => new Response(JSON.stringify({ id: 123 })))
    };
    const client = createApiClient({ transport });
    await expect(client.getModelDetail("model")).rejects.toThrowError("Expected string");
  });

  it("maps canonical preflight and build envelopes and sends canonical requests", async () => {
    const request = vi.fn(async (path: string, init?: RequestInit) => {
      if (path === "/api/models/preflight") {
        return new Response(
          JSON.stringify({
            cache_key: "cache-1",
            ok: true,
            cached: false,
            result: {
              candidate: {
                huggingface_model_id: "owner/model",
                modality: "llm",
                recommended_mobius_dtype: "f16",
                recommended_olive_precision: "int4"
              },
              blockers: []
            }
          })
        );
      }
      return new Response(
        JSON.stringify({
          idempotent_replay: false,
          job: {
            job_id: "job-1",
            state: "queued",
            request: {
              candidate: { huggingface_model_id: "owner/model", modality: "llm" }
            },
            started_utc: "2026-08-30T00:00:00Z",
            events: [],
            artifacts: [],
            validations: [],
            result_artifact_id: null
          }
        })
      );
    });
    const client = createApiClient({ transport: { request } });

    const preflight = await client.preflightModel({ modelId: "owner/model", task: "llm", target: "cpu" });
    const build = await client.startBuild(
      {
        modelId: "owner/model",
        task: "llm",
        target: "cpu",
        optimization: { strategy: "mobius-olive", precision: "int4" }
      },
      "idem-1"
    );

    expect(preflight).toMatchObject({
      modelId: "owner/model",
      task: "llm",
      buildable: true,
      defaultPrecision: "int4"
    });
    expect(build).toMatchObject({ jobId: "job-1", modelId: "owner/model", stage: "queued", cancellable: true });
    const preflightBody = JSON.parse(String(request.mock.calls[0][1]?.body));
    const buildBody = JSON.parse(String(request.mock.calls[1][1]?.body));
    expect(preflightBody).toEqual({
      model_id: "owner/model",
      task: "llm",
      task_profile: "llm-cpu-default",
      allow_experimental: false
    });
    expect(buildBody).toEqual({
      model_id: "owner/model",
      task: "llm",
      task_profile: "llm-cpu-int4",
      skip_olive: false,
      allow_experimental: false,
      optimization_strategy: "mobius-olive",
      optimization_precision: "int4"
    });
  });

  it("parses final ASR blocker schema and normalizes legacy classification", async () => {
    const request = vi.fn(async (path: string) => {
      if (path.startsWith("/api/models/detail")) {
        return new Response(
          JSON.stringify({
            model_id: "distil-whisper/distil-medium.en",
            revision: "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
            gated: false,
            requires_remote_code: false,
            buildable: false,
            build_blockers: ["recipe_blocked"],
            task_hints: ["asr"],
            card_data: { license: "apache-2.0" },
            foundry_catalog_matches: [],
            warnings: [],
            verification: { status: "not_verified" },
            recipe_status: "blocked",
            recipe_reason:
              "Blocked: deterministic config adaptation reaches OGA parser/model-load, but OGA and Foundry transcription still fail with Missing Input: position_ids because WhisperDecoderState does not bind/update position_ids.",
            recipe: {
              id: "distil-whisper-cpu-fp16",
              version: "1.0.0",
              status: "blocked",
              supported_optimizations: []
            },
            candidate_outcome: {
              model_id: "distil-whisper/distil-medium.en",
              revision: "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
              profile: "cpu/ort-genai; mobius=f32; deterministic-adapter=parser+model-load",
              status: "blocked",
              tested_status: "not_verified",
              failed_stage: "inferencing",
              classification: "source_runtime_contract_incompatible",
              error_summary:
                "Decoder ONNX requires position_ids, but OGA WhisperDecoderState does not bind/update it; OGA and Foundry Local transcription fail with Missing Input: position_ids.",
              versions: {
                mobius: "0.1.0",
                olive: "0.13.0",
                onnxruntime_genai: "0.15.2"
              },
              gate_outcomes: [
                { stage: "mobius_building", status: "passed", summary: "Mobius CPU ort-genai f32 build succeeded." },
                { stage: "runtime_validating", status: "passed", summary: "ONNX checker and ORT CPU load succeeded." },
                {
                  stage: "inferencing",
                  status: "failed",
                  summary:
                    "OGA and Foundry Local transcription fail with Missing Input: position_ids (WhisperDecoderState does not bind/update position_ids)."
                }
              ],
              evidence_reference: "docs/asr-contract-repair.md#irreducible-failure-boundary",
              capability_owner: "Primary owner: microsoft/onnxruntime-genai Whisper runtime",
              next_action:
                "Implement optional position_ids binding/updates from prompt + past sequence length, then rerun OGA + Foundry Local SDK transcription."
            }
          })
        );
      }
      return new Response(
        JSON.stringify({
          cache_key: "cache-asr",
          ok: false,
          cached: false,
          result: {
            candidate: {
              huggingface_model_id: "distil-whisper/distil-medium.en",
              modality: "asr"
            },
            blockers: [
              {
                stage: "inferencing",
                classification: "source_runtime_contract_incompatible",
                message:
                  "Decoder ONNX requires position_ids, but OGA WhisperDecoderState does not bind/update it; OGA and Foundry Local transcription fail with Missing Input: position_ids."
              }
            ]
          },
          recipe_status: "blocked",
          recipe_reason:
            "Blocked: deterministic config adaptation reaches OGA parser/model-load, but OGA and Foundry transcription still fail with Missing Input: position_ids because WhisperDecoderState does not bind/update position_ids.",
          candidate_outcome: {
            model_id: "distil-whisper/distil-medium.en",
            revision: "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
            profile: "cpu/ort-genai; mobius=f32; deterministic-adapter=parser+model-load",
            status: "blocked",
            tested_status: "not_verified",
            failed_stage: "inferencing",
            classification: "oga_runtime_contract_incompatible",
            error_summary:
              "Decoder ONNX requires position_ids, but OGA WhisperDecoderState does not bind/update it; OGA and Foundry Local transcription fail with Missing Input: position_ids.",
            versions: {},
            gate_outcomes: [],
            evidence_reference: "docs/asr-contract-repair.md#irreducible-failure-boundary",
            capability_owner: "Primary owner: microsoft/onnxruntime-genai Whisper runtime",
            next_action:
              "Implement optional position_ids binding/updates from prompt + past sequence length, then rerun OGA + Foundry Local SDK transcription."
          }
        })
      );
    });
    const client = createApiClient({
      transport: { request },
      baseUrl: "http://127.0.0.1:8080"
    });

    const detail = await client.getModelDetail("distil-whisper/distil-medium.en");
    const preflight = await client.preflightModel({
      modelId: "distil-whisper/distil-medium.en",
      task: "asr",
      target: "cpu"
    });

    expect(detail.recipeStatus).toBe("blocked");
    expect(detail.candidateOutcome?.classification).toBe("source_runtime_contract_incompatible");
    expect(detail.candidateOutcome?.failedStage).toBe("inferencing");
    expect(detail.candidateOutcome?.errorSummary).toContain("Missing Input: position_ids");

    expect(preflight.buildable).toBe(false);
    expect(preflight.task).toBe("asr");
    expect(preflight.blockedReason).toContain("Missing Input: position_ids");
    expect(preflight.candidateOutcome?.classification).toBe("source_runtime_contract_incompatible");
    expect(preflight.strategies).toEqual([]);
    expect(preflight.precisions).toEqual([]);
  });

  it("preserves generated attempt gate statuses for baseline evidence integrity", async () => {
    const request = vi.fn(async (path: string) => {
      if (path === "/api/recipes/generated/attempts") {
        return new Response(
          JSON.stringify({
            idempotent_replay: false,
            job: {
              job_id: "job-1",
              state: "queued",
              request: {
                candidate: {
                  huggingface_model_id: "owner/model",
                  modality: "llm"
                }
              }
            },
            attempt: {
              attempt_id: "attempt-1",
              recipe_fingerprint: "2".repeat(64),
              state: "failed",
              build_job_id: "job-1",
              gates: [
                {
                  sequence: 1,
                  gate: "mobius_build",
                  status: "passed",
                  evidence_ref: "job://job-1/mobius_build/passed",
                  metrics_ref: null,
                  started_utc: "2026-01-01T00:00:00Z",
                  finished_utc: "2026-01-01T00:00:01Z"
                },
                {
                  sequence: 2,
                  gate: "quality_validation",
                  status: "unavailable",
                  evidence_ref: "quality://job-1/quality_validation/baseline-unavailable",
                  metrics_ref: null,
                  started_utc: "2026-01-01T00:00:02Z",
                  finished_utc: "2026-01-01T00:00:03Z"
                },
                {
                  sequence: 3,
                  gate: "ort_validation",
                  status: "not_run",
                  evidence_ref: "quality://job-1/quality_validation/baseline-not-run",
                  metrics_ref: null,
                  started_utc: "2026-01-01T00:00:04Z",
                  finished_utc: "2026-01-01T00:00:05Z"
                }
              ],
              quality_validation: {
                recipe_integrity: {
                  status: "blocked",
                  gate_status: "failed",
                  runtime_functional: true,
                  baseline_available: true,
                  regression_free: false,
                  can_promote: false,
                  integrity_failures: ["baseline_passed_optimized_failed:factual-red-planet"]
                },
                model_capability: {
                  checks_passed: 3,
                  total_checks: 4,
                  warnings: ["factual-red-planet:shared_capability_failure"],
                  confidence: {
                    level: "low",
                    determinism_supported: false,
                    reasons: ["optimized:factual-red-planet:determinism_unsupported:seed"]
                  }
                }
              },
              failure: {
                classification: "validation_failed",
                stage: "succeeded",
                message: "Quality baseline unavailable.",
                evidence_refs: ["job://job-1"],
                source_owner: "fl-onboarding",
                next_action: "Provide a baseline package."
              }
            }
          })
        );
      }
      return new Response(JSON.stringify({}));
    });
    const client = createApiClient({
      transport: { request },
      baseUrl: "http://127.0.0.1:8080"
    });

    const response = await client.startGeneratedRecipeAttempt(
      {
        modelId: "owner/model",
        recipeFingerprint: "2".repeat(64),
        confirmAutomaticRecipeAttempt: true
      },
      "idem-generated-1"
    );

    expect(response.attempt.gates[1].status).toBe("unavailable");
    expect(response.attempt.gates[2].status).toBe("not_run");
    expect(response.attempt.failure?.sourceOwner).toBe("fl-onboarding");
    expect(response.attempt.qualityValidation?.recipeIntegrity.status).toBe("blocked");
    expect(response.attempt.qualityValidation?.modelCapability?.checksPassed).toBe(3);
    expect(response.attempt.qualityValidation?.modelCapability?.warnings[0]).toBe(
      "factual-red-planet:shared_capability_failure"
    );
  });

  it("parses candidate_plan on a generated recipe preview response", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            generated_recipe: {
              eligible_for_automatic_recipe_attempt: true,
              requires_explicit_attempt_confirmation: true,
              experimental_until_verified: true,
              fingerprint: "2".repeat(64),
              capability: { outcome: "exact", reason_code: "resolved", reason: "ok", matched_aliases: [] },
              validation_gates: [],
              candidate_plan: {
                policy_id: "cpu-int4-recipe-selection-v1",
                policy_version: "1.0.0",
                policy_fingerprint: "b6b2e91a",
                max_candidates: 2,
                candidates: [
                  {
                    candidate_index: 0,
                    candidate_id: "default-int4",
                    role: "default",
                    quantization_override: null,
                    eligibility_trigger: null
                  },
                  {
                    candidate_index: 1,
                    candidate_id: "int4-block-size-64",
                    role: "quality_retry",
                    quantization_override: { block_size: 64 },
                    eligibility_trigger: "retryable_optimized_structural_regression"
                  }
                ]
              }
            }
          })
        )
      )
    };
    const client = createApiClient({ transport });

    const preview = await client.getGeneratedRecipePreview("owner/model", "llm");

    expect(preview.candidatePlan?.policyId).toBe("cpu-int4-recipe-selection-v1");
    expect(preview.candidatePlan?.maxCandidates).toBe(2);
    expect(preview.candidatePlan?.candidates[0].role).toBe("default");
    expect(preview.candidatePlan?.candidates[1].role).toBe("quality_retry");
    expect(preview.candidatePlan?.candidates[1].quantizationOverride?.blockSize).toBe(64);
    expect(preview.candidatePlan?.candidates[1].eligibilityTrigger).toBe(
      "retryable_optimized_structural_regression"
    );
  });

  it("treats a preview response with no candidate_plan key as legacy (undefined, not null-shaped)", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            generated_recipe: {
              eligible_for_automatic_recipe_attempt: false,
              requires_explicit_attempt_confirmation: true,
              experimental_until_verified: true,
              capability: { outcome: "not-eligible", reason_code: "unsupported-task", reason: "n/a", matched_aliases: [] },
              validation_gates: []
            }
          })
        )
      )
    };
    const client = createApiClient({ transport });

    const preview = await client.getGeneratedRecipePreview("owner/model", "llm");

    expect(preview.candidatePlan).toBeUndefined();
  });

  it("parses workflow_outcome selected + candidate_selection for a verified fallback with a visibly failed default", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            attempt_id: "attempt-default",
            recipe_fingerprint: "d".repeat(64),
            state: "failed",
            gates: [],
            workflow_outcome: "selected",
            candidate_selection: {
              policy_id: "cpu-int4-recipe-selection-v1",
              policy_version: "1.0.0",
              policy_fingerprint: "b6b2e91a",
              max_candidates: 2,
              lineage_selection_state: "selected",
              selected_candidate: {
                candidate_attempt_id: "cand-1",
                attempt_id: "attempt-fallback",
                candidate_index: 1,
                candidate_id: "int4-block-size-64",
                selected_by: "validation",
                selection_reason: "Candidate 1 ('int4-block-size-64') verified.",
                selected_utc: "2026-01-01T00:00:00Z"
              },
              candidates: [
                {
                  candidate_attempt_id: "cand-0",
                  attempt_id: "attempt-default",
                  candidate_index: 0,
                  candidate_id: "default-int4",
                  role: "default",
                  attempt_state: "failed",
                  recipe_fingerprint: "d".repeat(64),
                  quantization_override: null,
                  eligibility_trigger: null,
                  disposition: "retryable_optimized_structural_regression",
                  disposition_reasons: ["json_format_invalid"],
                  selection_status: "not_selected",
                  artifact_ref: null,
                  package_ref: null,
                  invocation_counters: {
                    mobius_build_invocation_count: 1,
                    olive_optimize_invocation_count: 1,
                    total_invocation_count: 2,
                    wall_clock_seconds: null,
                    estimated_cost_usd: null
                  },
                  validated_scope: {
                    target_device: "cpu",
                    target_ep: null,
                    toolchain_fingerprint: null,
                    environment_scope: null
                  }
                },
                {
                  candidate_attempt_id: "cand-1",
                  attempt_id: "attempt-fallback",
                  candidate_index: 1,
                  candidate_id: "int4-block-size-64",
                  role: "quality_retry",
                  attempt_state: "succeeded",
                  recipe_fingerprint: "e".repeat(64),
                  quantization_override: { block_size: 64 },
                  eligibility_trigger: "retryable_optimized_structural_regression",
                  disposition: null,
                  disposition_reasons: [],
                  selection_status: "selected",
                  artifact_ref: "job://job-1/artifact/artifact-1",
                  package_ref: "job://job-1/package",
                  invocation_counters: {
                    mobius_build_invocation_count: 0,
                    olive_optimize_invocation_count: 1,
                    total_invocation_count: 1,
                    wall_clock_seconds: null,
                    estimated_cost_usd: null
                  },
                  validated_scope: {
                    target_device: "cpu",
                    target_ep: "CPUExecutionProvider",
                    toolchain_fingerprint: "tc-1",
                    environment_scope: "foundry-local-onboarding:1"
                  }
                }
              ],
              aggregate_invocation_counters: {
                mobius_build_invocation_count: 1,
                olive_optimize_invocation_count: 2,
                total_invocation_count: 3,
                wall_clock_seconds: null,
                estimated_cost_usd: null
              },
              reuse: null
            }
          })
        )
      )
    };
    const client = createApiClient({ transport });

    const attempt = await client.getGeneratedRecipeAttempt("attempt-default");

    expect(attempt.workflowOutcome).toBe("selected");
    expect(attempt.candidateSelection?.selectedCandidate?.candidateIndex).toBe(1);
    expect(attempt.candidateSelection?.candidates[0].attemptState).toBe("failed");
    expect(attempt.candidateSelection?.candidates[0].selectionStatus).toBe("not_selected");
    // A real, measured zero must survive as 0 -- never coerced to "unmeasured".
    expect(attempt.candidateSelection?.candidates[1].invocationCounters.mobiusBuildInvocationCount).toBe(0);
    expect(attempt.candidateSelection?.aggregateInvocationCounters?.mobiusBuildInvocationCount).toBe(1);
    // An unmeasured field must stay undefined -- never coerced to 0.
    expect(attempt.candidateSelection?.aggregateInvocationCounters?.wallClockSeconds).toBeUndefined();
  });

  it("defaults workflow_outcome to not_applicable for a legacy attempt payload with no 3C1 fields", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            attempt_id: "attempt-legacy",
            recipe_fingerprint: "a".repeat(64),
            state: "succeeded",
            gates: []
          })
        )
      )
    };
    const client = createApiClient({ transport });

    const attempt = await client.getGeneratedRecipeAttempt("attempt-legacy");

    expect(attempt.workflowOutcome).toBe("not_applicable");
    expect(attempt.candidateSelection).toBeUndefined();
  });

  it("parses a reused attempt with an empty timeline and zeroed, explicit no-build reuse evidence", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            attempt_id: "attempt-reused",
            recipe_fingerprint: "f".repeat(64),
            state: "succeeded",
            gates: [],
            workflow_outcome: "reused",
            candidate_selection: {
              policy_id: null,
              policy_version: null,
              policy_fingerprint: null,
              max_candidates: null,
              lineage_selection_state: null,
              selected_candidate: null,
              candidates: [],
              aggregate_invocation_counters: null,
              reuse: {
                reused_without_build: true,
                source_attempt_id: "attempt-winner",
                source_candidate_attempt_id: "cand-winner",
                source_parent_attempt_id: "attempt-parent",
                policy_id: "cpu-int4-recipe-selection-v1",
                policy_version: "1.0.0",
                policy_fingerprint: "b6b2e91a",
                quality_profile_fingerprint: "q-profile",
                runner_dispatch_count: 0,
                mobius_invocation_count: 0,
                olive_invocation_count: 0,
                recorded_utc: "2026-01-01T00:00:00Z"
              }
            }
          })
        )
      )
    };
    const client = createApiClient({ transport });

    const attempt = await client.getGeneratedRecipeAttempt("attempt-reused");

    expect(attempt.workflowOutcome).toBe("reused");
    expect(attempt.candidateSelection?.candidates).toHaveLength(0);
    expect(attempt.candidateSelection?.lineageSelectionState).toBeUndefined();
    expect(attempt.candidateSelection?.aggregateInvocationCounters).toBeUndefined();
    expect(attempt.candidateSelection?.reuse?.reusedWithoutBuild).toBe(true);
    expect(attempt.candidateSelection?.reuse?.mobiusInvocationCount).toBe(0);
    expect(attempt.candidateSelection?.reuse?.oliveInvocationCount).toBe(0);
  });

  it("rejects a malformed candidate_selection whose candidates field is not an array", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            attempt_id: "attempt-malformed",
            recipe_fingerprint: "a".repeat(64),
            state: "running",
            gates: [],
            workflow_outcome: "pending",
            candidate_selection: {
              candidates: "not-an-array"
            }
          })
        )
      )
    };
    const client = createApiClient({ transport });

    await expect(client.getGeneratedRecipeAttempt("attempt-malformed")).rejects.toThrowError(
      "Expected candidates array"
    );
  });

  it("rejects an attempt payload with an unrecognized workflow_outcome code", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            attempt_id: "attempt-bad-outcome",
            recipe_fingerprint: "a".repeat(64),
            state: "running",
            gates: [],
            workflow_outcome: "mystery_outcome"
          })
        )
      )
    };
    const client = createApiClient({ transport });

    await expect(client.getGeneratedRecipeAttempt("attempt-bad-outcome")).rejects.toThrowError(
      'Unknown workflow_outcome "mystery_outcome"'
    );
  });

  it("rejects a candidate timeline entry with an unrecognized role code", async () => {
    const transport: Transport = {
      request: vi.fn(async () =>
        new Response(
          JSON.stringify({
            attempt_id: "attempt-bad-role",
            recipe_fingerprint: "a".repeat(64),
            state: "running",
            gates: [],
            workflow_outcome: "pending",
            candidate_selection: {
              candidates: [
                {
                  candidate_attempt_id: "cand-0",
                  attempt_id: "attempt-bad-role",
                  candidate_index: 0,
                  candidate_id: "default-int4",
                  role: "mystery_role",
                  attempt_state: "running",
                  recipe_fingerprint: "a".repeat(64),
                  eligibility_trigger: null,
                  disposition: null,
                  disposition_reasons: [],
                  selection_status: "not_selected",
                  artifact_ref: null,
                  package_ref: null,
                  invocation_counters: {},
                  validated_scope: {}
                }
              ]
            }
          })
        )
      )
    };
    const client = createApiClient({ transport });

    await expect(client.getGeneratedRecipeAttempt("attempt-bad-role")).rejects.toThrowError(
      'Unknown candidate role "mystery_role"'
    );
  });
});

