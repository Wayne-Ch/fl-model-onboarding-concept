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
});
