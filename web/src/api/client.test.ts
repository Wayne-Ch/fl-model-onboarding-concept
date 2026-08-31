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
              { id: "microsoft/Phi-3-mini-4k-instruct", display_name: "Phi", tested_status: "tested", task: "llm" }
            ]
          })
        )
      )
    };

    const client = createApiClient({ transport, baseUrl: "http://127.0.0.1:8080" });
    const health = await client.getHealth();

    expect(health.status).toBe("ok");
    expect(health.testedModels).toHaveLength(1);
    expect(health.testedModels[0].id).toBe("microsoft/Phi-3-mini-4k-instruct");
  });

  it("uses the event cursor for incremental polling", async () => {
    const request = vi.fn(async (path: string) => {
      if (path.includes("/events")) {
        return new Response(JSON.stringify({ events: [] }));
      }
      return new Response(
        JSON.stringify({
          job_id: "job-1",
          model_id: "microsoft/Phi-3-mini-4k-instruct",
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
});
