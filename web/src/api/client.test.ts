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

  it("throws parse error for malformed payloads", async () => {
    const transport: Transport = {
      request: vi.fn(async () => new Response(JSON.stringify({ id: 123 })))
    };
    const client = createApiClient({ transport });
    await expect(client.getModelDetail("model")).rejects.toThrowError("Expected string");
  });
});
