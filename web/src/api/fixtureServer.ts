import type { BuildRequest, BuildStage, ModelTask, TestedStatus } from "./types";

export interface Transport {
  request(path: string, init?: RequestInit): Promise<Response>;
}

interface FixtureModel {
  id: string;
  displayName: string;
  task: ModelTask;
  revision: string;
  modality: string;
  license: string;
  gated: boolean;
  requiresRemoteCode: boolean;
  estimatedSizeMb: number;
  likelyCatalogMatch: string;
  mobiusSupport: string;
  mobiusRisk: string;
  testedStatus: TestedStatus;
  buildable: boolean;
  blockedReason?: string;
  strategies: string[];
  precisions: string[];
  verifiedAudioFormats: string[];
}

interface FixtureEvent {
  sequence: number;
  stage: BuildStage;
  message: string;
  timestamp: string;
}

interface FixtureJob {
  jobId: string;
  model: FixtureModel;
  stageIndex: number;
  currentStage: BuildStage;
  cancellable: boolean;
  events: FixtureEvent[];
  nextSequence: number;
  artifactId?: string;
  failure?: {
    stage: BuildStage;
    classification: string;
    message: string;
    retryable: boolean;
    logTail: string[];
  };
}

const PIPELINE: BuildStage[] = [
  "queued",
  "preflight",
  "downloading",
  "mobius_building",
  "mobius_validating",
  "olive_optimizing",
  "packaging",
  "runtime_validating",
  "fl_loading",
  "inferencing",
  "succeeded"
];

const MODELS: FixtureModel[] = [
  {
    id: "microsoft/Phi-3-mini-4k-instruct",
    displayName: "Phi-3 Mini 4K Instruct",
    task: "llm",
    revision: "8f4d2a5",
    modality: "text",
    license: "mit",
    gated: false,
    requiresRemoteCode: false,
    estimatedSizeMb: 2320,
    likelyCatalogMatch: "phi-3-mini-4k-instruct",
    mobiusSupport: "supported",
    mobiusRisk: "low",
    testedStatus: "tested",
    buildable: true,
    strategies: ["Auto", "Olive"],
    precisions: ["INT4", "FP16"],
    verifiedAudioFormats: []
  },
  {
    id: "openai/whisper-small.en",
    displayName: "Whisper Small English",
    task: "asr",
    revision: "31c1d9e",
    modality: "audio",
    license: "apache-2.0",
    gated: false,
    requiresRemoteCode: false,
    estimatedSizeMb: 967,
    likelyCatalogMatch: "whisper-small-en",
    mobiusSupport: "supported",
    mobiusRisk: "medium",
    testedStatus: "tested",
    buildable: true,
    strategies: ["Auto", "Olive"],
    precisions: ["FP32"],
    verifiedAudioFormats: ["audio/wav", "audio/flac"]
  },
  {
    id: "meta-llama/Llama-3.1-8B-Instruct",
    displayName: "Llama 3.1 8B Instruct",
    task: "llm",
    revision: "c913bb2",
    modality: "text",
    license: "llama3.1",
    gated: true,
    requiresRemoteCode: true,
    estimatedSizeMb: 12840,
    likelyCatalogMatch: "llama-3.1-8b-instruct",
    mobiusSupport: "partial",
    mobiusRisk: "high",
    testedStatus: "not_tested",
    buildable: false,
    blockedReason: "Gated model access is rejected in this POC.",
    strategies: ["Auto"],
    precisions: ["INT4"],
    verifiedAudioFormats: []
  }
];

function nowIso(): string {
  return new Date().toISOString();
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json"
    }
  });
}

function findModel(modelId: string): FixtureModel | undefined {
  return MODELS.find((model) => model.id === modelId);
}

export function createFixtureTransport(): Transport {
  const jobs = new Map<string, FixtureJob>();
  let nextJobId = 1;

  function addEvent(job: FixtureJob, stage: BuildStage, message: string): void {
    const sequence = job.nextSequence;
    job.nextSequence += 1;
    job.events.push({
      sequence,
      stage,
      message,
      timestamp: nowIso()
    });
  }

  function currentStatus(job: FixtureJob): Record<string, unknown> {
    return {
      job_id: job.jobId,
      model_id: job.model.id,
      task: job.model.task,
      stage: job.currentStage,
      cancellable: job.cancellable,
      artifact_id: job.artifactId,
      artifact_summary: job.artifactId
        ? {
            artifact_id: job.artifactId,
            package_path: `C:\\fake\\artifacts\\${job.artifactId}.zip`,
            checksum: `sha256:${job.artifactId}`
          }
        : undefined,
      reproducibility: job.artifactId
        ? {
            recipe_id: "fixture-recipe-v1",
            mobius_version: "0.9.0-fixture",
            olive_version: "0.8.0-fixture"
          }
        : undefined,
      failure: job.failure,
      updated_at: nowIso()
    };
  }

  function advance(job: FixtureJob): void {
    if (job.currentStage === "succeeded" || job.currentStage === "failed" || job.currentStage === "cancelled") {
      job.cancellable = false;
      return;
    }

    if (job.model.id.includes("broken") && job.currentStage === "mobius_validating") {
      job.currentStage = "failed";
      job.cancellable = false;
      job.failure = {
        stage: "mobius_validating",
        classification: "compatibility",
        message: "Validation failed against Mobius calibration checks.",
        retryable: true,
        logTail: ["E: kernel mismatch for op=MatMul", "E: calibration artifacts incompatible with selected precision"]
      };
      addEvent(job, "failed", "Build failed during Mobius validation.");
      return;
    }

    const nextIndex = Math.min(job.stageIndex + 1, PIPELINE.length - 1);
    job.stageIndex = nextIndex;
    job.currentStage = PIPELINE[nextIndex];
    job.cancellable = job.currentStage !== "succeeded";
    addEvent(job, job.currentStage, `Entered ${job.currentStage} stage.`);

    if (job.currentStage === "succeeded") {
      job.cancellable = false;
      job.artifactId = `artifact-${job.jobId}`;
    }
  }

  return {
    async request(path: string, init?: RequestInit): Promise<Response> {
      const method = (init?.method ?? "GET").toUpperCase();
      const url = new URL(path, "http://fixture.local");
      const pathname = url.pathname;

      if (pathname === "/api/health" && method === "GET") {
        const tested = MODELS.filter((model) => model.testedStatus === "tested").map((model) => ({
          id: model.id,
          display_name: model.displayName,
          task: model.task,
          tested_status: model.testedStatus,
          gated: model.gated
        }));
        return jsonResponse(200, {
          status: "ok",
          service: "fixture-local-onboarding",
          compatibility_index: tested
        });
      }

      if (pathname === "/api/models/search" && method === "GET") {
        const query = (url.searchParams.get("q") ?? "").toLowerCase();
        const limit = Number(url.searchParams.get("limit") ?? "25");
        const filtered = MODELS.filter((model) => {
          if (!query) {
            return true;
          }
          return model.id.toLowerCase().includes(query) || model.displayName.toLowerCase().includes(query);
        })
          .slice(0, Number.isFinite(limit) ? limit : 25)
          .map((model) => ({
            id: model.id,
            display_name: model.displayName,
            task: model.task,
            tested_status: model.testedStatus,
            gated: model.gated
          }));
        return jsonResponse(200, { results: filtered });
      }

      if (pathname === "/api/models/detail" && method === "GET") {
        const modelId = url.searchParams.get("id");
        if (!modelId) {
          return jsonResponse(400, { error: "Missing id query parameter." });
        }
        const model = findModel(modelId);
        if (!model) {
          return jsonResponse(404, { error: "Model not found." });
        }
        return jsonResponse(200, {
          id: model.id,
          display_name: model.displayName,
          revision: model.revision,
          task: model.task,
          modality: model.modality,
          license: model.license,
          gated: model.gated,
          requires_remote_code: model.requiresRemoteCode,
          estimated_size_mb: model.estimatedSizeMb,
          likely_catalog_match: model.likelyCatalogMatch,
          mobius_support: model.mobiusSupport,
          mobius_risk: model.mobiusRisk,
          tested_status: model.testedStatus
        });
      }

      if (pathname === "/api/models/preflight" && method === "POST") {
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        const modelId = body.model_id ?? body.modelId ?? body.id;
        if (typeof modelId !== "string") {
          return jsonResponse(400, { error: "Missing model_id." });
        }
        const model = findModel(modelId);
        if (!model) {
          return jsonResponse(404, { error: "Model not found." });
        }
        return jsonResponse(200, {
          model_id: model.id,
          task: model.task,
          target: "cpu",
          buildable: model.buildable,
          blocked_reason: model.blockedReason,
          supported_optimizations: {
            strategies: model.strategies,
            precisions: model.precisions,
            verified_audio_formats: model.verifiedAudioFormats
          },
          defaults: {
            strategy: model.strategies[0],
            precision: model.precisions[0],
            audio_format: model.verifiedAudioFormats[0]
          }
        });
      }

      if (pathname === "/api/builds" && method === "POST") {
        const idempotencyKey = init?.headers ? new Headers(init.headers).get("Idempotency-Key") : null;
        if (!idempotencyKey) {
          return jsonResponse(400, { error: "Idempotency-Key header is required." });
        }
        const body = init?.body ? (JSON.parse(String(init.body)) as Partial<BuildRequest>) : {};
        const modelId = body.modelId ?? (body as Record<string, unknown>).model_id;
        if (typeof modelId !== "string") {
          return jsonResponse(400, { error: "Missing model id." });
        }
        const model = findModel(modelId);
        if (!model) {
          return jsonResponse(404, { error: "Model not found." });
        }
        if (!model.buildable || model.gated) {
          return jsonResponse(400, {
            error: "Gated model access is rejected in this POC.",
            classification: "gated_model"
          });
        }
        const jobId = `job-${nextJobId}`;
        nextJobId += 1;
        const job: FixtureJob = {
          jobId,
          model,
          stageIndex: 0,
          currentStage: PIPELINE[0],
          cancellable: true,
          events: [],
          nextSequence: 1
        };
        addEvent(job, "queued", "Build queued.");
        jobs.set(jobId, job);
        return jsonResponse(202, currentStatus(job));
      }

      const buildStatusMatch = pathname.match(/^\/api\/builds\/([^/]+)$/);
      if (buildStatusMatch && method === "GET") {
        const job = jobs.get(decodeURIComponent(buildStatusMatch[1]));
        if (!job) {
          return jsonResponse(404, { error: "Build job not found." });
        }
        advance(job);
        return jsonResponse(200, currentStatus(job));
      }

      const buildEventsMatch = pathname.match(/^\/api\/builds\/([^/]+)\/events$/);
      if (buildEventsMatch && method === "GET") {
        const job = jobs.get(decodeURIComponent(buildEventsMatch[1]));
        if (!job) {
          return jsonResponse(404, { error: "Build job not found." });
        }
        const after = Number(url.searchParams.get("after") ?? "0");
        if (job.currentStage !== "succeeded" && job.currentStage !== "failed" && job.currentStage !== "cancelled") {
          advance(job);
        }
        const events = job.events.filter((event) => event.sequence > after);
        return jsonResponse(200, { events });
      }

      const cancelMatch = pathname.match(/^\/api\/builds\/([^/]+)\/cancel$/);
      if (cancelMatch && method === "POST") {
        const job = jobs.get(decodeURIComponent(cancelMatch[1]));
        if (!job) {
          return jsonResponse(404, { error: "Build job not found." });
        }
        if (job.currentStage === "succeeded" || job.currentStage === "failed" || job.currentStage === "cancelled") {
          return jsonResponse(409, { error: "Job already completed." });
        }
        job.currentStage = "cancelled";
        job.cancellable = false;
        addEvent(job, "cancelled", "Build cancelled by user.");
        return jsonResponse(200, currentStatus(job));
      }

      const inferTextMatch = pathname.match(/^\/api\/artifacts\/([^/]+)\/infer\/text$/);
      if (inferTextMatch && method === "POST") {
        const artifactId = decodeURIComponent(inferTextMatch[1]);
        const ownerJob = Array.from(jobs.values()).find((job) => job.artifactId === artifactId);
        if (!ownerJob || ownerJob.currentStage !== "succeeded") {
          return jsonResponse(404, { error: "Artifact not available for inference." });
        }
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        const prompt = typeof body.prompt === "string" ? body.prompt : "";
        return jsonResponse(200, {
          artifact_id: artifactId,
          output: `Fixture response: ${prompt || "Hello from Foundry Local"}`
        });
      }

      const inferAsrMatch = pathname.match(/^\/api\/artifacts\/([^/]+)\/infer\/asr$/);
      if (inferAsrMatch && method === "POST") {
        const artifactId = decodeURIComponent(inferAsrMatch[1]);
        const ownerJob = Array.from(jobs.values()).find((job) => job.artifactId === artifactId);
        if (!ownerJob || ownerJob.currentStage !== "succeeded") {
          return jsonResponse(404, { error: "Artifact not available for inference." });
        }
        if (!(init?.body instanceof FormData)) {
          return jsonResponse(400, { error: "Expected multipart form data." });
        }
        const filePart = init.body.get("audio");
        const filename = filePart instanceof File ? filePart.name : "unknown.wav";
        return jsonResponse(200, {
          artifact_id: artifactId,
          transcript: `Fixture transcript for ${filename}`
        });
      }

      return jsonResponse(404, { error: `Fixture route not found for ${method} ${pathname}` });
    }
  };
}
