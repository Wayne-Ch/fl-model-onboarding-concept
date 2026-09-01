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
    testedStatus: "not_verified",
    buildable: true,
    strategies: ["Auto", "Olive"],
    precisions: ["INT4", "FP16"],
    verifiedAudioFormats: []
  },
  {
    id: "distil-whisper/distil-medium.en",
    displayName: "Distil Whisper Medium English",
    task: "asr",
    revision: "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
    modality: "audio",
    license: "apache-2.0",
    gated: false,
    requiresRemoteCode: false,
    estimatedSizeMb: 967,
    likelyCatalogMatch: "whisper-small-en",
    mobiusSupport: "supported",
    mobiusRisk: "medium",
    testedStatus: "not_verified",
    buildable: false,
    blockedReason: "Generated Whisper genai_config rejected while parsing decoder_input_ids.",
    strategies: [],
    precisions: [],
    verifiedAudioFormats: []
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
    testedStatus: "not_verified",
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

function candidateOutcome(model: FixtureModel): Record<string, unknown> | null {
  if (model.id !== "distil-whisper/distil-medium.en") {
    return null;
  }
  return {
    model_id: model.id,
    revision: model.revision,
    profile: "cpu/ort-genai; mobius=f32; olive=existing-onnx-decoder/fp32",
    status: "blocked",
    tested_status: "not_verified",
    failed_stage: "fl_loading",
    classification: "oga_runtime_contract_incompatible",
    error_summary: model.blockedReason,
    versions: {
      mobius: "0.1.0",
      olive: "0.13.0",
      onnx: "1.22.0",
      onnxruntime: "1.29.0",
      onnxruntime_genai: "0.15.2",
      foundry_local_sdk: "1.2.4",
      foundry_cli: "0.11.0"
    },
    gate_outcomes: [
      { stage: "mobius_building", status: "passed", summary: "Mobius CPU ort-genai f32 build succeeded." },
      { stage: "olive_optimizing", status: "passed", summary: "Olive decoder FP32 optimization succeeded." },
      { stage: "runtime_validating", status: "passed", summary: "ONNX checker and ORT CPU load succeeded." },
      { stage: "fl_loading", status: "failed", summary: "OGA and Foundry Local SDK rejected decoder_input_ids." }
    ],
    evidence_reference: "docs/contract-probe-results.md#candidate-outcome-summary (run 20260830-225442-66553c73)",
    capability_owner: "Mobius-generated Whisper config <-> OGA/Foundry Local runtime contract integration",
    next_action: "Compare generated config to the pinned OGA Whisper schema/runtime and rerun the same profile."
  };
}

export function createFixtureTransport(): Transport {
  const jobs = new Map<string, FixtureJob>();
  const testedArtifacts = new Map<string, string>();
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
      state: job.currentStage,
      request: {
        candidate: {
          key: job.model.id.replaceAll("/", "-"),
          huggingface_model_id: job.model.id,
          modality: job.model.task
        },
        workspace_root: "C:\\fake\\workspace",
        model_cache_dir: "C:\\fake\\cache",
        output_dir: `C:\\fake\\jobs\\${job.jobId}`,
        task_profile: `${job.model.task}-cpu-${job.model.precisions[0]}`,
        runtime: "foundry-local"
      },
      started_utc: nowIso(),
      finished_utc: ["succeeded", "failed", "cancelled"].includes(job.currentStage) ? nowIso() : null,
      events: job.events.map((event) => ({
        sequence: event.sequence,
        state: event.stage,
        message: event.message,
        timestamp_utc: event.timestamp
      })),
      artifacts: job.artifactId
        ? [{
            artifact_id: job.artifactId,
            kind: "model",
            path: `C:\\fake\\artifacts\\${job.artifactId}.zip`,
            description: "Fixture model artifact",
            sha256: job.artifactId
          }]
        : [],
      validations: [],
      failure: job.failure,
      result_artifact_id: job.artifactId ?? null
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
        const tested = MODELS.filter((model) => testedArtifacts.has(model.id)).map((model) => ({
          model_id: model.id,
          display_name: model.displayName,
          task: model.task,
          tested_status: "tested",
          artifact_id: testedArtifacts.get(model.id),
          verified_utc: nowIso(),
          evidence: "successful_fl_inference"
        }));
        return jsonResponse(200, {
          status: "ok",
          service: "fixture-local-onboarding",
          active_job_id: null,
          jobs_total: jobs.size,
          storage_path: "fixture://state",
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
            model_id: model.id,
            downloads: 100,
            likes: 10,
            last_modified: nowIso(),
            verification: {
              status: testedArtifacts.has(model.id) ? "tested" : "not_verified",
              evidence: testedArtifacts.has(model.id) ? "successful_fl_inference" : "none",
              artifact_id: testedArtifacts.get(model.id) ?? null,
              verified_utc: testedArtifacts.has(model.id) ? nowIso() : null
            },
            candidate_outcome: candidateOutcome(model)
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
          model_id: model.id,
          revision: model.revision,
          gated: model.gated,
          requires_remote_code: model.requiresRemoteCode,
          buildable: model.buildable,
          build_blockers: model.blockedReason ? [model.blockedReason] : [],
          task_hints: [model.task],
          safetensors_total_bytes: model.estimatedSizeMb * 1024 * 1024,
          card_data: { license: model.license },
          foundry_catalog_matches: [{
            alias: model.likelyCatalogMatch,
            model_or_variant_id: model.likelyCatalogMatch,
            source_schema: "models",
            confidence: "medium",
            reason: "Fixture catalog match"
          }],
          warnings: [model.mobiusRisk],
          verification: {
            status: testedArtifacts.has(model.id) ? "tested" : "not_verified",
            evidence: testedArtifacts.has(model.id) ? "successful_fl_inference" : "none",
            artifact_id: testedArtifacts.get(model.id) ?? null,
            verified_utc: testedArtifacts.has(model.id) ? nowIso() : null
          },
          candidate_outcome: candidateOutcome(model)
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
          cache_key: `fixture-${model.id}`,
          ok: model.buildable,
          cached: false,
          result: {
            candidate: {
              huggingface_model_id: model.id,
              modality: model.task,
              recommended_mobius_dtype: model.precisions[0],
              recommended_olive_precision: model.precisions[0]
            },
            blockers: model.blockedReason ? [{ message: model.blockedReason }] : []
          },
          candidate_outcome: candidateOutcome(model)
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
        return jsonResponse(200, { idempotent_replay: false, job: currentStatus(job) });
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
        const events = job.events.filter((event) => event.sequence > after).map((event) => ({
          sequence: event.sequence,
          state: event.stage,
          message: event.message,
          timestamp_utc: event.timestamp
        }));
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
        if (ownerJob.model.task !== "llm") {
          return jsonResponse(409, { error: "Artifact task does not support text inference." });
        }
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        const prompt = typeof body.prompt === "string" ? body.prompt : "";
        if (!prompt.trim()) {
          return jsonResponse(400, { error: "Prompt must not be empty." });
        }
        testedArtifacts.set(ownerJob.model.id, artifactId);
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
        if (ownerJob.model.task !== "asr") {
          return jsonResponse(409, { error: "Artifact task does not support ASR inference." });
        }
        if (!(init?.body instanceof FormData)) {
          return jsonResponse(400, { error: "Expected multipart form data." });
        }
        const filePart = init.body.get("audio");
        if (!(filePart instanceof File) || filePart.size === 0) {
          return jsonResponse(400, { error: "Audio payload must not be empty." });
        }
        const filename = filePart instanceof File ? filePart.name : "unknown.wav";
        testedArtifacts.set(ownerJob.model.id, artifactId);
        return jsonResponse(200, {
          artifact_id: artifactId,
          transcript: `Fixture transcript for ${filename}`
        });
      }

      return jsonResponse(404, { error: `Fixture route not found for ${method} ${pathname}` });
    }
  };
}
