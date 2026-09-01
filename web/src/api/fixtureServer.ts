import type { BuildRequest, BuildStage, ModelTask, RecipeStatus, SupportedOptimization, TestedStatus } from "./types";

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
  recipeStatus: RecipeStatus;
  recipeReason: string;
  recipeId?: string;
  recipeVersion?: string;
  requiresExperimentalOptIn: boolean;
  buildableWithExperimentalOptIn: boolean;
  supportedOptimizations: SupportedOptimization[];
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
  attemptId?: string;
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

interface FixtureAttemptGate {
  sequence: number;
  gate: string;
  status: "passed" | "failed";
  evidence_ref: string;
  started_utc: string;
  finished_utc: string;
}

interface FixtureAttempt {
  attemptId: string;
  modelId: string;
  recipeFingerprint: string;
  state: "generated" | "running" | "succeeded" | "failed" | "cancelled";
  buildJobId?: string;
  gates: FixtureAttemptGate[];
  failure?: {
    classification: string;
    stage: string;
    message: string;
    evidence_refs: string[];
    source_owner: string;
    next_action: string;
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
    id: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    displayName: "SmolLM2 1.7B Instruct",
    task: "llm",
    revision: "31b70e2e869a7173562077fd711b654946d38674",
    modality: "text",
    license: "apache-2.0",
    gated: false,
    requiresRemoteCode: false,
    estimatedSizeMb: 1750,
    likelyCatalogMatch: "smollm2-1.7b-instruct",
    mobiusSupport: "verified",
    mobiusRisk: "low",
    testedStatus: "not_verified",
    buildable: true,
    strategies: ["mobius-olive"],
    precisions: ["int4"],
    verifiedAudioFormats: [],
    recipeStatus: "verified",
    recipeReason: "Verified Mobius->Olive->runtime->Foundry Local SDK chat path for the pinned SmolLM2 revision.",
    recipeId: "smollm2-1.7b-cpu-int4",
    recipeVersion: "1.0.0",
    requiresExperimentalOptIn: false,
    buildableWithExperimentalOptIn: false,
    supportedOptimizations: [
      { strategy: "mobius-olive", precision: "int4", taskProfile: "llm-cpu-int4", skipOlive: false, default: true }
    ]
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
    blockedReason:
      "Decoder ONNX requires position_ids, but OGA WhisperDecoderState does not bind/update it; OGA and Foundry Local transcription fail with Missing Input: position_ids.",
    strategies: [],
    precisions: [],
    verifiedAudioFormats: [],
    recipeStatus: "blocked",
    recipeReason:
      "Blocked: deterministic config adaptation reaches OGA parser/model-load, but OGA and Foundry transcription still fail with Missing Input: position_ids because WhisperDecoderState does not bind/update position_ids.",
    recipeId: "distil-whisper-cpu-fp16",
    recipeVersion: "1.0.0",
    requiresExperimentalOptIn: false,
    buildableWithExperimentalOptIn: false,
    supportedOptimizations: []
  },
  {
    id: "ibm-granite/granite-3.3-2b-instruct",
    displayName: "Granite 3.3 2B Instruct",
    task: "llm",
    revision: "707f574c62054322f6b5b04b6d075f0a8f05e0f0",
    modality: "text",
    license: "apache-2.0",
    gated: false,
    requiresRemoteCode: false,
    estimatedSizeMb: 2400,
    likelyCatalogMatch: "granite-3.3-2b-instruct",
    mobiusSupport: "verified",
    mobiusRisk: "low",
    testedStatus: "not_verified",
    buildable: true,
    strategies: ["mobius-olive"],
    precisions: ["int4"],
    verifiedAudioFormats: [],
    recipeStatus: "verified",
    recipeReason:
      "Verified direct Mobius->Olive->runtime->Foundry Local SDK chat inference path for granite-3.3-2b pinned revision 707f574c62054322f6b5b04b6d075f0a8f05e0f0.",
    recipeId: "granite-3.3-2b-cpu-int4",
    recipeVersion: "1.0.0",
    requiresExperimentalOptIn: false,
    buildableWithExperimentalOptIn: false,
    supportedOptimizations: [
      { strategy: "mobius-olive", precision: "int4", taskProfile: "llm-cpu-int4", skipOlive: false, default: true }
    ]
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
    strategies: [],
    precisions: [],
    verifiedAudioFormats: [],
    recipeStatus: "unregistered",
    recipeReason: "No recipe is registered for model 'meta-llama/Llama-3.1-8B-Instruct' (llm). Build remains blocked until a recipe is added.",
    recipeVersion: undefined,
    recipeId: undefined,
    requiresExperimentalOptIn: false,
    buildableWithExperimentalOptIn: false,
    supportedOptimizations: []
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

function recipePayload(model: FixtureModel): Record<string, unknown> | null {
  if (!model.recipeId || !model.recipeVersion) {
    return null;
  }
  return {
    id: model.recipeId,
    version: model.recipeVersion,
    status: model.recipeStatus,
    reason: model.recipeReason,
    model_id: model.id,
    task_profile: model.task === "asr" ? "asr-cpu-fp16" : "llm-cpu-int4",
    modality: model.task,
    verified_revision: model.recipeStatus === "verified" ? model.revision : null,
    preferred_revision: model.revision,
    runtime_validation:
      model.task === "asr"
        ? "onnx-checker + onnxruntime-cpu-load + deterministic-parser/model-load-adaptation + oga/fl-transcription (blocked: Missing Input: position_ids)"
        : "onnx-checker + onnxruntime-cpu-load + onnxruntime-genai-generation",
    inference_modality: model.task,
    mobius: {
      task: model.task === "asr" ? "automatic-speech-recognition" : "text-generation",
      ep: "cpu",
      runtime: "ort-genai",
      dtype: "f32"
    },
    olive: model.task === "asr"
      ? {
          input_source: "mobius-decoder-onnx",
          task: "automatic-speech-recognition",
          precision: "fp32",
          device: "cpu",
          provider: "CPUExecutionProvider",
          log_level: "1"
        }
      : {
          input_source: "mobius-output-dir",
          task: "text-generation-with-past",
          precision: "int4",
          device: "cpu",
          provider: "CPUExecutionProvider",
          log_level: "1"
        },
    ancillary_files: [],
    supported_optimizations: model.supportedOptimizations.map((item) => ({
      strategy: item.strategy,
      precision: item.precision,
      task_profile: item.taskProfile,
      skip_olive: item.skipOlive,
      default: item.default
    })),
    requires_experimental_opt_in: model.requiresExperimentalOptIn
  };
}

function preflightBlockers(model: FixtureModel): Array<Record<string, unknown>> {
  if (!model.blockedReason) {
    return [];
  }
  if (model.id === "distil-whisper/distil-medium.en") {
    return [{
      stage: "inferencing",
      classification: "source_runtime_contract_incompatible",
      message: model.blockedReason,
      detail: {
        required_input: "position_ids",
        runtime_component: "WhisperDecoderState",
        runtime_gap: "position_ids_not_bound_or_updated",
        error_signature: "Missing Input: position_ids"
      }
    }];
  }
  return [{
    stage: "preflight",
    classification: model.gated ? "gated_model" : "invalid_request",
    message: model.blockedReason
  }];
}

function candidateOutcome(model: FixtureModel): Record<string, unknown> | null {
  if (model.id !== "distil-whisper/distil-medium.en") {
    return null;
  }
  return {
    model_id: model.id,
    revision: model.revision,
    profile: "cpu/ort-genai; mobius=f32; deterministic-adapter=parser+model-load",
    status: "blocked",
    tested_status: "not_verified",
    failed_stage: "inferencing",
    classification: "source_runtime_contract_incompatible",
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
      { stage: "runtime_validating", status: "passed", summary: "ONNX checker and ORT CPU load succeeded." },
      { stage: "fl_loading", status: "passed", summary: "Deterministic config adaptation advanced OGA parser/model-load gates." },
      {
        stage: "inferencing",
        status: "failed",
        summary: "OGA and Foundry Local transcription fail with Missing Input: position_ids (WhisperDecoderState does not bind/update position_ids)."
      }
    ],
    evidence_reference: "docs/asr-contract-repair.md#irreducible-failure-boundary (run 20260831-124030-fc016713)",
    capability_owner: "Primary owner: microsoft/onnxruntime-genai Whisper runtime; coordinate Mobius Whisper regression coverage.",
    next_action: "Implement optional position_ids binding/updates from prompt + past sequence length, regression-test a Mobius-exported Whisper package, then rerun OGA + Foundry Local SDK transcription."
  };
}

function generatedRecipePayload(model: FixtureModel): Record<string, unknown> {
  const eligible = model.task === "llm" && !model.gated && model.recipeStatus === "unregistered";
  const fingerprint = eligible ? "1111111111111111111111111111111111111111111111111111111111111111" : null;
  return {
    eligible_for_automatic_recipe_attempt: eligible,
    requires_explicit_attempt_confirmation: true,
    experimental_until_verified: true,
    fingerprint,
    compile_error: eligible ? null : model.blockedReason ?? "Generated recipe unavailable for this fixture model.",
    capability: {
      outcome: eligible ? "exact" : "not-eligible",
      reason_code: eligible ? "resolved" : "unsupported-task",
      reason: eligible
        ? "Resolved to fixture capability."
        : "Generated recipe flow is unavailable for this fixture model.",
      matched_aliases: [model.id.split("/").pop() ?? model.id],
      capability: eligible
        ? {
            capability_id: "fixture-llm-cpu-auto",
            status: "tool-supported-unverified"
          }
        : null
    },
    argument_confidence: eligible
      ? {
          mobius_dtype_confidence: "candidate-unverified",
          olive_precision_confidence: "candidate-unverified",
          contains_unverified_arguments: true
        }
      : null,
    validation_gates: [
      "mobius_build",
      "olive_optimize",
      "onnx_validation",
      "ort_validation",
      "oga_validation",
      "fl_sdk_inference",
      "quality_validation"
    ],
    verified_reuse: null
  };
}

function attemptResponse(attempt: FixtureAttempt): Record<string, unknown> {
  return {
    attempt_id: attempt.attemptId,
    recipe_fingerprint: attempt.recipeFingerprint,
    state: attempt.state,
    build_job_id: attempt.buildJobId ?? null,
    gates: attempt.gates,
    failure: attempt.failure ?? null
  };
}

export function createFixtureTransport(): Transport {
  const jobs = new Map<string, FixtureJob>();
  const attempts = new Map<string, FixtureAttempt>();
  const testedArtifacts = new Map<string, string>();
  let nextJobId = 1;
  let nextAttemptId = 1;

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

  function syncAttemptFromJob(job: FixtureJob): void {
    if (!job.attemptId) {
      return;
    }
    const attempt = attempts.get(job.attemptId);
    if (!attempt) {
      return;
    }
    if (job.currentStage === "succeeded") {
      attempt.state = "succeeded";
      if (attempt.gates.length === 0) {
        const now = nowIso();
        attempt.gates = [
          "mobius_build",
          "olive_optimize",
          "onnx_validation",
          "ort_validation",
          "oga_validation",
          "fl_sdk_inference",
          "quality_validation"
        ].map((gate, index) => ({
          sequence: index + 1,
          gate,
          status: "passed",
          evidence_ref: `fixture://${attempt.attemptId}/${gate}`,
          started_utc: now,
          finished_utc: now
        }));
      }
      return;
    }
    if (job.currentStage === "failed" || job.currentStage === "cancelled") {
      attempt.state = job.currentStage === "failed" ? "failed" : "cancelled";
      attempt.failure = {
        classification: job.currentStage === "failed" ? "gate_failed" : "cancelled",
        stage: job.currentStage,
        message: job.failure?.message ?? `Build ${job.currentStage}.`,
        evidence_refs: [`job://${job.jobId}`],
        source_owner: "fixture",
        next_action: "Retry with a new idempotency key."
      };
    }
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
      syncAttemptFromJob(job);
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
      syncAttemptFromJob(job);
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
          candidate_outcome: candidateOutcome(model),
          recipe: recipePayload(model),
          recipe_status: model.recipeStatus,
          recipe_reason: model.recipeReason,
          requires_experimental_opt_in: model.requiresExperimentalOptIn,
          buildable_with_experimental_opt_in: model.buildableWithExperimentalOptIn,
          generated_recipe: generatedRecipePayload(model),
          supported_optimizations: model.supportedOptimizations.map((item) => ({
            strategy: item.strategy,
            precision: item.precision,
            task_profile: item.taskProfile,
            skip_olive: item.skipOlive,
            default: item.default
          }))
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
            blockers: preflightBlockers(model)
          },
          candidate_outcome: candidateOutcome(model),
          recipe: recipePayload(model),
          recipe_status: model.recipeStatus,
          recipe_reason: model.recipeReason,
          requires_experimental_opt_in: model.requiresExperimentalOptIn,
          generated_recipe: generatedRecipePayload(model),
          supported_optimizations: model.supportedOptimizations.map((item) => ({
            strategy: item.strategy,
            precision: item.precision,
            task_profile: item.taskProfile,
            skip_olive: item.skipOlive,
            default: item.default
          }))
        });
      }

      if (pathname === "/api/recipes/generated/preview" && method === "GET") {
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
          task: model.task,
          generated_recipe: generatedRecipePayload(model)
        });
      }

      if (pathname === "/api/recipes/generated/attempts" && method === "POST") {
        const idempotencyKey = init?.headers ? new Headers(init.headers).get("Idempotency-Key") : null;
        if (!idempotencyKey) {
          return jsonResponse(400, { error: "Idempotency-Key header is required." });
        }
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        const modelId = typeof body.model_id === "string" ? body.model_id : undefined;
        const fingerprint = typeof body.recipe_fingerprint === "string" ? body.recipe_fingerprint : undefined;
        const confirmed = body.confirm_automatic_recipe_attempt === true;
        if (!modelId || !fingerprint) {
          return jsonResponse(400, { error: "Missing model_id or recipe_fingerprint." });
        }
        if (!confirmed) {
          return jsonResponse(400, { error: "Automatic recipe attempt confirmation is required." });
        }
        const model = findModel(modelId);
        if (!model) {
          return jsonResponse(404, { error: "Model not found." });
        }
        const preview = generatedRecipePayload(model);
        if (!preview.eligible_for_automatic_recipe_attempt) {
          return jsonResponse(409, { error: "Model is not eligible for automatic recipe attempts." });
        }
        const attemptId = `attempt-${nextAttemptId}`;
        nextAttemptId += 1;
        const attempt: FixtureAttempt = {
          attemptId,
          modelId: model.id,
          recipeFingerprint: fingerprint,
          state: "running",
          gates: []
        };
        const jobId = `job-${nextJobId}`;
        nextJobId += 1;
        const job: FixtureJob = {
          jobId,
          model,
          attemptId,
          stageIndex: 0,
          currentStage: PIPELINE[0],
          cancellable: true,
          events: [],
          nextSequence: 1
        };
        attempt.buildJobId = jobId;
        addEvent(job, "queued", "Generated recipe attempt queued.");
        attempts.set(attemptId, attempt);
        jobs.set(jobId, job);
        return jsonResponse(200, {
          idempotent_replay: false,
          job: currentStatus(job),
          attempt: attemptResponse(attempt)
        });
      }

      const generatedAttemptMatch = pathname.match(/^\/api\/recipes\/generated\/attempts\/([^/]+)$/);
      if (generatedAttemptMatch && method === "GET") {
        const attempt = attempts.get(decodeURIComponent(generatedAttemptMatch[1]));
        if (!attempt) {
          return jsonResponse(404, { error: "Recipe attempt not found." });
        }
        const job = attempt.buildJobId ? jobs.get(attempt.buildJobId) : undefined;
        if (job) {
          syncAttemptFromJob(job);
        }
        return jsonResponse(200, attemptResponse(attempt));
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
          const blockedMessage = model.gated
            ? "Gated model access is rejected in this POC."
            : model.blockedReason ?? "Model is blocked by backend policy.";
          return jsonResponse(400, {
            error: blockedMessage,
            classification: model.gated ? "gated_model" : model.recipeStatus
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
        syncAttemptFromJob(job);
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
        syncAttemptFromJob(job);
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
        syncAttemptFromJob(job);
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
