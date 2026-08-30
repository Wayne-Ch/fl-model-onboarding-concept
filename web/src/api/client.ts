import { ApiParseError, asRecord, readBoolean, readNumber, readOptionalString, readRecord, readString, readStringArray, readUnknown } from "./runtime";
import { createFixtureTransport, type Transport } from "./fixtureServer";
import type {
  ApiClient,
  ApiClientConfig,
  ArtifactSummary,
  AsrInferenceResult,
  BuildFailure,
  BuildRequest,
  BuildStatus,
  HealthSnapshot,
  JobEvent,
  ModelDetail,
  ModelPreflight,
  ModelSummary,
  ModelTask,
  ReproducibilitySummary,
  TestedStatus,
  TextInferenceResult
} from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8080";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface CreateApiClientOptions {
  baseUrl?: string;
  allowNonLoopback?: boolean;
  forceFixtureMode?: boolean;
  transport?: Transport;
}

interface ParsedResponse<T> {
  data: T;
}

function isLoopbackHost(urlValue: string): boolean {
  try {
    const parsed = new URL(urlValue);
    const host = parsed.hostname.toLowerCase();
    return host === "localhost" || host === "::1" || host === "[::1]" || host === "127.0.0.1" || host.startsWith("127.");
  } catch {
    return false;
  }
}

function normalizeTask(value: string | undefined): ModelTask {
  const normalized = (value ?? "").toLowerCase();
  if (normalized === "llm") {
    return "llm";
  }
  if (normalized === "asr") {
    return "asr";
  }
  return "unknown";
}

function normalizeTestedStatus(value: string | undefined): TestedStatus {
  const normalized = (value ?? "").toLowerCase();
  if (normalized === "tested") {
    return "tested";
  }
  if (normalized === "not_tested" || normalized === "untested") {
    return "not_tested";
  }
  if (normalized === "failed") {
    return "failed";
  }
  return "unknown";
}

function parseModelSummary(input: unknown): ModelSummary {
  const record = asRecord(input, "model summary");
  return {
    id: readString(record, ["id", "model_id", "hf_id"]),
    displayName: readString(record, ["display_name", "name", "id"]),
    task: normalizeTask(readOptionalString(record, ["task", "model_task"])),
    testedStatus: normalizeTestedStatus(readOptionalString(record, ["tested_status", "status"])),
    gated: readBoolean(record, ["gated"], false)
  };
}

function parseHealth(input: unknown): HealthSnapshot {
  const record = asRecord(input, "health response");
  const compatibility = readUnknown(record, ["compatibility_index", "compatibilityIndex"]);

  let testedModelsRaw: unknown[] = [];
  if (Array.isArray(compatibility)) {
    testedModelsRaw = compatibility;
  } else if (compatibility && typeof compatibility === "object") {
    const compatibilityRecord = asRecord(compatibility, "compatibility index");
    const models = readUnknown(compatibilityRecord, ["models", "tested_models", "tested"]);
    if (Array.isArray(models)) {
      testedModelsRaw = models;
    }
  }

  return {
    status: readString(record, ["status"], "unknown"),
    service: readString(record, ["service", "name"], "foundry-local"),
    message: readOptionalString(record, ["message", "detail"]),
    testedModels: testedModelsRaw.map(parseModelSummary).filter((model) => model.testedStatus === "tested")
  };
}

function parseModelDetail(input: unknown): ModelDetail {
  const record = asRecord(input, "model detail");
  return {
    id: readString(record, ["id", "model_id", "hf_id"]),
    displayName: readString(record, ["display_name", "name", "id"]),
    revision: readString(record, ["revision"], "unknown"),
    task: normalizeTask(readOptionalString(record, ["task", "model_task"])),
    modality: readString(record, ["modality"], "unknown"),
    license: readString(record, ["license"], "unknown"),
    gated: readBoolean(record, ["gated"], false),
    requiresRemoteCode: readBoolean(record, ["requires_remote_code", "remote_code"], false),
    estimatedSizeMb: readNumber(record, ["estimated_size_mb", "estimatedSizeMb"]),
    likelyCatalogMatch: readString(record, ["likely_catalog_match", "catalog_match"], "unknown"),
    mobiusSupport: readString(record, ["mobius_support"], "unknown"),
    mobiusRisk: readString(record, ["mobius_risk"], "unknown"),
    testedStatus: normalizeTestedStatus(readOptionalString(record, ["tested_status", "status"]))
  };
}

function parsePreflight(input: unknown): ModelPreflight {
  const record = asRecord(input, "preflight response");
  const optimizations = readRecord(record, ["supported_optimizations", "optimization"]) ?? {};
  const defaults = readRecord(record, ["defaults"]) ?? {};

  return {
    modelId: readString(record, ["model_id", "id", "modelId"]),
    task: normalizeTask(readOptionalString(record, ["task", "model_task"])),
    target: "cpu",
    buildable: readBoolean(record, ["buildable"], true),
    blockedReason: readOptionalString(record, ["blocked_reason", "reason"]),
    strategies: readStringArray(optimizations, ["strategies", "strategy_choices", "supported_strategies"]),
    precisions: readStringArray(optimizations, ["precisions", "precision_choices", "supported_precisions"]),
    verifiedAudioFormats: readStringArray(optimizations, ["verified_audio_formats", "supported_audio_formats"]),
    defaultStrategy: readOptionalString(defaults, ["strategy", "default_strategy"]),
    defaultPrecision: readOptionalString(defaults, ["precision", "default_precision"]),
    defaultAudioFormat: readOptionalString(defaults, ["audio_format", "default_audio_format"])
  };
}

function parseFailure(input: unknown): BuildFailure | undefined {
  if (!input) {
    return undefined;
  }
  const record = asRecord(input, "build failure");
  return {
    stage: readString(record, ["stage"], "failed"),
    classification: readString(record, ["classification"], "unknown"),
    message: readString(record, ["message", "detail"], "Build failed."),
    retryable: readBoolean(record, ["retryable"], false),
    logTail: readStringArray(record, ["log_tail", "logTail"])
  };
}

function parseArtifactSummary(input: unknown): ArtifactSummary | undefined {
  if (!input) {
    return undefined;
  }
  const record = asRecord(input, "artifact summary");
  return {
    artifactId: readString(record, ["artifact_id", "artifactId"]),
    packagePath: readOptionalString(record, ["package_path", "packagePath"]),
    checksum: readOptionalString(record, ["checksum"])
  };
}

function parseReproducibility(input: unknown): ReproducibilitySummary | undefined {
  if (!input) {
    return undefined;
  }
  const record = asRecord(input, "reproducibility summary");
  return {
    recipeId: readOptionalString(record, ["recipe_id", "recipeId"]),
    mobiusVersion: readOptionalString(record, ["mobius_version", "mobiusVersion"]),
    oliveVersion: readOptionalString(record, ["olive_version", "oliveVersion"])
  };
}

function parseBuildStatus(input: unknown): BuildStatus {
  const record = asRecord(input, "build status");
  const artifactSummary = parseArtifactSummary(readUnknown(record, ["artifact_summary"]));
  const artifactId = readOptionalString(record, ["artifact_id", "artifactId"]) ?? artifactSummary?.artifactId;

  return {
    jobId: readString(record, ["job_id", "id", "jobId"]),
    modelId: readString(record, ["model_id", "modelId"]),
    task: normalizeTask(readOptionalString(record, ["task", "model_task"])),
    stage: readString(record, ["stage", "status"], "unknown"),
    cancellable: readBoolean(record, ["cancellable"], false),
    artifactId,
    artifactSummary,
    reproducibility: parseReproducibility(readUnknown(record, ["reproducibility"])),
    failure: parseFailure(readUnknown(record, ["failure", "error"])),
    updatedAt: readOptionalString(record, ["updated_at", "updatedAt"])
  };
}

function parseEvent(input: unknown): JobEvent {
  const record = asRecord(input, "job event");
  const sequence = readNumber(record, ["sequence"]);
  if (sequence === undefined) {
    throw new ApiParseError("Job event sequence is required.");
  }
  return {
    sequence,
    stage: readString(record, ["stage"], "unknown"),
    message: readString(record, ["message", "detail"], ""),
    timestamp: readOptionalString(record, ["timestamp", "created_at"]),
    classification: readOptionalString(record, ["classification"])
  };
}

function parseEvents(input: unknown): JobEvent[] {
  if (Array.isArray(input)) {
    return input.map(parseEvent).sort((left, right) => left.sequence - right.sequence);
  }
  const record = asRecord(input, "events response");
  const events = readUnknown(record, ["events"]);
  if (!Array.isArray(events)) {
    return [];
  }
  return events.map(parseEvent).sort((left, right) => left.sequence - right.sequence);
}

function parseTextInference(input: unknown): TextInferenceResult {
  const record = asRecord(input, "text inference");
  return {
    artifactId: readString(record, ["artifact_id", "artifactId"]),
    output: readString(record, ["output", "response"])
  };
}

function parseAsrInference(input: unknown): AsrInferenceResult {
  const record = asRecord(input, "asr inference");
  return {
    artifactId: readString(record, ["artifact_id", "artifactId"]),
    transcript: readString(record, ["transcript"])
  };
}

function readErrorMessage(payload: unknown): string | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }
  const record = payload as Record<string, unknown>;
  const candidate = record.message ?? record.error ?? record.detail;
  if (typeof candidate === "string" && candidate.length > 0) {
    return candidate;
  }
  return undefined;
}

async function parseResponse<T>(
  transport: Transport,
  path: string,
  init: RequestInit | undefined,
  parse: (input: unknown) => T
): Promise<ParsedResponse<T>> {
  let response: Response;
  try {
    response = await transport.request(path, init);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Request failed.";
    throw new ApiError(0, message.includes("fetch") ? "Local service unavailable" : message);
  }

  const bodyText = await response.text();
  let body: unknown = undefined;
  if (bodyText.length > 0) {
    try {
      body = JSON.parse(bodyText);
    } catch {
      body = bodyText;
    }
  }

  if (!response.ok) {
    const message = readErrorMessage(body) ?? response.statusText ?? `Request failed (${response.status})`;
    throw new ApiError(response.status, message);
  }

  return { data: parse(body) };
}

function createHttpTransport(baseUrl: string): Transport {
  return {
    async request(path: string, init?: RequestInit): Promise<Response> {
      const trimmedPath = path.startsWith("/") ? path : `/${path}`;
      return fetch(`${baseUrl}${trimmedPath}`, init);
    }
  };
}

function resolveConfig(options: CreateApiClientOptions): { config: ApiClientConfig; transport: Transport } {
  const envBase = import.meta.env.VITE_API_BASE_URL as string | undefined;
  const envAllowNonLoopback = import.meta.env.VITE_ALLOW_NON_LOOPBACK_API === "true";
  const envFixture = import.meta.env.DEV && import.meta.env.VITE_USE_FIXTURE_API === "true";

  const requestedBase = (options.baseUrl ?? envBase ?? DEFAULT_API_BASE_URL).trim();
  const allowNonLoopback = options.allowNonLoopback ?? envAllowNonLoopback;
  const fixtureMode = options.forceFixtureMode ?? envFixture;

  if (options.transport) {
    return {
      config: {
        baseUrl: requestedBase,
        fixtureMode,
        warning: undefined
      },
      transport: options.transport
    };
  }

  if (fixtureMode) {
    return {
      config: {
        baseUrl: "fixture://local",
        fixtureMode: true,
        warning: "Development fixture mode is enabled."
      },
      transport: createFixtureTransport()
    };
  }

  if (!isLoopbackHost(requestedBase) && !allowNonLoopback) {
    return {
      config: {
        baseUrl: DEFAULT_API_BASE_URL,
        fixtureMode: false,
        warning: `Non-loopback API base "${requestedBase}" was ignored. Using ${DEFAULT_API_BASE_URL}.`
      },
      transport: createHttpTransport(DEFAULT_API_BASE_URL)
    };
  }

  return {
    config: {
      baseUrl: requestedBase,
      fixtureMode: false,
      warning: isLoopbackHost(requestedBase)
        ? undefined
        : `Non-loopback API base is enabled: ${requestedBase}.`
    },
    transport: createHttpTransport(requestedBase)
  };
}

export function createApiClient(options: CreateApiClientOptions = {}): ApiClient {
  const { config, transport } = resolveConfig(options);

  return {
    config,
    async getHealth(): Promise<HealthSnapshot> {
      const { data } = await parseResponse(transport, "/api/health", undefined, parseHealth);
      return data;
    },
    async searchModels(query: string, limit = 20): Promise<ModelSummary[]> {
      const encodedQuery = encodeURIComponent(query);
      const { data } = await parseResponse(
        transport,
        `/api/models/search?q=${encodedQuery}&limit=${limit}`,
        undefined,
        (input) => {
          if (Array.isArray(input)) {
            return input.map(parseModelSummary);
          }
          const record = asRecord(input, "model search response");
          const results = readUnknown(record, ["results", "models"]);
          if (!Array.isArray(results)) {
            return [];
          }
          return results.map(parseModelSummary);
        }
      );
      return data;
    },
    async getModelDetail(modelId: string): Promise<ModelDetail> {
      const encodedId = encodeURIComponent(modelId);
      const { data } = await parseResponse(transport, `/api/models/detail?id=${encodedId}`, undefined, parseModelDetail);
      return data;
    },
    async preflightModel(request: { modelId: string; target: "cpu" }): Promise<ModelPreflight> {
      const { data } = await parseResponse(
        transport,
        "/api/models/preflight",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model_id: request.modelId,
            target: request.target
          })
        },
        parsePreflight
      );
      return data;
    },
    async startBuild(request: BuildRequest, idempotencyKey: string): Promise<BuildStatus> {
      const { data } = await parseResponse(
        transport,
        "/api/builds",
        {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "Idempotency-Key": idempotencyKey
          },
          body: JSON.stringify({
            model_id: request.modelId,
            target: request.target,
            optimization: request.optimization
          })
        },
        parseBuildStatus
      );
      return data;
    },
    async getBuildStatus(jobId: string): Promise<BuildStatus> {
      const encoded = encodeURIComponent(jobId);
      const { data } = await parseResponse(transport, `/api/builds/${encoded}`, undefined, parseBuildStatus);
      return data;
    },
    async getBuildEvents(jobId: string, afterSequence: number): Promise<JobEvent[]> {
      const encoded = encodeURIComponent(jobId);
      const query = afterSequence > 0 ? `?after=${afterSequence}` : "?after=0";
      const { data } = await parseResponse(transport, `/api/builds/${encoded}/events${query}`, undefined, parseEvents);
      return data;
    },
    async cancelBuild(jobId: string): Promise<BuildStatus> {
      const encoded = encodeURIComponent(jobId);
      const { data } = await parseResponse(
        transport,
        `/api/builds/${encoded}/cancel`,
        { method: "POST" },
        parseBuildStatus
      );
      return data;
    },
    async inferText(artifactId: string, prompt: string): Promise<TextInferenceResult> {
      const encoded = encodeURIComponent(artifactId);
      const { data } = await parseResponse(
        transport,
        `/api/artifacts/${encoded}/infer/text`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ prompt })
        },
        parseTextInference
      );
      return data;
    },
    async inferAsr(artifactId: string, audioFile: File): Promise<AsrInferenceResult> {
      const encoded = encodeURIComponent(artifactId);
      const body = new FormData();
      body.set("audio", audioFile);
      const { data } = await parseResponse(
        transport,
        `/api/artifacts/${encoded}/infer/asr`,
        {
          method: "POST",
          body
        },
        parseAsrInference
      );
      return data;
    }
  };
}
