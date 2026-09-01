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
  CandidateGateOutcome,
  CandidateOutcome,
  GeneratedRecipePreview,
  HealthSnapshot,
  JobEvent,
  ModelDetail,
  ModelPreflight,
  ModelSummary,
  ModelTask,
  RecipeAttemptStatus,
  RecipeStatus,
  SupportedOptimization,
  TestedStatus,
  TextInferenceResult
} from "./types";

const DEFAULT_API_BASE_URL = "";

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
  if (urlValue === "") {
    return true;
  }
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
  return "not_verified";
}

function normalizeRecipeStatus(value: string | undefined): RecipeStatus {
  const normalized = (value ?? "").toLowerCase();
  if (normalized === "verified") {
    return "verified";
  }
  if (normalized === "experimental") {
    return "experimental";
  }
  if (normalized === "blocked") {
    return "blocked";
  }
  return "unregistered";
}

function normalizeCandidateClassification(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized === "oga_runtime_contract_incompatible") {
    return "source_runtime_contract_incompatible";
  }
  return normalized;
}

function parseSupportedOptimizations(input: unknown): SupportedOptimization[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const parsed: SupportedOptimization[] = [];
  for (const item of input) {
    const record = asRecord(item, "supported optimization");
    const strategy = readOptionalString(record, ["strategy"]);
    const precision = readOptionalString(record, ["precision"]);
    const taskProfile = readOptionalString(record, ["task_profile", "taskProfile"]);
    if (!strategy || !precision || !taskProfile) {
      continue;
    }
    parsed.push({
      strategy,
      precision,
      taskProfile,
      skipOlive: readBoolean(record, ["skip_olive", "skipOlive"], false),
      default: readBoolean(record, ["default"], false)
    });
  }
  return parsed;
}

function parseModelSummary(input: unknown): ModelSummary {
  const record = asRecord(input, "model summary");
  const verification = readRecord(record, ["verification"]) ?? {};
  return {
    id: readString(record, ["id", "model_id", "hf_id"]),
    displayName: readString(record, ["display_name", "name", "model_id", "id"]),
    task: normalizeTask(readOptionalString(record, ["task", "model_task"])),
    testedStatus: normalizeTestedStatus(
      readOptionalString(verification, ["status"]) ?? readOptionalString(record, ["tested_status"])
    ),
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
  const cardData = readRecord(record, ["card_data"]) ?? {};
  const taskHints = readUnknown(record, ["task_hints"]);
  const matches = readUnknown(record, ["foundry_catalog_matches"]);
  const firstMatch = Array.isArray(matches) && matches.length > 0 ? asRecord(matches[0], "catalog match") : {};
  const verification = readRecord(record, ["verification"]) ?? {};
  const recipe = readRecord(record, ["recipe"]) ?? {};
  const recipeStatus = normalizeRecipeStatus(
    readOptionalString(record, ["recipe_status"]) ?? readOptionalString(recipe, ["status"])
  );
  const recipeReason =
    readOptionalString(record, ["recipe_reason"]) ??
    readOptionalString(recipe, ["reason"]) ??
    "No recipe is registered for this model profile.";
  const topLevelSupported = readUnknown(record, ["supported_optimizations"]);
  const recipeSupported = readUnknown(recipe, ["supported_optimizations"]);
  const supportedOptimizations = parseSupportedOptimizations(
    Array.isArray(topLevelSupported) ? topLevelSupported : recipeSupported
  );
  const bytes = readNumber(record, ["safetensors_total_bytes"]);
  const requiresExperimentalOptIn = readBoolean(
    record,
    ["requires_experimental_opt_in"],
    recipeStatus === "experimental"
  );
  const buildableWithExperimentalOptIn = readBoolean(
    record,
    ["buildable_with_experimental_opt_in"],
    recipeStatus === "experimental"
  );
  const mobiusSupport =
    recipeStatus === "verified"
      ? "verified"
      : recipeStatus === "experimental"
        ? requiresExperimentalOptIn
          ? "experimental (opt-in required)"
          : "experimental"
        : recipeStatus === "blocked"
          ? "blocked"
          : "not registered";
  return {
    id: readString(record, ["id", "model_id", "hf_id"]),
    displayName: readString(record, ["display_name", "name", "model_id", "id"]),
    revision: readString(record, ["revision"], "unknown"),
    task: normalizeTask(
      readOptionalString(record, ["task", "model_task"]) ??
        (Array.isArray(taskHints) && typeof taskHints[0] === "string" ? taskHints[0] : undefined)
    ),
    modality: Array.isArray(taskHints) && typeof taskHints[0] === "string" ? taskHints[0] : "unknown",
    license: readString(cardData, ["license"], "unknown"),
    gated: readBoolean(record, ["gated"], false),
    requiresRemoteCode: readBoolean(record, ["requires_remote_code", "remote_code"], false),
    estimatedSizeMb: bytes === undefined ? undefined : Math.round(bytes / (1024 * 1024)),
    likelyCatalogMatch: readString(firstMatch, ["model_or_variant_id"], "none"),
    mobiusSupport,
    mobiusRisk: readStringArray(record, ["warnings", "build_blockers"]).join("; ") || recipeReason,
    testedStatus: normalizeTestedStatus(readOptionalString(verification, ["status"])),
    recipeStatus,
    recipeReason,
    recipeId: readOptionalString(recipe, ["id"]),
    recipeVersion: readOptionalString(recipe, ["version"]),
    requiresExperimentalOptIn,
    buildableWithExperimentalOptIn,
    supportedOptimizations,
    candidateOutcome: parseCandidateOutcome(readUnknown(record, ["candidate_outcome"])),
    generatedRecipe: parseGeneratedRecipePreview(readUnknown(record, ["generated_recipe"]))
  };
}

function parseCandidateOutcome(input: unknown): CandidateOutcome | undefined {
  if (!input) {
    return undefined;
  }
  const record = asRecord(input, "candidate outcome");
  const versionsRaw = readRecord(record, ["versions"]) ?? {};
  const versions = Object.fromEntries(
    Object.entries(versionsRaw).filter((entry): entry is [string, string] => typeof entry[1] === "string")
  );
  const gatesRaw = readUnknown(record, ["gate_outcomes"]);
  const gateOutcomes = Array.isArray(gatesRaw)
    ? gatesRaw.map((gate): CandidateGateOutcome => {
        const gateRecord = asRecord(gate, "candidate gate outcome");
        const status = readString(gateRecord, ["status"]);
        if (status !== "passed" && status !== "failed") {
          throw new ApiParseError(`Unknown candidate gate status "${status}".`);
        }
        return {
          stage: readString(gateRecord, ["stage"]),
          status,
          summary: readString(gateRecord, ["summary"])
        };
      })
    : [];
  return {
    modelId: readString(record, ["model_id"]),
    revision: readString(record, ["revision"]),
    profile: readString(record, ["profile"]),
    status: "blocked",
    testedStatus: "not_verified",
    failedStage: readString(record, ["failed_stage"]),
    classification: normalizeCandidateClassification(readString(record, ["classification"])),
    errorSummary: readString(record, ["error_summary"]),
    versions,
    gateOutcomes,
    evidenceReference: readString(record, ["evidence_reference"]),
    capabilityOwner: readString(record, ["capability_owner"]),
    nextAction: readString(record, ["next_action"])
  };
}

function parseGeneratedRecipePreview(input: unknown): GeneratedRecipePreview | undefined {
  if (!input || typeof input !== "object") {
    return undefined;
  }
  const record = asRecord(input, "generated recipe preview");
  const capabilityRecord = readRecord(record, ["capability"]) ?? {};
  const nestedCapability = readRecord(capabilityRecord, ["capability"]) ?? {};
  const argument = readRecord(record, ["argument_confidence"]) ?? {};
  const verifiedReuse = readRecord(record, ["verified_reuse"]);
  const validationGates = readStringArray(record, ["validation_gates"]);
  const capabilityId =
    readOptionalString(nestedCapability, ["capability_id"]) ??
    readOptionalString(capabilityRecord, ["capability_id"]);
  const capabilityStatus =
    readOptionalString(nestedCapability, ["status"]) ??
    readOptionalString(capabilityRecord, ["status"]);
  return {
    eligibleForAutomaticRecipeAttempt: readBoolean(
      record,
      ["eligible_for_automatic_recipe_attempt"],
      false
    ),
    requiresExplicitAttemptConfirmation: readBoolean(
      record,
      ["requires_explicit_attempt_confirmation"],
      true
    ),
    experimentalUntilVerified: readBoolean(record, ["experimental_until_verified"], true),
    fingerprint: readOptionalString(record, ["fingerprint"]),
    compileError: readOptionalString(record, ["compile_error"]),
    capability: {
      outcome: readString(capabilityRecord, ["outcome"], "not-eligible"),
      reasonCode: readString(capabilityRecord, ["reason_code"], "unknown"),
      reason: readString(capabilityRecord, ["reason"], ""),
      matchedAliases: readStringArray(capabilityRecord, ["matched_aliases"]),
      capabilityId: capabilityId ?? undefined,
      status: capabilityStatus ?? undefined
    },
    argumentConfidence: Object.keys(argument).length > 0
      ? {
          mobiusDtypeConfidence:
            readString(argument, ["mobius_dtype_confidence"], "candidate-unverified"),
          olivePrecisionConfidence:
            readString(argument, ["olive_precision_confidence"], "candidate-unverified"),
          containsUnverifiedArguments: readBoolean(
            argument,
            ["contains_unverified_arguments"],
            true
          )
        }
      : undefined,
    validationGates,
    verifiedReuse: verifiedReuse
      ? {
          available: readBoolean(verifiedReuse, ["available"], false),
          verifiedFingerprint: readString(verifiedReuse, ["verified_fingerprint"]),
          sourceRecipeFingerprint: readString(verifiedReuse, ["source_recipe_fingerprint"]),
          attemptId: readString(verifiedReuse, ["attempt_id"]),
          promotedUtc: readString(verifiedReuse, ["promoted_utc"]),
          recipe: readRecord(verifiedReuse, ["recipe"]) ?? undefined
        }
      : undefined
  };
}

function parseRecipeAttempt(input: unknown): RecipeAttemptStatus {
  const record = asRecord(input, "recipe attempt");
  const gatesRaw = readUnknown(record, ["gates"]);
  const gates = Array.isArray(gatesRaw)
    ? gatesRaw.map((row): RecipeAttemptStatus["gates"][number] => {
        const gate = asRecord(row, "recipe attempt gate");
        const status = readString(gate, ["status"], "failed");
        const gateStatus: "passed" | "failed" = status === "passed" ? "passed" : "failed";
        const sequence = readNumber(gate, ["sequence"]);
        return {
          sequence: sequence ?? 0,
          gate: readString(gate, ["gate"]),
          status: gateStatus,
          evidenceRef: readString(gate, ["evidence_ref"]),
          metricsRef: readOptionalString(gate, ["metrics_ref"]) ?? undefined,
          startedUtc: readString(gate, ["started_utc"]),
          finishedUtc: readString(gate, ["finished_utc"])
        };
      })
    : [];
  const failureRecord = readRecord(record, ["failure"]);
  return {
    attemptId: readString(record, ["attempt_id"]),
    recipeFingerprint: readString(record, ["recipe_fingerprint"]),
    state: readString(record, ["state"]) as RecipeAttemptStatus["state"],
    buildJobId: readOptionalString(record, ["build_job_id"]) ?? undefined,
    gates,
    failure: failureRecord
      ? {
          classification: readString(failureRecord, ["classification"]),
          stage: readString(failureRecord, ["stage"]),
          message: readString(failureRecord, ["message"]),
          evidenceRefs: readStringArray(failureRecord, ["evidence_refs"]),
          sourceOwner: readString(failureRecord, ["source_owner"]),
          nextAction: readString(failureRecord, ["next_action"])
        }
      : undefined
  };
}

function parsePreflight(input: unknown): ModelPreflight {
  const record = asRecord(input, "preflight response");
  const result = readRecord(record, ["result"]) ?? record;
  const candidate = readRecord(result, ["candidate"]) ?? {};
  const recipe = readRecord(record, ["recipe"]) ?? readRecord(result, ["recipe"]) ?? {};
  const recipeStatus = normalizeRecipeStatus(
    readOptionalString(record, ["recipe_status"]) ?? readOptionalString(recipe, ["status"])
  );
  const recipeReason =
    readOptionalString(record, ["recipe_reason"]) ??
    readOptionalString(recipe, ["reason"]) ??
    "No recipe is registered for this model profile.";
  const requiresExperimentalOptIn = readBoolean(
    record,
    ["requires_experimental_opt_in"],
    recipeStatus === "experimental"
  );
  const blockers = readUnknown(result, ["blockers"]);
  const firstBlocker =
    Array.isArray(blockers) && blockers.length > 0 ? asRecord(blockers[0], "preflight blocker") : undefined;
  const task = normalizeTask(readOptionalString(candidate, ["modality"]));
  const topLevelSupported = readUnknown(record, ["supported_optimizations"]);
  const recipeSupported = readUnknown(recipe, ["supported_optimizations"]);
  const supportedOptimizations = parseSupportedOptimizations(
    Array.isArray(topLevelSupported) ? topLevelSupported : recipeSupported
  );
  const fallbackOlivePrecision = readOptionalString(candidate, ["recommended_olive_precision"]);
  const fallbackMobiusPrecision = readOptionalString(candidate, ["recommended_mobius_dtype"]);
  const defaultOptimization = supportedOptimizations.find((item) => item.default) ?? supportedOptimizations[0];
  const strategies = Array.from(new Set(supportedOptimizations.map((item) => item.strategy)));
  const precisions = Array.from(new Set(supportedOptimizations.map((item) => item.precision)));
  const fallbackPrecisions = Array.from(
    new Set([fallbackOlivePrecision, fallbackMobiusPrecision].filter((value): value is string => !!value))
  );
  const resolvedStrategies =
    strategies.length > 0 ? strategies : fallbackPrecisions.length > 0 ? ["mobius-olive"] : [];
  const resolvedPrecisions = precisions.length > 0 ? precisions : fallbackPrecisions;

  return {
    modelId: readString(candidate, ["huggingface_model_id"]),
    task,
    target: "cpu",
    buildable: readBoolean(record, ["ok"], false),
    blockedReason: firstBlocker ? readString(firstBlocker, ["message"], "Preflight blocked.") : undefined,
    strategies: task === "asr" ? [] : resolvedStrategies,
    precisions: task === "asr" ? [] : resolvedPrecisions,
    verifiedAudioFormats: [],
    defaultStrategy: task === "asr" ? undefined : defaultOptimization?.strategy ?? resolvedStrategies[0],
    defaultPrecision:
      task === "asr"
        ? undefined
        : defaultOptimization?.precision ?? fallbackOlivePrecision ?? fallbackMobiusPrecision ?? "default",
    defaultAudioFormat: undefined,
    recipeStatus,
    recipeReason,
    recipeId: readOptionalString(recipe, ["id"]),
    recipeVersion: readOptionalString(recipe, ["version"]),
    requiresExperimentalOptIn,
    supportedOptimizations,
    candidateOutcome: parseCandidateOutcome(readUnknown(record, ["candidate_outcome"])),
    generatedRecipe: parseGeneratedRecipePreview(readUnknown(record, ["generated_recipe"]))
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
    packagePath: readOptionalString(record, ["path", "package_path", "packagePath"]),
    checksum: readOptionalString(record, ["sha256", "checksum"])
  };
}

function parseBuildStatus(input: unknown): BuildStatus {
  const envelope = asRecord(input, "build status");
  const record = readRecord(envelope, ["job"]) ?? envelope;
  const request = readRecord(record, ["request"]) ?? {};
  const candidate = readRecord(request, ["candidate"]) ?? {};
  const artifacts = readUnknown(record, ["artifacts"]);
  const resultArtifactId = readOptionalString(record, ["result_artifact_id"]);
  const resultArtifact =
    Array.isArray(artifacts) && resultArtifactId
      ? artifacts.find((item) => {
          const artifact = asRecord(item, "artifact");
          return readOptionalString(artifact, ["artifact_id"]) === resultArtifactId;
        })
      : undefined;
  const artifactSummary = parseArtifactSummary(resultArtifact);
  const stage = readString(record, ["state"], "unknown");
  const recipeId = readOptionalString(request, ["recipe_id", "recipeId"]);

  return {
    jobId: readString(record, ["job_id", "id", "jobId"]),
    modelId: readString(candidate, ["huggingface_model_id"]),
    task: normalizeTask(readOptionalString(candidate, ["modality"])),
    stage,
    cancellable: !["succeeded", "failed", "cancelled"].includes(stage),
    artifactId: resultArtifactId,
    artifactSummary,
    reproducibility: recipeId ? { recipeId } : undefined,
    failure: parseFailure(readUnknown(record, ["failure", "error"])),
    updatedAt: readOptionalString(record, ["finished_utc", "started_utc"])
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
    stage: readString(record, ["state", "stage"], "unknown"),
    message: readString(record, ["message", "detail"], ""),
    timestamp: readOptionalString(record, ["timestamp_utc", "timestamp", "created_at"]),
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
    async getGeneratedRecipePreview(modelId: string, task: "llm" | "asr"): Promise<GeneratedRecipePreview> {
      const encodedId = encodeURIComponent(modelId);
      const { data } = await parseResponse(
        transport,
        `/api/recipes/generated/preview?id=${encodedId}&task=${encodeURIComponent(task)}`,
        undefined,
        (input) => {
          const record = asRecord(input, "generated recipe preview response");
          const nested = readUnknown(record, ["generated_recipe"]);
          const parsed = parseGeneratedRecipePreview(nested ?? input);
          if (!parsed) {
            throw new ApiParseError("Generated recipe preview payload is missing.");
          }
          return parsed;
        }
      );
      return data;
    },
    async preflightModel(request: {
      modelId: string;
      task: "llm" | "asr";
      target: "cpu";
      allowExperimental?: boolean;
    }): Promise<ModelPreflight> {
      const { data } = await parseResponse(
        transport,
        "/api/models/preflight",
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model_id: request.modelId,
            task: request.task,
            task_profile: `${request.task}-cpu-default`,
            allow_experimental: request.allowExperimental === true
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
            task: request.task,
            task_profile: `${request.task}-cpu-${request.optimization.precision}`,
            skip_olive: request.optimization.strategy === "mobius-only",
            allow_experimental: request.allowExperimental === true,
            optimization_strategy: request.optimization.strategy,
            optimization_precision: request.optimization.precision
          })
        },
        parseBuildStatus
      );
      return data;
    },
    async startGeneratedRecipeAttempt(
      request: { modelId: string; recipeFingerprint: string; confirmAutomaticRecipeAttempt: boolean },
      idempotencyKey: string
    ): Promise<{ idempotentReplay: boolean; build: BuildStatus; attempt: RecipeAttemptStatus }> {
      const { data } = await parseResponse(
        transport,
        "/api/recipes/generated/attempts",
        {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "Idempotency-Key": idempotencyKey
          },
          body: JSON.stringify({
            recipe_fingerprint: request.recipeFingerprint,
            model_id: request.modelId,
            confirm_automatic_recipe_attempt: request.confirmAutomaticRecipeAttempt
          })
        },
        (input) => {
          const record = asRecord(input, "generated recipe attempt response");
          const job = parseBuildStatus(readUnknown(record, ["job"]) ?? record);
          const attemptRaw = readUnknown(record, ["attempt"]);
          const attempt = parseRecipeAttempt(attemptRaw);
          return {
            idempotentReplay: readBoolean(record, ["idempotent_replay"], false),
            build: job,
            attempt
          };
        }
      );
      return data;
    },
    async getGeneratedRecipeAttempt(attemptId: string): Promise<RecipeAttemptStatus> {
      const encoded = encodeURIComponent(attemptId);
      const { data } = await parseResponse(
        transport,
        `/api/recipes/generated/attempts/${encoded}`,
        undefined,
        parseRecipeAttempt
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
