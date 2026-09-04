export type ModelTask = "llm" | "asr" | "unknown";
export type TestedStatus = "tested" | "not_verified";
export type BuildStage = string;
export type RecipeStatus = "verified" | "experimental" | "blocked" | "unregistered";

export interface ModelSummary {
  id: string;
  displayName: string;
  task: ModelTask;
  testedStatus: TestedStatus;
  gated: boolean;
}

export interface SupportedOptimization {
  strategy: string;
  precision: string;
  taskProfile: string;
  skipOlive: boolean;
  default: boolean;
}

export interface CandidateGateOutcome {
  stage: BuildStage;
  status: "passed" | "failed";
  summary: string;
}

export interface CandidateOutcome {
  modelId: string;
  revision: string;
  profile: string;
  status: "blocked";
  testedStatus: "not_verified";
  failedStage: BuildStage;
  classification: string;
  errorSummary: string;
  versions: Record<string, string>;
  gateOutcomes: CandidateGateOutcome[];
  evidenceReference: string;
  capabilityOwner: string;
  nextAction: string;
}

export interface GeneratedRecipeCapability {
  outcome: string;
  reasonCode: string;
  reason: string;
  matchedAliases: string[];
  capabilityId?: string;
  status?: string;
}

export interface GeneratedRecipeVerifiedReuse {
  available: boolean;
  verifiedFingerprint: string;
  sourceRecipeFingerprint: string;
  attemptId: string;
  promotedUtc: string;
  recipe?: Record<string, unknown>;
}

/**
 * Stable, machine-readable role of a planned candidate under the approved
 * CPU INT4 selection policy. Frontend primary copy must translate these:
 * "default" -> "First recipe", "quality_retry" -> "Automatic quality retry".
 * Never surface this raw code outside a "Technical details" disclosure.
 */
export type CandidateRole = "default" | "quality_retry";

export interface CandidateQuantizationOverride {
  blockSize: number;
}

export interface CandidatePlanEntry {
  candidateIndex: number;
  candidateId: string;
  role: CandidateRole;
  quantizationOverride?: CandidateQuantizationOverride;
  eligibilityTrigger?: string;
}

/**
 * Additive (Slice 3C1) static projection of the approved CPU INT4 candidate
 * plan. Present on `generatedRecipe.candidatePlan` only when the compiled
 * recipe is CPU INT4-eligible; absent/undefined for every other generated
 * recipe (wrong device/precision, compile failure, or a legacy/static
 * recipe/response that predates this field).
 */
export interface CandidateSelectionPlan {
  policyId: string;
  policyVersion: string;
  policyFingerprint: string;
  maxCandidates: number;
  candidates: CandidatePlanEntry[];
}

export type CandidateLineageSelectionState = "pending" | "selected" | "exhausted";

export type CandidateAttemptState = "generated" | "running" | "succeeded" | "failed" | "cancelled";

export type CandidateSelectionStatus = "not_selected" | "selected";

/**
 * Nullable, typed invocation/cost evidence. A field is `undefined` exactly
 * when it has never been measured/instrumented -- callers must never coerce
 * an unmeasured field to `0`, and must never treat a real `0` as "unmeasured".
 */
export interface CandidateInvocationCounters {
  mobiusBuildInvocationCount?: number;
  oliveOptimizeInvocationCount?: number;
  totalInvocationCount?: number;
  wallClockSeconds?: number;
  estimatedCostUsd?: number;
}

/**
 * Nullable-until-verified selection-scope provenance for a candidate.
 * Absence of any single field must never be read as an implicit match for
 * that scope.
 */
export interface CandidateValidatedScope {
  targetDevice?: string;
  targetEp?: string;
  toolchainFingerprint?: string;
  environmentScope?: string;
}

export interface CandidateTimelineEntry {
  candidateAttemptId: string;
  attemptId: string;
  candidateIndex: number;
  candidateId: string;
  role: CandidateRole;
  attemptState: CandidateAttemptState;
  recipeFingerprint: string;
  quantizationOverride?: CandidateQuantizationOverride;
  eligibilityTrigger?: string;
  disposition?: string;
  dispositionReasons: string[];
  selectionStatus: CandidateSelectionStatus;
  artifactRef?: string;
  packageRef?: string;
  invocationCounters: CandidateInvocationCounters;
  validatedScope: CandidateValidatedScope;
}

export interface CandidateSelectedSummary {
  candidateAttemptId: string;
  attemptId: string;
  candidateIndex: number;
  candidateId: string;
  selectedBy?: string;
  selectionReason?: string;
  selectedUtc?: string;
}

/**
 * Durable, measured-zero candidate-selection-reuse dispatch evidence: the
 * returned job/artifact aliases a previously selected winner's own build --
 * it never implies a new build ran for this attempt.
 */
export interface CandidateReuseEvidence {
  reusedWithoutBuild: boolean;
  sourceAttemptId: string;
  sourceCandidateAttemptId: string;
  sourceParentAttemptId: string;
  policyId: string;
  policyVersion: string;
  policyFingerprint: string;
  qualityProfileFingerprint: string;
  runnerDispatchCount: number;
  mobiusInvocationCount: number;
  oliveInvocationCount: number;
  recordedUtc: string;
}

/**
 * Additive (Slice 3C1) candidate-selection/timeline/reuse summary. `undefined`
 * on the parent attempt exactly when `workflowOutcome` is `not_applicable`.
 */
export interface RecipeAttemptCandidateSelection {
  policyId?: string;
  policyVersion?: string;
  policyFingerprint?: string;
  maxCandidates?: number;
  lineageSelectionState?: CandidateLineageSelectionState;
  selectedCandidate?: CandidateSelectedSummary;
  candidates: CandidateTimelineEntry[];
  aggregateInvocationCounters?: CandidateInvocationCounters;
  reuse?: CandidateReuseEvidence;
}

/**
 * Overall candidate-selection workflow outcome for an attempt. Always one of
 * these five stable codes; distinguishes the *workflow's* outcome (e.g.
 * `selected` once a fallback candidate is verified) from the attempt's own,
 * never-rewritten `state` (which stays `failed` for a regressed default even
 * when `workflowOutcome` is `selected` via its fallback sibling).
 */
export type WorkflowOutcome = "not_applicable" | "pending" | "selected" | "exhausted" | "reused";

export interface GeneratedRecipePreview {
  eligibleForAutomaticRecipeAttempt: boolean;
  requiresExplicitAttemptConfirmation: boolean;
  experimentalUntilVerified: boolean;
  fingerprint?: string;
  compileError?: string;
  capability: GeneratedRecipeCapability;
  argumentConfidence?: {
    mobiusDtypeConfidence: string;
    olivePrecisionConfidence: string;
    containsUnverifiedArguments: boolean;
  };
  validationGates: string[];
  verifiedReuse?: GeneratedRecipeVerifiedReuse;
  candidatePlan?: CandidateSelectionPlan;
}

export interface RecipeAttemptGate {
  sequence: number;
  gate: string;
  status: "passed" | "failed" | "not_run" | "unavailable";
  evidenceRef: string;
  metricsRef?: string;
  startedUtc: string;
  finishedUtc: string;
}

export interface RecipeIntegritySummary {
  status: "verified" | "blocked" | "inconclusive";
  gateStatus?: "passed" | "failed" | "missing" | "unavailable";
  runtimeFunctional?: boolean;
  baselineAvailable?: boolean;
  regressionFree?: boolean;
  canPromote?: boolean;
  integrityFailures?: string[];
}

export interface ModelCapabilityConfidenceSummary {
  level?: "high" | "low";
  determinismSupported?: boolean;
  reasons?: string[];
}

export interface ModelCapabilitySummary {
  checksPassed: number;
  totalChecks: number;
  warnings: string[];
  confidence?: ModelCapabilityConfidenceSummary;
}

export interface RecipeAttemptQualityValidation {
  recipeIntegrity: RecipeIntegritySummary;
  modelCapability?: ModelCapabilitySummary;
}

export interface RecipeAttemptStatus {
  attemptId: string;
  recipeFingerprint: string;
  state: "generated" | "running" | "succeeded" | "failed" | "cancelled";
  buildJobId?: string;
  gates: RecipeAttemptGate[];
  qualityValidation?: RecipeAttemptQualityValidation;
  failure?: {
    classification: string;
    stage: string;
    message: string;
    evidenceRefs: string[];
    sourceOwner: string;
    nextAction: string;
  };
  /**
   * Always present. Defaults to "not_applicable" when parsing a legacy
   * response that predates Slice 3C1 (the field is additive/omittable on
   * the wire, but never omitted by clients of this parser).
   */
  workflowOutcome: WorkflowOutcome;
  candidateSelection?: RecipeAttemptCandidateSelection;
}

export interface HealthSnapshot {
  status: string;
  service: string;
  testedModels: ModelSummary[];
  message?: string;
}

export interface ModelDetail {
  id: string;
  displayName: string;
  revision: string;
  task: ModelTask;
  modality: string;
  license: string;
  gated: boolean;
  requiresRemoteCode: boolean;
  estimatedSizeMb?: number;
  likelyCatalogMatch: string;
  mobiusSupport: string;
  mobiusRisk: string;
  testedStatus: TestedStatus;
  recipeStatus: RecipeStatus;
  recipeReason: string;
  recipeId?: string;
  recipeVersion?: string;
  requiresExperimentalOptIn: boolean;
  buildableWithExperimentalOptIn: boolean;
  supportedOptimizations: SupportedOptimization[];
  candidateOutcome?: CandidateOutcome;
  generatedRecipe?: GeneratedRecipePreview;
}

export interface ModelPreflight {
  modelId: string;
  task: ModelTask;
  target: "cpu";
  buildable: boolean;
  blockedReason?: string;
  strategies: string[];
  precisions: string[];
  verifiedAudioFormats: string[];
  defaultStrategy?: string;
  defaultPrecision?: string;
  defaultAudioFormat?: string;
  recipeStatus: RecipeStatus;
  recipeReason: string;
  recipeId?: string;
  recipeVersion?: string;
  requiresExperimentalOptIn: boolean;
  supportedOptimizations: SupportedOptimization[];
  candidateOutcome?: CandidateOutcome;
  generatedRecipe?: GeneratedRecipePreview;
}

export interface BuildRequest {
  modelId: string;
  task: Exclude<ModelTask, "unknown">;
  target: "cpu";
  allowExperimental?: boolean;
  optimization: {
    strategy: string;
    precision: string;
    audioFormat?: string;
  };
}

export interface BuildFailure {
  stage: BuildStage;
  classification: string;
  message: string;
  retryable: boolean;
  logTail: string[];
}

export interface ArtifactSummary {
  artifactId: string;
  packagePath?: string;
  checksum?: string;
}

export interface ReproducibilitySummary {
  recipeId?: string;
  mobiusVersion?: string;
  oliveVersion?: string;
}

export interface BuildStatus {
  jobId: string;
  modelId: string;
  task: ModelTask;
  stage: BuildStage;
  cancellable: boolean;
  artifactId?: string;
  artifactSummary?: ArtifactSummary;
  reproducibility?: ReproducibilitySummary;
  failure?: BuildFailure;
  updatedAt?: string;
}

export interface JobEvent {
  sequence: number;
  stage: BuildStage;
  message: string;
  timestamp?: string;
  classification?: string;
}

export interface TextInferenceResult {
  artifactId: string;
  output: string;
}

export interface AsrInferenceResult {
  artifactId: string;
  transcript: string;
}

export interface ApiClientConfig {
  baseUrl: string;
  warning?: string;
  fixtureMode: boolean;
}

export interface ApiClient {
  readonly config: ApiClientConfig;
  getHealth(): Promise<HealthSnapshot>;
  searchModels(query: string, limit?: number): Promise<ModelSummary[]>;
  getModelDetail(modelId: string): Promise<ModelDetail>;
  getGeneratedRecipePreview(modelId: string, task: Exclude<ModelTask, "unknown">): Promise<GeneratedRecipePreview>;
  preflightModel(request: {
    modelId: string;
    task: Exclude<ModelTask, "unknown">;
    target: "cpu";
    allowExperimental?: boolean;
  }): Promise<ModelPreflight>;
  startBuild(request: BuildRequest, idempotencyKey: string): Promise<BuildStatus>;
  startGeneratedRecipeAttempt(
    request: { modelId: string; recipeFingerprint: string; confirmAutomaticRecipeAttempt: boolean },
    idempotencyKey: string
  ): Promise<{ idempotentReplay: boolean; build: BuildStatus; attempt: RecipeAttemptStatus }>;
  getGeneratedRecipeAttempt(attemptId: string): Promise<RecipeAttemptStatus>;
  getBuildStatus(jobId: string): Promise<BuildStatus>;
  getBuildEvents(jobId: string, afterSequence: number): Promise<JobEvent[]>;
  cancelBuild(jobId: string): Promise<BuildStatus>;
  inferText(artifactId: string, prompt: string): Promise<TextInferenceResult>;
  inferAsr(artifactId: string, audioFile: File): Promise<AsrInferenceResult>;
}
