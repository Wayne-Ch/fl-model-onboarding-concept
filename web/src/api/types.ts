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
}

export interface RecipeAttemptGate {
  sequence: number;
  gate: string;
  status: "passed" | "failed";
  evidenceRef: string;
  metricsRef?: string;
  startedUtc: string;
  finishedUtc: string;
}

export interface RecipeAttemptStatus {
  attemptId: string;
  recipeFingerprint: string;
  state: "generated" | "running" | "succeeded" | "failed" | "cancelled";
  buildJobId?: string;
  gates: RecipeAttemptGate[];
  failure?: {
    classification: string;
    stage: string;
    message: string;
    evidenceRefs: string[];
    sourceOwner: string;
    nextAction: string;
  };
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
