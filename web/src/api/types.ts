export type ModelTask = "llm" | "asr" | "unknown";
export type TestedStatus = "tested" | "not_tested" | "failed" | "unknown";
export type BuildStage = string;

export interface ModelSummary {
  id: string;
  displayName: string;
  task: ModelTask;
  testedStatus: TestedStatus;
  gated: boolean;
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
}

export interface BuildRequest {
  modelId: string;
  target: "cpu";
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
  preflightModel(request: { modelId: string; target: "cpu" }): Promise<ModelPreflight>;
  startBuild(request: BuildRequest, idempotencyKey: string): Promise<BuildStatus>;
  getBuildStatus(jobId: string): Promise<BuildStatus>;
  getBuildEvents(jobId: string, afterSequence: number): Promise<JobEvent[]>;
  cancelBuild(jobId: string): Promise<BuildStatus>;
  inferText(artifactId: string, prompt: string): Promise<TextInferenceResult>;
  inferAsr(artifactId: string, audioFile: File): Promise<AsrInferenceResult>;
}
