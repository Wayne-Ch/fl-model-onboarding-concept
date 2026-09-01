import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { OnboardingShell } from "./App";
import type {
  ApiClient,
  AsrInferenceResult,
  BuildStatus,
  CandidateOutcome,
  GeneratedRecipePreview,
  HealthSnapshot,
  JobEvent,
  ModelDetail,
  ModelPreflight,
  ModelSummary,
  RecipeAttemptStatus,
  TextInferenceResult
} from "./api/types";

const llmModel: ModelSummary = {
  id: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  displayName: "SmolLM2 1.7B Instruct",
  task: "llm",
  testedStatus: "tested",
  gated: false
};

const graniteModel: ModelSummary = {
  id: "ibm-granite/granite-3.3-2b-instruct",
  displayName: "Granite 3.3 2B Instruct",
  task: "llm",
  testedStatus: "tested",
  gated: false
};

const asrModel: ModelSummary = {
  id: "openai/whisper-small.en",
  displayName: "Whisper Small",
  task: "asr",
  testedStatus: "not_verified",
  gated: false
};

const gatedModel: ModelSummary = {
  id: "meta-llama/Llama-3.1-8B-Instruct",
  displayName: "Llama 3.1 8B",
  task: "llm",
  testedStatus: "not_verified",
  gated: true
};

const experimentalModel: ModelSummary = {
  id: "contoso/experimental-llm-2b-instruct",
  displayName: "Experimental LLM 2B",
  task: "llm",
  testedStatus: "not_verified",
  gated: false
};

const generatedEligibleModel: ModelSummary = {
  id: "owner/unregistered-eligible",
  displayName: "Unregistered eligible model",
  task: "llm",
  testedStatus: "not_verified",
  gated: false
};

const asrBlockedOutcome: CandidateOutcome = {
  modelId: "distil-whisper/distil-medium.en",
  revision: "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
  profile: "cpu/ort-genai; mobius=f32; deterministic-adapter=parser+model-load",
  status: "blocked",
  testedStatus: "not_verified",
  failedStage: "inferencing",
  classification: "source_runtime_contract_incompatible",
  errorSummary:
    "Decoder ONNX requires position_ids, but OGA WhisperDecoderState does not bind/update it; OGA and Foundry Local transcription fail with Missing Input: position_ids.",
  versions: {
    mobius: "0.1.0",
    olive: "0.13.0",
    onnx: "1.22.0",
    onnxruntime: "1.29.0",
    onnxruntime_genai: "0.15.2",
    foundry_local_sdk: "1.2.4",
    foundry_cli: "0.11.0"
  },
  gateOutcomes: [
    { stage: "mobius_building", status: "passed", summary: "Mobius CPU ort-genai f32 build succeeded." },
    { stage: "runtime_validating", status: "passed", summary: "ONNX checker and ORT CPU load succeeded." },
    { stage: "fl_loading", status: "passed", summary: "Deterministic config adaptation advanced OGA parser/model-load gates." },
    {
      stage: "inferencing",
      status: "failed",
      summary: "OGA and Foundry Local transcription fail with Missing Input: position_ids (WhisperDecoderState does not bind/update position_ids)."
    }
  ],
  evidenceReference: "docs/asr-contract-repair.md#irreducible-failure-boundary (run 20260831-124030-fc016713)",
  capabilityOwner: "Primary owner: microsoft/onnxruntime-genai Whisper runtime; coordinate Mobius Whisper regression coverage.",
  nextAction:
    "Implement optional position_ids binding/updates from prompt + past sequence length, regression-test a Mobius-exported Whisper package, then rerun OGA + Foundry Local SDK transcription."
};

function detailFor(model: ModelSummary): ModelDetail {
  const isExperimental = model.id === experimentalModel.id;
  const isGranite = model.id === graniteModel.id;
  const isAsr = model.task === "asr";
  const isGeneratedEligible = model.id === generatedEligibleModel.id;
  return {
    id: model.id,
    displayName: model.displayName,
    revision: "abc123",
    task: model.task,
    modality: model.task === "asr" ? "audio" : "text",
    license: "mit",
    gated: model.gated,
    requiresRemoteCode: false,
    estimatedSizeMb: 1200,
    likelyCatalogMatch: model.id,
    mobiusSupport: isGeneratedEligible
      ? "not registered"
      : isExperimental
        ? "experimental (opt-in required)"
        : isAsr
          ? "blocked"
          : "verified",
    mobiusRisk: "low",
    testedStatus: model.testedStatus,
    recipeStatus: isGeneratedEligible ? "unregistered" : isExperimental ? "experimental" : isAsr ? "blocked" : "verified",
    recipeReason: isGeneratedEligible
      ? "No recipe is registered for this model profile."
      : isExperimental
      ? "Recipe requires explicit experimental opt-in."
      : isAsr
        ? asrBlockedOutcome.errorSummary
        : isGranite
          ? "Verified direct Mobius->Olive->runtime->Foundry Local SDK chat inference path for granite-3.3-2b pinned revision 707f574c62054322f6b5b04b6d075f0a8f05e0f0."
          : "Verified Mobius->Olive->runtime->Foundry Local SDK chat path for the pinned SmolLM2 revision.",
    recipeId: isGeneratedEligible
      ? undefined
      : isExperimental
      ? "experimental-llm-2b-cpu-int4"
      : isAsr
        ? "distil-whisper-cpu-fp16"
        : isGranite
          ? "granite-3.3-2b-cpu-int4"
          : "smollm2-1.7b-cpu-int4",
    recipeVersion: isGeneratedEligible ? undefined : "1.0.0",
    requiresExperimentalOptIn: isExperimental,
    buildableWithExperimentalOptIn: isExperimental,
    supportedOptimizations: isAsr || isGeneratedEligible
      ? []
      : [{ strategy: "mobius-olive", precision: "int4", taskProfile: "llm-cpu-int4", skipOlive: false, default: true }]
  };
}

function preflightFor(model: ModelSummary): ModelPreflight {
  const isExperimental = model.id === experimentalModel.id;
  const isGeneratedEligible = model.id === generatedEligibleModel.id;
  if (model.task === "asr") {
    return {
      modelId: model.id,
      task: "asr",
      target: "cpu",
      buildable: !model.gated,
      blockedReason: model.gated ? "Gated model access is rejected in this POC." : undefined,
      strategies: ["Auto", "Olive"],
      precisions: ["FP32"],
      verifiedAudioFormats: ["audio/wav", "audio/flac"],
      defaultStrategy: "Auto",
      defaultPrecision: "FP32",
      defaultAudioFormat: "audio/wav",
      recipeStatus: "blocked",
      recipeReason: "Blocked for this ASR profile.",
      recipeId: "distil-whisper-cpu-fp16",
      recipeVersion: "1.0.0",
      requiresExperimentalOptIn: false,
      supportedOptimizations: [
        { strategy: "Auto", precision: "FP32", taskProfile: "asr-cpu-fp16", skipOlive: false, default: true }
      ]
    };
  }
  return {
    modelId: model.id,
    task: "llm",
    target: "cpu",
    buildable: !model.gated && !isExperimental && !isGeneratedEligible,
    blockedReason: model.gated
      ? "Gated model access is rejected in this POC."
      : isGeneratedEligible
        ? "No recipe is registered for this model profile."
        : undefined,
    strategies: isGeneratedEligible ? [] : ["mobius-olive"],
    precisions: isGeneratedEligible ? [] : ["int4"],
    verifiedAudioFormats: [],
    defaultStrategy: isGeneratedEligible ? undefined : "mobius-olive",
    defaultPrecision: isGeneratedEligible ? undefined : "int4",
    recipeStatus: isGeneratedEligible ? "unregistered" : isExperimental ? "experimental" : "verified",
    recipeReason: isGeneratedEligible
      ? "No recipe is registered for this model profile."
      : isExperimental
        ? "Recipe requires explicit experimental opt-in."
        : "Verified recipe.",
    recipeId: isGeneratedEligible
      ? undefined
      : isExperimental
        ? "experimental-llm-2b-cpu-int4"
        : model.id === graniteModel.id
          ? "granite-3.3-2b-cpu-int4"
          : "smollm2-1.7b-cpu-int4",
    recipeVersion: isGeneratedEligible ? undefined : "1.0.0",
    requiresExperimentalOptIn: isExperimental,
    supportedOptimizations: isGeneratedEligible
      ? []
      : isExperimental
      ? [{ strategy: "mobius-olive", precision: "int4", taskProfile: "llm-cpu-int4", skipOlive: false, default: true }]
      : [{ strategy: "mobius-olive", precision: "int4", taskProfile: "llm-cpu-int4", skipOlive: false, default: true }]
  };
}

function generatedFor(model: ModelSummary): GeneratedRecipePreview {
  const eligible = model.id === "owner/unregistered-eligible";
  return {
    eligibleForAutomaticRecipeAttempt: eligible,
    requiresExplicitAttemptConfirmation: true,
    experimentalUntilVerified: true,
    fingerprint: eligible
      ? "2222222222222222222222222222222222222222222222222222222222222222"
      : undefined,
    compileError: eligible ? undefined : "Generated recipe unavailable.",
    capability: {
      outcome: eligible ? "exact" : "not-eligible",
      reasonCode: eligible ? "resolved" : "unsupported-task",
      reason: eligible ? "Resolved." : "Unavailable.",
      matchedAliases: [model.id]
    },
    validationGates: [
      "mobius_build",
      "olive_optimize",
      "onnx_validation",
      "ort_validation",
      "oga_validation",
      "fl_sdk_inference",
      "quality_validation"
    ],
    verifiedReuse: undefined
  };
}

function statusFor(model: ModelSummary, stage: string, overrides?: Partial<BuildStatus>): BuildStatus {
  return {
    jobId: "job-1",
    modelId: model.id,
    task: model.task,
    stage,
    cancellable: stage !== "succeeded" && stage !== "failed" && stage !== "cancelled",
    ...overrides
  };
}

function eventsAt(...events: JobEvent[]): JobEvent[] {
  return events;
}

function createClient(overrides: Partial<ApiClient> = {}): ApiClient {
  const health: HealthSnapshot = {
    status: "ok",
    service: "local",
    testedModels: [llmModel, graniteModel]
  };

  const getModelDetail = vi.fn(async (modelId: string) => {
    if (modelId === asrModel.id) {
      return detailFor(asrModel);
    }
    if (modelId === gatedModel.id) {
      return detailFor(gatedModel);
    }
    if (modelId === graniteModel.id) {
      return detailFor(graniteModel);
    }
    if (modelId === experimentalModel.id) {
      return detailFor(experimentalModel);
    }
    if (modelId === generatedEligibleModel.id) {
      return detailFor(generatedEligibleModel);
    }
    return detailFor(llmModel);
  });

  const preflightModel = vi.fn(
    async ({ modelId, allowExperimental }: { modelId: string; allowExperimental?: boolean }) => {
      if (modelId === asrModel.id) {
        return preflightFor(asrModel);
      }
      if (modelId === gatedModel.id) {
        return preflightFor(gatedModel);
      }
      if (modelId === graniteModel.id) {
        return preflightFor(graniteModel);
      }
      if (modelId === experimentalModel.id) {
        return {
          ...preflightFor(experimentalModel),
          buildable: Boolean(allowExperimental),
          requiresExperimentalOptIn: !allowExperimental,
          recipeReason: allowExperimental
            ? "Experimental recipe opt-in enabled."
            : "Recipe requires explicit experimental opt-in."
        };
      }
      if (modelId === generatedEligibleModel.id) {
        return preflightFor(generatedEligibleModel);
      }
      return preflightFor(llmModel);
    }
  );
  const getGeneratedRecipePreview = vi.fn(async (modelId: string) => {
    const model =
      [llmModel, graniteModel, asrModel, gatedModel, experimentalModel, generatedEligibleModel].find(
        (row) => row.id === modelId
      ) ??
      llmModel;
    return generatedFor(model);
  });
  const startGeneratedRecipeAttempt = vi.fn(
    async ({ modelId }: { modelId: string }): Promise<{ idempotentReplay: boolean; build: BuildStatus; attempt: RecipeAttemptStatus }> => ({
      idempotentReplay: false,
      build: statusFor({ ...llmModel, id: modelId }, "queued"),
      attempt: {
        attemptId: "attempt-1",
        recipeFingerprint: "2222222222222222222222222222222222222222222222222222222222222222",
        state: "running",
        buildJobId: "job-1",
        gates: []
      }
    })
  );
  const getGeneratedRecipeAttempt = vi.fn(async (): Promise<RecipeAttemptStatus> => ({
    attemptId: "attempt-1",
    recipeFingerprint: "2222222222222222222222222222222222222222222222222222222222222222",
    state: "running",
    buildJobId: "job-1",
    gates: []
  }));

  const base: ApiClient = {
    config: { baseUrl: "http://127.0.0.1:8080", fixtureMode: false },
    getHealth: vi.fn(async () => health),
    searchModels: vi.fn(async () => [asrModel, gatedModel, experimentalModel, generatedEligibleModel]),
    getModelDetail,
    getGeneratedRecipePreview,
    preflightModel,
    startBuild: vi.fn(async () => statusFor(llmModel, "queued")),
    startGeneratedRecipeAttempt,
    getGeneratedRecipeAttempt,
    getBuildStatus: vi.fn(async () => statusFor(llmModel, "succeeded", { artifactId: "artifact-1" })),
    getBuildEvents: vi.fn(async () => eventsAt({ sequence: 1, stage: "succeeded", message: "done" })),
    cancelBuild: vi.fn(async () => statusFor(llmModel, "cancelled")),
    inferText: vi.fn(async () => ({ artifactId: "artifact-1", output: "hello" } as TextInferenceResult)),
    inferAsr: vi.fn(async () => ({ artifactId: "artifact-1", transcript: "test transcript" } as AsrInferenceResult))
  };

  return { ...base, ...overrides };
}

describe("OnboardingShell", () => {
  it("loads metadata/preflight and reuses cache when re-selecting a model", async () => {
    const client = createClient();
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    expect(screen.getByRole("option", { name: /SmolLM2 1.7B Instruct/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Granite 3.3 2B Instruct/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByRole("option", { name: /Whisper Small/ });

    const modelSelect = screen.getByLabelText("Model");
    await user.selectOptions(modelSelect, llmModel.id);
    await screen.findByText(/Metadata and preflight/);
    await user.selectOptions(modelSelect, asrModel.id);
    await screen.findByText("Verified ASR audio format");
    await user.selectOptions(modelSelect, llmModel.id);

    await waitFor(() => {
      expect(client.getModelDetail).toHaveBeenCalledTimes(2);
      expect(client.preflightModel).toHaveBeenCalledTimes(2);
    });
  });

  it("renders task-aware optimization without unsupported options", async () => {
    const client = createClient();
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    const modelSelect = screen.getByLabelText("Model");
    await user.selectOptions(modelSelect, asrModel.id);
    await screen.findByText("Verified ASR audio format");

    const precisionSelect = screen.getByLabelText("Precision");
    expect(screen.getByRole("option", { name: "FP32" })).toBeInTheDocument();
    expect(precisionSelect).not.toHaveTextContent("INT4");
    expect(screen.getByRole("option", { name: "audio/wav" })).toBeInTheDocument();
  });

  it("blocks gated models from build", async () => {
    const client = createClient();
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.selectOptions(screen.getByLabelText("Model"), gatedModel.id);

    await screen.findByText("Gated model access is rejected in this POC.");
    expect(screen.getByRole("button", { name: "Build for CPU" })).toBeDisabled();
  });

  it("requires explicit opt-in before experimental recipes are buildable", async () => {
    const client = createClient();
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.selectOptions(screen.getByLabelText("Model"), experimentalModel.id);

    const optInWarnings = await screen.findAllByText(/Recipe requires explicit experimental opt-in/);
    expect(optInWarnings.length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Build for CPU" })).toBeDisabled();

    await user.click(screen.getByLabelText("Enable experimental recipe opt-in for this model"));
    await waitFor(() =>
      expect(client.preflightModel).toHaveBeenCalledWith(
        expect.objectContaining({ modelId: experimentalModel.id, allowExperimental: true })
      )
    );
    await waitFor(() => expect(screen.getByRole("button", { name: "Build for CPU" })).not.toBeDisabled());
  });

  it("requires explicit confirmation before automatic recipe attempt", async () => {
    const startGeneratedRecipeAttempt = vi.fn(
      async (): Promise<{ idempotentReplay: boolean; build: BuildStatus; attempt: RecipeAttemptStatus }> => ({
        idempotentReplay: false,
        build: statusFor(generatedEligibleModel, "queued"),
        attempt: {
          attemptId: "attempt-1",
          recipeFingerprint: "2222222222222222222222222222222222222222222222222222222222222222",
          state: "running",
          buildJobId: "job-1",
          gates: []
        }
      })
    );
    const client = createClient({ startGeneratedRecipeAttempt });
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.selectOptions(screen.getByLabelText("Model"), generatedEligibleModel.id);

    const runAttemptButton = await screen.findByRole("button", { name: "Run automatic recipe attempt" });
    expect(runAttemptButton).toBeDisabled();
    expect(screen.getByText("Automatic recipe attempts require explicit confirmation.")).toBeInTheDocument();

    await user.click(
      screen.getByLabelText(
        "Confirm automatic recipe attempt for fingerprint 2222222222222222222222222222222222222222222222222222222222222222"
      )
    );
    await waitFor(() => expect(runAttemptButton).not.toBeDisabled());

    await user.click(runAttemptButton);
    await waitFor(() => expect(startGeneratedRecipeAttempt).toHaveBeenCalledTimes(1));
    expect(client.startBuild).not.toHaveBeenCalled();
  });

  it("shows baseline-unavailable recipe gate status from generated attempts", async () => {
    const startGeneratedRecipeAttempt = vi.fn(
      async (): Promise<{ idempotentReplay: boolean; build: BuildStatus; attempt: RecipeAttemptStatus }> => ({
        idempotentReplay: false,
        build: statusFor(generatedEligibleModel, "queued"),
        attempt: {
          attemptId: "attempt-1",
          recipeFingerprint: "2222222222222222222222222222222222222222222222222222222222222222",
          state: "running",
          buildJobId: "job-1",
          gates: []
        }
      })
    );
    const getGeneratedRecipeAttempt = vi.fn(async (): Promise<RecipeAttemptStatus> => ({
      attemptId: "attempt-1",
      recipeFingerprint: "2222222222222222222222222222222222222222222222222222222222222222",
      state: "failed",
      buildJobId: "job-1",
      gates: [
        {
          sequence: 1,
          gate: "mobius_build",
          status: "passed",
          evidenceRef: "job://job-1/mobius_build/passed",
          startedUtc: "2026-01-01T00:00:00Z",
          finishedUtc: "2026-01-01T00:00:01Z"
        },
        {
          sequence: 2,
          gate: "quality_validation",
          status: "unavailable",
          evidenceRef: "quality://job-1/quality_validation/baseline-unavailable",
          startedUtc: "2026-01-01T00:00:02Z",
          finishedUtc: "2026-01-01T00:00:03Z"
        }
      ],
      failure: {
        classification: "validation_failed",
        stage: "succeeded",
        message: "Quality baseline unavailable.",
        evidenceRefs: ["job://job-1"],
        sourceOwner: "fl-onboarding",
        nextAction: "Produce a baseline package."
      }
    }));
    const client = createClient({
      startGeneratedRecipeAttempt,
      getGeneratedRecipeAttempt,
      getBuildStatus: vi.fn(async () => statusFor(generatedEligibleModel, "succeeded", { artifactId: "artifact-1" })),
      getBuildEvents: vi.fn(async () => eventsAt({ sequence: 1, stage: "succeeded", message: "done" }))
    });
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.selectOptions(screen.getByLabelText("Model"), generatedEligibleModel.id);
    await user.click(
      screen.getByLabelText(
        "Confirm automatic recipe attempt for fingerprint 2222222222222222222222222222222222222222222222222222222222222222"
      )
    );
    await user.click(await screen.findByRole("button", { name: "Run automatic recipe attempt" }));

    await screen.findByText("Recipe attempt gates");
    expect(screen.getByText("quality_validation")).toBeInTheDocument();
    expect(screen.getByText("unavailable")).toBeInTheDocument();
    expect(screen.getByText(/Quality baseline unavailable/)).toBeInTheDocument();
  });

  it("shows baseline-passed recipe gate status from generated attempts", async () => {
    const startGeneratedRecipeAttempt = vi.fn(
      async (): Promise<{ idempotentReplay: boolean; build: BuildStatus; attempt: RecipeAttemptStatus }> => ({
        idempotentReplay: false,
        build: statusFor(generatedEligibleModel, "queued"),
        attempt: {
          attemptId: "attempt-1",
          recipeFingerprint: "2222222222222222222222222222222222222222222222222222222222222222",
          state: "running",
          buildJobId: "job-1",
          gates: []
        }
      })
    );
    const getGeneratedRecipeAttempt = vi.fn(async (): Promise<RecipeAttemptStatus> => ({
      attemptId: "attempt-1",
      recipeFingerprint: "2222222222222222222222222222222222222222222222222222222222222222",
      state: "succeeded",
      buildJobId: "job-1",
      gates: [
        {
          sequence: 1,
          gate: "quality_validation",
          status: "passed",
          evidenceRef: "quality://job-1/quality_validation/baseline-passed",
          startedUtc: "2026-01-01T00:00:02Z",
          finishedUtc: "2026-01-01T00:00:03Z"
        }
      ]
    }));
    const client = createClient({
      startGeneratedRecipeAttempt,
      getGeneratedRecipeAttempt,
      getBuildStatus: vi.fn(async () => statusFor(generatedEligibleModel, "succeeded", { artifactId: "artifact-1" })),
      getBuildEvents: vi.fn(async () => eventsAt({ sequence: 1, stage: "succeeded", message: "done" }))
    });
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.selectOptions(screen.getByLabelText("Model"), generatedEligibleModel.id);
    await user.click(
      screen.getByLabelText(
        "Confirm automatic recipe attempt for fingerprint 2222222222222222222222222222222222222222222222222222222222222222"
      )
    );
    await user.click(await screen.findByRole("button", { name: "Run automatic recipe attempt" }));

    await screen.findByText("Recipe attempt gates");
    expect(screen.getByText("quality://job-1/quality_validation/baseline-passed")).toBeInTheDocument();
    expect(screen.getByText("passed")).toBeInTheDocument();
  });

  it("keeps the verified ASR blocker visible and out of tested success", async () => {
    const blockedAsr = { ...asrModel, id: asrBlockedOutcome.modelId };
    const client = createClient({
      searchModels: vi.fn(async () => [blockedAsr]),
      getModelDetail: vi.fn(async () => ({
        ...detailFor(blockedAsr),
        revision: asrBlockedOutcome.revision,
        candidateOutcome: asrBlockedOutcome
      })),
      preflightModel: vi.fn(async () => ({
        ...preflightFor(blockedAsr),
        buildable: false,
        blockedReason: asrBlockedOutcome.errorSummary,
        strategies: [],
        precisions: [],
        verifiedAudioFormats: [],
        candidateOutcome: asrBlockedOutcome
      }))
    });
    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.selectOptions(screen.getByLabelText("Model"), blockedAsr.id);

    expect(await screen.findByText("Candidate evidence history")).toBeInTheDocument();
    expect(screen.getByText(/Blocked \/ Not tested successfully/)).toBeInTheDocument();
    expect(screen.getByText(/source_runtime_contract_incompatible/)).toBeInTheDocument();
    expect(screen.getAllByText(/Missing Input: position_ids/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Mobius CPU ort-genai f32 build succeeded/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Build for CPU" })).toBeDisabled();
  });

  it("shows backend event stream and supports cancellation with cursor polling", async () => {
    const getBuildStatus = vi
      .fn()
      .mockResolvedValueOnce(statusFor(llmModel, "downloading"))
      .mockResolvedValueOnce(statusFor(llmModel, "cancelled", { cancellable: false }));
    const getBuildEvents = vi
      .fn()
      .mockResolvedValueOnce(
        eventsAt(
          { sequence: 1, stage: "queued", message: "queued" },
          { sequence: 2, stage: "downloading", message: "downloading" }
        )
      )
      .mockResolvedValueOnce(eventsAt({ sequence: 3, stage: "cancelled", message: "cancelled" }));

    const client = createClient({
      startBuild: vi.fn(async () => statusFor(llmModel, "queued")),
      getBuildStatus,
      getBuildEvents,
      cancelBuild: vi.fn(async () => statusFor(llmModel, "cancelled", { cancellable: false }))
    });

    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.selectOptions(screen.getByLabelText("Model"), llmModel.id);
    await screen.findByText(/Task-aware optimization for CPU/);

    await user.click(screen.getByRole("button", { name: "Build for CPU" }));
    await screen.findByText(
      (_, element) => element?.tagName.toLowerCase() === "p" && element.textContent?.includes("Current stage: Downloading") === true
    );
    await user.click(screen.getByRole("button", { name: "Cancel build" }));
    await screen.findByText(
      (_, element) => element?.tagName.toLowerCase() === "p" && element.textContent?.includes("Current stage: Cancelled") === true
    );

    expect(screen.getByText(/Retrying starts a new build job/)).toBeInTheDocument();
    expect(getBuildEvents).toHaveBeenCalledWith("job-1", 0);
    expect(getBuildEvents).toHaveBeenCalledWith("job-1", 2);
  });

  it("shows failure classification and sanitized log tail", async () => {
    const client = createClient({
      startBuild: vi.fn(async () => statusFor(llmModel, "queued")),
      getBuildStatus: vi.fn(async () =>
        statusFor(llmModel, "failed", {
          cancellable: false,
          failure: {
            stage: "mobius_validating",
            classification: "compatibility",
            message: "Validation failed.",
            retryable: true,
            logTail: ["E: kernel mismatch", "E: calibration failed"]
          }
        })
      ),
      getBuildEvents: vi.fn(async () => eventsAt({ sequence: 1, stage: "failed", message: "failed" }))
    });

    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.selectOptions(screen.getByLabelText("Model"), llmModel.id);
    await user.click(screen.getByRole("button", { name: "Build for CPU" }));

    await screen.findByText("Failure details");
    expect(screen.getByText(/compatibility/i)).toBeInTheDocument();
    expect(screen.getByText(/Validation failed\./)).toBeInTheDocument();
    expect(screen.getByText(/kernel mismatch/i)).toBeInTheDocument();
  });

  it("supports text inference after a successful LLM build", async () => {
    const inferText = vi.fn(async () => ({ artifactId: "artifact-1", output: "Generated text." }));
    const client = createClient({
      startBuild: vi.fn(async () => statusFor(llmModel, "queued")),
      getBuildStatus: vi.fn(async () =>
        statusFor(llmModel, "succeeded", {
          cancellable: false,
          artifactId: "artifact-1",
          artifactSummary: { artifactId: "artifact-1", packagePath: "C:\\pkg.zip", checksum: "sha256:abc" }
        })
      ),
      getBuildEvents: vi.fn(async () => eventsAt({ sequence: 1, stage: "succeeded", message: "done" })),
      inferText
    });

    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.selectOptions(screen.getByLabelText("Model"), llmModel.id);
    await user.click(screen.getByRole("button", { name: "Build for CPU" }));
    await screen.findByText("Success and inference");

    await user.click(screen.getByRole("button", { name: "Run text inference" }));
    await screen.findByText("Generated text.");
    expect(inferText).toHaveBeenCalledWith("artifact-1", expect.any(String));
  });

  it("supports ASR file inference on successful ASR builds", async () => {
    const inferAsr = vi.fn(async () => ({ artifactId: "artifact-2", transcript: "Hello from ASR." }));
    const client = createClient({
      getBuildStatus: vi.fn(async () =>
        statusFor(asrModel, "succeeded", {
          cancellable: false,
          artifactId: "artifact-2"
        })
      ),
      getBuildEvents: vi.fn(async () => eventsAt({ sequence: 1, stage: "succeeded", message: "done" })),
      startBuild: vi.fn(async () => statusFor(asrModel, "queued")),
      inferAsr
    });

    const user = userEvent.setup();
    render(<OnboardingShell client={client} />);

    await screen.findByText(/Connected/);
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.selectOptions(screen.getByLabelText("Model"), asrModel.id);
    await user.click(screen.getByRole("button", { name: "Build for CPU" }));
    await screen.findByText("Success and inference");

    const file = new File(["fake"], "sample.wav", { type: "audio/wav" });
    await user.upload(screen.getByLabelText("Audio file"), file);
    await user.click(screen.getByRole("button", { name: "Run ASR inference" }));

    await screen.findByText("Hello from ASR.");
    expect(inferAsr).toHaveBeenCalledWith("artifact-2", expect.any(File));
  });
});
