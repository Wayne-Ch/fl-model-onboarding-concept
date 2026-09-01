import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { OnboardingShell } from "./App";
import type { ApiClient, AsrInferenceResult, BuildStatus, CandidateOutcome, HealthSnapshot, JobEvent, ModelDetail, ModelPreflight, ModelSummary, TextInferenceResult } from "./api/types";

const llmModel: ModelSummary = {
  id: "microsoft/Phi-3-mini-4k-instruct",
  displayName: "Phi-3 Mini",
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

const asrBlockedOutcome: CandidateOutcome = {
  modelId: "distil-whisper/distil-medium.en",
  revision: "6e61418885eaf4d5cc9f64e508e80ac5b4c052b7",
  profile: "cpu/ort-genai; mobius=f32; olive=existing-onnx-decoder/fp32",
  status: "blocked",
  testedStatus: "not_verified",
  failedStage: "fl_loading",
  classification: "oga_runtime_contract_incompatible",
  errorSummary: "Generated Whisper genai_config rejected while parsing decoder_input_ids.",
  versions: {
    mobius: "0.1.0",
    olive: "0.13.0",
    onnxruntime_genai: "0.15.2",
    foundry_local_sdk: "1.2.4"
  },
  gateOutcomes: [
    { stage: "mobius_building", status: "passed", summary: "Mobius build succeeded." },
    { stage: "fl_loading", status: "failed", summary: "OGA and Foundry Local SDK rejected the config." }
  ],
  evidenceReference: "docs/contract-probe-results.md (run 20260830-225442-66553c73)",
  capabilityOwner: "Mobius-generated Whisper config <-> OGA/Foundry Local runtime contract integration",
  nextAction: "Compare against the pinned OGA Whisper schema and rerun the same profile."
};

function detailFor(model: ModelSummary): ModelDetail {
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
    mobiusSupport: "supported",
    mobiusRisk: "low",
    testedStatus: model.testedStatus
  };
}

function preflightFor(model: ModelSummary): ModelPreflight {
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
      defaultAudioFormat: "audio/wav"
    };
  }
  return {
    modelId: model.id,
    task: "llm",
    target: "cpu",
    buildable: !model.gated,
    blockedReason: model.gated ? "Gated model access is rejected in this POC." : undefined,
    strategies: ["Auto", "Olive"],
    precisions: ["INT4", "FP16"],
    verifiedAudioFormats: [],
    defaultStrategy: "Auto",
    defaultPrecision: "INT4"
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
    testedModels: [llmModel]
  };

  const getModelDetail = vi.fn(async (modelId: string) => {
    if (modelId === asrModel.id) {
      return detailFor(asrModel);
    }
    if (modelId === gatedModel.id) {
      return detailFor(gatedModel);
    }
    return detailFor(llmModel);
  });

  const preflightModel = vi.fn(async ({ modelId }: { modelId: string }) => {
    if (modelId === asrModel.id) {
      return preflightFor(asrModel);
    }
    if (modelId === gatedModel.id) {
      return preflightFor(gatedModel);
    }
    return preflightFor(llmModel);
  });

  const base: ApiClient = {
    config: { baseUrl: "http://127.0.0.1:8080", fixtureMode: false },
    getHealth: vi.fn(async () => health),
    searchModels: vi.fn(async () => [asrModel, gatedModel]),
    getModelDetail,
    preflightModel,
    startBuild: vi.fn(async () => statusFor(llmModel, "queued")),
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
    expect(screen.getByText(/oga_runtime_contract_incompatible/)).toBeInTheDocument();
    expect(screen.getAllByText(/decoder_input_ids/)).toHaveLength(2);
    expect(screen.getByText(/Mobius build succeeded/)).toBeInTheDocument();
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
