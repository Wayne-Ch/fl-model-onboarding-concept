import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, createApiClient } from "./api/client";
import type {
  ApiClient,
  BuildRequest,
  BuildStatus,
  CandidateOutcome,
  JobEvent,
  ModelPreflight,
  ModelSummary
} from "./api/types";

const defaultClient = createApiClient();
const POLL_INTERVAL_MS = 2500;

interface LoadedModel {
  detailLoaded: boolean;
  preflight: ModelPreflight;
  summary: ModelSummary;
  metadata: {
    revision: string;
    modality: string;
    license: string;
    requiresRemoteCode: boolean;
    estimatedSizeMb?: number;
    likelyCatalogMatch: string;
    mobiusSupport: string;
    mobiusRisk: string;
    candidateOutcome?: CandidateOutcome;
  };
}

function readableStage(stage: string): string {
  return stage
    .split("_")
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

function isTerminalStage(stage: string): boolean {
  return stage === "succeeded" || stage === "failed" || stage === "cancelled";
}

function createIdempotencyKey(): string {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `build-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function asMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error.";
}

function chooseOption(options: string[], preferred: Array<string | undefined>): string {
  for (const candidate of preferred) {
    if (!candidate) {
      continue;
    }
    const match = options.find((option) => option.toLowerCase() === candidate.toLowerCase());
    if (match) {
      return match;
    }
  }
  return options[0] ?? "";
}

function defaultsFromPreflight(preflight: ModelPreflight): { strategy: string; precision: string; audioFormat: string } {
  const taskStrategyPreference = preflight.task === "asr" ? "Auto" : "Auto";
  const taskPrecisionPreference = preflight.task === "asr" ? "FP32" : "INT4";
  return {
    strategy: chooseOption(preflight.strategies, [preflight.defaultStrategy, taskStrategyPreference]),
    precision: chooseOption(preflight.precisions, [preflight.defaultPrecision, taskPrecisionPreference]),
    audioFormat: chooseOption(preflight.verifiedAudioFormats, [preflight.defaultAudioFormat])
  };
}

function mergeEvents(current: JobEvent[], incoming: JobEvent[]): JobEvent[] {
  const bySequence = new Map<number, JobEvent>();
  for (const event of current) {
    bySequence.set(event.sequence, event);
  }
  for (const event of incoming) {
    bySequence.set(event.sequence, event);
  }
  return Array.from(bySequence.values()).sort((left, right) => left.sequence - right.sequence);
}

function CandidateOutcomeDetails({ outcome }: { outcome: CandidateOutcome }): JSX.Element {
  return (
    <div className="candidate-outcome">
      <h3>Candidate evidence history</h3>
      <p>
        <strong>Model revision:</strong> {outcome.modelId}@{outcome.revision}
      </p>
      <p>
        <strong>Profile:</strong> {outcome.profile}
      </p>
      <p>
        <strong>Failed stage:</strong> {readableStage(outcome.failedStage)} ({outcome.classification})
      </p>
      <p>
        <strong>Error:</strong> {outcome.errorSummary}
      </p>
      <ol>
        {outcome.gateOutcomes.map((gate) => (
          <li key={`${gate.stage}-${gate.status}`}>
            <strong>{readableStage(gate.stage)}:</strong> {gate.status} - {gate.summary}
          </li>
        ))}
      </ol>
      <p>
        <strong>Versions:</strong>{" "}
        {Object.entries(outcome.versions)
          .map(([name, version]) => `${name}=${version}`)
          .join(", ")}
      </p>
      <p>
        <strong>Evidence:</strong> {outcome.evidenceReference}
      </p>
      <p>
        <strong>Capability owner:</strong> {outcome.capabilityOwner}
      </p>
      <p>
        <strong>Next action:</strong> {outcome.nextAction}
      </p>
    </div>
  );
}

export function OnboardingShell({ client }: { client: ApiClient }): JSX.Element {
  const [healthState, setHealthState] = useState<"loading" | "ready" | "unavailable">("loading");
  const [healthMessage, setHealthMessage] = useState<string>("Connecting to local service...");
  const [testedModels, setTestedModels] = useState<ModelSummary[]>([]);

  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searching, setSearching] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | undefined>();
  const [searchResults, setSearchResults] = useState<ModelSummary[]>([]);

  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [selectionLoading, setSelectionLoading] = useState<boolean>(false);
  const [selectionError, setSelectionError] = useState<string | undefined>();
  const [modelCache, setModelCache] = useState<Record<string, LoadedModel>>({});

  const [strategy, setStrategy] = useState<string>("");
  const [precision, setPrecision] = useState<string>("");
  const [audioFormat, setAudioFormat] = useState<string>("");

  const [startingBuild, setStartingBuild] = useState<boolean>(false);
  const [buildError, setBuildError] = useState<string | undefined>();
  const [currentJob, setCurrentJob] = useState<BuildStatus | undefined>();
  const [events, setEvents] = useState<JobEvent[]>([]);

  const [textPrompt, setTextPrompt] = useState<string>("Write a concise product value statement.");
  const [textOutput, setTextOutput] = useState<string>("");
  const [asrFile, setAsrFile] = useState<File | undefined>();
  const [asrOutput, setAsrOutput] = useState<string>("");

  const [announcement, setAnnouncement] = useState<string>("Ready.");

  const selectionTokenRef = useRef(0);
  const idempotencyRef = useRef<{ signature: string; key: string } | undefined>();
  const eventCursorRef = useRef<Record<string, number>>({});

  const selectedModel = useMemo(() => {
    if (!selectedModelId) {
      return undefined;
    }
    return modelCache[selectedModelId];
  }, [modelCache, selectedModelId]);

  const blockedReason = useMemo(() => {
    if (!selectedModel) {
      return undefined;
    }
    if (selectedModel.summary.gated) {
      return "Gated model access is rejected in this POC.";
    }
    if (!selectedModel.preflight.buildable) {
      return selectedModel.preflight.blockedReason ?? "Backend marked this model as not buildable.";
    }
    return undefined;
  }, [selectedModel]);

  useEffect(() => {
    let cancelled = false;
    async function loadHealth(): Promise<void> {
      try {
        const health = await client.getHealth();
        if (cancelled) {
          return;
        }
        setTestedModels(health.testedModels);
        setHealthState("ready");
        setHealthMessage(`Connected (${health.service})`);
      } catch {
        if (cancelled) {
          return;
        }
        setHealthState("unavailable");
        setHealthMessage("Local service unavailable");
      }
    }
    void loadHealth();
    return () => {
      cancelled = true;
    };
  }, [client]);

  const applyPreflightDefaults = useCallback((preflight: ModelPreflight) => {
    const defaults = defaultsFromPreflight(preflight);
    setStrategy(defaults.strategy);
    setPrecision(defaults.precision);
    setAudioFormat(defaults.audioFormat);
  }, []);

  const loadModelContext = useCallback(
    async (model: ModelSummary) => {
      const existing = modelCache[model.id];
      if (existing) {
        applyPreflightDefaults(existing.preflight);
        return;
      }

      const token = selectionTokenRef.current + 1;
      selectionTokenRef.current = token;
      setSelectionLoading(true);
      setSelectionError(undefined);

      try {
        const detail = await client.getModelDetail(model.id);
        const preflight = await client.preflightModel({
          modelId: model.id,
          task: detail.task === "asr" ? "asr" : "llm",
          target: "cpu"
        });
        if (selectionTokenRef.current !== token) {
          return;
        }

        const resolvedPreflight: ModelPreflight = model.gated
          ? {
              ...preflight,
              buildable: false,
              blockedReason: "Gated model access is rejected in this POC."
            }
          : preflight;

        const loaded: LoadedModel = {
          detailLoaded: true,
          summary: {
            id: detail.id,
            displayName: detail.displayName,
            task: detail.task,
            testedStatus: detail.testedStatus,
            gated: detail.gated
          },
          preflight: resolvedPreflight,
          metadata: {
            revision: detail.revision,
            modality: detail.modality,
            license: detail.license,
            requiresRemoteCode: detail.requiresRemoteCode,
            estimatedSizeMb: detail.estimatedSizeMb,
            likelyCatalogMatch: detail.likelyCatalogMatch,
            mobiusSupport: detail.mobiusSupport,
            mobiusRisk: detail.mobiusRisk,
            candidateOutcome: detail.candidateOutcome ?? preflight.candidateOutcome
          }
        };

        setModelCache((current) => ({ ...current, [model.id]: loaded }));
        applyPreflightDefaults(resolvedPreflight);
      } catch (error) {
        if (selectionTokenRef.current !== token) {
          return;
        }
        setSelectionError(asMessage(error));
      } finally {
        if (selectionTokenRef.current === token) {
          setSelectionLoading(false);
        }
      }
    },
    [applyPreflightDefaults, client, modelCache]
  );

  const onSearch = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setSearchError(undefined);
      setSearching(true);
      try {
        const results = await client.searchModels(searchQuery.trim(), 20);
        const testedIds = new Set(testedModels.map((model) => model.id));
        setSearchResults(results.filter((model) => !testedIds.has(model.id)));
      } catch (error) {
        setSearchError(asMessage(error));
      } finally {
        setSearching(false);
      }
    },
    [client, searchQuery, testedModels]
  );

  const refreshJob = useCallback(
    async (jobId: string) => {
      const afterSequence = eventCursorRef.current[jobId] ?? 0;
      const [status, incrementalEvents] = await Promise.all([
        client.getBuildStatus(jobId),
        client.getBuildEvents(jobId, afterSequence)
      ]);

      setCurrentJob(status);
      if (incrementalEvents.length > 0) {
        setEvents((current) => mergeEvents(current, incrementalEvents));
        eventCursorRef.current[jobId] = incrementalEvents[incrementalEvents.length - 1].sequence;
      }

      if (status.stage === "succeeded") {
        setAnnouncement("Build succeeded.");
      } else if (status.stage === "failed") {
        setAnnouncement("Build failed.");
      } else if (status.stage === "cancelled") {
        setAnnouncement("Build cancelled.");
      } else {
        setAnnouncement(`Build stage: ${readableStage(status.stage)}`);
      }
    },
    [client]
  );

  useEffect(() => {
    if (!currentJob || isTerminalStage(currentJob.stage)) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshJob(currentJob.jobId).catch((error) => {
        setBuildError(asMessage(error));
      });
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [currentJob, refreshJob]);

  const onStartBuild = useCallback(async () => {
    if (!selectedModel) {
      return;
    }
    if (blockedReason) {
      return;
    }
    if (!strategy || !precision) {
      setBuildError("Select strategy and precision before starting build.");
      return;
    }

    const request: BuildRequest = {
      modelId: selectedModel.summary.id,
      task: selectedModel.summary.task === "asr" ? "asr" : "llm",
      target: "cpu",
      optimization: {
        strategy,
        precision,
        audioFormat: selectedModel.preflight.task === "asr" ? audioFormat || undefined : undefined
      }
    };
    const signature = JSON.stringify(request);
    const key = idempotencyRef.current?.signature === signature ? idempotencyRef.current.key : createIdempotencyKey();
    idempotencyRef.current = { signature, key };

    setStartingBuild(true);
    setBuildError(undefined);
    setTextOutput("");
    setAsrOutput("");
    try {
      const started = await client.startBuild(request, key);
      idempotencyRef.current = undefined;
      eventCursorRef.current[started.jobId] = 0;
      setEvents([]);
      setCurrentJob(started);
      await refreshJob(started.jobId);
    } catch (error) {
      setBuildError(asMessage(error));
    } finally {
      setStartingBuild(false);
    }
  }, [audioFormat, blockedReason, client, refreshJob, selectedModel, strategy, precision]);

  const onCancelBuild = useCallback(async () => {
    if (!currentJob || isTerminalStage(currentJob.stage) || !currentJob.cancellable) {
      return;
    }
    setBuildError(undefined);
    try {
      const cancelled = await client.cancelBuild(currentJob.jobId);
      setCurrentJob(cancelled);
      await refreshJob(currentJob.jobId);
    } catch (error) {
      setBuildError(asMessage(error));
    }
  }, [client, currentJob, refreshJob]);

  const onRunTextInference = useCallback(async () => {
    if (!currentJob?.artifactId) {
      return;
    }
    setBuildError(undefined);
    try {
      const result = await client.inferText(currentJob.artifactId, textPrompt);
      setTextOutput(result.output);
    } catch (error) {
      setBuildError(asMessage(error));
    }
  }, [client, currentJob?.artifactId, textPrompt]);

  const onRunAsrInference = useCallback(async () => {
    if (!currentJob?.artifactId || !asrFile) {
      return;
    }
    setBuildError(undefined);
    try {
      const result = await client.inferAsr(currentJob.artifactId, asrFile);
      setAsrOutput(result.transcript);
    } catch (error) {
      setBuildError(asMessage(error));
    }
  }, [asrFile, client, currentJob?.artifactId]);

  const allOptions = useMemo(() => {
    const byId = new Map<string, ModelSummary>();
    for (const tested of testedModels) {
      byId.set(tested.id, tested);
    }
    for (const result of searchResults) {
      byId.set(result.id, result);
    }
    return byId;
  }, [searchResults, testedModels]);

  const canStartBuild = healthState === "ready" && Boolean(selectedModel) && !blockedReason && !startingBuild;

  return (
    <div className="app-shell">
      <p className="sr-only" role="status" aria-live="polite">
        {announcement}
      </p>
      <aside className="panel stepper" aria-label="Workflow">
        <h2>Build flow</h2>
        <ol>
          <li>Select model</li>
          <li>Review preflight and choose optimization</li>
          <li>Build for CPU</li>
          <li>Review backend events/status</li>
          <li>Run task-specific inference</li>
        </ol>
      </aside>

      <main className="content">
        <header className="panel">
          <h1>Foundry Local model onboarding</h1>
          <p className="muted">Backend-connected UI shell for model preflight, build orchestration, and artifact inference.</p>
          <p className="service-state">
            <strong>Service:</strong> {healthMessage}
          </p>
          {client.config.warning ? <p className="warning">{client.config.warning}</p> : null}
          {healthState === "unavailable" ? (
            <p className="error" role="alert">
              Local service unavailable
            </p>
          ) : null}
        </header>

        <section className="panel">
          <h2>1. Model selection</h2>
          <form className="search-row" onSubmit={onSearch}>
            <label htmlFor="hf-search">Search Hugging Face</label>
            <input
              id="hf-search"
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="e.g. phi, whisper"
            />
            <button type="submit" disabled={searching || healthState !== "ready"}>
              {searching ? "Searching..." : "Search"}
            </button>
          </form>
          {searchError ? (
            <p className="error" role="alert">
              {searchError}
            </p>
          ) : null}

          <label htmlFor="model-select">Model</label>
          <select
            id="model-select"
            value={selectedModelId}
            onChange={(event) => {
              const nextId = event.target.value;
              setSelectedModelId(nextId);
              setCurrentJob(undefined);
              setEvents([]);
              setBuildError(undefined);
              setTextOutput("");
              setAsrOutput("");
              if (!nextId) {
                return;
              }
              const model = allOptions.get(nextId);
              if (model) {
                void loadModelContext(model);
              }
            }}
            disabled={healthState !== "ready"}
          >
            <option value="">Choose a model</option>
            <optgroup label="Tested successfully">
              {testedModels.map((model) => (
                <option key={`tested-${model.id}`} value={model.id}>
                  {model.displayName} ({model.id})
                </option>
              ))}
            </optgroup>
            <optgroup label="Search Hugging Face">
              {searchResults.map((model) => (
                <option key={`search-${model.id}`} value={model.id}>
                  {model.displayName} ({model.id}){model.gated ? " [gated]" : ""}
                </option>
              ))}
            </optgroup>
          </select>
          {selectionLoading ? <p className="muted">Loading model metadata and preflight...</p> : null}
          {selectionError ? (
            <p className="error" role="alert">
              {selectionError}
            </p>
          ) : null}
        </section>

        {selectedModel ? (
          <section className="panel">
            <h2>2. Metadata and preflight</h2>
            <dl className="metadata-grid">
              <div>
                <dt>HF revision</dt>
                <dd>{selectedModel.metadata.revision}</dd>
              </div>
              <div>
                <dt>Task</dt>
                <dd>{selectedModel.summary.task}</dd>
              </div>
              <div>
                <dt>Modality</dt>
                <dd>{selectedModel.metadata.modality}</dd>
              </div>
              <div>
                <dt>License</dt>
                <dd>{selectedModel.metadata.license}</dd>
              </div>
              <div>
                <dt>Gated</dt>
                <dd>{selectedModel.summary.gated ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Remote code required</dt>
                <dd>{selectedModel.metadata.requiresRemoteCode ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Estimated size</dt>
                <dd>
                  {selectedModel.metadata.estimatedSizeMb !== undefined
                    ? `${selectedModel.metadata.estimatedSizeMb.toLocaleString()} MB`
                    : "Unknown"}
                </dd>
              </div>
              <div>
                <dt>Likely FL catalog match</dt>
                <dd>{selectedModel.metadata.likelyCatalogMatch}</dd>
              </div>
              <div>
                <dt>Mobius support</dt>
                <dd>{selectedModel.metadata.mobiusSupport}</dd>
              </div>
              <div>
                <dt>Mobius risk</dt>
                <dd>{selectedModel.metadata.mobiusRisk}</dd>
              </div>
              <div>
                <dt>Tested status</dt>
                <dd>
                  {selectedModel.metadata.candidateOutcome
                    ? "Blocked / Not tested successfully"
                    : selectedModel.summary.testedStatus}
                </dd>
              </div>
            </dl>
            {selectedModel.metadata.candidateOutcome ? (
              <CandidateOutcomeDetails outcome={selectedModel.metadata.candidateOutcome} />
            ) : null}
            {blockedReason ? (
              <p className="warning" role="alert">
                {blockedReason}
              </p>
            ) : null}
          </section>
        ) : null}

        {selectedModel ? (
          <section className="panel">
            <h2>3. Task-aware optimization for CPU</h2>
            <div className="grid-two">
              <div>
                <label htmlFor="strategy">Strategy</label>
                <select
                  id="strategy"
                  value={strategy}
                  onChange={(event) => setStrategy(event.target.value)}
                  disabled={selectedModel.preflight.strategies.length === 0}
                >
                  {selectedModel.preflight.strategies.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="precision">Precision</label>
                <select
                  id="precision"
                  value={precision}
                  onChange={(event) => setPrecision(event.target.value)}
                  disabled={selectedModel.preflight.precisions.length === 0}
                >
                  {selectedModel.preflight.precisions.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {selectedModel.preflight.task === "asr" ? (
              <div>
                <label htmlFor="audio-format">Verified ASR audio format</label>
                <select
                  id="audio-format"
                  value={audioFormat}
                  onChange={(event) => setAudioFormat(event.target.value)}
                  disabled={selectedModel.preflight.verifiedAudioFormats.length === 0}
                >
                  {selectedModel.preflight.verifiedAudioFormats.map((format) => (
                    <option key={format} value={format}>
                      {format}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}
          </section>
        ) : null}

        {selectedModel ? (
          <section className="panel">
            <h2>4. Build execution</h2>
            <div className="button-row">
              <button type="button" onClick={() => void onStartBuild()} disabled={!canStartBuild}>
                {startingBuild ? "Submitting..." : "Build for CPU"}
              </button>
              <button
                type="button"
                onClick={() => void onCancelBuild()}
                disabled={!currentJob || !currentJob.cancellable || isTerminalStage(currentJob.stage)}
              >
                Cancel build
              </button>
              <button
                type="button"
                onClick={() => {
                  if (currentJob) {
                    void refreshJob(currentJob.jobId).catch((error) => setBuildError(asMessage(error)));
                  }
                }}
                disabled={!currentJob}
              >
                Refresh status
              </button>
            </div>
            <p className="muted">Retrying starts a new build job; partial artifacts from cancelled or failed jobs are not usable.</p>
            {buildError ? (
              <p className="error" role="alert">
                {buildError}
              </p>
            ) : null}
          </section>
        ) : null}

        {currentJob ? (
          <section className="panel">
            <h2>5. Backend-reported status and events</h2>
            <p>
              <strong>Job:</strong> {currentJob.jobId}
            </p>
            <p>
              <strong>Current stage:</strong> {readableStage(currentJob.stage)}
            </p>
            <p>
              <strong>Cancellable:</strong> {currentJob.cancellable ? "Yes" : "No"}
            </p>
            {currentJob.artifactId ? (
              <p>
                <strong>Artifact ID:</strong> {currentJob.artifactId}
              </p>
            ) : null}

            {events.length > 0 ? (
              <table className="event-table">
                <thead>
                  <tr>
                    <th scope="col">Seq</th>
                    <th scope="col">Stage</th>
                    <th scope="col">Message</th>
                    <th scope="col">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event) => (
                    <tr key={event.sequence}>
                      <td>{event.sequence}</td>
                      <td>{readableStage(event.stage)}</td>
                      <td>{event.message || "—"}</td>
                      <td>{event.timestamp ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">No events received yet.</p>
            )}
          </section>
        ) : null}

        {currentJob?.stage === "failed" && currentJob.failure ? (
          <section className="panel">
            <h2>Failure details</h2>
            <p>
              <strong>Failed stage:</strong> {readableStage(currentJob.failure.stage)}
            </p>
            <p>
              <strong>Classification:</strong> {currentJob.failure.classification}
            </p>
            <p>
              <strong>Message:</strong> {currentJob.failure.message}
            </p>
            <p>
              <strong>Retryable:</strong> {currentJob.failure.retryable ? "Yes" : "No"}
            </p>
            {currentJob.failure.logTail.length > 0 ? (
              <details>
                <summary>Sanitized log tail</summary>
                <pre>{currentJob.failure.logTail.join("\n")}</pre>
              </details>
            ) : null}
          </section>
        ) : null}

        {currentJob?.stage === "succeeded" ? (
          <section className="panel">
            <h2>Success and inference</h2>
            <p>
              <strong>Artifact:</strong> {currentJob.artifactSummary?.artifactId ?? currentJob.artifactId ?? "n/a"}
            </p>
            {currentJob.artifactSummary?.packagePath ? (
              <p>
                <strong>Package:</strong> {currentJob.artifactSummary.packagePath}
              </p>
            ) : null}
            {currentJob.artifactSummary?.checksum ? (
              <p>
                <strong>Checksum:</strong> {currentJob.artifactSummary.checksum}
              </p>
            ) : null}
            {currentJob.reproducibility?.recipeId ? (
              <p>
                <strong>Recipe:</strong> {currentJob.reproducibility.recipeId}
              </p>
            ) : null}

            {currentJob.task === "llm" ? (
              <div className="inference-panel">
                <label htmlFor="text-prompt">Prompt</label>
                <textarea
                  id="text-prompt"
                  rows={4}
                  value={textPrompt}
                  onChange={(event) => setTextPrompt(event.target.value)}
                />
                <button type="button" disabled={!currentJob.artifactId} onClick={() => void onRunTextInference()}>
                  Run text inference
                </button>
                {textOutput ? (
                  <div>
                    <h3>Generated response</h3>
                    <pre>{textOutput}</pre>
                  </div>
                ) : null}
              </div>
            ) : null}

            {currentJob.task === "asr" ? (
              <div className="inference-panel">
                <label htmlFor="asr-audio">Audio file</label>
                <input
                  id="asr-audio"
                  type="file"
                  accept="audio/*"
                  onChange={(event) => setAsrFile(event.target.files?.[0])}
                />
                <button type="button" disabled={!currentJob.artifactId || !asrFile} onClick={() => void onRunAsrInference()}>
                  Run ASR inference
                </button>
                {asrOutput ? (
                  <div>
                    <h3>Transcript</h3>
                    <pre>{asrOutput}</pre>
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>
        ) : null}
      </main>
    </div>
  );
}

export function App(): JSX.Element {
  return <OnboardingShell client={defaultClient} />;
}
