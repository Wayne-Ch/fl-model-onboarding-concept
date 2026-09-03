import type {
  CandidateInvocationCounters,
  CandidateRole,
  CandidateSelectionPlan,
  CandidateTimelineEntry,
  RecipeAttemptCandidateSelection,
  RecipeAttemptStatus
} from "../api/types";

/**
 * Plain-language label for a candidate role. Never surface the raw
 * `role`/`candidate_id` codes in primary copy -- those belong inside the
 * "Technical details" disclosure only.
 */
export function candidateRoleLabel(role: CandidateRole): string {
  return role === "quality_retry" ? "Automatic quality retry" : "First recipe";
}

function formatMeasured(value: number | undefined, suffix = ""): string {
  return value === undefined ? "Not measured" : `${value}${suffix}`;
}

function formatMeasuredCost(value: number | undefined): string {
  return value === undefined ? "Not measured" : `$${value.toFixed(2)}`;
}

function candidateStatusDisplay(state: CandidateTimelineEntry["attemptState"]): { icon: string; label: string } {
  switch (state) {
    case "generated":
      return { icon: "○", label: "Not run" };
    case "running":
      return { icon: "◐", label: "Running" };
    case "succeeded":
      return { icon: "✓", label: "Verified" };
    case "failed":
      return { icon: "✕", label: "Failed" };
    case "cancelled":
      return { icon: "⊘", label: "Cancelled" };
    default:
      return { icon: "?", label: state };
  }
}

function candidateOutcomeNote(candidate: CandidateTimelineEntry): string {
  if (candidate.selectionStatus === "selected") {
    return "Selected as the verified recipe.";
  }
  switch (candidate.attemptState) {
    case "failed":
      return "This recipe ran but did not pass the required quality checks.";
    case "running":
      return "Still running.";
    case "generated":
      return "Not started yet.";
    case "cancelled":
      return "Cancelled before completion.";
    default:
      return "A different candidate was selected.";
  }
}

/**
 * Defensive display order: the API already returns `candidates` ordered by
 * `candidate_index` ascending, but this sorts a copy (stable) so a
 * misordered payload still displays sensibly. It never dedupes -- duplicate
 * candidate_index values (an integrity error) stay visible rather than
 * being silently hidden.
 */
function sortCandidatesForDisplay(candidates: CandidateTimelineEntry[]): CandidateTimelineEntry[] {
  return candidates
    .map((candidate, index) => ({ candidate, index }))
    .sort((left, right) => left.candidate.candidateIndex - right.candidate.candidateIndex || left.index - right.index)
    .map((entry) => entry.candidate);
}

function resolveSelectedCandidateRole(selection: RecipeAttemptCandidateSelection): CandidateRole | undefined {
  const selected = selection.selectedCandidate;
  if (!selected) {
    return undefined;
  }
  const match = selection.candidates.find((candidate) => candidate.candidateAttemptId === selected.candidateAttemptId);
  if (match) {
    return match.role;
  }
  return selected.candidateIndex === 0 ? "default" : "quality_retry";
}

/**
 * Plain-language headline for `workflow_outcome` + `candidate_selection`.
 * Returns `undefined` for `not_applicable` (legacy/non-eligible attempts) --
 * callers must leave existing UI unchanged in that case rather than
 * rendering an empty/placeholder panel.
 */
export function workflowOutcomeHeadline(attempt: RecipeAttemptStatus): string | undefined {
  const { workflowOutcome, candidateSelection } = attempt;
  switch (workflowOutcome) {
    case "not_applicable":
      return undefined;
    case "pending":
      return "Build and validation in progress.";
    case "exhausted":
      return "No recipe passed validation.";
    case "reused":
      return "Previously verified recipe reused — no build ran.";
    case "selected": {
      const role = candidateSelection ? resolveSelectedCandidateRole(candidateSelection) : undefined;
      return role === "quality_retry" ? "Automatic quality retry verified and selected." : "First recipe verified.";
    }
    default:
      return undefined;
  }
}

function InvocationCountersFields({ counters }: { counters: CandidateInvocationCounters | undefined }): JSX.Element {
  return (
    <dl>
      <div>
        <dt>Mobius builds</dt>
        <dd>{formatMeasured(counters?.mobiusBuildInvocationCount)}</dd>
      </div>
      <div>
        <dt>Olive optimizations</dt>
        <dd>{formatMeasured(counters?.oliveOptimizeInvocationCount)}</dd>
      </div>
      <div>
        <dt>Elapsed time</dt>
        <dd>{formatMeasured(counters?.wallClockSeconds, "s")}</dd>
      </div>
      <div>
        <dt>Estimated cost</dt>
        <dd>{formatMeasuredCost(counters?.estimatedCostUsd)}</dd>
      </div>
    </dl>
  );
}

/** Aggregate "Build work performed" section: real counts, never a guess. */
function BuildWorkPerformed({
  counters,
  noBuildRan
}: {
  counters: CandidateInvocationCounters | undefined;
  noBuildRan: boolean;
}): JSX.Element {
  return (
    <div className="build-evidence">
      <h4>Build work performed</h4>
      {noBuildRan ? (
        <p className="muted">No build ran for this attempt — a previously verified recipe was reused.</p>
      ) : null}
      <InvocationCountersFields counters={counters} />
    </div>
  );
}

/** Semantic ordered list of at most two candidate cards. */
export function CandidateTimeline({ candidates }: { candidates: CandidateTimelineEntry[] }): JSX.Element {
  const ordered = sortCandidatesForDisplay(candidates);
  return (
    <ol className="candidate-timeline" aria-label="Recipe build attempts">
      {ordered.map((candidate) => {
        const status = candidateStatusDisplay(candidate.attemptState);
        return (
          <li key={candidate.candidateAttemptId} className="candidate-card">
            <h4>{candidateRoleLabel(candidate.role)}</h4>
            <p className="candidate-status">
              <span className="status-glyph" aria-hidden="true">
                {status.icon}
              </span>
              {status.label}
            </p>
            <p>{candidateOutcomeNote(candidate)}</p>
            <InvocationCountersFields counters={candidate.invocationCounters} />
          </li>
        );
      })}
    </ol>
  );
}

/**
 * Accessible disclosure holding every internal/machine-readable identifier:
 * candidate ids, recipe fingerprints, policy id/version/fingerprint,
 * eligibility triggers, quantization block_size, artifact/package refs,
 * validated scope, and the selection reason/time. Nothing here is ever
 * duplicated into primary copy. Refs are trusted as already API-sanitized
 * (`job://<job_id>/...`); no raw private paths are read or displayed here.
 */
export function TechnicalDetails({ selection }: { selection: RecipeAttemptCandidateSelection }): JSX.Element {
  return (
    <details className="technical-details">
      <summary>Technical details</summary>
      <dl>
        <div>
          <dt>Policy</dt>
          <dd>
            {selection.policyId ?? "—"}
            {selection.policyVersion ? `@${selection.policyVersion}` : ""}
          </dd>
        </div>
        <div>
          <dt>Policy fingerprint</dt>
          <dd>{selection.policyFingerprint ?? "—"}</dd>
        </div>
        <div>
          <dt>Lineage selection state</dt>
          <dd>{selection.lineageSelectionState ?? "—"}</dd>
        </div>
        {selection.selectedCandidate ? (
          <>
            <div>
              <dt>Selected candidate</dt>
              <dd>
                {selection.selectedCandidate.candidateId} (index {selection.selectedCandidate.candidateIndex})
              </dd>
            </div>
            <div>
              <dt>Selected by</dt>
              <dd>{selection.selectedCandidate.selectedBy ?? "—"}</dd>
            </div>
            <div>
              <dt>Selection reason</dt>
              <dd>{selection.selectedCandidate.selectionReason ?? "—"}</dd>
            </div>
            <div>
              <dt>Selected at</dt>
              <dd>{selection.selectedCandidate.selectedUtc ?? "—"}</dd>
            </div>
          </>
        ) : null}
      </dl>
      {selection.candidates.length > 0 ? (
        <ul>
          {selection.candidates.map((candidate) => (
            <li key={candidate.candidateAttemptId}>
              <strong>{candidateRoleLabel(candidate.role)}</strong> (candidate_id: {candidate.candidateId}, index{" "}
              {candidate.candidateIndex})
              <br />
              recipe_fingerprint: {candidate.recipeFingerprint}
              {candidate.eligibilityTrigger ? <>, eligibility_trigger: {candidate.eligibilityTrigger}</> : null}
              {candidate.quantizationOverride ? <>, block_size: {candidate.quantizationOverride.blockSize}</> : null}
              {candidate.disposition ? <>, disposition: {candidate.disposition}</> : null}
              {candidate.dispositionReasons.length > 0 ? (
                <>, disposition_reasons: {candidate.dispositionReasons.join("; ")}</>
              ) : null}
              <br />
              artifact_ref: {candidate.artifactRef ?? "—"}, package_ref: {candidate.packageRef ?? "—"}
              <br />
              validated_scope: target_device={candidate.validatedScope.targetDevice ?? "—"}, target_ep=
              {candidate.validatedScope.targetEp ?? "—"}, toolchain_fingerprint=
              {candidate.validatedScope.toolchainFingerprint ?? "—"}, environment_scope=
              {candidate.validatedScope.environmentScope ?? "—"}
            </li>
          ))}
        </ul>
      ) : null}
      {selection.reuse ? (
        <dl>
          <div>
            <dt>Reused from attempt</dt>
            <dd>{selection.reuse.sourceAttemptId}</dd>
          </div>
          <div>
            <dt>Reuse policy</dt>
            <dd>
              {selection.reuse.policyId}@{selection.reuse.policyVersion}
            </dd>
          </div>
          <div>
            <dt>Reuse policy fingerprint</dt>
            <dd>{selection.reuse.policyFingerprint}</dd>
          </div>
          <div>
            <dt>Quality profile fingerprint</dt>
            <dd>{selection.reuse.qualityProfileFingerprint}</dd>
          </div>
          <div>
            <dt>Recorded</dt>
            <dd>{selection.reuse.recordedUtc}</dd>
          </div>
        </dl>
      ) : null}
    </details>
  );
}

/**
 * Preview "Build plan" shown before confirmation, when
 * `generatedRecipe.candidatePlan` is non-null. Only ever describes the
 * static, approved policy shape (never implies a fallback always runs).
 */
export function BuildPlanPreview({ plan }: { plan: CandidateSelectionPlan }): JSX.Element {
  const hasDefault = plan.candidates.some((candidate) => candidate.role === "default");
  const hasFallback = plan.candidates.some((candidate) => candidate.role === "quality_retry");
  return (
    <div className="build-plan">
      <h3>Build plan</h3>
      <ol>
        {hasDefault ? (
          <li>
            <strong>First recipe</strong> — standard CPU INT4 build.
          </li>
        ) : null}
        {hasFallback ? (
          <li>
            <strong>Automatic quality retry</strong> — conditional; same CPU INT4 build, attempted at most once only
            if the first recipe runs but damages a strict output format.
          </li>
        ) : null}
      </ol>
      <details className="technical-details">
        <summary>Technical details</summary>
        <dl>
          <div>
            <dt>Policy</dt>
            <dd>
              {plan.policyId}@{plan.policyVersion}
            </dd>
          </div>
          <div>
            <dt>Policy fingerprint</dt>
            <dd>{plan.policyFingerprint}</dd>
          </div>
          <div>
            <dt>Max candidates</dt>
            <dd>{plan.maxCandidates}</dd>
          </div>
        </dl>
        <ul>
          {plan.candidates.map((candidate) => (
            <li key={candidate.candidateId}>
              <strong>{candidateRoleLabel(candidate.role)}</strong> — candidate_id: {candidate.candidateId}
              {candidate.eligibilityTrigger ? `, eligibility_trigger: ${candidate.eligibilityTrigger}` : ""}
              {candidate.quantizationOverride ? `, block_size: ${candidate.quantizationOverride.blockSize}` : ""}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

/**
 * Attempt/result panel driven by `workflow_outcome` + `candidate_selection`.
 * Returns `null` for `not_applicable` so legacy/non-eligible attempts leave
 * the surrounding (pre-3C1) UI completely unchanged.
 */
export function CandidateSelectionSummary({ attempt }: { attempt: RecipeAttemptStatus }): JSX.Element | null {
  const headline = workflowOutcomeHeadline(attempt);
  if (!headline) {
    return null;
  }
  const selection = attempt.candidateSelection;
  const isReused = attempt.workflowOutcome === "reused";
  const aggregateCounters: CandidateInvocationCounters | undefined = isReused
    ? selection?.reuse
      ? {
          mobiusBuildInvocationCount: selection.reuse.mobiusInvocationCount,
          oliveOptimizeInvocationCount: selection.reuse.oliveInvocationCount
        }
      : undefined
    : selection?.aggregateInvocationCounters;

  return (
    <section className="panel" aria-label="Recipe build result">
      <h2>Recipe build result</h2>
      <p className={isReused ? "notice-banner" : undefined} role="status" aria-live="polite">
        {headline}
      </p>
      {selection && selection.candidates.length > 0 ? <CandidateTimeline candidates={selection.candidates} /> : null}
      <BuildWorkPerformed counters={aggregateCounters} noBuildRan={isReused} />
      {attempt.workflowOutcome === "exhausted" && attempt.failure ? (
        <p className="warning" role="alert">
          {attempt.failure.message} {attempt.failure.nextAction}
        </p>
      ) : null}
      {selection ? <TechnicalDetails selection={selection} /> : null}
    </section>
  );
}
