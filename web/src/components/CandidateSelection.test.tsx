import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { BuildPlanPreview, CandidateSelectionSummary } from "./CandidateSelection";
import type {
  CandidateSelectionPlan,
  CandidateTimelineEntry,
  RecipeAttemptCandidateSelection,
  RecipeAttemptStatus
} from "../api/types";

function baseAttempt(overrides: Partial<RecipeAttemptStatus> = {}): RecipeAttemptStatus {
  return {
    attemptId: "attempt-1",
    recipeFingerprint: "f".repeat(64),
    state: "succeeded",
    gates: [],
    workflowOutcome: "not_applicable",
    ...overrides
  };
}

function candidateEntry(overrides: Partial<CandidateTimelineEntry> = {}): CandidateTimelineEntry {
  return {
    candidateAttemptId: "cand-0",
    attemptId: "attempt-1",
    candidateIndex: 0,
    candidateId: "default-int4",
    role: "default",
    attemptState: "succeeded",
    recipeFingerprint: "f".repeat(64),
    dispositionReasons: [],
    selectionStatus: "selected",
    invocationCounters: {},
    validatedScope: {},
    ...overrides
  };
}

function selection(overrides: Partial<RecipeAttemptCandidateSelection> = {}): RecipeAttemptCandidateSelection {
  return {
    candidates: [],
    ...overrides
  };
}

const plan: CandidateSelectionPlan = {
  policyId: "cpu-int4-recipe-selection-v1",
  policyVersion: "1.0.0",
  policyFingerprint: "b6b2e91a",
  maxCandidates: 2,
  candidates: [
    { candidateIndex: 0, candidateId: "default-int4", role: "default" },
    {
      candidateIndex: 1,
      candidateId: "int4-block-size-64",
      role: "quality_retry",
      quantizationOverride: { blockSize: 64 },
      eligibilityTrigger: "retryable_optimized_structural_regression"
    }
  ]
};

describe("BuildPlanPreview", () => {
  it("shows plain-language plan copy and keeps jargon inside Technical details", () => {
    const { container } = render(<BuildPlanPreview plan={plan} />);

    const primaryList = container.querySelector(".build-plan > ol") as HTMLElement;
    expect(screen.getByText("Build plan")).toBeInTheDocument();
    expect(within(primaryList).getByText(/First recipe/)).toBeInTheDocument();
    expect(within(primaryList).getByText(/standard CPU INT4 build/)).toBeInTheDocument();
    expect(within(primaryList).getByText(/Automatic quality retry/)).toBeInTheDocument();
    expect(within(primaryList).getByText(/conditional; same CPU INT4 build/)).toBeInTheDocument();

    // Never expose internal ids/trigger/block_size in primary (non-disclosure) copy.
    expect(within(primaryList).queryByText(/retryable_optimized_structural_regression/)).not.toBeInTheDocument();
    expect(within(primaryList).queryByText(/block_size: 64/)).not.toBeInTheDocument();
    expect(within(primaryList).queryByText(/int4-block-size-64/)).not.toBeInTheDocument();

    // The disclosure exists but its content is collapsed (not visible) until opened.
    const details = screen.getByText("Technical details").closest("details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");
    expect(screen.getByText(/retryable_optimized_structural_regression/)).not.toBeVisible();

    // Once opened, the ids/trigger/block_size are all present inside it.
    expect(details).toHaveTextContent("retryable_optimized_structural_regression");
    expect(details).toHaveTextContent("block_size: 64");
    expect(details).toHaveTextContent("int4-block-size-64");
    expect(details).toHaveTextContent("b6b2e91a");
  });
});

describe("CandidateSelectionSummary", () => {
  it("returns nothing for not_applicable (legacy) attempts", () => {
    const { container } = render(<CandidateSelectionSummary attempt={baseAttempt()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows 'First recipe verified.' when the default candidate is selected", () => {
    const attempt = baseAttempt({
      workflowOutcome: "selected",
      candidateSelection: selection({
        lineageSelectionState: "selected",
        maxCandidates: 2,
        selectedCandidate: {
          candidateAttemptId: "cand-0",
          attemptId: "attempt-1",
          candidateIndex: 0,
          candidateId: "default-int4"
        },
        candidates: [
          candidateEntry({
            invocationCounters: { mobiusBuildInvocationCount: 1, oliveOptimizeInvocationCount: 1 }
          })
        ],
        aggregateInvocationCounters: { mobiusBuildInvocationCount: 1, oliveOptimizeInvocationCount: 1 }
      })
    });

    render(<CandidateSelectionSummary attempt={attempt} />);

    const headline = screen.getByText("First recipe verified.");
    expect(headline).toBeInTheDocument();
    expect(headline).toHaveAttribute("role", "status");
    expect(screen.getByText("Selected as the verified recipe.")).toBeInTheDocument();
    expect(screen.getByText("Build work performed")).toBeInTheDocument();
  });

  it("shows fallback-selected headline and keeps the failed First recipe visible with a concise reason", () => {
    const attempt = baseAttempt({
      workflowOutcome: "selected",
      candidateSelection: selection({
        lineageSelectionState: "selected",
        maxCandidates: 2,
        selectedCandidate: {
          candidateAttemptId: "cand-1",
          attemptId: "attempt-2",
          candidateIndex: 1,
          candidateId: "int4-block-size-64"
        },
        candidates: [
          candidateEntry({
            candidateAttemptId: "cand-0",
            attemptId: "attempt-1",
            candidateIndex: 0,
            candidateId: "default-int4",
            role: "default",
            attemptState: "failed",
            selectionStatus: "not_selected"
          }),
          candidateEntry({
            candidateAttemptId: "cand-1",
            attemptId: "attempt-2",
            candidateIndex: 1,
            candidateId: "int4-block-size-64",
            role: "quality_retry",
            attemptState: "succeeded",
            selectionStatus: "selected"
          })
        ],
        aggregateInvocationCounters: { mobiusBuildInvocationCount: 1, oliveOptimizeInvocationCount: 2 }
      })
    });

    render(<CandidateSelectionSummary attempt={attempt} />);

    expect(screen.getByText("Automatic quality retry verified and selected.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "First recipe", level: 4 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Automatic quality retry", level: 4 })).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("This recipe ran but did not pass the required quality checks.")).toBeInTheDocument();
    expect(screen.getByText("Selected as the verified recipe.")).toBeInTheDocument();
  });

  it("shows 'No recipe passed validation.' and both attempts with an actionable failure reason when exhausted", () => {
    const attempt = baseAttempt({
      workflowOutcome: "exhausted",
      failure: {
        classification: "validation_failed",
        stage: "succeeded",
        message: "Both candidates failed quality validation.",
        evidenceRefs: [],
        sourceOwner: "fl-onboarding",
        nextAction: "Review the structural regression details before retrying manually."
      },
      candidateSelection: selection({
        lineageSelectionState: "exhausted",
        maxCandidates: 2,
        candidates: [
          candidateEntry({
            candidateAttemptId: "cand-0",
            candidateIndex: 0,
            role: "default",
            attemptState: "failed",
            selectionStatus: "not_selected"
          }),
          candidateEntry({
            candidateAttemptId: "cand-1",
            candidateIndex: 1,
            candidateId: "int4-block-size-64",
            role: "quality_retry",
            attemptState: "failed",
            selectionStatus: "not_selected"
          })
        ],
        aggregateInvocationCounters: { mobiusBuildInvocationCount: 1, oliveOptimizeInvocationCount: 2 }
      })
    });

    render(<CandidateSelectionSummary attempt={attempt} />);

    expect(screen.getByText("No recipe passed validation.")).toBeInTheDocument();
    expect(screen.getAllByText("Failed")).toHaveLength(2);
    expect(screen.getByText(/Both candidates failed quality validation\./)).toBeInTheDocument();
    expect(screen.getByText(/Review the structural regression details/)).toBeInTheDocument();
  });

  it("shows the pending headline without a failure banner", () => {
    const attempt = baseAttempt({
      workflowOutcome: "pending",
      candidateSelection: selection({
        lineageSelectionState: "pending",
        maxCandidates: 2,
        candidates: [candidateEntry({ attemptState: "running", selectionStatus: "not_selected" })]
      })
    });

    render(<CandidateSelectionSummary attempt={attempt} />);

    expect(screen.getByText("Build and validation in progress.")).toBeInTheDocument();
    expect(screen.getByText("Running")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("renders the reused banner with no timeline and explicit 0/0 no-build evidence", () => {
    const attempt = baseAttempt({
      workflowOutcome: "reused",
      candidateSelection: selection({
        candidates: [],
        reuse: {
          reusedWithoutBuild: true,
          sourceAttemptId: "attempt-winner",
          sourceCandidateAttemptId: "cand-winner",
          sourceParentAttemptId: "attempt-parent",
          policyId: "cpu-int4-recipe-selection-v1",
          policyVersion: "1.0.0",
          policyFingerprint: "b6b2e91a",
          qualityProfileFingerprint: "q-profile",
          runnerDispatchCount: 0,
          mobiusInvocationCount: 0,
          oliveInvocationCount: 0,
          recordedUtc: "2026-01-01T00:00:00Z"
        }
      })
    });

    render(<CandidateSelectionSummary attempt={attempt} />);

    expect(screen.getByText("Previously verified recipe reused — no build ran.")).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Recipe build attempts" })).not.toBeInTheDocument();
    expect(
      screen.getByText("No build ran for this attempt — a previously verified recipe was reused.")
    ).toBeInTheDocument();

    const mobiusValue = screen.getByText("Mobius builds").closest("div")?.querySelector("dd");
    const oliveValue = screen.getByText("Olive optimizations").closest("div")?.querySelector("dd");
    expect(mobiusValue).toHaveTextContent("0");
    expect(oliveValue).toHaveTextContent("0");
  });

  it("displays 'Not measured' for unmeasured counters and never coerces them to zero", () => {
    const attempt = baseAttempt({
      workflowOutcome: "selected",
      candidateSelection: selection({
        lineageSelectionState: "selected",
        selectedCandidate: {
          candidateAttemptId: "cand-0",
          attemptId: "attempt-1",
          candidateIndex: 0,
          candidateId: "default-int4"
        },
        candidates: [candidateEntry({ invocationCounters: {} })]
      })
    });

    render(<CandidateSelectionSummary attempt={attempt} />);

    const mobiusValues = screen.getAllByText("Mobius builds").map((dt) => dt.closest("div")?.querySelector("dd"));
    for (const value of mobiusValues) {
      expect(value).toHaveTextContent("Not measured");
    }
    expect(screen.queryByText("0")).not.toBeInTheDocument();
  });

  it("keeps Recipe integrity separate from Model capability by never claiming model quality", () => {
    const attempt = baseAttempt({
      workflowOutcome: "selected",
      candidateSelection: selection({
        lineageSelectionState: "selected",
        selectedCandidate: {
          candidateAttemptId: "cand-0",
          attemptId: "attempt-1",
          candidateIndex: 0,
          candidateId: "default-int4"
        },
        candidates: [candidateEntry()]
      })
    });

    render(<CandidateSelectionSummary attempt={attempt} />);

    expect(screen.queryByText(/model capability/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/model quality/i)).not.toBeInTheDocument();
  });

  it("keeps ids/fingerprints/trigger/block_size/selection reason inside an accessible, keyboard-operable Technical details disclosure", async () => {
    const user = userEvent.setup();
    const attempt = baseAttempt({
      workflowOutcome: "selected",
      candidateSelection: selection({
        policyId: "cpu-int4-recipe-selection-v1",
        policyVersion: "1.0.0",
        policyFingerprint: "b6b2e91a",
        lineageSelectionState: "selected",
        selectedCandidate: {
          candidateAttemptId: "cand-0",
          attemptId: "attempt-1",
          candidateIndex: 0,
          candidateId: "default-int4",
          selectedBy: "validation",
          selectionReason: "Candidate 0 ('default-int4') verified.",
          selectedUtc: "2026-01-01T00:00:00Z"
        },
        candidates: [
          candidateEntry({
            candidateId: "default-int4",
            quantizationOverride: { blockSize: 64 },
            eligibilityTrigger: "retryable_optimized_structural_regression"
          })
        ],
        aggregateInvocationCounters: { mobiusBuildInvocationCount: 1, oliveOptimizeInvocationCount: 1 }
      })
    });

    const { container } = render(<CandidateSelectionSummary attempt={attempt} />);

    const details = screen.getByText("Technical details").closest("details") as HTMLDetailsElement;
    expect(details).not.toHaveAttribute("open");

    // Disclosure content exists (native <details>/<summary>, no extra tabindex plumbing
    // needed) but is not visible to the user until opened.
    expect(within(details).getByText(/b6b2e91a/)).not.toBeVisible();

    // Outside the disclosure, primary copy never surfaces the raw ids/fingerprints.
    const primary = container.querySelector("section") as HTMLElement;
    const outsideDetails = Array.from(primary.children).filter((child) => child.tagName !== "DETAILS");
    for (const node of outsideDetails) {
      expect(node.textContent).not.toMatch(/b6b2e91a/);
      expect(node.textContent).not.toMatch(/retryable_optimized_structural_regression/);
      expect(node.textContent).not.toMatch(/block_size: 64/);
    }

    // <summary> is natively focusable/actionable per the HTML spec (no extra tabindex
    // needed); jsdom does not simulate its built-in Enter/Space activation, so this
    // exercises the equivalent activation via a user gesture instead.
    const summary = screen.getByText("Technical details");
    summary.focus();
    expect(document.activeElement).toBe(summary);
    await user.click(summary);

    expect(details).toHaveAttribute("open");
    expect(details).toHaveTextContent("b6b2e91a");
    expect(details).toHaveTextContent("retryable_optimized_structural_regression");
    expect(details).toHaveTextContent("block_size: 64");
    expect(details).toHaveTextContent("Candidate 0 ('default-int4') verified.");
  });
});
