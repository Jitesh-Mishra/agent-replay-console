import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ApprovalPanel } from "../src/components/ApprovalPanel";
import { StepTimeline } from "../src/components/StepTimeline";
import { summarize } from "../src/summary";
import type { Step } from "../src/types";

const step = (node: string, detail: object, status = "ok"): Step =>
  ({ runId: "r", seq: node.length, node, status, detail: JSON.stringify(detail), latencyMs: 3.2 });

describe("StepTimeline", () => {
  it("renders each streamed step with a readable summary", () => {
    render(<StepTimeline running steps={[
      step("classify", { classify: { queue: "IT Support", confidence: 0.71, priority: "high" } }),
      step("retrieve", { retrieve: { hits: [{}, {}, {}], top_sim: 0.82, kb: [{ title: "Account access and VPN" }] } }),
    ]} />);
    expect(screen.getByText(/IT Support · confidence 0.71 · priority high/)).toBeInTheDocument();
    expect(screen.getByText(/3 past answers · top match 0.82 · KB "Account access and VPN"/)).toBeInTheDocument();
    expect(screen.getByText(/streaming/)).toBeInTheDocument();
  });

  it("surfaces fault flags on the step", () => {
    expect(summarize(step("draft", { draft: { text: "I'm sorry" }, fault_flags: ["llm_refusal"] })))
      .toContain("⚠ llm_refusal");
  });
});

describe("ApprovalPanel", () => {
  it("never offers approve on an auto-sent run", () => {
    render(<ApprovalPanel outcome={{ decision: "auto_send", reason: "all_checks_passed", draft: "Hi" }} onDecide={() => {}} />);
    expect(screen.queryByText("Approve and send")).toBeNull();
  });

  it("requires a non-empty reply before approving an escalated run", () => {
    const onDecide = vi.fn();
    render(<ApprovalPanel outcome={{ decision: "escalate", reason: "fault:llm_refusal", draft: "I'm sorry, but I can't help." }} onDecide={onDecide} />);
    expect(screen.getByLabelText("Draft reply")).toHaveValue("");
    const approve = screen.getByText("Approve and send");
    expect(approve).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Draft reply"), { target: { value: "Here is the fix." } });
    fireEvent.click(screen.getByText("Approve and send"));
    expect(onDecide).toHaveBeenCalledWith(true, "Here is the fix.");
  });
});
