import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import demos from "../src/demoTickets.json";
import { DemoPicker } from "../src/components/DemoPicker";
import { DrafterBadge } from "../src/components/DrafterBadge";
import { FaultToggle } from "../src/components/FaultToggle";
import { ApprovalPanel } from "../src/components/ApprovalPanel";
import { StepTimeline } from "../src/components/StepTimeline";
import { describeDrafter, drafterOf, explainDecision, formatMs } from "../src/summary";
import type { Demo, Step, Ticket } from "../src/types";

const step = (node: string, detail: object, status = "ok", ms = 3.2, seq = 1): Step =>
  ({ runId: "r", seq, node, status, detail: JSON.stringify(detail), latencyMs: ms });
const ticket = (id: string): Ticket => ({ id, subject: `Subject ${id}`, body: "body", queue: "IT Support", priority: "low" });

describe("which drafter wrote the reply", () => {
  it("names Groq with the model, the offline drafter, the fallback and an injected fault", () => {
    expect(describeDrafter("groq:openai/gpt-oss-20b")).toMatchObject({ kind: "groq", short: "Groq · gpt-oss-20b" });
    expect(describeDrafter("extractive").kind).toBe("extractive");
    expect(describeDrafter("fallback:extractive")).toMatchObject({ kind: "fallback", short: "Groq down: extractive fallback" });
    expect(describeDrafter("injected fault: llm_refusal").kind).toBe("injected");
    expect(describeDrafter("none (nothing retrieved)").kind).toBe("none");
    expect(describeDrafter(undefined).kind).toBe("unknown");          // older agent build without the label
  });

  it("is read from the draft step and shown in the trace", () => {
    const steps = [step("draft", { draft: { text: "Hi", drafter: "groq:openai/gpt-oss-20b" } })];
    expect(drafterOf(steps)).toBe("groq:openai/gpt-oss-20b");
    render(<StepTimeline running={false} steps={steps} />);
    expect(screen.getByText("Groq · gpt-oss-20b")).toBeInTheDocument();
  });

  it("says plainly when Groq was down", () => {
    render(<DrafterBadge label="fallback:extractive" />);
    expect(screen.getByText("Groq down: extractive fallback")).toHaveAttribute("title", expect.stringContaining("unavailable"));
  });
});

describe("gate decision in plain words", () => {
  it("explains each outcome", () => {
    expect(explainDecision("needs_approval", "low_route_confidence")).toBe("Held for human: routing confidence is low");
    expect(explainDecision("needs_approval", "low_route_confidence,high_priority"))
      .toBe("Held for human: routing confidence is low; the ticket looks high priority");
    expect(explainDecision("auto_send", "all_checks_passed")).toBe("Sent automatically: every check passed");
    expect(explainDecision("escalate", "fault:llm_refusal")).toBe("Escalated to a human: the model refused or gave no usable reply");
    expect(explainDecision("escalate", "fault:drafter_fallback")).toContain("Groq was unavailable");
    expect(explainDecision("escalate", "fault:something_new")).toBe("Escalated to a human: something new");   // unknown codes still readable
  });

  it("shows the plain words and the raw code on the gate step, and latency on every step", () => {
    render(<StepTimeline running={false} steps={[
      step("classify", { classify: { queue: "IT Support", confidence: 0.4, priority: "low" } }, "ok", 6.34, 1),
      step("draft", { draft: { text: "Hello", drafter: "extractive" } }, "ok", 1240, 2),
      step("gate", { decision: "needs_approval", reason: "low_route_confidence" }, "ok", 0.04, 3),
    ]} />);
    expect(screen.getByText("Held for human: routing confidence is low")).toBeInTheDocument();
    expect(screen.getByText("needs_approval · low_route_confidence")).toBeInTheDocument();
    expect(screen.getAllByLabelText("step latency").map((e) => e.textContent)).toEqual(["6.3 ms", "1.2 s", "0.0 ms"]);
  });

  it("formats latency", () => {
    expect([formatMs(0.04), formatMs(6.34), formatMs(412.6), formatMs(1999)]).toEqual(["0.0 ms", "6.3 ms", "413 ms", "2.0 s"]);
  });
});

describe("ApprovalPanel labels", () => {
  it("shows the plain-words verdict and who drafted it", () => {
    render(<ApprovalPanel drafter="groq:openai/gpt-oss-20b" onDecide={() => {}}
      outcome={{ decision: "needs_approval", reason: "low_route_confidence", draft: "Hello" }} />);
    expect(screen.getByText("Held for human: routing confidence is low")).toBeInTheDocument();
    expect(screen.getByText("Groq · gpt-oss-20b")).toBeInTheDocument();
  });

  it("still never offers approve on an auto-sent run, and still blanks an escalated draft", () => {
    const { unmount } = render(<ApprovalPanel drafter="extractive" onDecide={() => {}}
      outcome={{ decision: "auto_send", reason: "all_checks_passed", draft: "Hi" }} />);
    expect(screen.queryByText("Approve and send")).toBeNull();
    unmount();
    render(<ApprovalPanel drafter="fallback:extractive" onDecide={() => {}}
      outcome={{ decision: "escalate", reason: "fault:drafter_fallback", draft: "offline draft" }} />);
    expect(screen.getByLabelText("Draft reply")).toHaveValue("");
  });
});

describe("DemoPicker", () => {
  const list = demos as Demo[];

  it("lists the four demo tickets by what they show, and picks one", () => {
    expect(list).toHaveLength(4);
    const onPick = vi.fn();
    render(<DemoPicker demos={list} tickets={list.map((d) => ticket(d.id))} onPick={onPick} />);
    for (const d of list) expect(screen.getByText(d.label)).toBeInTheDocument();
    fireEvent.click(screen.getByText(list[2].label));
    expect(onPick).toHaveBeenCalledWith(list[2], expect.objectContaining({ id: list[2].id }));
  });

  it("disables a demo whose ticket is not loaded yet", () => {
    render(<DemoPicker demos={list} tickets={[]} onPick={() => {}} />);
    expect(screen.getByText(list[0].label).closest("button")).toBeDisabled();
  });
});

describe("FaultToggle", () => {
  it("shows OFF/ON state, toggles a fault and turns everything off", () => {
    const onChange = vi.fn();
    const { rerender } = render(<FaultToggle faults={[]} onChange={onChange} />);
    expect(screen.getByTestId("fault-state")).toHaveTextContent("OFF: normal run");
    fireEvent.click(screen.getByText("LLM refusal"));
    expect(onChange).toHaveBeenCalledWith(["llm_refusal"]);
    rerender(<FaultToggle faults={["llm_refusal"]} onChange={onChange} />);
    expect(screen.getByTestId("fault-state")).toHaveTextContent("ON: LLM refusal");
    expect(screen.getByText("LLM refusal")).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByText("Turn off"));
    expect(onChange).toHaveBeenLastCalledWith([]);
  });
});

// ---- the whole click path with the network mocked: pick demo 3 -> fault is preset -> run -> labelled result
const api = vi.hoisted(() => ({ gql: vi.fn(), subscribeRun: vi.fn() }));
vi.mock("../src/api", () => api);

describe("App click path", () => {
  beforeEach(() => { api.gql.mockReset(); api.subscribeRun.mockReset(); });

  it("pick a demo, run it with its preset fault, see the labelled outcome, fault switches off", async () => {
    const list = demos as Demo[];
    api.gql.mockImplementation(async (q: string) => q.includes("tickets") ? { tickets: list.map((d) => ticket(d.id)) } : { startRun: "run1" });
    api.subscribeRun.mockImplementation((_id: string, onStep: (s: Step) => void) => {
      onStep(step("classify", { classify: { queue: "Billing and Payments", confidence: 0.7, priority: "low" } }, "ok", 5, 1));
      onStep(step("draft", { draft: { text: "I'm sorry, but I can't help with that request.", drafter: "injected fault: llm_refusal" }, fault_flags: ["llm_refusal"] }, "ok", 420, 2));
      onStep(step("gate", { decision: "escalate", reason: "fault:llm_refusal" }, "ok", 0.1, 3));
      onStep(step("end", { decision: "escalate", reason: "fault:llm_refusal", draft: "I'm sorry, but I can't help with that request." }, "escalate", 0, 99));
      return () => {};
    });
    const { default: App } = await import("../src/App");
    render(<App />);
    const demo3 = await screen.findByText(list[2].label);
    await waitFor(() => expect(demo3.closest("button")).not.toBeDisabled());
    fireEvent.click(demo3);
    expect(screen.getByTestId("fault-state")).toHaveTextContent("ON: LLM refusal");
    fireEvent.click(screen.getByText("Run agent"));
    await waitFor(() => expect(api.gql).toHaveBeenCalledWith(expect.stringContaining("startRun"), { t: list[2].id, f: ["llm_refusal"] }));
    expect((await screen.findAllByText("Escalated to a human: the model refused or gave no usable reply")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Injected fault").length).toBeGreaterThan(0);
    expect(screen.getByTestId("fault-state")).toHaveTextContent("OFF: normal run");
  });
});
