import type { Step } from "./types";

export type DrafterKind = "groq" | "extractive" | "fallback" | "injected" | "none" | "unknown";

/** Who wrote the draft, from the label the agent records on the draft step. */
export function describeDrafter(label?: string): { kind: DrafterKind; short: string; long: string } {
  if (!label) return { kind: "unknown", short: "Drafter unknown", long: "Drafter unknown (older agent build)" };
  if (label.startsWith("groq:")) {
    const model = label.slice(5);
    return { kind: "groq", short: `Groq · ${model.split("/").pop()}`, long: `Written by Groq (${model})` };
  }
  if (label === "extractive") return { kind: "extractive", short: "Extractive (offline)", long: "Written by the offline extractive drafter" };
  if (label.startsWith("fallback:"))
    return { kind: "fallback", short: "Groq down: extractive fallback", long: "Groq was unavailable, so the offline extractive drafter wrote this" };
  if (label.startsWith("injected")) return { kind: "injected", short: "Injected fault", long: "The reply was replaced by an injected fault (demo)" };
  if (label.startsWith("none")) return { kind: "none", short: "No draft", long: "No draft: nothing was retrieved to ground one" };
  return { kind: "unknown", short: label, long: label };
}

const REASON: Record<string, string> = {
  low_route_confidence: "routing confidence is low",
  high_priority: "the ticket looks high priority",
  weak_grounding: "the retrieved answers are only a weak match",
  all_checks_passed: "every check passed",
  llm_refusal: "the model refused or gave no usable reply",
  empty_draft: "the draft came back empty",
  retrieve_exception: "the retrieval tool failed or timed out",
  stale_schema: "the router returned a queue that no longer exists (stale schema)",
  malformed_tool_output: "the retrieval tool returned malformed data",
  empty_retrieval: "nothing was retrieved",
  drafter_fallback: "Groq was unavailable, so the offline drafter wrote this",
  draft_exception: "the drafter failed",
  classify_exception: "the router failed",
};
const LEAD: Record<string, string> = { auto_send: "Sent automatically", needs_approval: "Held for human", escalate: "Escalated to a human" };

/** "Held for human: routing confidence is low", from the gate's decision and reason codes. */
export function explainDecision(decision: string, reason: string): string {
  const words = (reason ?? "").replace(/^fault:/, "").split(",").filter(Boolean).map((c) => REASON[c] ?? c.replaceAll("_", " "));
  return `${LEAD[decision] ?? decision}${words.length ? ": " + words.join("; ") : ""}`;
}

export function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${ms < 100 ? ms.toFixed(1) : Math.round(ms)} ms`;
}

/** The drafter label recorded on a run's draft step, if the run has reached it. */
export function drafterOf(steps: Step[]): string | undefined {
  const s = steps.find((x) => x.node === "draft");
  if (!s) return undefined;
  try { return JSON.parse(s.detail).draft?.drafter; } catch { return undefined; }
}

/** One-line, human summary of what a step did, from its JSON detail. */
export function summarize(step: Step): string {
  let d: any = {};
  try { d = JSON.parse(step.detail); } catch { return step.detail.slice(0, 160); }
  const flags: string[] = d.fault_flags ?? [];
  const flagTxt = flags.length ? `  ⚠ ${flags.join(", ")}` : "";
  switch (step.node) {
    case "classify": {
      const c = d.classify ?? {};
      if (c.error) return `error: ${c.error}${flagTxt}`;
      return `${c.queue} · confidence ${(c.confidence ?? 0).toFixed(2)} · priority ${c.priority}${flagTxt}`;
    }
    case "retrieve": {
      const r = d.retrieve ?? {};
      if (r.error) return `error: ${r.error}${flagTxt}`;
      const kb = r.kb?.[0] ? ` · KB "${r.kb[0].title}"` : "";
      return `${r.hits?.length ?? 0} past answers · top match ${(r.top_sim ?? 0).toFixed(2)}${kb}${flagTxt}`;
    }
    case "draft": {
      const t: string = d.draft?.text ?? "";
      return (t ? `"${t.slice(0, 140)}${t.length > 140 ? "…" : ""}"` : "(no draft)") + flagTxt;
    }
    case "gate":
    case "end":
      return explainDecision(d.decision, d.reason);
    default:
      return step.detail.slice(0, 160);
  }
}

/** The raw machine-readable decision shown next to the plain-words one, for people who want the trace. */
export function decisionCode(step: Step): string | undefined {
  if (step.node !== "gate" && step.node !== "end") return undefined;
  try { const d = JSON.parse(step.detail); return `${d.decision} · ${d.reason}`; } catch { return undefined; }
}
