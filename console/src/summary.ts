import type { Step } from "./types";

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
      return `${d.decision} · ${d.reason}`;
    case "end":
      return `${d.decision} · ${d.reason}`;
    default:
      return step.detail.slice(0, 160);
  }
}
