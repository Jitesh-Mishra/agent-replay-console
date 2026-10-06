import { summarize } from "../summary";
import type { Step } from "../types";

const LABEL: Record<string, string> = {
  classify: "Classify", retrieve: "Retrieve", draft: "Draft", gate: "Approval gate", end: "Run finished",
};

export function StepTimeline({ steps, running }: { steps: Step[]; running: boolean }) {
  return (
    <section className="panel timeline" aria-label="Live agent steps">
      <h2>Live trace {running && <span className="live">● streaming</span>}</h2>
      {steps.length === 0 && !running && <p className="muted">Pick a ticket and run the agent.</p>}
      <ol>
        {steps.map((s) => (
          <li key={`${s.runId}-${s.seq}`} className={`step ${s.status === "error" ? "err" : ""}`} data-node={s.node}>
            <div className="step-head">
              <span className="node">{LABEL[s.node] ?? s.node}</span>
              <span className={`pill st-${s.status}`}>{s.status}</span>
              {s.node !== "end" && <span className="muted mono">{s.latencyMs.toFixed(1)} ms</span>}
            </div>
            <p className="summary">{summarize(s)}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
