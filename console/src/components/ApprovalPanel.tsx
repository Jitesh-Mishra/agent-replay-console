import { useEffect, useState } from "react";
import { explainDecision } from "../summary";
import { DrafterBadge } from "./DrafterBadge";

export type Outcome = { decision: string; reason: string; draft: string };

export function ApprovalPanel({ outcome, onDecide, decided, drafter }: {
  outcome?: Outcome; onDecide: (approve: boolean, text: string) => void; decided?: boolean; drafter?: string;
}) {
  // an escalated run's draft is unsafe (refusal, no grounding...): start the human from a blank reply
  const initial = outcome && outcome.decision !== "escalate" ? outcome.draft : "";
  const [text, setText] = useState(initial);
  useEffect(() => setText(initial), [initial]);
  if (!outcome) return <section className="panel approval"><h2>Approval</h2><p className="muted">Waiting for the gate.</p></section>;
  const why = explainDecision(outcome.decision, outcome.reason);
  if (outcome.decision === "auto_send")
    return (
      <section className="panel approval">
        <h2>Approval</h2>
        <p className="ok verdict">{why}</p>
        <p className="who"><DrafterBadge label={drafter} /></p>
        <pre className="draft">{outcome.draft}</pre>
      </section>
    );
  const escalated = outcome.decision === "escalate";
  return (
    <section className="panel approval">
      <h2>Approval</h2>
      <p className={`verdict ${escalated ? "bad" : "warn"}`}>{why}</p>
      <p className="who"><DrafterBadge label={drafter} /> <span className="mono muted">{outcome.decision} · {outcome.reason}</span></p>
      {escalated && outcome.draft && <p className="muted">Rejected draft: <span className="mono">{outcome.draft.slice(0, 120)}</span></p>}
      <label htmlFor="draft">Draft reply</label>
      <textarea id="draft" value={text} onChange={(e) => setText(e.target.value)} rows={10}
        placeholder={escalated ? "No safe draft. Write the reply yourself." : ""} />
      <div className="actions">
        <button className="primary" disabled={decided || !text.trim()} onClick={() => onDecide(true, text)}>Approve and send</button>
        <button disabled={decided} onClick={() => onDecide(false, text)}>Reject</button>
      </div>
      {decided && <p className="muted">Decision recorded in the run's trace.</p>}
    </section>
  );
}
