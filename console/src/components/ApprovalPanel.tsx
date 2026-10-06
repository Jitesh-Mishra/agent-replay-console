import { useEffect, useState } from "react";

export type Outcome = { decision: string; reason: string; draft: string };

export function ApprovalPanel({ outcome, onDecide, decided }: {
  outcome?: Outcome; onDecide: (approve: boolean, text: string) => void; decided?: boolean;
}) {
  // an escalated run's draft is unsafe (refusal, no grounding...): start the human from a blank reply
  const initial = outcome && outcome.decision !== "escalate" ? outcome.draft : "";
  const [text, setText] = useState(initial);
  useEffect(() => setText(initial), [initial]);
  if (!outcome) return <section className="panel approval"><h2>Approval</h2><p className="muted">Waiting for the gate.</p></section>;
  if (outcome.decision === "auto_send")
    return (
      <section className="panel approval">
        <h2>Approval</h2>
        <p className="ok">Sent automatically: every gate check passed.</p>
        <pre className="draft">{outcome.draft}</pre>
      </section>
    );
  const escalated = outcome.decision === "escalate";
  return (
    <section className="panel approval">
      <h2>Approval</h2>
      <p className={escalated ? "bad" : "warn"}>
        {escalated ? "Escalated to a human: " : "Needs approval: "}<span className="mono">{outcome.reason}</span>
      </p>
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
