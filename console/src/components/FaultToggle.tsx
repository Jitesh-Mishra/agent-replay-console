import { FAULTS, type Fault } from "../types";

const NAME: Record<Fault, string> = {
  tool_timeout: "Tool timeout", stale_schema: "Stale schema", malformed_tool_output: "Malformed tool output",
  empty_retrieval: "Empty retrieval", llm_refusal: "LLM refusal",
};

/** Switches that inject a fault into the NEXT run only. All off means a normal run. */
export function FaultToggle({ faults, onChange }: { faults: Fault[]; onChange: (f: Fault[]) => void }) {
  return (
    <div className="faults" role="group" aria-label="Inject a fault on the next run">
      <div className="fault-head">
        <strong>Inject a fault on the next run only</strong>
        <span className={faults.length ? "warn" : "muted"} data-testid="fault-state">
          {faults.length ? `ON: ${faults.map((f) => NAME[f]).join(", ")}` : "OFF: normal run"}
        </span>
        {faults.length > 0 && <button className="link" onClick={() => onChange([])}>Turn off</button>}
      </div>
      <div className="chips">
        {FAULTS.map((f) => (
          <button key={f} className={`chip ${faults.includes(f) ? "on" : ""}`} aria-pressed={faults.includes(f)}
            onClick={() => onChange(faults.includes(f) ? faults.filter((x) => x !== f) : [...faults, f])}>
            {NAME[f]}
          </button>
        ))}
      </div>
    </div>
  );
}
