import type { Demo, Ticket } from "../types";

/** Four hand-picked tickets, each labelled by what it shows. Picking one selects the ticket and presets its fault. */
export function DemoPicker({ demos, tickets, activeId, onPick }: {
  demos: Demo[]; tickets: Ticket[]; activeId?: string; onPick: (demo: Demo, ticket: Ticket) => void;
}) {
  return (
    <section className="panel demo" aria-label="Demo tickets">
      <h2>Demo tickets <span className="muted">(pick one, then press Run agent)</span></h2>
      <div className="demo-grid">
        {demos.map((d) => {
          const t = tickets.find((x) => x.id === d.id);
          return (
            <button key={d.id} className={`demo-card ${d.id === activeId ? "active" : ""}`} disabled={!t}
              onClick={() => t && onPick(d, t)} aria-pressed={d.id === activeId}>
              <span className="demo-label">{d.label}</span>
              <span className="demo-shows">{d.shows}</span>
              <span className="muted mono">{d.id}{d.fault ? ` · fault: ${d.fault.replaceAll("_", " ")}` : ""}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
