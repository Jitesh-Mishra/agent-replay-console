import type { Ticket } from "../types";

export function TicketQueue({ tickets, selected, onSelect }: {
  tickets: Ticket[]; selected?: string; onSelect: (t: Ticket) => void;
}) {
  return (
    <section className="panel queue" aria-label="Ticket queue">
      <h2>Tickets <span className="muted">({tickets.length})</span></h2>
      <ul>
        {tickets.map((t) => (
          <li key={t.id}>
            <button className={`ticket ${t.id === selected ? "active" : ""}`} onClick={() => onSelect(t)}>
              <span className="subject">{t.subject || "(no subject)"}</span>
              <span className="meta">
                <span className={`pill prio-${t.priority}`}>{t.priority}</span>
                <span className="muted">{t.queue}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
