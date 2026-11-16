import { useEffect, useRef, useState } from "react";
import { gql, subscribeRun } from "./api";
import { ApprovalPanel, type Outcome } from "./components/ApprovalPanel";
import { StepTimeline } from "./components/StepTimeline";
import { DemoPicker } from "./components/DemoPicker";
import { FaultToggle } from "./components/FaultToggle";
import { TicketQueue } from "./components/TicketQueue";
import demos from "./demoTickets.json";
import { drafterOf } from "./summary";
import type { Demo, Fault, Step, Ticket } from "./types";

export default function App() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [ticket, setTicket] = useState<Ticket>();
  const [faults, setFaults] = useState<Fault[]>([]);
  const [steps, setSteps] = useState<Step[]>([]);
  const [runId, setRunId] = useState<string>();
  const [running, setRunning] = useState(false);
  const [decided, setDecided] = useState(false);
  const [error, setError] = useState<string>();
  const unsub = useRef<() => void>();

  useEffect(() => {
    gql<{ tickets: Ticket[] }>("{ tickets(limit: 200) { id subject body queue priority } }")
      .then((d) => setTickets(d.tickets)).catch((e) => setError(String(e)));
    return () => unsub.current?.();
  }, []);

  async function run() {
    if (!ticket) return;
    unsub.current?.();
    setSteps([]); setDecided(false); setError(undefined); setRunning(true);
    try {
      const { startRun } = await gql<{ startRun: string }>(
        "mutation($t: ID!, $f: [Fault!]) { startRun(ticketId: $t, faults: $f) }", { t: ticket.id, f: faults });
      setRunId(startRun);
      setFaults([]);               // a fault applies to this run only
      unsub.current = subscribeRun(startRun, (s) => {
        setSteps((prev) => [...prev, s]);
        if (s.node === "end") setRunning(false);
      }, (e) => { setError(e); setRunning(false); });
    } catch (e) { setError(String(e)); setRunning(false); }
  }

  async function decide(approve: boolean, text: string) {
    if (!runId) return;
    await gql("mutation($r: ID!, $a: Boolean!, $t: String) { decide(runId: $r, approve: $a, text: $t) }", { r: runId, a: approve, t: text });
    setDecided(true);
  }

  function pickDemo(d: Demo, t: Ticket) {
    unsub.current?.();
    setTicket(t); setFaults(d.fault ? [d.fault] : []); setSteps([]); setRunId(undefined); setDecided(false); setRunning(false);
  }

  const end = steps.find((s) => s.node === "end");
  const outcome: Outcome | undefined = end ? JSON.parse(end.detail) : undefined;

  return (
    <div className="app">
      <header>
        <h1>Agent Replay Console</h1>
        <p className="muted">Support-ticket agent · every step traced, streamed live, and gated before anything is sent</p>
      </header>
      {error && <p className="bad banner">{error}</p>}
      <DemoPicker demos={demos as Demo[]} tickets={tickets} activeId={ticket?.id} onPick={pickDemo} />
      <main>
        <TicketQueue tickets={tickets.slice(0, 25)} selected={ticket?.id} onSelect={(t) => { setTicket(t); setSteps([]); setRunId(undefined); }} />
        <div className="center">
          {ticket && (
            <section className="panel ticket-view">
              <h2>{ticket.subject || "(no subject)"}</h2>
              <p className="body">{ticket.body}</p>
              <FaultToggle faults={faults} onChange={setFaults} />
              <button className="primary" onClick={run} disabled={running}>{running ? "Running…" : "Run agent"}</button>
            </section>
          )}
          <StepTimeline steps={steps} running={running} />
        </div>
        <ApprovalPanel outcome={outcome} onDecide={decide} decided={decided} drafter={drafterOf(steps)} />
      </main>
    </div>
  );
}
