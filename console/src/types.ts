export interface Ticket { id: string; subject: string; body: string; queue: string; priority: string }
export interface Step { runId: string; seq: number; node: string; status: string; detail: string; latencyMs: number }
export const FAULTS = ["tool_timeout", "stale_schema", "malformed_tool_output", "empty_retrieval", "llm_refusal"] as const;
export type Fault = (typeof FAULTS)[number];
/** One of the four hand-picked demo tickets (console/src/demoTickets.json, checked by scripts/verify_demo.py). */
export interface Demo { id: string; label: string; shows: string; fault: Fault | null; expect: string }
