export interface Ticket { id: string; subject: string; body: string; queue: string; priority: string }
export interface Step { runId: string; seq: number; node: string; status: string; detail: string; latencyMs: number }
export const FAULTS = ["tool_timeout", "stale_schema", "malformed_tool_output", "empty_retrieval", "llm_refusal"] as const;
export type Fault = (typeof FAULTS)[number];
