/**
 * GraphQL gateway in front of the Python agent service.
 * Queries/mutations proxy REST calls; the runSteps subscription turns the Python SSE stream into a
 * GraphQL subscription (served to the console over SSE by GraphQL Yoga).
 */
import { createSchema, createYoga } from "graphql-yoga";
import { EventSourceParserStream } from "eventsource-parser/stream";

export type Fetch = typeof fetch;

export interface Step {
  runId: string;
  seq: number;
  node: string;
  status: string;
  detail: string;
  latencyMs: number;
}

export const typeDefs = /* GraphQL */ `
  enum Fault { tool_timeout stale_schema malformed_tool_output empty_retrieval llm_refusal }

  type Ticket { id: ID!  subject: String!  body: String!  queue: String!  priority: String! }

  type Step { runId: ID!  seq: Int!  node: String!  status: String!  detail: String!  latencyMs: Float! }

  type Approval { approved: Boolean!  text: String }

  type Run {
    id: ID!
    ticketId: String!
    decision: String
    reason: String
    status: String!
    steps: [Step!]!
    approval: Approval
  }

  type Query {
    tickets(limit: Int = 20): [Ticket!]!
    run(id: ID!): Run
  }

  type Mutation {
    startRun(ticketId: ID!, faults: [Fault!] = []): ID!
    decide(runId: ID!, approve: Boolean!, text: String): Boolean!
  }

  type Subscription {
    "Live agent steps for one run; completes after the final 'end' step."
    runSteps(runId: ID!): Step!
  }
`;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`agent service ${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

export async function* streamSteps(api: string, runId: string, fetchImpl: Fetch): AsyncGenerator<Step> {
  const res = await fetchImpl(`${api}/runs/${runId}/events`, { headers: { accept: "text/event-stream" } });
  if (!res.ok || !res.body) throw new Error(`agent service ${res.status}`);
  const events = res.body.pipeThrough(new TextDecoderStream()).pipeThrough(new EventSourceParserStream());
  for await (const ev of events) {
    if (!ev.data) continue;
    const step = JSON.parse(ev.data) as Step;
    yield step;
    if (step.node === "end") return;
  }
}

export function createGateway(api = process.env.ARC_API ?? "http://127.0.0.1:8000", fetchImpl: Fetch = fetch) {
  const resolvers = {
    Query: {
      tickets: async (_: unknown, { limit }: { limit: number }) => json(await fetchImpl(`${api}/tickets?limit=${limit}`)),
      run: async (_: unknown, { id }: { id: string }) => {
        const res = await fetchImpl(`${api}/runs/${id}`);
        if (res.status === 404) return null;
        const r = await json<any>(res);
        return {
          id: r.run_id, ticketId: r.ticket_id, decision: r.decision, reason: r.reason, status: r.status,
          approval: r.approval,
          steps: r.steps.map((s: any) => ({
            runId: r.run_id, seq: s.seq, node: s.node, status: s.error ? "error" : "ok",
            detail: JSON.stringify(s.output), latencyMs: s.latency_ms,
          })),
        };
      },
    },
    Mutation: {
      startRun: async (_: unknown, a: { ticketId: string; faults: string[] }) => {
        const r = await json<{ run_id: string }>(
          await fetchImpl(`${api}/runs`, {
            method: "POST", headers: { "content-type": "application/json" },
            body: JSON.stringify({ ticket_id: a.ticketId, faults: a.faults ?? [] }),
          }),
        );
        return r.run_id;
      },
      decide: async (_: unknown, a: { runId: string; approve: boolean; text?: string }) => {
        await json(
          await fetchImpl(`${api}/runs/${a.runId}/decision`, {
            method: "POST", headers: { "content-type": "application/json" },
            body: JSON.stringify({ approve: a.approve, text: a.text ?? null }),
          }),
        );
        return true;
      },
    },
    Subscription: {
      runSteps: {
        subscribe: (_: unknown, { runId }: { runId: string }) => streamSteps(api, runId, fetchImpl),
        resolve: (step: Step) => step,
      },
    },
  };
  return createYoga({ schema: createSchema({ typeDefs, resolvers }), graphqlEndpoint: "/graphql", cors: true, logging: false });
}
