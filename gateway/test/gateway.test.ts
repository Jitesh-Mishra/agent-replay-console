import { describe, expect, it } from "vitest";
import { createGateway, type Fetch } from "../src/gateway";

/** Fake agent service: records calls and serves canned JSON + an SSE stream. */
function fakeApi() {
  const calls: { url: string; body?: unknown }[] = [];
  const steps = ["classify", "retrieve", "draft", "gate"].map((node, i) => ({
    runId: "r1", seq: i + 1, node, status: "ok", detail: "{}", latencyMs: 1.5,
  }));
  steps.push({ runId: "r1", seq: 99, node: "end", status: "needs_approval", detail: "{}", latencyMs: 0 });
  const fetchImpl: Fetch = async (input, init) => {
    const url = String(input);
    calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    if (url.endsWith("/tickets?limit=2"))
      return Response.json([{ id: "T1", subject: "VPN down", body: "b", queue: "IT Support", priority: "high" }]);
    if (url.endsWith("/runs") && init?.method === "POST") return Response.json({ run_id: "r1" });
    if (url.endsWith("/runs/r1/decision")) return Response.json({ run_id: "r1", approved: true });
    if (url.endsWith("/runs/missing")) return new Response("nope", { status: 404 });
    if (url.endsWith("/runs/r1/events")) {
      const enc = new TextEncoder();
      const body = new ReadableStream({
        start(c) {
          for (const s of steps) c.enqueue(enc.encode(`event: step\ndata: ${JSON.stringify(s)}\n\n`));
          c.close();
        },
      });
      return new Response(body, { headers: { "content-type": "text/event-stream" } });
    }
    return new Response("not found", { status: 404 });
  };
  return { calls, fetchImpl };
}

const gql = (yoga: ReturnType<typeof createGateway>, query: string, variables = {}) =>
  yoga.fetch("http://x/graphql", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ query, variables }),
  }).then((r) => r.json());

describe("gateway contract", () => {
  it("lists tickets through the agent API", async () => {
    const { fetchImpl } = fakeApi();
    const res = await gql(createGateway("http://api", fetchImpl), "{ tickets(limit: 2) { id queue priority } }");
    expect(res.data.tickets).toEqual([{ id: "T1", queue: "IT Support", priority: "high" }]);
  });

  it("startRun sends ticket and faults in the API's snake_case contract", async () => {
    const { fetchImpl, calls } = fakeApi();
    const res = await gql(createGateway("http://api", fetchImpl),
      'mutation { startRun(ticketId: "T1", faults: [llm_refusal]) }');
    expect(res.data.startRun).toBe("r1");
    expect(calls.at(-1)!.body).toEqual({ ticket_id: "T1", faults: ["llm_refusal"] });
  });

  it("rejects faults outside the schema before calling the API", async () => {
    const { fetchImpl, calls } = fakeApi();
    const res = await gql(createGateway("http://api", fetchImpl), 'mutation { startRun(ticketId: "T1", faults: [rm_rf]) }');
    expect(res.errors).toBeDefined();
    expect(calls.length).toBe(0);
  });

  it("decide forwards the human decision", async () => {
    const { fetchImpl, calls } = fakeApi();
    const res = await gql(createGateway("http://api", fetchImpl), 'mutation { decide(runId: "r1", approve: true, text: "ok") }');
    expect(res.data.decide).toBe(true);
    expect(calls.at(-1)!.body).toEqual({ approve: true, text: "ok" });
  });

  it("returns null for an unknown run", async () => {
    const { fetchImpl } = fakeApi();
    const res = await gql(createGateway("http://api", fetchImpl), '{ run(id: "missing") { id } }');
    expect(res.data.run).toBeNull();
  });

  it("streams agent steps as a GraphQL subscription over SSE and completes after 'end'", async () => {
    const { fetchImpl } = fakeApi();
    const yoga = createGateway("http://api", fetchImpl);
    const q = encodeURIComponent('subscription { runSteps(runId: "r1") { seq node status } }');
    const res = await yoga.fetch(`http://x/graphql?query=${q}`, { headers: { accept: "text/event-stream" } });
    const text = await res.text();
    const nodes = [...text.matchAll(/"node":"(\w+)"/g)].map((m) => m[1]);
    expect(nodes).toEqual(["classify", "retrieve", "draft", "gate", "end"]);
  });
});
