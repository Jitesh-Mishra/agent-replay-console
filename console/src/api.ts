import type { Step } from "./types";

export async function gql<T>(query: string, variables: Record<string, unknown> = {}): Promise<T> {
  const res = await fetch("/graphql", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ query, variables }),
  });
  const body = await res.json();
  if (body.errors) throw new Error(body.errors[0].message);
  return body.data as T;
}

/** GraphQL subscription over SSE (GraphQL Yoga serves subscriptions to a plain EventSource). */
export function subscribeRun(runId: string, onStep: (s: Step) => void, onError: (e: string) => void): () => void {
  const q = "subscription($id: ID!) { runSteps(runId: $id) { runId seq node status detail latencyMs } }";
  const url = `/graphql?query=${encodeURIComponent(q)}&variables=${encodeURIComponent(JSON.stringify({ id: runId }))}`;
  const es = new EventSource(url);
  const handle = (e: MessageEvent) => {
    if (!e.data) return;
    const msg = JSON.parse(e.data);
    if (msg.errors) return onError(msg.errors[0].message);
    const step: Step = msg.data.runSteps;
    onStep(step);
    if (step.node === "end") es.close();
  };
  es.addEventListener("next", handle as EventListener);   // GraphQL-over-SSE "distinct connections" events
  es.onmessage = handle;                                    // legacy Yoga SSE (unnamed events)
  es.addEventListener("complete", () => es.close());
  es.onerror = () => { es.close(); };
  return () => es.close();
}
