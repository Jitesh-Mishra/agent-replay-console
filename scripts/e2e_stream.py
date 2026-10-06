"""Full-stack streaming check: Python agent service -> TypeScript GraphQL gateway -> SSE subscription.

Start both services first:
  uvicorn arc.api:app --port 8000 &
  (cd gateway && npm start) &
python scripts/e2e_stream.py --runs 50   ->  reports/e2e_stream.json
Measures, through the gateway, the time from startRun to the first streamed step and to the final step.
"""
import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np

GW = "http://127.0.0.1:4000/graphql"
SUB = "subscription($id: ID!) { runSteps(runId: $id) { seq node status } }"


def gql(c, q, v=None):
    r = c.post(GW, json={"query": q, "variables": v or {}}).json()
    if "errors" in r:
        raise RuntimeError(r["errors"])
    return r["data"]


def main(runs: int):
    first, total, nodes_ok = [], [], 0
    with httpx.Client(timeout=60) as c:
        tickets = gql(c, "{ tickets(limit: 50) { id } }")["tickets"]
        for i in range(runs):
            t0 = time.perf_counter()
            rid = gql(c, "mutation($t: ID!) { startRun(ticketId: $t) }", {"t": tickets[i % len(tickets)]["id"]})["startRun"]
            seen, t_first = [], None
            with c.stream("GET", GW, params={"query": SUB, "variables": json.dumps({"id": rid})},
                          headers={"accept": "text/event-stream"}) as r:
                for line in r.iter_lines():
                    if not line.startswith("data:") or line.strip() == "data:":
                        continue
                    msg = json.loads(line[5:])
                    if "data" not in msg:
                        continue
                    t_first = t_first or time.perf_counter()
                    seen.append(msg["data"]["runSteps"]["node"])
                    if seen[-1] == "end":
                        break
            first.append((t_first - t0) * 1000); total.append((time.perf_counter() - t0) * 1000)
            nodes_ok += seen == ["classify", "retrieve", "draft", "gate", "end"]
    out = {"runs": runs, "complete_ordered_streams": nodes_ok,
           "ms_to_first_step": {"p50": float(np.percentile(first, 50)), "p95": float(np.percentile(first, 95))},
           "ms_to_final_step": {"p50": float(np.percentile(total, 50)), "p95": float(np.percentile(total, 95))},
           "setup": "local: uvicorn (1 worker) + Node gateway + client on one 2-core VM"}
    Path("reports/e2e_stream.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=50)
    main(ap.parse_args().runs)
