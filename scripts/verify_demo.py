"""Check the four demo tickets end to end through the GraphQL gateway (agent API and gateway must be running).

    python scripts/verify_demo.py                 # 3 runs per ticket, paced for Groq's 8,000 tokens/min limit
    python scripts/verify_demo.py --runs 1 --only T00648

Each run must stream classify, retrieve, draft, gate, end in order and end with the expected decision. Tickets are in
console/src/demoTickets.json, the same file the console picker reads. Exit code 1 if anything is off.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

DEMO = Path(__file__).resolve().parents[1] / "console" / "src" / "demoTickets.json"
SUB = "subscription($id: ID!) { runSteps(runId: $id) { seq node status detail latencyMs } }"
ORDER = ["classify", "retrieve", "draft", "gate", "end"]


def gql(c, gw, q, v=None):
    r = c.post(gw, json={"query": q, "variables": v or {}}).json()
    if "errors" in r:
        raise RuntimeError(r["errors"][0]["message"])
    return r["data"]


def one_run(c, gw, ticket):
    faults = [ticket["fault"]] if ticket["fault"] else []
    rid = gql(c, gw, "mutation($t: ID!, $f: [Fault!]) { startRun(ticketId: $t, faults: $f) }", {"t": ticket["id"], "f": faults})["startRun"]
    steps = []
    with c.stream("GET", gw, params={"query": SUB, "variables": json.dumps({"id": rid})}, headers={"accept": "text/event-stream"}) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and line.strip() != "data:":
                steps.append(json.loads(line[5:])["data"]["runSteps"])
    by = {s["node"]: s for s in steps}
    end = json.loads(by["end"]["detail"]) if "end" in by else {}
    draft = json.loads(by["draft"]["detail"]).get("draft", {}) if "draft" in by else {}
    return {"nodes": [s["node"] for s in steps], "decision": end.get("decision"), "reason": end.get("reason"),
            "drafter": draft.get("drafter"), "draft_ms": by.get("draft", {}).get("latencyMs"),
            "draft_chars": len(draft.get("text", "")), "sent_to_groq": str(draft.get("drafter", "")).startswith("groq")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:4000/graphql")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--pause", type=float, default=12.0, help="seconds to wait after a run that used Groq")
    ap.add_argument("--only", default="", help="comma-separated ticket ids")
    a = ap.parse_args()
    tickets = [t for t in json.loads(DEMO.read_text()) if not a.only or t["id"] in a.only.split(",")]
    bad = 0
    with httpx.Client(timeout=90) as c:
        for t in tickets:
            for k in range(a.runs):
                r = one_run(c, a.gateway, t)
                ok = r["nodes"] == ORDER and r["decision"] == t["expect"] and r["drafter"] is not None and "fallback" not in str(r["drafter"])
                bad += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {t['id']} run {k + 1}  {r['decision']:<14} {str(r['reason']):<28} drafter={r['drafter']}  "
                      f"draft {r['draft_ms'] or 0:.0f} ms  steps={'>'.join(r['nodes'])}", flush=True)
                if r["sent_to_groq"]:
                    time.sleep(a.pause)
    print("ALL PASS" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
