"""FastAPI service: start agent runs, stream their steps (SSE), record human approval decisions.

uvicorn arc.api:app --port 8000          (ARC_SMALL=1 trains on a 3k-ticket subset for fast dev/tests)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent import FAULTS, Agent, Components, ExtractiveDrafter, TraceStore, Version
from .components import build_index, fit_classifier, load_kb, load_tickets, split

STATE: dict = {}


def build_state(small: bool = False):
    df = load_tickets()
    tr, te = split(df)
    if small:
        tr = tr.sample(3000, random_state=0)
    router, _ = fit_classifier(tr["text"], tr["queue"], C=1.0)
    prio, _ = fit_classifier(tr["text"], tr["priority"], C=1.0)
    comp = Components(router, prio, build_index(tr, load_kb(), name="api"), ExtractiveDrafter())
    store = TraceStore()
    store.db.execute("CREATE TABLE IF NOT EXISTS approvals(run_id TEXT PRIMARY KEY, approved INT, text TEXT, ts REAL)")
    return {"tickets": te.head(200).set_index("ticket_id"), "comp": comp, "store": store, "runs": {}, "lock": threading.Lock()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not STATE:
        STATE.update(build_state(small=os.getenv("ARC_SMALL") == "1"))
    STATE["loop"] = asyncio.get_running_loop()
    yield


app = FastAPI(title="Agent Replay Console API", lifespan=lifespan)


class StartRun(BaseModel):
    ticket_id: str
    faults: list[str] = []


class Decide(BaseModel):
    approve: bool
    text: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "tickets": len(STATE["tickets"])}


@app.get("/tickets")
def tickets(limit: int = 20):
    t = STATE["tickets"].head(limit).reset_index()
    return [{"id": r.ticket_id, "subject": r.subject or r.body[:60], "body": r.body, "queue": r.queue,
             "priority": r.priority} for r in t.itertuples()]


@app.post("/runs")
def start_run(req: StartRun):
    if req.ticket_id not in STATE["tickets"].index:
        raise HTTPException(404, "unknown ticket")
    bad = set(req.faults) - set(FAULTS)
    if bad:
        raise HTTPException(422, f"unknown faults {sorted(bad)}")
    run_id = uuid.uuid4().hex[:12]
    rec = {"events": [], "done": False, "waiters": []}
    STATE["runs"][run_id] = rec
    loop = STATE["loop"]

    def emit(ev):
        rec["events"].append(ev)
        for q in list(rec["waiters"]):
            loop.call_soon_threadsafe(q.put_nowait, ev)

    def work():
        t = STATE["tickets"].loc[req.ticket_id].to_dict() | {"ticket_id": req.ticket_id}
        with STATE["lock"]:   # one agent at a time; the trace store is a single sqlite connection
            out = Agent(STATE["comp"], Version(), STATE["store"], on_event=emit).run(
                t, faults=req.faults, seed=int(time.time()), run_id=run_id)
        emit({"runId": run_id, "seq": 99, "node": "end", "status": out["decision"],
              "detail": json.dumps({"decision": out["decision"], "reason": out["reason"],
                                    "draft": (out.get("draft") or {}).get("text", "")}), "latencyMs": 0})
        rec["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id}


@app.get("/runs/{run_id}/events")
async def events(run_id: str):
    rec = STATE["runs"].get(run_id)
    if rec is None:
        raise HTTPException(404, "unknown run")
    q: asyncio.Queue = asyncio.Queue()
    rec["waiters"].append(q)

    async def gen():
        try:
            sent = 0
            for ev in list(rec["events"]):          # replay what already happened
                sent += 1
                yield {"event": "step", "data": json.dumps(ev)}
                if ev["node"] == "end":
                    return
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=30)
                if ev["seq"] <= sent and ev["node"] != "end":
                    continue
                yield {"event": "step", "data": json.dumps(ev)}
                if ev["node"] == "end":
                    return
        finally:
            rec["waiters"].remove(q)

    return EventSourceResponse(gen())


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = STATE["store"].run(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    ap = STATE["store"].db.execute("SELECT approved, text FROM approvals WHERE run_id=?", (run_id,)).fetchone()
    return run | {"steps": STATE["store"].steps_for(run_id),
                  "approval": {"approved": bool(ap[0]), "text": ap[1]} if ap else None}


@app.post("/runs/{run_id}/decision")
def decide(run_id: str, d: Decide):
    run = STATE["store"].run(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    if run["decision"] == "auto_send":
        raise HTTPException(409, "already auto-sent")
    STATE["store"].db.execute("INSERT OR REPLACE INTO approvals VALUES (?,?,?,?)", (run_id, int(d.approve), d.text, time.time()))
    STATE["store"].db.commit()
    return {"run_id": run_id, "approved": d.approve}
