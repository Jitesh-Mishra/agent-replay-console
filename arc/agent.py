"""The agent: LangGraph pipeline classify -> retrieve -> draft -> gate, with step-level tracing to SQLite,
pluggable drafter, fault injection, and validators (guards) that can be switched off for ablation."""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from .components import QUEUES, Index, search

FAULTS = ("tool_timeout", "stale_schema", "malformed_tool_output", "empty_retrieval", "llm_refusal")
# v0 matched any "I'm sorry", which flagged empathetic openers ("I'm sorry to hear...") in 0.6% of
# historical answers and escalated 8 of 1,000 clean runs. Empathy phrases are now excluded.
REFUSAL = re.compile(r"\b(i'?m sorry,? but|i'?m sorry(?! to hear| for| about| that)|i cannot help|i can'?t help|"
                     r"as an ai\b|i am unable to (help|assist))", re.I)


# ----------------------------------------------------------------------------- config / version
@dataclass
class Version:
    """Everything that defines an agent build. Replay compares two of these."""
    name: str = "v1"
    conf_threshold: float = 0.6          # route confidence needed to auto-send
    min_retrieval_sim: float = 0.5       # grounding needed to auto-send
    gate_checks_priority: bool = True    # predicted high priority always needs a human
    temperature: float = 0.3             # drafter sampling over top-3 retrieved answers
    retrieval_field: str = "text"        # text = subject + body
    drop_closing: bool = False
    label_swap: tuple = ()               # (a, b) output-label bug
    guards: bool = True                  # validators on tool / model outputs


@dataclass
class Components:
    router: object
    priority: object
    index: Index
    drafter: "Drafter"
    _cache: dict = field(default_factory=dict, repr=False)

    def route(self, text):
        """Deterministic model calls are memoised per component set (replay re-runs them many times)."""
        k = ("route", text)
        if k not in self._cache:
            proba = self.router.predict_proba([text])[0]
            self._cache[k] = (self.router.classes_[int(proba.argmax())], float(proba.max()),
                              self.priority.predict([text])[0])
        return self._cache[k]

    def search(self, text):
        k = ("search", text)
        if k not in self._cache:
            self._cache[k] = search(self.index, text)
        return self._cache[k]


# ----------------------------------------------------------------------------- drafter (pluggable LLM client)
class Drafter:
    """Interface: draft(ticket, hits, kb, rng, version) -> str."""

    def draft(self, ticket, hits, kb, rng, v: Version) -> str:  # pragma: no cover
        raise NotImplementedError


class ExtractiveDrafter(Drafter):
    """Offline baseline used in the proof: samples one of the top-3 retrieved historical answers
    (softmax over similarity / temperature) and adds a KB pointer. No LLM involved."""

    def draft(self, ticket, hits, kb, rng, v: Version) -> str:
        sims = np.array([h["sim"] for h in hits])
        if v.temperature <= 0:
            k = int(sims.argmax())
        else:
            p = np.exp((sims - sims.max()) / v.temperature); p /= p.sum()
            k = int(rng.choice(len(hits), p=p))
        text = hits[k]["answer"]
        if kb and kb[0]["sim"] >= 0.3:
            text += f"\n\nYou can also find guidance in our help article \"{kb[0]['title']}\"."
        if v.drop_closing:
            parts = re.split(r"(?<=[.!?])\s+", text.strip())
            text = " ".join(parts[:-1]) if len(parts) > 1 else text
        return text


class GroqDrafter(Drafter):
    """LLM drafter for the deeper build (needs GROQ_API_KEY and network access to api.groq.com)."""

    def __init__(self, model="openai/gpt-oss-20b"):
        from langchain_groq import ChatGroq  # optional dependency
        self.llm = ChatGroq(model=model, temperature=0.3)

    def draft(self, ticket, hits, kb, rng, v: Version) -> str:
        ctx = "\n---\n".join(h["answer"] for h in hits)
        msg = (f"Draft a support reply to this ticket using only the reference answers.\nTicket:\n{ticket['text']}\n"
               f"Reference answers:\n{ctx}")
        return self.llm.invoke(msg).content


# ----------------------------------------------------------------------------- tracing
class TraceStore:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, ticket_id TEXT, version TEXT, faults TEXT, seed INT,
            decision TEXT, reason TEXT, status TEXT, started REAL, ended REAL);
        CREATE TABLE IF NOT EXISTS steps(run_id TEXT, seq INT, node TEXT, input TEXT, output TEXT, error TEXT,
            latency_ms REAL, ts REAL, PRIMARY KEY(run_id, seq));""")

    def start(self, run_id, ticket_id, version, faults, seed):
        self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (run_id, ticket_id, version, json.dumps(sorted(faults)), seed, None, None, "running", time.time(), None))

    def step(self, run_id, seq, node, inp, out, err, ms):
        self.db.execute("INSERT INTO steps VALUES (?,?,?,?,?,?,?,?)",
                        (run_id, seq, node, json.dumps(inp, default=str), json.dumps(out, default=str), err, ms, time.time()))

    def end(self, run_id, decision, reason, status):
        self.db.execute("UPDATE runs SET decision=?, reason=?, status=?, ended=? WHERE run_id=?",
                        (decision, reason, status, time.time(), run_id))
        self.db.commit()

    def steps_for(self, run_id):
        cur = self.db.execute("SELECT seq,node,input,output,error,latency_ms FROM steps WHERE run_id=? ORDER BY seq", (run_id,))
        return [{"seq": s, "node": n, "input": json.loads(i), "output": json.loads(o), "error": e, "latency_ms": ms}
                for s, n, i, o, e, ms in cur.fetchall()]

    def run(self, run_id):
        cols = ["run_id", "ticket_id", "version", "faults", "seed", "decision", "reason", "status", "started", "ended"]
        r = self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(zip(cols, r)) if r else None


# ----------------------------------------------------------------------------- the gate (pure function of step outputs)
def gate_decision(classify_out: dict, retrieve_out: dict, draft_out: dict, v: Version, fault_flags: list) -> tuple:
    """auto_send | needs_approval | escalate, with a reason. Pure: trace reconstruction re-runs it."""
    if fault_flags:
        return "escalate", "fault:" + ",".join(sorted(fault_flags))
    reasons = []
    if classify_out.get("confidence", 0) < v.conf_threshold:
        reasons.append("low_route_confidence")
    if v.gate_checks_priority and classify_out.get("priority") == "high":
        reasons.append("high_priority")
    if retrieve_out.get("top_sim", 0) < v.min_retrieval_sim:
        reasons.append("weak_grounding")
    if reasons:
        return "needs_approval", ",".join(reasons)
    return "auto_send", "all_checks_passed"


class State(TypedDict, total=False):
    run_id: str
    ticket: dict
    classify: dict
    retrieve: dict
    draft: dict
    fault_flags: list
    decision: str
    reason: str


# ----------------------------------------------------------------------------- agent
class Agent:
    def __init__(self, comp: Components, version: Version, store: TraceStore,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.c, self.v, self.store, self.on_event = comp, version, store, on_event
        g = StateGraph(State)
        for name in ("classify", "retrieve", "draft", "gate"):
            g.add_node(name, self._traced(name, getattr(self, f"_{name}")))
        g.set_entry_point("classify")
        g.add_edge("classify", "retrieve"); g.add_edge("retrieve", "draft"); g.add_edge("draft", "gate")
        g.add_edge("gate", END)
        self.graph = g.compile()

    # each node: (state, ctx) -> partial state; ctx carries rng + injected faults
    def _traced(self, name, fn):
        def node(state: State):
            ctx = self._ctx
            t0 = time.perf_counter()
            err, out = None, {}
            try:
                out = fn(state, ctx)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                if not self.v.guards:
                    ctx["crashed"] = True
                    raise
                out = {name: {"error": err}, "fault_flags": state.get("fault_flags", []) + [f"{name}_exception"]}
            ms = (time.perf_counter() - t0) * 1000
            ctx["seq"] += 1
            self.store.step(state["run_id"], ctx["seq"], name, _summ(state, name), out, err, ms)
            if self.on_event:
                self.on_event({"runId": state["run_id"], "seq": ctx["seq"], "node": name,
                               "status": "error" if err else "ok", "detail": json.dumps(_compact(out), default=str),
                               "latencyMs": round(ms, 2)})
            return out
        return node

    def _classify(self, s: State, ctx):
        t = s["ticket"]["text"]
        label, conf, prio = self.c.route(t)
        if self.v.label_swap and label in self.v.label_swap:
            a, b = self.v.label_swap
            label = b if label == a else a
        if "stale_schema" in ctx["faults"]:
            label = "Tech Support (legacy)"            # label from an old queue schema
        out = {"queue": label, "confidence": conf, "priority": prio}
        flags = list(s.get("fault_flags", []))
        if self.v.guards and label not in QUEUES:
            flags.append("stale_schema")
        return {"classify": out, "fault_flags": flags}

    def _retrieve(self, s: State, ctx):
        if "tool_timeout" in ctx["faults"]:
            raise TimeoutError("retrieval tool exceeded 2s budget")
        tk = s["ticket"]
        q = tk["subject"] if self.v.retrieval_field == "subject" and tk["subject"] else tk["text"]
        hits, kb = self.c.search(q)
        hits = [dict(h) for h in hits]
        if "empty_retrieval" in ctx["faults"]:
            hits = []
        if "malformed_tool_output" in ctx["faults"]:
            hits = [{"doc": None, "score": "n/a"}]       # wrong schema from the tool
        flags = list(s.get("fault_flags", []))
        if self.v.guards:
            if not hits:
                flags.append("empty_retrieval")
            elif not all(isinstance(h, dict) and isinstance(h.get("answer"), str) and isinstance(h.get("sim"), float)
                         for h in hits):
                flags.append("malformed_tool_output")
        good = [h for h in hits if isinstance(h.get("answer"), str)]
        top = max((h["sim"] for h in good if isinstance(h.get("sim"), float)), default=0.0)
        return {"retrieve": {"hits": good, "kb": kb, "top_sim": float(top), "n_raw": len(hits)}, "fault_flags": flags}

    def _draft(self, s: State, ctx):
        r = s.get("retrieve", {})
        hits = r.get("hits") or []
        flags = list(s.get("fault_flags", []))
        if not hits:
            text = ""          # nothing to ground on
        else:
            text = self.c.drafter.draft(s["ticket"], hits, r.get("kb", []), ctx["rng"], self.v)
        if "llm_refusal" in ctx["faults"]:
            text = "I'm sorry, but I can't help with that request."
        if self.v.guards:
            if REFUSAL.search(text):
                flags.append("llm_refusal")
            if not text.strip():
                flags.append("empty_draft")
        return {"draft": {"text": text}, "fault_flags": flags}

    def _gate(self, s: State, ctx):
        d, why = gate_decision(s.get("classify", {}), s.get("retrieve", {}), s.get("draft", {}), self.v,
                               s.get("fault_flags", []))
        if d == "auto_send":
            ctx["outbox"].append({"run_id": s["run_id"], "text": s["draft"]["text"]})  # mocked write
        return {"decision": d, "reason": why}

    def run(self, ticket: dict, faults=(), seed=0, run_id=None) -> dict:
        run_id = run_id or uuid.uuid4().hex[:12]
        self._ctx = {"faults": set(faults), "rng": np.random.default_rng(seed), "seq": 0, "outbox": [], "crashed": False}
        self.store.start(run_id, ticket["ticket_id"], self.v.name, faults, seed)
        try:
            final = self.graph.invoke({"run_id": run_id, "ticket": ticket, "fault_flags": []})
            self.store.end(run_id, final.get("decision"), final.get("reason"), "done")
        except Exception as e:  # unguarded crash
            final = {"decision": "crashed", "reason": f"{type(e).__name__}"}
            self.store.end(run_id, "crashed", final["reason"], "crashed")
        final["run_id"], final["outbox"] = run_id, self._ctx["outbox"]
        return final


def _compact(out: dict) -> dict:
    """Event payload for the live stream: same structure as the trace, long texts shortened."""
    o = json.loads(json.dumps(out, default=str))
    r = o.get("retrieve")
    if isinstance(r, dict) and "hits" in r:
        r["hits"] = [{"ticket_id": h.get("ticket_id"), "queue": h.get("queue"), "sim": round(h.get("sim", 0), 3),
                      "answer": (h.get("answer") or "")[:160]} for h in r["hits"]]
    d = o.get("draft")
    if isinstance(d, dict) and isinstance(d.get("text"), str) and len(d["text"]) > 1500:
        d["text"] = d["text"][:1500] + "…"
    return o


def _summ(state, node):
    """Small, JSON-safe view of the input to a node for the trace."""
    if node == "classify":
        return {"ticket_id": state["ticket"]["ticket_id"]}
    return {k: state.get(k) for k in ("classify", "retrieve") if k in state and node != "classify"} | {
        "fault_flags": state.get("fault_flags", [])}


def reconstruct(store: TraceStore, run_id: str, v: Version) -> bool:
    """Rebuild the final decision from the step trace alone and compare with the recorded run."""
    steps = store.steps_for(run_id)
    run = store.run(run_id)
    if [s["seq"] for s in steps] != list(range(1, len(steps) + 1)):
        return False
    state = {"fault_flags": []}
    for s in steps:
        if s["node"] == "gate":
            d, _ = gate_decision(state.get("classify", {}), state.get("retrieve", {}), state.get("draft", {}), v,
                                 state.get("fault_flags", []))
            return d == run["decision"] == s["output"].get("decision")
        state.update(s["output"])
    return run["status"] == "crashed"
