import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc.agent import (FAULTS, Agent, Components, ExtractiveDrafter, TraceStore, Version,  # noqa: E402
                       gate_decision, reconstruct)
from arc.components import build_index, fit_classifier, load_kb, load_tickets, near_dup_groups, split  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    df = load_tickets()
    tr, te = split(df)
    tr = tr.sample(2500, random_state=0)
    comp = Components(fit_classifier(tr["text"], tr["queue"], C=1.0)[0], fit_classifier(tr["text"], tr["priority"], C=1.0)[0],
                      build_index(tr, load_kb(), name="t"), ExtractiveDrafter())
    return comp, te.head(30)


def test_grouped_split_has_no_shared_paraphrase_family():
    df = load_tickets()
    g = pd.Series(near_dup_groups(df), index=df["ticket_id"])
    tr, te = split(df)
    assert not set(g.loc[tr["ticket_id"]]) & set(g.loc[te["ticket_id"]])


def test_gate_policy():
    v = Version()
    ok = ({"confidence": 0.9, "priority": "low"}, {"top_sim": 0.8}, {"text": "hi"})
    assert gate_decision(*ok, v, [])[0] == "auto_send"
    assert gate_decision({"confidence": 0.9, "priority": "high"}, *ok[1:], v, [])[0] == "needs_approval"
    assert gate_decision({"confidence": 0.3, "priority": "low"}, *ok[1:], v, [])[0] == "needs_approval"
    assert gate_decision(*ok, v, ["llm_refusal"])[0] == "escalate"


@pytest.mark.parametrize("fault", FAULTS)
def test_every_fault_is_escalated_never_auto_sent(setup, fault):
    comp, te = setup
    store = TraceStore()
    agent = Agent(comp, Version(), store)
    for t in te.to_dict("records")[:10]:
        r = agent.run(t, faults=[fault], seed=1)
        assert r["decision"] == "escalate", (fault, r["reason"])
        assert not r["outbox"]
        assert reconstruct(store, r["run_id"], agent.v)


def test_without_guards_faults_slip_through_or_crash(setup):
    comp, te = setup
    agent = Agent(comp, Version(guards=False), TraceStore())
    t = te.to_dict("records")[0]
    assert agent.run(t, faults=["tool_timeout"])["decision"] == "crashed"
    assert agent.run(t, faults=["llm_refusal"])["decision"] != "escalate"


def test_trace_has_ordered_steps_and_events(setup):
    comp, te = setup
    store, events = TraceStore(), []
    r = Agent(comp, Version(), store, on_event=events.append).run(te.to_dict("records")[0], seed=3)
    steps = store.steps_for(r["run_id"])
    assert [s["node"] for s in steps] == ["classify", "retrieve", "draft", "gate"]
    assert [e["node"] for e in events] == ["classify", "retrieve", "draft", "gate"]
    assert reconstruct(store, r["run_id"], Version())


def test_api_streams_steps_and_records_approval(monkeypatch):
    monkeypatch.setenv("ARC_SMALL", "1")
    from fastapi.testclient import TestClient
    from arc import api
    api.STATE.clear()
    with TestClient(api.app) as c:
        tid = c.get("/tickets?limit=1").json()[0]["id"]
        rid = c.post("/runs", json={"ticket_id": tid, "faults": ["llm_refusal"]}).json()["run_id"]
        nodes = []
        with c.stream("GET", f"/runs/{rid}/events") as resp:
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    ev = json.loads(line[5:])
                    nodes.append(ev["node"])
                    if ev["node"] == "end":
                        assert ev["status"] == "escalate"
                        break
        assert nodes == ["classify", "retrieve", "draft", "gate", "end"]
        assert c.post(f"/runs/{rid}/decision", json={"approve": False}).status_code == 200
        assert c.get(f"/runs/{rid}").json()["approval"]["approved"] is False
        assert c.post("/runs", json={"ticket_id": tid, "faults": ["bogus"]}).status_code == 422


def test_refusal_detector_ignores_empathy_but_catches_refusals():
    from arc.agent import REFUSAL
    assert REFUSAL.search("I'm sorry, but I can't help with that request.")
    assert REFUSAL.search("As an AI, I cannot do that.")
    assert not REFUSAL.search("I'm sorry to hear about the outage. We are on it.")
    assert not REFUSAL.search("I'm sorry for the inconvenience.")
