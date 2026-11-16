"""LLM drafter: the ARC_DRAFTER switch, the Groq failure path (mocked, no network), and the real-LLM refusal/empty guards."""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc.agent import (REFUSAL_V1, Agent, Components, DrafterError, ExtractiveDrafter, FallbackDrafter,  # noqa: E402
                       GroqDrafter, TraceStore, Version, has_content, is_refusal, make_drafter, reconstruct)
from arc.components import ROOT, build_index, fit_classifier, load_kb, load_tickets, split  # noqa: E402


# ----------------------------------------------------------------------------- fakes (no network)
class Resp:
    def __init__(self, content, finish="stop"):
        self.content, self.response_metadata = content, {"finish_reason": finish}
        self.usage_metadata = {"input_tokens": 100, "output_tokens": 20}


class ProviderError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class FakeLLM:
    """Plays a script: each item is a Resp to return or an Exception to raise; the last item repeats."""

    def __init__(self, *script):
        self.script, self.calls = list(script), 0

    def invoke(self, msgs):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def drafter(primary, fallback=None, **kw):
    llms = {"primary": primary, "fallback": fallback or primary}
    sleeps = []
    d = GroqDrafter(model="primary", fallback_model="fallback", client_factory=lambda m: llms[m], sleep=sleeps.append,
                    **kw)
    d.sleeps = sleeps
    return d


GOOD = Resp("Thanks for reaching out. We will review your invoice and follow up with next steps.\n\nCustomer Support Team")
# Relaxed policy so a clean draft WOULD be auto-sent: lets the tests prove failures are what stop the send.
OPEN_GATE = Version(conf_threshold=0.0, min_retrieval_sim=0.0, gate_checks_priority=False)


@pytest.fixture(scope="module")
def setup():
    df = load_tickets()
    tr, te = split(df)
    tr = tr.sample(2500, random_state=0)
    comp = Components(fit_classifier(tr["text"], tr["queue"], C=1.0)[0], fit_classifier(tr["text"], tr["priority"], C=1.0)[0],
                      build_index(tr, load_kb(), name="llm"), ExtractiveDrafter())
    return comp, te.head(5).to_dict("records")


def run_with(setup, llm_drafter, v=OPEN_GATE, faults=()):
    comp, tickets = setup
    store = TraceStore()
    agent = Agent(replace(comp, drafter=llm_drafter), v, store)
    return agent, store, agent.run(tickets[0], faults=faults, seed=1)


# ----------------------------------------------------------------------------- the switch
def test_switch_defaults_to_extractive_and_selects_groq(monkeypatch):
    monkeypatch.delenv("ARC_DRAFTER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert isinstance(make_drafter(load_env=False), ExtractiveDrafter)
    assert isinstance(make_drafter("extractive", load_env=False), ExtractiveDrafter)
    monkeypatch.setenv("ARC_DRAFTER", "groq")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):          # fails at startup, not on every ticket
        make_drafter(load_env=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    d = make_drafter(load_env=False)                                    # building it makes no network call
    assert isinstance(d, FallbackDrafter) and isinstance(d.primary, GroqDrafter) and isinstance(d.fallback, ExtractiveDrafter)
    monkeypatch.setenv("ARC_DRAFTER", "gpt5")
    with pytest.raises(ValueError, match="ARC_DRAFTER"):
        make_drafter(load_env=False)


def test_model_name_and_fallback_come_from_one_place(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ARC_GROQ_MODEL", "m-primary")
    monkeypatch.setenv("ARC_GROQ_FALLBACK_MODEL", "m-fallback")
    assert GroqDrafter().models == ["m-primary", "m-fallback"]


# ----------------------------------------------------------------------------- Groq failure path
def test_control_working_llm_would_auto_send(setup):
    _, _, r = run_with(setup, drafter(FakeLLM(GOOD)))
    assert r["decision"] == "auto_send" and len(r["outbox"]) == 1


@pytest.mark.parametrize("err", [ProviderError(429), ProviderError(500), ProviderError(503), ProviderError(401),
                                 TimeoutError("read timed out"), ConnectionError("dns")])
def test_groq_total_failure_does_not_crash_and_never_auto_sends(setup, err):
    d = drafter(FakeLLM(err))
    agent, store, r = run_with(setup, d)
    assert r["decision"] == "escalate" and r["reason"].startswith("fault:draft_exception")
    assert not r["outbox"]
    assert [s["node"] for s in store.steps_for(r["run_id"])] == ["classify", "retrieve", "draft", "gate"]
    assert "DrafterError" in store.steps_for(r["run_id"])[2]["error"]
    assert reconstruct(store, r["run_id"], agent.v)
    assert d.stats["failures"] == 1


def test_retry_backs_off_then_succeeds():
    d = drafter(FakeLLM(ProviderError(429), ProviderError(429), GOOD), retries=2)
    assert "Customer Support Team" in d.draft({"text": "t"}, [{"answer": "a", "sim": 0.9, "queue": "q"}], [], None, Version())
    assert d.stats["retries"] == 2 and d.stats["ok"] == 1 and d.stats["fallbacks"] == 0
    assert len(d.sleeps) == 2 and d.sleeps[1] > d.sleeps[0]          # exponential backoff
    assert d.stats["prompt_tokens"] == 100 and d.stats["completion_tokens"] == 20   # only the successful call reports usage


def test_fallback_model_after_5xx():
    primary, fallback = FakeLLM(ProviderError(500)), FakeLLM(GOOD)
    d = drafter(primary, fallback, retries=1)
    assert d.draft({"text": "t"}, [{"answer": "a", "sim": 0.9, "queue": "q"}], [], None, Version())
    assert primary.calls == 2 and fallback.calls == 1 and d.stats["fallbacks"] == 1 and d.stats["ok_fallback"] == 1


def test_bad_key_fails_fast_without_retry_or_fallback():
    primary, fallback = FakeLLM(ProviderError(401)), FakeLLM(GOOD)
    d = drafter(primary, fallback)
    with pytest.raises(DrafterError):
        d.draft({"text": "t"}, [{"answer": "a", "sim": 0.9, "queue": "q"}], [], None, Version())
    assert primary.calls == 1 and fallback.calls == 0


def test_truncated_reply_is_never_used(setup):
    d = drafter(FakeLLM(Resp("Thanks for reaching out. We will", finish="length")))
    _, _, r = run_with(setup, d)
    assert r["decision"] == "escalate" and not r["outbox"] and d.stats["truncated"] >= 1


def test_prompt_uses_kb_article_and_past_answers():
    d = drafter(FakeLLM(GOOD))
    kb_doc = load_kb()[0]
    hits = [{"answer": "PAST-ANSWER-ONE", "sim": 0.8, "queue": "Billing and Payments"},
            {"answer": "PAST-ANSWER-TWO", "sim": 0.6, "queue": "Billing and Payments"}]
    text = d.messages({"text": "TICKET-BODY"}, hits, [{"id": kb_doc["id"], "title": kb_doc["title"], "sim": 0.7}])[1][1]
    assert "TICKET-BODY" in text and "PAST-ANSWER-ONE" in text and "PAST-ANSWER-TWO" in text
    assert kb_doc["text"][:60] in text and kb_doc["title"] in text
    weak = d.messages({"text": "x"}, hits, [{"id": kb_doc["id"], "title": kb_doc["title"], "sim": 0.1}])[1][1]
    assert kb_doc["text"][:60] not in weak                          # a barely related article is not shown as grounding


# ----------------------------------------------------------------------------- refusal / empty guards on real LLM output
LLM_REFUSALS = ["I can’t help with that.", "I’m sorry, but I can’t assist with that request.",
                "I can't assist with this.", "I'm unable to provide that information.",
                "I’m not able to help with this request.", "I must decline to answer.", "Sorry, I can’t do that."]
LEGIT = ["I'm sorry to hear about the outage. We are on it.", "I'm sorry for the inconvenience.",
         "Thanks for reaching out. We will review your invoice and follow up."]


@pytest.mark.parametrize("text", LLM_REFUSALS)
def test_refusal_guard_catches_real_llm_refusals(text):
    assert is_refusal(text)


def test_old_guard_missed_all_of_them():            # the "before" of the before/after in reports/llm_drafter.md
    assert not any(REFUSAL_V1.search(t) for t in LLM_REFUSALS)
    assert not any(is_refusal(t) for t in LEGIT)


def test_new_refusal_patterns_flag_no_historical_answer():
    """Keeps every existing measured result unchanged: the extractive drafter only emits historical answers."""
    answers = pd.read_csv(ROOT / "data" / "tickets_en.csv.gz")["answer"].fillna("")
    assert not answers.map(is_refusal).any()


@pytest.mark.parametrize("text", LLM_REFUSALS)
def test_llm_refusal_is_escalated_end_to_end(setup, text):
    _, _, r = run_with(setup, drafter(FakeLLM(Resp(text))))
    assert r["decision"] == "escalate" and "llm_refusal" in r["reason"] and not r["outbox"]


@pytest.mark.parametrize("text", ["", "   \n\t", "...", "—", "**"])
def test_empty_llm_reply_is_escalated_end_to_end(setup, text):
    assert not has_content(text)
    _, _, r = run_with(setup, drafter(FakeLLM(Resp(text))))
    assert r["decision"] == "escalate" and "empty_draft" in r["reason"] and not r["outbox"]


def test_non_string_content_does_not_slip_through(setup):
    _, _, r = run_with(setup, drafter(FakeLLM(Resp(None))))        # SDK returned no text at all
    assert r["decision"] == "escalate" and not r["outbox"]


# ----------------------------------------------------------------------------- offline fallback (Groq down / wrong key)
def with_fallback(llm):
    return FallbackDrafter(drafter(llm), ExtractiveDrafter())


def draft_of(store, r):
    return store.steps_for(r["run_id"])[2]["output"]["draft"]


def test_working_groq_is_labelled_and_not_flagged(setup):
    _, store, r = run_with(setup, with_fallback(FakeLLM(GOOD)))
    assert draft_of(store, r)["drafter"] == "groq:primary" and r["decision"] == "auto_send"
    assert "drafter_fallback" not in r["reason"]


@pytest.mark.parametrize("err", [ProviderError(401), ProviderError(503), ConnectionError("offline"), TimeoutError("slow")])
def test_groq_failure_falls_back_to_extractive_and_is_escalated_not_sent(setup, err):
    agent, store, r = run_with(setup, with_fallback(FakeLLM(err)))     # OPEN_GATE: a clean draft would auto-send
    d = draft_of(store, r)
    assert d["drafter"] == "fallback:extractive" and d["text"].strip()  # the run completed with a real draft
    assert r["decision"] == "escalate" and r["reason"] == "fault:drafter_fallback"
    assert not r["outbox"]                                              # never auto-sent
    assert [s["node"] for s in store.steps_for(r["run_id"])] == ["classify", "retrieve", "draft", "gate"]
    assert reconstruct(store, r["run_id"], agent.v)


def test_extractive_is_labelled(setup):
    comp, tickets = setup
    store = TraceStore()
    r = Agent(comp, OPEN_GATE, store).run(tickets[0], seed=1)
    assert draft_of(store, r)["drafter"] == "extractive" and r["decision"] == "auto_send"


def test_nothing_retrieved_has_no_drafter(setup):
    _, store, r = run_with(setup, with_fallback(FakeLLM(GOOD)), faults=["empty_retrieval"])
    assert draft_of(store, r)["drafter"].startswith("none") and r["decision"] == "escalate"


def test_injected_refusal_is_not_credited_to_the_model(setup):
    _, store, r = run_with(setup, with_fallback(FakeLLM(GOOD)), faults=["llm_refusal"])
    assert draft_of(store, r)["drafter"] == "injected fault: llm_refusal" and r["decision"] == "escalate"
