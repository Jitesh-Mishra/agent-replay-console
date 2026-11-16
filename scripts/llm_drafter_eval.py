"""Measure the extractive and the Groq drafter on one fixed sample, then fault-inject the Groq drafter.

    python scripts/llm_drafter_eval.py                  # needs GROQ_API_KEY (.env); about 5 minutes
    python scripts/llm_drafter_eval.py --budget 150     # stop after ~150 s of work; run again to resume

Every finished run is appended to a checkpoint file, so an interrupted run resumes instead of repeating Groq calls.
Writes reports/llm_drafter.json and reports/llm_drafter.md. Nothing is tuned on this sample: the prompt, the guards
and the models are fixed before the first call. Numbers are reported exactly as measured.
"""
import argparse
import hashlib
import json
import platform
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc.agent import (FAULTS, REFUSAL_V1, SYSTEM_PROMPT, Agent, Components, ExtractiveDrafter, TraceStore,  # noqa: E402
                       Version, gate_decision, is_refusal, make_drafter, reconstruct)
from arc.components import (ROOT, SEED, AnswerSim, build_index, fit_classifier, load_kb, load_tickets, split)  # noqa: E402

SAMPLE_SEED, N_TICKETS, N_FAULT_RUNS = 2026, 50, 20
PLACEHOLDER = re.compile(r"\[[A-Za-z][A-Za-z ]{1,30}\]|\{\{.*?\}\}|<[A-Za-z][A-Za-z ]{1,30}>")
DONE_CLAIM = re.compile(r"\b(i'?ve|i have|we'?ve|we have|has been|have been)\s+(already\s+)?(processed|issued|refunded|"
                        r"credited|reset|fixed|resolved|escalated|corrected|reversed)\b", re.I)
URL_OR_PHONE = re.compile(r"https?://|www\.|\b\d{3}[-. ]\d{3}[-. ]\d{4}\b|\b1-800")
FAULT_FLAG = {"tool_timeout": "retrieve_exception", "stale_schema": "stale_schema", "malformed_tool_output": "malformed",
              "empty_retrieval": "empty_retrieval", "llm_refusal": "llm_refusal"}
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


# ----------------------------------------------------------------------------- running
def build():
    df = load_tickets()
    tr, te = split(df)
    router = fit_classifier(tr["text"], tr["queue"], C=1.0)[0]       # same config as the console (arc/api.py)
    prio = fit_classifier(tr["text"], tr["priority"], C=1.0)[0]
    comp = Components(router, prio, build_index(tr, load_kb(), name="llm_eval"), ExtractiveDrafter())
    sample = te.sample(N_TICKETS, random_state=SAMPLE_SEED).reset_index(drop=True)
    return comp, sample, AnswerSim(tr["answer"]), len(tr)


def tasks(sample):
    ids = sample["ticket_id"].tolist()
    out = [("groq", "clean", i) for i in range(len(ids))] + [("extractive", "clean", i) for i in range(len(ids))]
    out += [("groq", f, i) for f in FAULTS for i in range(N_FAULT_RUNS)]
    return [(d, c, i, f"{d}|{c}|{ids[i]}") for d, c, i in out]


def run_one(agent, store, drafter, sim, t, i, cond):
    before = Counter(getattr(drafter, "stats", {}))
    r = agent.run(t, faults=[] if cond == "clean" else [cond], seed=i)
    delta = dict(Counter(getattr(drafter, "stats", {})) - before)
    steps = {s["node"]: s for s in store.steps_for(r["run_id"])}
    text = (r.get("draft") or {}).get("text", "") or ""
    c, rt = r.get("classify") or {}, r.get("retrieve") or {}
    return {"ticket_id": t["ticket_id"], "cond": cond, "decision": r["decision"], "reason": r["reason"],
            "flags": r.get("fault_flags", []), "route": c.get("queue"), "conf": c.get("confidence"),
            "priority_pred": c.get("priority"), "top_sim": rt.get("top_sim"), "true_queue": t["queue"],
            "true_priority": t["priority"], "n_steps": len(steps), "outbox": len(r["outbox"]),
            "reconstructs": bool(reconstruct(store, r["run_id"], agent.v)),
            "draft_ms": steps["draft"]["latency_ms"] if "draft" in steps else None,
            "draft_error": steps["draft"]["error"] if "draft" in steps else None,
            "draft": text, "draft_chars": len(text),
            "quality": float(sim([text], [t["answer"]])[0]) if text else 0.0,
            "placeholder": bool(PLACEHOLDER.search(text)), "done_claim": bool(DONE_CLAIM.search(text)),
            "url_or_phone": bool(URL_OR_PHONE.search(text)), "stats": delta}


def collect(ckpt: Path, budget: float):
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    comp, sample, sim, n_train = build()
    log(f"built (train {n_train}); {len(done)} runs already in checkpoint")
    groq = make_drafter("groq").primary          # the measurement is of Groq itself, without the offline fallback
    agents = {}
    for d, drafter in (("groq", groq), ("extractive", ExtractiveDrafter())):
        store = TraceStore()
        agents[d] = (Agent(replace(comp, drafter=drafter), Version(), store), store, drafter)
    todo = [t for t in tasks(sample) if t[3] not in done]
    for d, cond, i, key in todo:
        if time.time() - T0 > budget:
            log(f"budget reached: {len(done)}/{len(tasks(sample))} done; run again to resume")
            return None, sample, n_train, groq
        agent, store, drafter = agents[d]
        rec = run_one(agent, store, drafter, sim, sample.iloc[i].to_dict(), i, cond) | {"key": key, "drafter": d}
        with ckpt.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        done[key] = rec
        if len(done) % 10 == 0:
            log(f"{len(done)}/{len(tasks(sample))}  last: {d} {cond} -> {rec['decision']} ({rec['draft_ms'] or 0:.0f} ms)")
    return list(done.values()), sample, n_train, groq


# ----------------------------------------------------------------------------- summarising
def policy_violations(rows):
    """Independent audit: re-apply the v1 gate policy to each auto-sent run's recorded step outputs."""
    n = 0
    for r in rows:
        if r["decision"] != "auto_send":
            continue
        d, _ = gate_decision({"confidence": r["conf"], "priority": r["priority_pred"]}, {"top_sim": r["top_sim"]}, {},
                             Version(), r["flags"])
        n += d != "auto_send"
    return n


def summarise_clean(rows):
    dec = Counter(r["decision"] for r in rows)
    ms = [r["draft_ms"] for r in rows if r["draft_ms"] is not None]
    auto = [r for r in rows if r["decision"] == "auto_send"]
    bad = [r for r in auto if r["route"] != r["true_queue"] or r["true_priority"] == "high"]
    return {"n": len(rows),
            "decisions": {k: dec.get(k, 0) for k in ("auto_send", "needs_approval", "escalate", "crashed")},
            "flags": dict(Counter(f for r in rows for f in r["flags"])),
            "route_accuracy": float(np.mean([r["route"] == r["true_queue"] for r in rows])),
            "auto_sent_wrong_route_or_high_priority": len(bad), "policy_violations": policy_violations(rows),
            "unsafe_auto_sends": len(bad) + policy_violations(rows),
            "draft_latency_ms": {"p50": pct(ms, 50), "p95": pct(ms, 95), "max": max(ms) if ms else None},
            "draft_quality_vs_historical_answer": float(np.mean([r["quality"] for r in rows])),
            "mean_draft_chars": float(np.mean([r["draft_chars"] for r in rows])),
            "drafts_with_placeholder": sum(r["placeholder"] for r in rows),
            "drafts_claiming_completed_action_heuristic": sum(r["done_claim"] for r in rows),
            "drafts_with_url_or_phone_heuristic": sum(r["url_or_phone"] for r in rows),
            "trace_reconstruct_rate": float(np.mean([r["reconstructs"] for r in rows])),
            "all_runs_streamed_4_steps": all(r["n_steps"] == 4 for r in rows)}


def summarise_groq_stats(rows):
    tot = Counter()
    for r in rows:
        tot.update(r["stats"])
    calls = tot.get("calls", 0)
    return {"api_calls": calls, "ok_primary": tot.get("ok", 0), "ok_after_fallback": tot.get("ok_fallback", 0),
            "errors": tot.get("errors", 0), "retries": tot.get("retries", 0), "fallbacks": tot.get("fallbacks", 0),
            "drafts_failed_completely": tot.get("failures", 0), "truncated_replies": tot.get("truncated", 0),
            "error_codes": {k[6:]: v for k, v in tot.items() if k.startswith("error_")},
            "prompt_tokens": tot.get("prompt_tokens", 0), "completion_tokens": tot.get("completion_tokens", 0),
            "approx_tokens_per_successful_call": (tot.get("prompt_tokens", 0) + tot.get("completion_tokens", 0))
            / max(1, tot.get("ok", 0) + tot.get("ok_fallback", 0))}


def latency_split(rows):
    out = {}
    for name, sel in (("no_retry", [r for r in rows if not r["stats"].get("retries")]),
                      ("with_retry", [r for r in rows if r["stats"].get("retries")])):
        ms = [r["draft_ms"] for r in sel if r["draft_ms"] is not None]
        out[name] = {"n": len(ms), "p50": pct(ms, 50) or 0.0, "p95": pct(ms, 95) or 0.0}
    return out


def summarise_faults(rows):
    out = []
    for f in FAULTS:
        rs = [r for r in rows if r["cond"] == f]
        calls = sum(r["stats"].get("calls", 0) for r in rs)
        out.append({"fault": f, "n": len(rs), "escalated": sum(r["decision"] == "escalate" for r in rs),
                    "auto_sent_while_faulty": sum(r["decision"] == "auto_send" for r in rs),
                    "crashed": sum(r["decision"] == "crashed" for r in rs),
                    "needs_approval": sum(r["decision"] == "needs_approval" for r in rs),
                    "detected_flag": float(np.mean([any(FAULT_FLAG[f] in x for x in r["flags"]) for r in rs])),
                    "runs_that_called_groq": sum(r["stats"].get("calls", 0) > 0 for r in rs), "groq_api_calls": calls,
                    "all_runs_streamed_4_steps": all(r["n_steps"] == 4 for r in rs),
                    "trace_reconstruct_rate": float(np.mean([r["reconstructs"] for r in rs]))})
    return out


EMPATHY = ("I'm sorry to hear about the outage. We are on it.", "I'm sorry for the inconvenience.")
GUARD_SAMPLES = ["I can’t help with that.", "I’m sorry, but I can’t assist with that request.",
                 "I can't assist with this.", "I'm unable to provide that information.",
                 "I’m not able to help with this request.", "I must decline to answer.",
                 "Sorry, I can’t do that.", "I'm sorry to hear about the outage. We are on it.",
                 "I'm sorry for the inconvenience."]


def finalize(rows, sample, n_train, groq):
    clean = {d: [r for r in rows if r["drafter"] == d and r["cond"] == "clean"] for d in ("extractive", "groq")}
    faults = [r for r in rows if r["drafter"] == "groq" and r["cond"] != "clean"]
    R = {"meta": {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "sample": {"n": N_TICKETS, "seed": SAMPLE_SEED, "from": "grouped held-out test split (arc.components.split)",
                             "n_train_for_router_and_index": n_train},
                  "models": groq.models, "temperature": 0.3, "reasoning_effort": "low (gpt-oss models)",
                  "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
                  "guards": True, "gate_policy": "Version() defaults (conf 0.6, grounding 0.5, high priority needs a human)",
                  "python": platform.python_version(), "run": "single pass, no retries of the experiment, nothing tuned on this sample",
                  "fault_runs_per_type": N_FAULT_RUNS},
         "clean_50_tickets": {d: summarise_clean(rs) for d, rs in clean.items()},
         "groq_provider": summarise_groq_stats(clean["groq"]),
         "groq_latency_split": latency_split(clean["groq"]),
         "groq_faults": summarise_faults(faults),
         "groq_faults_provider": summarise_groq_stats(faults),
         "guard_before_after": [{"text": t, "is_refusal": t not in EMPATHY, "v1_flags": bool(REFUSAL_V1.search(t)), "v2_flags": is_refusal(t)}
                                for t in GUARD_SAMPLES],
         "examples": [{"ticket_id": r["ticket_id"], "decision": r["decision"], "reason": r["reason"], "draft": r["draft"]}
                      for r in clean["groq"][:3]],
         "flagged_groq_drafts": [{"ticket_id": r["ticket_id"], "flags": r["flags"], "draft": r["draft"][:300]}
                                 for r in clean["groq"] if any(f in ("llm_refusal", "empty_draft") for f in r["flags"])],
         "per_run_groq": [{k: r[k] for k in ("ticket_id", "decision", "reason", "flags", "draft_ms", "draft_chars", "quality",
                                             "placeholder", "done_claim", "url_or_phone", "stats", "draft_error")}
                          for r in clean["groq"]]}
    return R


def render_md(R):
    m, e, g, pv, ft = R["meta"], R["clean_50_tickets"]["extractive"], R["clean_50_tickets"]["groq"], R["groq_provider"], R["groq_faults"]
    lat = R["groq_latency_split"]

    def d(x):
        return f"{x['decisions']['auto_send']} / {x['decisions']['needs_approval']} / {x['decisions']['escalate']}"

    def ms(x):
        return f"{x['draft_latency_ms']['p50']:.1f} / {x['draft_latency_ms']['p95']:.1f}"
    L = [f"# LLM drafter: measured results\n",
         f"Generated {m['generated_utc']}. Models: `{m['models'][0]}` (fallback `{m['models'][1] if len(m['models']) > 1 else 'none'}`), "
         f"temperature {m['temperature']}. Sample: {m['sample']['n']} tickets drawn with seed {m['sample']['seed']} from the grouped "
         f"held-out split. Guards on. Gate policy: {m['gate_policy']}. Single pass, nothing tuned on this sample.\n",
         "## 50 clean tickets, both drafters\n", "| Metric | Extractive | Groq |", "|---|---:|---:|",
         f"| Decisions (auto_send / needs_approval / escalate) | {d(e)} | {d(g)} |",
         f"| Unsafe auto-sends (auto-sent with wrong route or true high priority, plus policy violations) | {e['unsafe_auto_sends']} | {g['unsafe_auto_sends']} |",
         f"| Policy violations (audit re-applying the gate to recorded outputs) | {e['policy_violations']} | {g['policy_violations']} |",
         f"| Draft step latency p50 / p95 (ms) | {ms(e)} | {ms(g)} |",
         f"| Guard flags raised | {e['flags'] or 'none'} | {g['flags'] or 'none'} |",
         f"| Draft similarity to historical answer (TF-IDF cosine) | {e['draft_quality_vs_historical_answer']:.3f} | {g['draft_quality_vs_historical_answer']:.3f} |",
         f"| Mean draft length (chars) | {e['mean_draft_chars']:.0f} | {g['mean_draft_chars']:.0f} |",
         f"| Drafts with leftover placeholders | {e['drafts_with_placeholder']} | {g['drafts_with_placeholder']} |",
         f"| Drafts claiming a completed action (regex heuristic) | {e['drafts_claiming_completed_action_heuristic']} | {g['drafts_claiming_completed_action_heuristic']} |",
         f"| Drafts containing a URL or phone number (regex heuristic) | {e['drafts_with_url_or_phone_heuristic']} | {g['drafts_with_url_or_phone_heuristic']} |",
         f"| All 4 steps recorded and trace reconstructs the decision | {e['trace_reconstruct_rate']:.2f} | {g['trace_reconstruct_rate']:.2f} |", "",
         "## Groq provider behaviour on those 50 runs\n", "| | |", "|---|---:|",
         f"| API calls | {pv['api_calls']} |", f"| Errors | {pv['errors']} {pv['error_codes'] or ''} |",
         f"| Retries | {pv['retries']} |", f"| Fallbacks to the second model | {pv['fallbacks']} |",
         f"| Drafts produced by the fallback model | {pv['ok_after_fallback']} |",
         f"| Drafts that failed completely (escalated) | {pv['drafts_failed_completely']} |",
         f"| Truncated replies rejected | {pv['truncated_replies']} |",
         f"| Tokens (prompt / completion) | {pv['prompt_tokens']} / {pv['completion_tokens']} |",
         f"| Approx tokens per successful call (prompt + completion) | {pv['approx_tokens_per_successful_call']:.0f} |", "",
         f"Draft latency for the Groq runs with no retry: n={lat['no_retry']['n']}, p50 {lat['no_retry']['p50']:.0f} ms, p95 {lat['no_retry']['p95']:.0f} ms. "
         f"With at least one retry (a 429 and a backoff wait): n={lat['with_retry']['n']}, p50 {lat['with_retry']['p50']:.0f} ms, p95 {lat['with_retry']['p95']:.0f} ms.", "",
         f"## Fault injection against the Groq drafter ({m['fault_runs_per_type']} runs per fault)\n",
         "| Fault | Runs | Escalated | Auto-sent while faulty | Crashed | Flag detected | Runs that called Groq |", "|---|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {x['fault']} | {x['n']} | {x['escalated']} | {x['auto_sent_while_faulty']} | {x['crashed']} | {x['detected_flag']:.2f} | {x['runs_that_called_groq']} |"
          for x in ft]
    L += ["", "## Guard before / after on realistic LLM refusals\n", "| Reply | Old guard | New guard |", "|---|---|---|"]
    L += [f"| {x['text']} ({'refusal' if x['is_refusal'] else 'not a refusal'}) | "
          f"{'flagged' if x['v1_flags'] else ('missed' if x['is_refusal'] else 'not flagged')} | "
          f"{'flagged' if x['v2_flags'] else ('missed' if x['is_refusal'] else 'not flagged')} |" for x in R["guard_before_after"]]
    if not g["flags"]:
        L += ["", "No real Groq reply triggered the refusal or empty guard in these 50 runs, so the guard fix is validated on "
              "hand-written refusals and mocks, not on refusals that occurred naturally."]
    L += ["", "## Limits\n",
          "- 50 tickets and one pass: small samples, and Groq output is not deterministic, so a rerun will differ.",
          "- The tickets are likely synthetic. The knowledge base is 12 self-written documents.",
          "- `Unsafe auto-send` here means the routing and priority policy was violated. It does NOT mean the reply text is correct.",
          "- The gate does not read the draft, so both drafters get identical gate decisions. Reply quality is only checked by the refusal and "
          "empty guards plus a similarity score against one historical answer, a weak proxy (it measures word overlap, not correctness). "
          "No human or LLM judge has rated these replies.",
          "- Groq returned x-ratelimit-limit-tokens 8000 per minute and 1000 requests per day for the key used (read on 2026-11-16). "
          "At roughly 700 tokens per draft, back-to-back runs exceed the per-minute cap, which is where the 429 retries and the slow "
          "p95 come from. One ticket at a time, as in the console, is well under it.",
          "- The unsafe auto-send is the same ticket for both drafters: the route was right but the priority model predicted medium for a "
          "truly high-priority ticket. That is the existing priority-model gap, independent of the drafter.",
          "- The completed-action, placeholder and URL/phone rows are regex heuristics, not a faithfulness measurement.",
          "- Latency is measured from one machine against Groq's public API and includes retries and backoff.",
          "- Fault runs that stop before the draft step (tool_timeout, empty_retrieval, malformed_tool_output) never call Groq, "
          "so only stale_schema and llm_refusal exercise the LLM; llm_refusal overwrites the model reply with the injected refusal.\n"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9, help="seconds of work before stopping (resume by re-running)")
    ap.add_argument("--ckpt", default=str(Path(tempfile.gettempdir()) / "arc_llm_drafter_ckpt.jsonl"))
    a = ap.parse_args()
    rows, sample, n_train, groq = collect(Path(a.ckpt), a.budget)
    if rows is None:
        sys.exit(3)
    R = finalize(rows, sample, n_train, groq)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "llm_drafter.json").write_text(json.dumps(R, indent=2, default=float))
    (ROOT / "reports" / "llm_drafter.md").write_text(render_md(R))
    log("DONE -> reports/llm_drafter.json, reports/llm_drafter.md")


if __name__ == "__main__":
    main()
