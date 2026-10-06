"""All measured numbers for the proof. python -m arc.evaluate  ->  reports/results.json (+ csvs)

Splits: GROUPED (no paraphrase family in both train and test) is the headline. The random split is
reported only to show how much near-duplicate leakage inflates routing scores on this dataset.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from .agent import FAULTS, Agent, Components, ExtractiveDrafter, TraceStore, Version, gate_decision, reconstruct
from .components import (ROOT, SEED, AnswerSim, build_index, fit_classifier, keyword_route, load_kb, load_tickets,
                         near_dup_groups, random_split, split)

REPORTS = ROOT / "reports"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def routing_table(df, groups):
    rows = []
    for split_name, fn in [("grouped (headline)", split), ("random (leaky, comparison only)", random_split)]:
        tr, te = fn(df)
        gtr = groups.loc[tr["ticket_id"]].to_numpy()
        m, info = fit_classifier(tr["text"], tr["queue"], groups=gtr)
        preds = {"majority": np.full(len(te), tr["queue"].mode()[0]),
                 "keyword rules": np.array([keyword_route(t) for t in te["text"]]),
                 "TF-IDF + logistic regression": m.predict(te["text"])}
        for name, p in preds.items():
            rows.append({"split": split_name, "model": name, "macro_f1": f1_score(te["queue"], p, average="macro"),
                         "accuracy": accuracy_score(te["queue"], p), "n_test": len(te),
                         "C": info.get("C") if "logistic" in name else None,
                         "grouped_cv_macro_f1": info.get("cv_macro_f1") if "logistic" in name else None})
        log(f"routing {split_name}: LR {rows[-1]['macro_f1']:.3f}")
    return pd.DataFrame(rows)


def build(tr, groups, kb, frac_router=1.0, index_fraction=1.0, lowercase=True, name="v1"):
    trr = tr.sample(frac=frac_router, random_state=SEED) if frac_router < 1 else tr
    g = groups.loc[trr["ticket_id"]].to_numpy()
    router, rinfo = fit_classifier(trr["text"], trr["queue"], groups=g, lowercase=lowercase)
    prio, pinfo = fit_classifier(trr["text"], trr["priority"], groups=g, lowercase=lowercase)
    idx = build_index(tr, kb, fraction=index_fraction, name=name)
    return Components(router, prio, idx, ExtractiveDrafter()), {"router": rinfo, "priority": pinfo}


def run_rows(agent, tickets, sim, faults=(), seed0=0):
    out = []
    for i, t in enumerate(tickets.to_dict("records")):
        r = agent.run(t, faults=faults, seed=seed0 + i)
        c = r.get("classify", {}) or {}
        draft = (r.get("draft") or {}).get("text", "")
        out.append({"ticket_id": t["ticket_id"], "run_id": r["run_id"], "decision": r["decision"],
                    "reason": r["reason"], "route": c.get("queue"), "conf": c.get("confidence"),
                    "prio_pred": c.get("priority"), "true_queue": t["queue"], "true_priority": t["priority"],
                    "draft": draft, "true_answer": t["answer"], "flags": ",".join(r.get("fault_flags", []))})
    df = pd.DataFrame(out)
    df["route_correct"] = df["route"] == df["true_queue"]
    df["quality"] = sim(df["draft"].fillna(""), df["true_answer"]) if len(df) else []
    df["auto"] = df["decision"] == "auto_send"
    df["bad_auto"] = df["auto"] & (~df["route_correct"] | (df["true_priority"] == "high"))
    return df


def policy_violations(store, df, v1: Version):
    """Independent audit: re-apply the v1 policy to each run's recorded step outputs; an auto_send that
    v1 would not allow is a violation."""
    n = 0
    for rid, dec in zip(df["run_id"], df["decision"]):
        if dec != "auto_send":
            continue
        st = {"fault_flags": []}
        for s in store.steps_for(rid):
            if s["node"] != "gate":
                st.update(s["output"])
        d, _ = gate_decision(st.get("classify", {}), st.get("retrieve", {}), st.get("draft", {}), v1, st.get("fault_flags", []))
        n += d != "auto_send"
    return n


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def compare(base: pd.DataFrame, cand: pd.DataFrame, viol: int, alpha=0.05) -> dict:
    """Flag a regression if any paired test is significant (Bonferroni over 3) or the policy invariant breaks."""
    p_route = mcnemar_p(base["route_correct"].to_numpy(), cand["route_correct"].to_numpy())
    dq = cand["quality"].to_numpy() - base["quality"].to_numpy()
    p_quality = 1.0 if np.allclose(dq, 0) else float(stats.ttest_rel(cand["quality"], base["quality"]).pvalue)
    p_bad = mcnemar_p(base["bad_auto"].to_numpy(), cand["bad_auto"].to_numpy())
    flag = min(p_route, p_quality, p_bad) < alpha / 3 or viol > 0
    return {"p_route": p_route, "p_quality": p_quality, "p_bad_auto": p_bad, "violations": viol,
            "d_route_acc": float(cand["route_correct"].mean() - base["route_correct"].mean()),
            "d_quality": float(dq.mean()), "d_bad_auto": int(cand["bad_auto"].sum() - base["bad_auto"].sum()),
            "flagged": bool(flag)}


def main(n_e2e=1000, n_fault=300, n_replay=200, aa_reps=100, reg_reps=20):
    REPORTS.mkdir(exist_ok=True)
    df = load_tickets()
    groups = pd.Series(near_dup_groups(df), index=df["ticket_id"])
    R = {"data": {"tickets_en": len(df), "near_dup_groups": int(groups.nunique()),
                  "largest_group": int(groups.value_counts().max())}}

    # ---------- 1. routing
    rt = routing_table(df, groups)
    rt.to_csv(REPORTS / "routing.csv", index=False, float_format="%.4f")
    R["routing"] = rt.to_dict("records")

    tr, te = split(df)
    kb = load_kb()
    comp, info = build(tr, groups, kb)
    R["models"] = info
    sim = AnswerSim(tr["answer"])
    log("components built")

    # ---------- 2. retrieval
    rng = np.random.default_rng(SEED)
    top1 = [comp.search(t)[0][0]["answer"] for t in te["text"]]
    same_q = [tr.loc[tr["queue"] == q, "answer"].sample(1, random_state=int(rng.integers(1e9))).iloc[0] for q in te["queue"]]
    R["retrieval"] = {"top1_answer_sim": float(sim(top1, te["answer"]).mean()),
                      "random_answer_sim": float(sim(tr["answer"].sample(len(te), random_state=1, replace=True), te["answer"]).mean()),
                      "same_queue_random_sim": float(sim(same_q, te["answer"]).mean()), "n": len(te)}
    log(f"retrieval {R['retrieval']}")

    # ---------- 3. end-to-end, clean
    v1 = Version()
    store = TraceStore()
    agent = Agent(comp, v1, store)
    sample = te.sample(min(n_e2e, len(te)), random_state=SEED)
    e2e = run_rows(agent, sample, sim)
    lat = pd.read_sql("SELECT node, latency_ms FROM steps", store.db).groupby("node")["latency_ms"].describe(percentiles=[.5, .95])
    R["e2e"] = {"n": len(e2e), "decisions": e2e["decision"].value_counts().to_dict(),
                "auto_send_rate": float(e2e["auto"].mean()),
                "auto_send_route_accuracy": float(e2e.loc[e2e["auto"], "route_correct"].mean()) if e2e["auto"].any() else None,
                "overall_route_accuracy": float(e2e["route_correct"].mean()),
                "true_high_priority_auto_sent": int((e2e["auto"] & (e2e["true_priority"] == "high")).sum()),
                "bad_auto_sends": int(e2e["bad_auto"].sum()),
                "policy_violations": policy_violations(store, e2e, v1),
                "draft_quality_mean": float(e2e["quality"].mean()),
                "trace_reconstruct_rate": float(np.mean([reconstruct(store, r, v1) for r in e2e["run_id"]])),
                "latency_ms": {n: {"p50": float(r["50%"]), "p95": float(r["95%"])} for n, r in lat.iterrows()}}
    log(f"e2e {R['e2e']['decisions']} auto-acc {R['e2e']['auto_send_route_accuracy']}")

    # ---------- 4. fault injection, guards on vs off
    fl = []
    fs = te.sample(min(n_fault, len(te)), random_state=SEED + 1)
    for guards in (True, False):
        v = replace(v1, guards=guards, name=f"v1_guards_{guards}")
        st = TraceStore(); ag = Agent(comp, v, st)
        for f in FAULTS:
            rows = run_rows(ag, fs, sim, faults=[f], seed0=1000)
            rec = [reconstruct(st, r, v) for r in rows["run_id"]]
            fl.append({"guards": guards, "fault": f, "n": len(rows),
                       "detected": float(rows["flags"].str.contains(f.split("_")[0] if f != "tool_timeout" else "retrieve_exception").mean()),
                       "escalated": float((rows["decision"] == "escalate").mean()),
                       "crashed": float((rows["decision"] == "crashed").mean()),
                       "auto_sent_while_faulty": int((rows["decision"] == "auto_send").sum()),
                       "trace_reconstructed": float(np.mean(rec))})
    fdf = pd.DataFrame(fl)
    fdf.to_csv(REPORTS / "faults.csv", index=False, float_format="%.4f")
    R["faults"] = fl
    log("faults done")
    print(fdf.to_string())

    # ---------- 5. replay evaluation
    rp = te.sample(min(n_replay, len(te)), random_state=SEED + 2)
    base_store = TraceStore()
    base = run_rows(Agent(comp, v1, base_store), rp, sim, seed0=0)       # recorded baseline runs
    log("building regression components")
    comp_small, _ = build(tr, groups, kb, frac_router=0.2, name="r_small")
    comp_stale, _ = build(tr, groups, kb, index_fraction=0.5, name="r_stale")
    comp_nolower, _ = build(tr, groups, kb, lowercase=False, name="r_case")
    cands = {
        "A/A (same build, new seeds)": (comp, v1),
        "router retrained on 20% of data": (comp_small, v1),
        "confidence threshold 0.6 -> 0.3": (comp, replace(v1, conf_threshold=0.3)),
        "gate stops checking priority": (comp, replace(v1, gate_checks_priority=False)),
        "retrieval queries subject only": (comp, replace(v1, retrieval_field="subject")),
        "drafter temperature 0.3 -> 1.0": (comp, replace(v1, temperature=1.0)),
        "retrieval index 50% stale": (comp_stale, v1),
        "label map swaps two queues": (comp, replace(v1, label_swap=("Technical Support", "Product Support"))),
        "tokenizer stops lowercasing": (comp_nolower, v1),
        "drafter drops closing sentence": (comp, replace(v1, drop_closing=True)),
    }
    rep = []
    for name, (cp, v) in cands.items():
        reps = aa_reps if name.startswith("A/A") else reg_reps
        flags = []
        for k in range(reps):
            st = TraceStore()
            cand = run_rows(Agent(cp, replace(v, name=name), st), rp, sim, seed0=10_000 * (k + 1))
            res = compare(base, cand, policy_violations(st, cand, v1))
            flags.append(res)
        fr = pd.DataFrame(flags)
        rep.append({"candidate": name, "reps": reps, "flag_rate": float(fr["flagged"].mean()),
                    "mean_d_route_acc": float(fr["d_route_acc"].mean()), "mean_d_quality": float(fr["d_quality"].mean()),
                    "mean_d_bad_auto": float(fr["d_bad_auto"].mean()), "mean_violations": float(fr["violations"].mean())})
        log(f"replay {name}: flagged {rep[-1]['flag_rate']:.2f}")
    rdf = pd.DataFrame(rep)
    rdf.to_csv(REPORTS / "replay.csv", index=False, float_format="%.4f")
    regs = rdf[~rdf["candidate"].str.startswith("A/A")]
    R["replay"] = {"n_recorded_runs": len(base), "aa_false_alarm_rate": float(rdf.iloc[0]["flag_rate"]),
                   "aa_reps": aa_reps, "regression_reps": reg_reps,
                   "detection_rate_mean": float(regs["flag_rate"].mean()),
                   "regressions_caught_majority": int((regs["flag_rate"] >= 0.5).sum()), "n_regressions": len(regs),
                   "table": rep}
    R["runtime_s"] = time.time() - T0
    (REPORTS / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(rdf.to_string())
    log("done")


if __name__ == "__main__":
    main()
