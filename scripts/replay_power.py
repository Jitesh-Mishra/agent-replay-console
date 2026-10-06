"""Why did replay miss 5 regressions at 200 runs? Re-run them on 1,000 recorded tickets and compute,
from the observed paired differences, how many replay runs 80% power would need.
python scripts/replay_power.py  ->  reports/replay_power.csv
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc.agent import Agent, TraceStore, Version  # noqa: E402
from arc.components import SEED, AnswerSim, load_kb, load_tickets, near_dup_groups, split  # noqa: E402
from arc.evaluate import build, compare, policy_violations, run_rows  # noqa: E402

T0 = time.time()
df = load_tickets()
groups = pd.Series(near_dup_groups(df), index=df["ticket_id"])
tr, te = split(df)
kb = load_kb()
sim = AnswerSim(tr["answer"])
comp, _ = build(tr, groups, kb)
v1 = Version()
cands = {
    "A/A (same build, new seeds)": (comp, v1),
    "router retrained on 20% of data": (build(tr, groups, kb, frac_router=0.2, name="p_small")[0], v1),
    "drafter temperature 0.3 -> 1.0": (comp, replace(v1, temperature=1.0)),
    "retrieval index 50% stale": (build(tr, groups, kb, index_fraction=0.5, name="p_stale")[0], v1),
    "tokenizer stops lowercasing": (build(tr, groups, kb, lowercase=False, name="p_case")[0], v1),
    "drafter drops closing sentence": (comp, replace(v1, drop_closing=True)),
}
print(f"[{time.time()-T0:.0f}s] built", flush=True)
N, REPS = 1000, 10
rp = te.sample(N, random_state=SEED + 2)
base = run_rows(Agent(comp, v1, TraceStore()), rp, sim, seed0=0)
rows = []
for name, (cp, v) in cands.items():
    flags, d_q, d_r = [], [], []
    for k in range(REPS):
        st = TraceStore()
        cand = run_rows(Agent(cp, replace(v, name=name), st), rp, sim, seed0=10_000 * (k + 1))
        flags.append(compare(base, cand, policy_violations(st, cand, v1))["flagged"])
        d_q.append(cand["quality"].to_numpy() - base["quality"].to_numpy())
        d_r.append(cand["route_correct"].to_numpy().astype(int) - base["route_correct"].to_numpy().astype(int))
    dq, dr = np.concatenate(d_q), np.concatenate(d_r)
    z = stats.norm.ppf(1 - 0.05 / 3 / 2) + stats.norm.ppf(0.8)
    n_q = int(np.ceil((z * dq.std(ddof=1) / abs(dq.mean())) ** 2)) if abs(dq.mean()) > 1e-9 else None
    n_r = int(np.ceil((z * dr.std(ddof=1) / abs(dr.mean())) ** 2)) if abs(dr.mean()) > 1e-9 else None
    rows.append({"candidate": name, "replay_runs": N, "reps": REPS, "flag_rate": float(np.mean(flags)),
                 "mean_d_quality": float(dq.mean()), "sd_d_quality": float(dq.std(ddof=1)),
                 "mean_d_route_acc": float(dr.mean()),
                 "runs_for_80pct_power_quality": n_q, "runs_for_80pct_power_routing": n_r})
    print(f"[{time.time()-T0:.0f}s] {rows[-1]}", flush=True)


def vs_aa(out: pd.DataFrame) -> pd.DataFrame:
    """Every candidate is compared with ONE recorded baseline, whose own sampling draw shifts all comparisons
    by the same offset (visible in the A/A row). Effect = candidate shift minus the A/A shift."""
    aa = out.iloc[0]
    z = stats.norm.ppf(1 - 0.05 / 3 / 2) + stats.norm.ppf(0.8)
    out["effect_quality_vs_aa"] = out["mean_d_quality"] - aa["mean_d_quality"]
    out["runs_needed_quality_vs_aa"] = [
        int(np.ceil((z * sd / abs(e)) ** 2)) if abs(e) > 5e-4 and i > 0 else None
        for i, (sd, e) in enumerate(zip(out["sd_d_quality"], out["effect_quality_vs_aa"]))]
    return out


if __name__ == "__main__":
    vs_aa(pd.DataFrame(rows)).to_csv(Path(__file__).resolve().parents[1] / "reports" / "replay_power.csv",
                                     index=False, float_format="%.5g")
