"""
FPR cap, fixed alert budget and confidence-bound withdrawal (Table 10 and
the paragraphs "Caps and alert budgets" and "Confidence-bound audits" in
Section 5.6).

For each configuration, writes the per-round means and standard deviations
of every metric (as in the other result files) and, for each seed, the number
of rounds the detector was deployed, the precision of a flag at 10%
prevalence in the round of withdrawal, and the number of rounds in use while
that precision was below 1/2. Prints the values reported in the paper.

    python policy_revision_runs.py --seeds 20 --out policy_revision.json
"""
import argparse
import json

import numpy as np

import arms_race_sim as m

A = m.CALIBRATED["r=0.5"]
CB = dict(policy="withdraw", withdraw_rule="cb")
CONFIGS = {
    "default_cap": dict(policy="cap"),
    "default_budget": dict(policy="budget"),
    "default_withdraw": dict(policy="withdraw"),
    "default_cb_200_200": dict(CB, audit_nh=200, audit_na=200),
    "default_cb_2000_200": dict(CB, audit_nh=2000, audit_na=200),
    "default_cb_2000_200_simul": dict(CB, audit_nh=2000, audit_na=200, audit_delta=0.05 / 60),
    "anchored_cap": {**A, "policy": "cap"},
    "anchored_budget": {**A, "policy": "budget"},
    "anchored_withdraw": {**A, "policy": "withdraw"},
    "anchored_cb_200_200": {**A, **CB, "audit_nh": 200, "audit_na": 200},
    "anchored_cb_2000_200": {**A, **CB, "audit_nh": 2000, "audit_na": 200},
    "anchored_cb_2000_200_simul": {**A, **CB, "audit_nh": 2000, "audit_na": 200,
                                   "audit_delta": 0.05 / 60},
}


def summarise(runs):
    out = {k: (np.mean([r[k] for r in runs], 0).tolist(),
               np.std([r[k] for r in runs], 0).tolist()) for k in m.KEYS}
    out["final_by_seed"] = {k: [r[k][-1] for r in runs] for k in m.KEYS}
    dep, ppv_w, bad = [], [], []
    for r in runs:
        dep.append(sum(r["deployed"]))
        if 0.0 in r["deployed"]:
            ppv_w.append(r["ppv_true"][r["deployed"].index(0.0)])
        bad.append(sum(1 for d, v in zip(r["deployed"], r["ppv_true"]) if d == 1.0 and v < 0.5))
    out["rounds_deployed_by_seed"] = dep
    out["ppv_at_withdrawal_by_seed"] = ppv_w
    out["rounds_in_use_ppv_below_half_by_seed"] = bad
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="policy_revision.json")
    args = ap.parse_args()

    res = {k: summarise([m.run(seed=s, **c) for s in range(args.seeds)])
           for k, c in CONFIGS.items()}
    json.dump(res, open(args.out, "w"))

    print(f"{'configuration':28s} AUROC deployed (sd) meanFPR  driftG1 driftG2  ffG1  ffG2  "
          "PPV@withdrawal  rounds-in-use-PPV<0.5")
    for k, v in res.items():
        fin = v["final_by_seed"]
        dep = v["rounds_deployed_by_seed"]
        pw = v["ppv_at_withdrawal_by_seed"]
        print(f"{k:28s} {np.mean(fin['auroc']):.2f}  {np.mean(dep):5.2f} ({np.std(dep):.1f}) "
              f"{np.mean(v['fpr'][0]) * 100:5.2f}%  {np.mean(fin['drift_g1']):.2f}    "
              f"{np.mean(fin['drift_g2']):.2f}    {np.mean(fin['cum_ff_g1']):.2f}  "
              f"{np.mean(fin['cum_ff_g2']):.2f}  "
              f"{(np.mean(pw) if pw else float('nan')):.2f}            "
              f"{np.mean(v['rounds_in_use_ppv_below_half_by_seed']):.1f}")
