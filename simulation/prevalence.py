"""Withdrawal under a misspecified or estimated prevalence of machine text.

Before withdrawal, a detector under the withdrawal policy follows exactly the
same trajectory as under recalibration to a fixed FPR (the audit draws no
random numbers when it uses the full evaluation sample), and after
withdrawal it flags nothing. The withdrawal round under any assumed
prevalence can therefore be computed from fixed-FPR trajectories, and the
fixed-FPR trajectory is also the counterfactual "kept in use".

For each seed and round we use TPR_t, FPR_t and the group rates. For an
assumed prevalence pi_hat and a true prevalence pi:
  W(pi_hat)  first round with PPV(pi_hat) < 1/2 (withdrawal round)
  W*(pi)     the same under the true prevalence (oracle withdrawal)
  excess     rounds in use while the true PPV < 1/2
  premature  rounds withdrawn while the true PPV would still be >= 1/2
  extra G2 false flags relative to oracle withdrawal.

Estimated prevalence: each round the operator sees N submissions with true
prevalence pi, counts flags, audits TPR and FPR on m labelled texts per
class, and estimates pi by adjusted classify-and-count,
  pi_hat = (flag rate - FPR_hat) / (TPR_hat - FPR_hat),
either as a point estimate or as a lower 5% bound from a parametric bootstrap.

Usage: python prevalence.py --sim-dir ../code/simulation --out prevalence.json
"""
import argparse
import json
import os
import sys

import numpy as np


def ppv(pi, tpr, fpr):
    den = pi * tpr + (1 - pi) * fpr
    return np.where(den > 0, pi * tpr / np.maximum(den, 1e-300), 0.0)


def withdrawal_round(pi_hat_t, tpr, fpr):
    below = ppv(pi_hat_t, tpr, fpr) < 0.5
    return int(np.argmax(below)) if below.any() else len(tpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default=".")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--N", type=int, default=2000, help="submissions per round")
    ap.add_argument("--audit", type=int, default=200, help="labelled texts per class")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default="prevalence.json")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.sim_dir))
    from arms_race_sim import run, CALIBRATED

    settings = {"default": {}, "anchored_r0.5": CALIBRATED["r=0.5"]}
    grid = [0.02, 0.05, 0.1, 0.2, 0.3]
    out = {}
    for sname, cfg in settings.items():
        traj = [run(seed=s, **cfg) for s in range(args.seeds)]
        tpr = np.array([r["tpr"] for r in traj]); fpr = np.array([r["fpr"] for r in traj])
        f2 = np.array([r["fpr_g2"] for r in traj]); f1 = np.array([r["fpr_g1"] for r in traj])
        T = tpr.shape[1]
        res = dict(misspecified={}, estimated={})
        for pi in grid:
            oracle = [withdrawal_round(np.full(T, pi), tpr[s], fpr[s]) for s in range(args.seeds)]
            for ph in grid:
                W = [withdrawal_round(np.full(T, ph), tpr[s], fpr[s]) for s in range(args.seeds)]
                true_bad = ppv(pi, tpr, fpr) < 0.5
                excess = [int(true_bad[s, :W[s]].sum()) for s in range(args.seeds)]
                prem = [int((~true_bad[s, W[s]:]).sum()) for s in range(args.seeds)]
                xg2 = [float(f2[s, :W[s]].sum() - f2[s, :oracle[s]].sum()) for s in range(args.seeds)]
                xg1 = [float(f1[s, :W[s]].sum() - f1[s, :oracle[s]].sum()) for s in range(args.seeds)]
                minppv = [float(ppv(pi, tpr[s, :W[s]], fpr[s, :W[s]]).min()) if W[s] > 0 else None
                          for s in range(args.seeds)]
                res["misspecified"][f"pi={pi},pi_hat={ph}"] = dict(
                    rounds_deployed=float(np.mean(W)), oracle_rounds=float(np.mean(oracle)),
                    excess_rounds=float(np.mean(excess)), premature_rounds=float(np.mean(prem)),
                    extra_flags_g2=float(np.mean(xg2)), extra_flags_g1=float(np.mean(xg1)),
                    min_true_ppv_in_use=float(np.mean([m for m in minppv if m is not None])),
                    bound_min_true_ppv=float(pi * (1 - ph) / (pi * (1 - ph) + ph * (1 - pi))))
        # estimated prevalence (adjusted classify-and-count)
        rng = np.random.default_rng(12345)
        for pi in [0.05, 0.1, 0.2]:
            for mode in ["point", "lower5"]:
                Ws, exc, xg2, bad_est = [], [], [], []
                for s in range(args.seeds):
                    oracle = withdrawal_round(np.full(T, pi), tpr[s], fpr[s])
                    W = T
                    for t in range(T):
                        k = rng.binomial(args.N, pi * tpr[s, t] + (1 - pi) * fpr[s, t])
                        a_t = rng.binomial(args.audit, tpr[s, t]) / args.audit
                        a_f = rng.binomial(args.audit, fpr[s, t]) / args.audit
                        def acc(kk, tt, ff):
                            return np.clip((kk / args.N - ff) / np.maximum(tt - ff, 1e-3), 1e-3, 1.0)
                        est = float(acc(k, a_t, a_f))
                        if mode == "lower5":
                            kb = rng.binomial(args.N, k / args.N, args.boot)
                            tb = rng.binomial(args.audit, a_t, args.boot) / args.audit
                            fb = rng.binomial(args.audit, a_f, args.boot) / args.audit
                            est = float(np.percentile(acc(kb, tb, fb), 5))
                        # the audit itself decides: PPV at the estimated prevalence
                        p_ = est * a_t / (est * a_t + (1 - est) * a_f) if est * a_t + (1 - est) * a_f > 0 else 0.0
                        if p_ < 0.5:
                            W = t; break
                    true_bad = ppv(pi, tpr[s], fpr[s]) < 0.5
                    Ws.append(W); exc.append(int(true_bad[:W].sum()))
                    xg2.append(float(f2[s, :W].sum() - f2[s, :oracle].sum()))
                res["estimated"][f"pi={pi},{mode}"] = dict(
                    rounds_deployed=float(np.mean(Ws)),
                    oracle_rounds=float(np.mean([withdrawal_round(np.full(T, pi), tpr[s], fpr[s])
                                                 for s in range(args.seeds)])),
                    excess_rounds=float(np.mean(exc)), extra_flags_g2=float(np.mean(xg2)))
        out[sname] = res
    json.dump(out, open(args.out, "w"), indent=2)
    for sname, res in out.items():
        print(f"\n== {sname}: misspecified prevalence (rows: true pi; cols: assumed pi_hat) ==")
        print("rounds deployed / excess rounds / extra G2 flags / min true PPV in use (bound)")
        for pi in grid:
            cells = []
            for ph in grid:
                r = res["misspecified"][f"pi={pi},pi_hat={ph}"]
                cells.append(f"{r['rounds_deployed']:4.1f}/{r['excess_rounds']:4.1f}/{r['extra_flags_g2']:+.2f}/"
                             f"{r['min_true_ppv_in_use']:.2f}({r['bound_min_true_ppv']:.2f})")
            print(f"pi={pi:<5}", "  ".join(cells))
        print("estimated:")
        for k, r in res["estimated"].items():
            print(f"  {k:16s} deployed {r['rounds_deployed']:.1f} (oracle {r['oracle_rounds']:.1f}), "
                  f"excess {r['excess_rounds']:.1f}, extra G2 flags {r['extra_flags_g2']:+.2f}")


if __name__ == "__main__":
    main()
