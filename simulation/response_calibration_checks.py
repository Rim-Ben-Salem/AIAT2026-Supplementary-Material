"""Alternative response functions and imperfect recalibration; writes
response_calibration_checks.json.

1. Response functions. The main model maps each group's pressure
   g = FPR_g / target FPR linearly to its avoidance and homogenisation steps.
   Here the mapping is a threshold (a group responds only once g reaches a
   given multiple of the operating point), a saturating function
   (resp_cap * tanh(g / resp_cap)), or a linear response scaled down for
   partial awareness (only a quarter of flagged writers learn of the flag).

2. Imperfect recalibration. Under the fixed-FPR policy the threshold is set
   on a calibration sample of human texts. Here that sample is small (200 or
   500 texts instead of 2,000), unrepresentative (G2 is 5% or 40% of it
   instead of 20%) or stale (drawn from the populations 3 or 10 rounds
   earlier).

For each configuration we report final AUROC, the realised mean pooled FPR,
cumulative expected false flags per writer and style drift by group, the
final group false-positive rates (whether flags have passed from G2 to G1),
and the G2-minus-G1 difference in false flags with a paired 95% t-interval
over seeds.

Usage: python response_calibration_checks.py [--seeds 20]
"""
import argparse
import json

import numpy as np
from scipy.stats import t as tdist

import arms_race_sim as A


def ci(x):
    x = np.asarray(x, float)
    h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def summarise(seeds, **cfg):
    runs = [A.run(seed=s, **cfg) for s in range(seeds)]
    fin = lambda k: [r[k][-1] for r in runs]
    return dict(
        auroc=float(np.mean(fin("auroc"))),
        mean_fpr=float(np.mean([np.mean(r["fpr"]) for r in runs])),
        cum_ff_g1=float(np.mean(fin("cum_ff_g1"))), cum_ff_g2=float(np.mean(fin("cum_ff_g2"))),
        drift_g1=float(np.mean(fin("drift_g1"))), drift_g2=float(np.mean(fin("drift_g2"))),
        final_fpr_g1=float(np.mean(fin("fpr_g1"))), final_fpr_g2=float(np.mean(fin("fpr_g2"))),
        g2_minus_g1_flags=ci(np.array(fin("cum_ff_g2")) - np.array(fin("cum_ff_g1"))),
        g2_minus_g1_drift=ci(np.array(fin("drift_g2")) - np.array(fin("drift_g1"))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="response_calibration_checks.json")
    args = ap.parse_args()
    anch = A.CALIBRATED["r=1.0"]
    cfgs = {
        "responses": {
            "default_linear": dict(),
            "threshold_2x": dict(response_fn="threshold", resp_threshold=2.0),
            "threshold_5x": dict(response_fn="threshold", resp_threshold=5.0),
            "saturating_cap2": dict(response_fn="saturating", resp_cap=2.0),
            "saturating_cap1": dict(response_fn="saturating", resp_cap=1.0),
            "awareness_25pct": dict(alpha=0.25 * A.DEFAULTS["alpha"], beta=0.25 * A.DEFAULTS["beta"]),
            "anchored_linear": dict(anch),
            "anchored_threshold_2x": {**anch, "response_fn": "threshold", "resp_threshold": 2.0},
            "anchored_saturating_cap2": {**anch, "response_fn": "saturating", "resp_cap": 2.0},
        },
        "calibration": {
            "default_full": dict(),
            "cal_n_500": dict(cal_n=500),
            "cal_n_200": dict(cal_n=200),
            "cal_g2_share_0.05": dict(cal_p2=0.05),
            "cal_g2_share_0.40": dict(cal_p2=0.40),
            "cal_stale_3": dict(cal_stale=3),
            "cal_stale_10": dict(cal_stale=10),
            "anchored_full": dict(anch),
            "anchored_cal_n_200": {**anch, "cal_n": 200},
            "anchored_cal_stale_3": {**anch, "cal_stale": 3},
        },
    }
    out = {}
    for group, cs in cfgs.items():
        out[group] = {}
        print(f"\n{group}")
        for name, cfg in cs.items():
            r = summarise(args.seeds, **cfg)
            out[group][name] = r
            print(f"  {name:26s} AUROC {r['auroc']:.2f}  mean FPR {100*r['mean_fpr']:.2f}%  "
                  f"flags G1 {r['cum_ff_g1']:.2f} G2 {r['cum_ff_g2']:.2f} "
                  f"(G2-G1 {r['g2_minus_g1_flags'][0]:+.2f} [{r['g2_minus_g1_flags'][1]:+.2f}, {r['g2_minus_g1_flags'][2]:+.2f}])  "
                  f"drift G1 {r['drift_g1']:.2f} G2 {r['drift_g2']:.2f}  "
                  f"final FPR G1 {100*r['final_fpr_g1']:.2f}% G2 {100*r['final_fpr_g2']:.2f}%", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
