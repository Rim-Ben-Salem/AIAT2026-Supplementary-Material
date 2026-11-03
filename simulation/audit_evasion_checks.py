"""Audit-sample and evasion-rule checks; writes audit_evasion_checks.json.

1. Audit sample. The withdrawal audit draws a fresh labelled sample each round
   (audit_source="independent") from a separate random stream, so that the
   trajectory before withdrawal does not depend on the audit. Every withdrawal
   configuration is also run with an audit that reuses the round's evaluation
   sample (audit_source="evaluation"), in which the decision to withdraw and
   the reported rates share their random draws. We report the distribution of
   withdrawal timing across seeds and the residual burden under both.

2. Constant evasion step. In the main model the evasion step scales with the
   true-positive rate, which builds in part of the effect of Proposition
   4.3(1b) on the adversary. Here the step is held at a constant
   TPR-equivalent rate c, equal to the mean true-positive rate over rounds and
   seeds of the matching no-response run, and the effect of human adaptation
   on final AUROC is compared with the TPR-scaled rule.

Usage: python audit_evasion_checks.py [--seeds 20]
"""
import argparse
import json

import numpy as np
from scipy.stats import t as tdist

import arms_race_sim as A


def per_seed(seeds, **cfg):
    runs = [A.run(seed=s, **cfg) for s in range(seeds)]
    return dict(
        rounds_deployed=[float(sum(r["deployed"])) for r in runs],
        mean_fpr=[float(np.mean(r["fpr"])) for r in runs],
        mean_tpr=[float(np.mean(r["tpr"])) for r in runs],
        auroc=[r["auroc"][-1] for r in runs],
        cum_ff_g1=[r["cum_ff_g1"][-1] for r in runs],
        cum_ff_g2=[r["cum_ff_g2"][-1] for r in runs],
        drift_g2=[r["drift_g2"][-1] for r in runs],
    )


def ci(x):
    x = np.asarray(x, float)
    h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def summary(r):
    return {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                    min=float(np.min(v)), max=float(np.max(v))) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="audit_evasion_checks.json")
    args = ap.parse_args()
    S = args.seeds
    anch = A.CALIBRATED["r=0.5"]
    out = {"independent_audit": {}, "constant_evasion": {}}

    # 1. independent audit
    withdraw_cfgs = {
        "default_pi0.1": dict(policy="withdraw"),
        "default_pi0.05": dict(policy="withdraw", withdraw_prevalence=0.05),
        "default_pi0.25": dict(policy="withdraw", withdraw_prevalence=0.25),
        "default_pi_drift": dict(policy="withdraw", withdraw_prevalence=(0.05, 0.3)),
        "default_auroc0.75": dict(policy="withdraw", withdraw_rule="auroc"),
        "default_audit200": dict(policy="withdraw", audit_n=200),
        "anchored_pi0.1": {**anch, "policy": "withdraw"},
        "anchored_audit200": {**anch, "policy": "withdraw", "audit_n": 200},
        "anchored_auroc0.75": {**anch, "policy": "withdraw", "withdraw_rule": "auroc"},
    }
    for name, cfg in withdraw_cfgs.items():
        ev = per_seed(S, **cfg, audit_source="evaluation")
        ind = per_seed(S, **cfg, audit_source="independent")
        out["independent_audit"][name] = dict(
            evaluation_audit=summary(ev), independent_audit=summary(ind),
            paired_diff={k: ci(np.array(ind[k]) - np.array(ev[k]))
                         for k in ["rounds_deployed", "cum_ff_g1", "cum_ff_g2", "mean_fpr"]})
        print(f"{name:20s} rounds {np.mean(ev['rounds_deployed']):5.1f} -> {np.mean(ind['rounds_deployed']):5.1f} "
              f"(sd {np.std(ind['rounds_deployed'], ddof=1):.1f}, range {min(ind['rounds_deployed']):.0f}-"
              f"{max(ind['rounds_deployed']):.0f}) | G2 flags {np.mean(ev['cum_ff_g2']):.3f} -> "
              f"{np.mean(ind['cum_ff_g2']):.3f} | G1 {np.mean(ev['cum_ff_g1']):.3f} -> {np.mean(ind['cum_ff_g1']):.3f}")

    # 2. constant evasion step
    settings = {
        "anchored": dict(none={**A.CALIBRATED["r=0.0"]}, response={**A.CALIBRATED["r=1.0"]}),
        "default": dict(none=dict(alpha=0.0, beta=0.0), response=dict()),
    }
    for sname, cfgs in settings.items():
        tpr_none = per_seed(S, **cfgs["none"])
        tpr_resp = per_seed(S, **cfgs["response"])
        res = dict(tpr_scaled=dict(
            auroc_none=ci(tpr_none["auroc"]), auroc_response=ci(tpr_resp["auroc"]),
            diff=ci(np.array(tpr_resp["auroc"]) - np.array(tpr_none["auroc"]))))
        for ref in ["none", "response"]:
            c = float(np.mean((tpr_none if ref == "none" else tpr_resp)["mean_tpr"]))
            cn = per_seed(S, **cfgs["none"], evasion="constant", evasion_rate=c)
            cr = per_seed(S, **cfgs["response"], evasion="constant", evasion_rate=c)
            res[f"constant_c_from_{ref}"] = dict(
                c=c, auroc_none=ci(cn["auroc"]), auroc_response=ci(cr["auroc"]),
                diff=ci(np.array(cr["auroc"]) - np.array(cn["auroc"])),
                flags_g2_none=ci(cn["cum_ff_g2"]), flags_g2_response=ci(cr["cum_ff_g2"]),
                flags_g1_none=ci(cn["cum_ff_g1"]), flags_g1_response=ci(cr["cum_ff_g1"]))
        out["constant_evasion"][sname] = res
        print(f"\n{sname}: response minus no response, final AUROC")
        for k, v in res.items():
            extra = f" (c = {v['c']:.3f})" if "c" in v else ""
            print(f"  {k:24s}{extra}: none {v['auroc_none'][0]:.3f}, response {v['auroc_response'][0]:.3f}, "
                  f"diff {v['diff'][0]:+.3f} [{v['diff'][1]:+.3f}, {v['diff'][2]:+.3f}]")
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
