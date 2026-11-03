"""Which fixed threshold does the simulation test, and does Proposition 4.4(2) hold?

In arms_race_sim.py the "fixed_threshold" policy stores the round-0 cut-off
on the logistic regression's decision function (its logit) and reuses that
number after every retraining. Because the predicted probability is the
logistic function of the logit, this is exactly a fixed cut-off on the
retrained model's *predicted probability*. It is not, in general, a fixed
cut-off on the *posterior* probability of machine authorship, which is what
Proposition 4.4(2) assumes: the model's probability equals the equal-prior
posterior only if the model is well specified and trained on balanced
classes. Here the human class is a two-group mixture and machine text has a
smaller variance, so the log-likelihood ratio is not linear, and a training
prior other than 1/2 shifts the logit by log(pi_tr / (1 - pi_tr)).

This script compares three ways of holding a threshold fixed, each with the
cut-off set at round 0 to give a 1% false-positive rate on human text:
  raw      the paper's policy: fixed cut-off on the model's own probability;
  platt    each round the retrained model's logit is recalibrated (Platt
           scaling) on a fresh labelled sample with equal class sizes, and the
           cut-off is fixed on that calibrated probability;
  oracle   evaluation only: the exact equal-prior posterior a/(a+h) of the
           current populations, computed from their Gaussian densities, with a
           cut-off fixed at round 0. It is applied to the populations of the
           platt trajectory and shows what Proposition 4.4(2) guarantees.

Each round it also computes TV(A, H) exactly enough by Monte Carlo
(TV = E_A[(1 - h/a)_+]) and the bound TV (1 - tau) / (2 tau - 1).

With policy "raw" the trajectory is identical to arms_race_sim.run(
policy="fixed_threshold"), which main() checks.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def log_gauss(X, mu, s):
    d = X.shape[1]
    return -0.5 * ((X - mu) ** 2).sum(1) / s ** 2 - d * np.log(s) - 0.5 * d * np.log(2 * np.pi)


def run(seed=0, threshold="raw", q=1.0, alpha=0.5, beta=0.05, rho=0.02, kappa=0.01, eta=0.6,
        m2=1.0, a0=3.0, d=16, T=60, n=2000, p2=0.2, fpr0=0.01, train_machine_frac=None,
        n_tv=20000):
    rng = np.random.default_rng(seed)
    rng_cal = np.random.default_rng([seed, 2])     # extra draws, kept off the main stream
    e = np.zeros(d); e[0] = 1.0
    mu1, mu2 = np.zeros(d), m2 * e
    muA = a0 * e + 0.5 * rng.standard_normal(d) / np.sqrt(d)
    s1 = s2 = 1.0; sA = 0.8
    mu1_0, mu2_0 = mu1.copy(), mu2.copy()
    cum1 = cum2 = 0.0
    thr_fixed = tau_oracle = None
    keys = ["auroc", "tpr", "fpr", "fpr_g1", "fpr_g2", "cum_ff_g1", "cum_ff_g2", "drift_g1", "drift_g2",
            "tv", "cutoff_prob", "oracle_fpr", "oracle_bound", "calib_err"]
    hist = {k: [] for k in keys}

    def humans(k, r=rng):
        k2 = int(k * p2)
        return (mu1 + s1 * r.standard_normal((k - k2, d)), mu2 + s2 * r.standard_normal((k2, d)))

    def machine(k, r=rng):
        return muA + sA * r.standard_normal((k, d))

    def log_h(X):
        return logsumexp(np.c_[np.log(1 - p2) + log_gauss(X, mu1, s1), np.log(p2) + log_gauss(X, mu2, s2)], axis=1)

    def posterior(X):              # equal-prior posterior of machine authorship
        return expit(log_gauss(X, muA, sA) - log_h(X))

    for t in range(T):
        if train_machine_frac is None:
            h1, h2 = humans(n); a = machine(n)
        else:
            frac = train_machine_frac
            if isinstance(frac, (tuple, list)):
                frac = frac[0] + (frac[1] - frac[0]) * t / max(T - 1, 1)
            n_a = int(round(2 * n * frac))
            h1, h2 = humans(2 * n - n_a); a = machine(n_a)
        X = np.vstack([h1, h2, a]); y = np.r_[np.zeros(len(h1) + len(h2)), np.ones(len(a))]
        clf = LogisticRegression(max_iter=500).fit(X, y)
        score = clf.decision_function
        if threshold == "platt":
            ch1, ch2 = humans(n, rng_cal); ca = machine(n, rng_cal)
            z = score(np.vstack([ch1, ch2, ca]))[:, None]
            platt = LogisticRegression(C=1e6, max_iter=500).fit(z, np.r_[np.zeros(n), np.ones(n)])
            score = lambda Z, f=clf.decision_function, p=platt: p.decision_function(f(Z)[:, None])

        c1, c2 = humans(n)
        thr = np.quantile(score(np.vstack([c1, c2])), 1 - fpr0)
        if thr_fixed is None:
            thr_fixed = thr
        thr = thr_fixed                       # held fixed on the chosen scale
        t1, t2 = humans(n); ta = machine(n)
        s_1, s_2, s_a = score(t1), score(t2), score(ta)
        s_h = np.r_[s_1, s_2]
        auroc = roc_auc_score(np.r_[np.zeros(len(s_h)), np.ones(len(s_a))], np.r_[s_h, s_a])
        tpr, fpr = (s_a > thr).mean(), (s_h > thr).mean()
        f1, f2 = (s_1 > thr).mean(), (s_2 > thr).mean()
        cum1 += f1; cum2 += f2
        # calibration against the equal-prior posterior on the evaluation
        # sample: mean absolute difference between the deployed probability
        # and the exact posterior a/(a+h), human and machine texts weighted equally
        Xh_e, Xa_e = np.vstack([t1, t2]), ta
        ce = 0.5 * (np.abs(expit(s_h) - posterior(Xh_e)).mean()
                    + np.abs(expit(s_a) - posterior(Xa_e)).mean())
        w = clf.coef_.ravel(); w_hat = w / np.linalg.norm(w)

        # exact-posterior reference on the current populations (no effect on the dynamics)
        xa = muA + sA * rng_cal.standard_normal((n_tv, d))
        tv = float(np.mean(np.clip(1 - np.exp(log_h(xa) - log_gauss(xa, muA, sA)), 0, None)))
        k2 = int(n_tv * p2)
        xh = np.vstack([mu1 + s1 * rng_cal.standard_normal((n_tv - k2, d)),
                        mu2 + s2 * rng_cal.standard_normal((k2, d))])
        ph = posterior(xh)
        if tau_oracle is None:
            tau_oracle = float(np.quantile(ph, 1 - fpr0))
        o_fpr = float((ph > tau_oracle).mean())
        bound = tv * (1 - tau_oracle) / (2 * tau_oracle - 1) if tau_oracle > 0.5 else np.inf

        for k, v in zip(keys, [auroc, tpr, fpr, f1, f2, cum1, cum2, np.linalg.norm(mu1 - mu1_0),
                               np.linalg.norm(mu2 - mu2_0), tv, float(expit(thr)), o_fpr, bound, ce]):
            hist[k].append(float(v))

        mu_h = (1 - p2) * mu1 + p2 * mu2
        muA = muA - eta * q * tpr * w_hat + rho * (mu_h - muA)
        g1, g2 = f1 / fpr0, f2 / fpr0
        mu1 = mu1 - 0.05 * alpha * g1 * w_hat + kappa * (muA - mu1)
        mu2 = mu2 - 0.05 * alpha * g2 * w_hat + kappa * (muA - mu2)
        s1 = max(0.3, s1 * (1 - 0.1 * beta * g1))
        s2 = max(0.3, s2 * (1 - 0.1 * beta * g2))
    hist["tau_oracle"] = tau_oracle
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="fixed_threshold_check.json")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import arms_race_sim as ars
    ref = ars.run(seed=0, policy="fixed_threshold"); mine = run(seed=0, threshold="raw")
    for k in ["auroc", "tpr", "fpr_g1", "fpr_g2", "cum_ff_g2", "drift_g2"]:
        assert np.allclose(ref[k], mine[k]), k
    print("threshold='raw' reproduces arms_race_sim.run(policy='fixed_threshold') exactly")

    anchored = {**ars.CALIBRATED["r=0.5"]}
    cfgs = {
        "default_raw": dict(threshold="raw"),
        "default_platt": dict(threshold="platt"),
        "default_raw_prior_drift_0.98": dict(threshold="raw", train_machine_frac=(0.5, 0.98)),
        "default_platt_prior_drift_0.98": dict(threshold="platt", train_machine_frac=(0.5, 0.98)),
        "anchored_raw": dict(threshold="raw", **anchored),
        "anchored_platt": dict(threshold="platt", **anchored),
        "anchored_platt_prior_drift_0.98": dict(threshold="platt", train_machine_frac=(0.5, 0.98), **anchored),
    }
    out = {}
    for name, cfg in cfgs.items():
        runs = [run(seed=s, **cfg) for s in range(args.seeds)]
        arr = {k: np.array([r[k] for r in runs]) for k in runs[0] if k != "tau_oracle"}
        viol = arr["oracle_fpr"] > arr["oracle_bound"] + 1e-12
        out[name] = dict(
            mean={k: v.mean(0).tolist() for k, v in arr.items()},
            final_by_seed={k: v[:, -1].tolist() for k, v in arr.items()},
            mean_fpr=float(arr["fpr"].mean()), max_round_fpr=float(arr["fpr"].mean(0).max()),
            cum_ff_g1=float(arr["cum_ff_g1"][:, -1].mean()), cum_ff_g2=float(arr["cum_ff_g2"][:, -1].mean()),
            final_auroc=float(arr["auroc"][:, -1].mean()),
            calib_err_round0=float(arr["calib_err"][:, 0].mean()),
            calib_err_mean=float(arr["calib_err"].mean()),
            calib_err_final=float(arr["calib_err"][:, -1].mean()),
            cutoff_prob_round0=float(arr["cutoff_prob"][:, 0].mean()),
            tau_oracle=float(np.mean([r["tau_oracle"] for r in runs])),
            oracle_mean_fpr=float(arr["oracle_fpr"].mean()),
            oracle_bound_violations=int(viol.sum()), oracle_rounds=int(viol.size),
            share_rounds_platt_fpr_below_bound=float((arr["fpr"] <= arr["oracle_bound"] + 1e-12).mean()))
        r = out[name]
        print(f"{name:34s} mean FPR {r['mean_fpr']:.2%}  max round FPR {r['max_round_fpr']:.2%}  "
              f"flags G1 {r['cum_ff_g1']:.2f} G2 {r['cum_ff_g2']:.2f}  final AUROC {r['final_auroc']:.3f}  "
              f"calibration error {r['calib_err_round0']:.3f}/{r['calib_err_mean']:.3f}/{r['calib_err_final']:.3f}  "
              f"cut-off p {r['cutoff_prob_round0']:.3f}  oracle tau {r['tau_oracle']:.3f} "
              f"oracle FPR {r['oracle_mean_fpr']:.3%} violations {r['oracle_bound_violations']}/{r['oracle_rounds']}  "
              f"deployed FPR<=bound {r['share_rounds_platt_fpr_below_bound']:.0%}", flush=True)
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
