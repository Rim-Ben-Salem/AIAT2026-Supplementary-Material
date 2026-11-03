"""The detection loop with a nonlinear detector and several generators.

Extends the default model of arms_race_sim.py (recalibration to a fixed FPR,
default response rule) in two ways:

  * detector: "logistic" (as in the paper) or "mlp", a one-hidden-layer
    network with tanh units whose score is the output logit. Responses use
    the score's *local* gradient: each human group moves along the mean
    normalised gradient at its flagged texts, and each generator along the
    mean normalised gradient at its caught texts (falling back to all of
    that population's texts when none is flagged). For a linear detector the
    gradient is constant, so this reproduces the paper's model exactly.
  * generators: 1 (as in the paper) or 2. The second generator starts at
    a0 e2 instead of a0 e1, so machine text forms two clusters in different
    directions and the Bayes boundary is no longer close to a hyperplane. Each
    generator evades separately, with its own true-positive rate.

With detector="logistic", n_gen=1 the trajectory is identical to
arms_race_sim.run() with default settings (checked in main()).
"""
import argparse
import json
import os
import sys

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

KEYS = ["auroc", "tpr", "fpr", "fpr_g1", "fpr_g2", "drift_g1", "drift_g2",
        "std_g1", "std_g2", "cum_ff_g1", "cum_ff_g2", "cos_g1_g2", "cos_g2_global"]


class Score:
    def __init__(self, clf):
        self.clf = clf

    def __call__(self, X):
        if isinstance(self.clf, LogisticRegression):
            return self.clf.decision_function(X)
        W1, W2 = self.clf.coefs_; b1, b2 = self.clf.intercepts_
        return (np.tanh(X @ W1 + b1) @ W2 + b2).ravel()

    def grad(self, X):
        if isinstance(self.clf, LogisticRegression):
            return np.tile(self.clf.coef_.ravel(), (len(X), 1))
        W1, W2 = self.clf.coefs_; b1, _ = self.clf.intercepts_
        h = np.tanh(X @ W1 + b1)
        return ((1 - h ** 2) * W2.ravel()) @ W1.T


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def mean_dir(score, X, mask):
    sel = X[mask] if mask.any() else X
    g = score.grad(sel)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return _unit(g.mean(0))


def run(seed=0, detector="logistic", n_gen=1, hidden=32, q=1.0, alpha=0.5, beta=0.05,
        rho=0.02, kappa=0.01, eta=0.6, m2=1.0, a0=3.0, d=16, T=60, n=2000, p2=0.2, fpr0=0.01):
    rng = np.random.default_rng(seed)
    e = np.zeros(d); e[0] = 1.0
    e2 = np.zeros(d); e2[1] = 1.0
    mu1, mu2 = np.zeros(d), m2 * e
    muA = [a0 * e + 0.5 * rng.standard_normal(d) / np.sqrt(d)]
    if n_gen == 2:
        muA.append(a0 * e2 + 0.5 * rng.standard_normal(d) / np.sqrt(d))
    s1 = s2 = 1.0; sA = 0.8
    mu1_0, mu2_0 = mu1.copy(), mu2.copy()
    cum1 = cum2 = 0.0
    hist = {k: [] for k in KEYS}

    def humans(k):
        k2 = int(k * p2)
        return (mu1 + s1 * rng.standard_normal((k - k2, d)),
                mu2 + s2 * rng.standard_normal((k2, d)))

    def machine(k):
        if n_gen == 1:
            return muA[0] + sA * rng.standard_normal((k, d)), np.zeros(k, int)
        comp = np.arange(k) % n_gen
        X = np.vstack([muA[c] for c in comp]) + sA * rng.standard_normal((k, d))
        return X, comp

    for t in range(T):
        h1, h2 = humans(n); a, _ = machine(n)
        X = np.vstack([h1, h2, a]); y = np.r_[np.zeros(len(h1) + len(h2)), np.ones(len(a))]
        if detector == "logistic":
            clf = LogisticRegression(max_iter=500).fit(X, y)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", alpha=1e-3,
                                max_iter=300, early_stopping=True, n_iter_no_change=20,
                                random_state=seed * 1000 + t).fit(X, y)
        sc = Score(clf)
        c1, c2 = humans(n)
        thr = np.quantile(sc(np.vstack([c1, c2])), 1 - fpr0)
        t1, t2 = humans(n); ta, comp = machine(n)
        s_1, s_2, s_a = sc(t1), sc(t2), sc(ta)
        s_h = np.r_[s_1, s_2]
        auroc = roc_auc_score(np.r_[np.zeros(len(s_h)), np.ones(len(s_a))], np.r_[s_h, s_a])
        tpr, fpr = (s_a > thr).mean(), (s_h > thr).mean()
        f1, f2 = (s_1 > thr).mean(), (s_2 > thr).mean()
        cum1 += f1; cum2 += f2
        u1 = mean_dir(sc, t1, s_1 > thr); u2 = mean_dir(sc, t2, s_2 > thr)
        if detector == "logistic":
            w = clf.coef_.ravel(); w_glob = w / np.linalg.norm(w)
        else:
            w_glob = mean_dir(sc, np.vstack([t1, t2, ta]), np.zeros(len(t1) + len(t2) + len(ta), bool))
        vals = [auroc, tpr, fpr, f1, f2, np.linalg.norm(mu1 - mu1_0), np.linalg.norm(mu2 - mu2_0),
                s1, s2, cum1, cum2, float(u1 @ u2), float(u2 @ w_glob)]
        for k, v in zip(KEYS, vals):
            hist[k].append(float(v))

        mu_h = (1 - p2) * mu1 + p2 * mu2
        for c in range(len(muA)):
            m = comp == c
            tpr_c = (s_a[m] > thr).mean()
            u_c = mean_dir(sc, ta[m], s_a[m] > thr)
            muA[c] = muA[c] - eta * q * tpr_c * u_c + rho * (mu_h - muA[c])
        muA_bar = np.mean(muA, 0)
        g1, g2 = f1 / fpr0, f2 / fpr0
        mu1 = mu1 - 0.05 * alpha * g1 * u1 + kappa * (muA_bar - mu1)
        mu2 = mu2 - 0.05 * alpha * g2 * u2 + kappa * (muA_bar - mu2)
        s1 = max(0.3, s1 * (1 - 0.1 * beta * g1))
        s2 = max(0.3, s2 * (1 - 0.1 * beta * g2))
    return hist


CONFIGS = {
    "logistic_1gen": dict(detector="logistic", n_gen=1),
    "mlp_1gen": dict(detector="mlp", n_gen=1),
    "logistic_2gen": dict(detector="logistic", n_gen=2),
    "mlp_2gen": dict(detector="mlp", n_gen=2),
    "logistic_1gen_noadapt": dict(detector="logistic", n_gen=1, alpha=0.0, beta=0.0),
    "mlp_1gen_noadapt": dict(detector="mlp", n_gen=1, alpha=0.0, beta=0.0),
    "logistic_2gen_noadapt": dict(detector="logistic", n_gen=2, alpha=0.0, beta=0.0),
    "mlp_2gen_noadapt": dict(detector="mlp", n_gen=2, alpha=0.0, beta=0.0),
    "mlp_1gen_static": dict(detector="mlp", n_gen=1, q=0.0, alpha=0.0, beta=0.0, rho=0.0, kappa=0.0),
    "mlp_2gen_static": dict(detector="mlp", n_gen=2, q=0.0, alpha=0.0, beta=0.0, rho=0.0, kappa=0.0),
}


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default=".")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="nonlinear.json")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.sim_dir))
    import arms_race_sim
    # check: logistic, one generator reproduces the paper's default model
    ref = arms_race_sim.run(seed=0)
    mine = run(seed=0)
    for k in ["auroc", "tpr", "fpr_g1", "fpr_g2", "drift_g2", "cum_ff_g2"]:
        assert np.allclose(ref[k], mine[k]), k
    print("logistic / one generator reproduces arms_race_sim.run() exactly")

    cfgs = CONFIGS if args.only is None else {k: CONFIGS[k] for k in args.only.split(",")}
    out = {}
    for name, cfg in cfgs.items():
        runs = [run(seed=s, **cfg) for s in range(args.seeds)]
        out[name] = {k: (np.mean([r[k] for r in runs], 0).tolist(), np.std([r[k] for r in runs], 0).tolist())
                     for k in KEYS}
        out[name]["final_by_seed"] = {k: [r[k][-1] for r in runs] for k in KEYS}
        out[name]["round0_by_seed"] = {k: [r[k][0] for r in runs] for k in KEYS}
        out[name]["below075_by_seed"] = [next((t for t, v in enumerate(r["auroc"]) if v < 0.75), None) for r in runs]
        f = out[name]["final_by_seed"]
        print(f"{name:24s} AUROC0 {np.mean(out[name]['round0_by_seed']['auroc']):.3f} "
              f"final {np.mean(f['auroc']):.3f}  drift G1 {np.mean(f['drift_g1']):.2f} G2 {np.mean(f['drift_g2']):.2f}  "
              f"flags G1 {np.mean(f['cum_ff_g1']):.2f} G2 {np.mean(f['cum_ff_g2']):.2f}  "
              f"disp0 {np.mean(np.array(out[name]['round0_by_seed']['fpr_g2'])/np.maximum(out[name]['round0_by_seed']['fpr_g1'],1e-9)):.1f}",
              flush=True)
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
