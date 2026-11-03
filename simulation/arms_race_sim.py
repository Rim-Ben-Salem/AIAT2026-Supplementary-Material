"""
Detector-generator-human co-evolution simulation.

Texts are points in a d-dimensional style space. Three populations:
  - Human writers: a majority group G1 and a minority group G2 whose baseline
    style lies closer to machine text along one axis.
  - Generator: the distribution of machine-generated text.
  - Detector: logistic regression, retrained every round.

Each round t:
  1. Draw a training sample (humans and machine text) and fit the detector.
  2. Draw a separate calibration sample of human text and set the threshold
     at its (1 - target_fpr) quantile.
  3. Draw an independent evaluation sample and measure AUROC, TPR, pooled
     FPR and per-group FPR at that threshold.
  4. Under the withdrawal policy, audit the detector on a fresh labelled
     sample and withdraw it once it fails the audit rule.
  5. Generator: evasion against the detector direction (scaled by query
     access q and by how often it is caught) and imitation of human text.
  6. Humans: avoidance of the flagged region and narrowing of style, both
     scaled by each group's FPR relative to the operating point, plus
     assimilation toward machine style.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

DEFAULTS = dict(q=1.0, alpha=0.5, beta=0.05, rho=0.02, kappa=0.01, eta=0.6,
                m2=1.0, a0=3.0, d=16, T=60, n=2000, p_g2=0.2, target_fpr=0.01,
                feedback="sample",
                # human-response variants (defaults reproduce the main model)
                response="detector",   # "detector" | "noisy" | "misdirected"
                dir_noise=0.0,         # noisy: scale of random error in direction
                misdirect_deg=0.0,     # misdirected: angle to the deployed direction
                avoid_budget=None,     # bounded: total avoidance displacement per group
                delay=0,               # delayed: rounds between a flag and the response
                resp_g2=1.0,           # heterogeneous: G2 response relative to G1
                open_loop=False,       # pressure and direction frozen at round 0
                response_fn="linear",  # pressure-to-response function: "linear" | "threshold" | "saturating"
                resp_threshold=2.0,    # threshold: respond only when FPR_g >= resp_threshold * target FPR
                resp_cap=2.0,          # saturating: pressure capped smoothly at resp_cap * target FPR
                efficacy_halflife=None,  # rewriting efficacy halves every h rounds
                # deployment policy
                policy="fixed_fpr",    # "fixed_fpr" | "fixed_threshold" | "withdraw"
                withdraw_prevalence=0.1,  # assumed prevalence, or (start, end) for a drift
                withdraw_rule="precision",  # "precision" | "auroc"
                withdraw_auroc=0.75,   # AUROC rule: withdraw below this audited AUROC
                audit_n=None,          # labelled texts per class in the audit (None: full sample)
                audit_source="independent",  # "independent": fresh audit sample each round
                                             # "evaluation": audit reuses the evaluation sample
                evasion="tpr",         # "tpr": evasion step scales with TPR (main model)
                                       # "constant": step scales with evasion_rate instead
                evasion_rate=None,     # TPR-equivalent rate for evasion="constant"
                train_machine_frac=None,  # machine share of training data, or (start, end)
                # recalibration sample (fixed-FPR policy)
                cal_n=None,            # human texts in the calibration sample (None: n)
                cal_p2=None,           # G2 share of the calibration sample (None: p_g2)
                cal_stale=0)           # calibration sample drawn from the populations k rounds earlier
KEYS = ["auroc", "tpr", "fpr", "fpr_g1", "fpr_g2", "drift_g1", "drift_g2",
        "std_g1", "std_g2", "cum_ff_g1", "cum_ff_g2", "deployed"]


def _unit(v):
    return v / np.linalg.norm(v)


def run(seed=0, **overrides):
    """One simulated trajectory.

    Order of operations in each round: (1) fit the detector on a training
    sample; (2) set the threshold on a calibration sample of human text;
    (3) evaluate on an independent sample; (4) generator update (evasion,
    then imitation of the current human mean); (5) human update (avoidance,
    then assimilation toward the updated machine mean); (6) homogenisation.
    """
    p = {**DEFAULTS, **overrides}
    if p["feedback"] == "analytic" and p["policy"] == "fixed_threshold":
        # The analytic branch solves a new threshold at the target pooled FPR
        # each round, which would silently turn a fixed threshold back into
        # recalibration. No reported configuration combines the two; the
        # analytic-feedback robustness check (Section S1.3) concerns
        # recalibration to a fixed FPR only.
        raise ValueError("feedback='analytic' is implemented for fixed-FPR deployment only")
    d, n, p2, fpr0 = p["d"], p["n"], p["p_g2"], p["target_fpr"]
    rng = np.random.default_rng(seed)
    # randomness used only by the response variants, kept separate so that
    # the default model draws exactly the same numbers as before
    rng_resp = np.random.default_rng([seed, 1])
    # independent audit samples use their own stream, so that the trajectory
    # before withdrawal is unchanged by the choice of audit source
    rng_audit = np.random.default_rng([seed, 2])
    e = np.zeros(d); e[0] = 1.0
    mu1, mu2 = np.zeros(d), p["m2"] * e
    muA = p["a0"] * e + 0.5 * rng.standard_normal(d) / np.sqrt(d)
    s1 = s2 = 1.0
    sA = 0.8
    mu1_0, mu2_0 = mu1.copy(), mu2.copy()
    cum1 = cum2 = 0.0
    used1 = used2 = 0.0
    hist = {k: [] for k in KEYS}
    past = []            # (g1, g2, w_hat) observed each round, for delayed responses
    pop_hist = []        # (mu1, mu2, s1, s2) at the start of each round, for stale calibration
    frozen = None        # (g1, g2, w_hat) at round 0, for the open-loop control
    thr_fixed = None
    withdrawn = False
    r_mis = rng_resp.standard_normal(d) if p["response"] == "misdirected" else None

    def humans(k):
        k2 = int(k * p2)
        return (mu1 + s1 * rng.standard_normal((k - k2, d)),
                mu2 + s2 * rng.standard_normal((k2, d)))

    def machine(k):
        return muA + sA * rng.standard_normal((k, d))

    def response_direction(w_hat):
        if p["response"] == "noisy":
            xi = rng_resp.standard_normal(d) / np.sqrt(d)
            return _unit(w_hat + p["dir_noise"] * xi)
        if p["response"] == "misdirected":
            v = _unit(r_mis - (r_mis @ w_hat) * w_hat)
            phi = np.deg2rad(p["misdirect_deg"])
            return np.cos(phi) * w_hat + np.sin(phi) * v
        return w_hat

    def schedule(v, t):
        if isinstance(v, (tuple, list)):
            return v[0] + (v[1] - v[0]) * t / max(p["T"] - 1, 1)
        return v

    for t in range(p["T"]):
        if p["train_machine_frac"] is None:
            h1, h2 = humans(n); a = machine(n)
        else:
            # training prior differs from 1/2, possibly drifting over rounds
            n_a = int(round(2 * n * schedule(p["train_machine_frac"], t)))
            h1, h2 = humans(2 * n - n_a); a = machine(n_a)
        X = np.vstack([h1, h2, a])
        y = np.r_[np.zeros(len(h1) + len(h2)), np.ones(len(a))]
        clf = LogisticRegression(max_iter=500).fit(X, y)

        pop_hist.append((mu1.copy(), mu2.copy(), s1, s2))
        if p["cal_n"] is None and p["cal_p2"] is None and p["cal_stale"] == 0:
            c1, c2 = humans(n)
        else:
            kc = n if p["cal_n"] is None else p["cal_n"]
            kc2 = int(kc * (p2 if p["cal_p2"] is None else p["cal_p2"]))
            m1c, m2c, s1c, s2c = pop_hist[max(0, t - p["cal_stale"])]
            c1 = m1c + s1c * rng.standard_normal((kc - kc2, d))
            c2 = m2c + s2c * rng.standard_normal((kc2, d))
        thr = np.quantile(clf.decision_function(np.vstack([c1, c2])), 1 - fpr0)
        if p["policy"] == "fixed_threshold":
            # the detector's probability cut-off is set once, at round 0, and kept
            if thr_fixed is None:
                thr_fixed = thr
            thr = thr_fixed

        t1, t2 = humans(n); ta = machine(n)
        s_1, s_2, s_a = (clf.decision_function(z) for z in (t1, t2, ta))
        s_h = np.r_[s_1, s_2]
        auroc = roc_auc_score(np.r_[np.zeros(len(s_h)), np.ones(len(s_a))], np.r_[s_h, s_a])
        tpr, fpr = (s_a > thr).mean(), (s_h > thr).mean()
        f1, f2 = (s_1 > thr).mean(), (s_2 > thr).mean()
        w = clf.coef_.ravel(); w_hat = w / np.linalg.norm(w)

        if p["feedback"] == "analytic":
            # Replace sample estimates by exact rates under the current
            # Gaussian populations: a linear score w.x + b of N(mu, s^2 I) is
            # N(w.mu + b, s^2 |w|^2). The threshold solves the pooled human
            # rate exactly, so the feedback carries no sampling noise.
            b, wn = clf.intercept_[0], np.linalg.norm(w)
            m1_, m2_, mA_ = w @ mu1 + b, w @ mu2 + b, w @ muA + b
            pooled = lambda t_: ((1 - p2) * norm.sf((t_ - m1_) / (s1 * wn))
                                 + p2 * norm.sf((t_ - m2_) / (s2 * wn)) - fpr0)
            thr = brentq(pooled, min(m1_, m2_) - 50 * wn, max(m1_, m2_) + 50 * wn)
            f1 = norm.sf((thr - m1_) / (s1 * wn))
            f2 = norm.sf((thr - m2_) / (s2 * wn))
            tpr = norm.sf((thr - mA_) / (sA * wn))
            fpr = (1 - p2) * f1 + p2 * f2

        if p["policy"] == "withdraw" and not withdrawn:
            # audit on a labelled sample: withdraw the detector, permanently,
            # once it fails the audit rule
            a_tpr, a_fpr, a_auc = tpr, fpr, auroc
            if p["audit_source"] == "independent":
                # fresh labelled texts from the current populations, scored by
                # the deployed detector at the deployed threshold
                k = n if p["audit_n"] is None else p["audit_n"]
                k2 = int(k * p2)
                ah = np.vstack([mu1 + s1 * rng_audit.standard_normal((k - k2, d)),
                                mu2 + s2 * rng_audit.standard_normal((k2, d))])
                aa = muA + sA * rng_audit.standard_normal((k, d))
                sh_a, sa_a = clf.decision_function(ah), clf.decision_function(aa)
                a_tpr, a_fpr = (sa_a > thr).mean(), (sh_a > thr).mean()
                a_auc = roc_auc_score(np.r_[np.zeros(len(sh_a)), np.ones(len(sa_a))],
                                      np.r_[sh_a, sa_a])
            elif p["audit_n"] is not None:
                ih = rng_resp.choice(len(s_h), p["audit_n"], replace=False)
                ia = rng_resp.choice(len(s_a), p["audit_n"], replace=False)
                a_tpr, a_fpr = (s_a[ia] > thr).mean(), (s_h[ih] > thr).mean()
                a_auc = roc_auc_score(np.r_[np.zeros(len(ih)), np.ones(len(ia))],
                                      np.r_[s_h[ih], s_a[ia]])
            if p["withdraw_rule"] == "auroc":
                withdrawn = a_auc < p["withdraw_auroc"]
            else:
                # a flag is more likely false than true at the assumed prevalence
                pi = schedule(p["withdraw_prevalence"], t)
                ppv = pi * a_tpr / (pi * a_tpr + (1 - pi) * a_fpr) if a_tpr + a_fpr > 0 else 0.0
                withdrawn = ppv < 0.5
        if withdrawn:
            tpr = fpr = f1 = f2 = 0.0
        cum1 += f1; cum2 += f2

        vals = [auroc, tpr, fpr, f1, f2, np.linalg.norm(mu1 - mu1_0),
                np.linalg.norm(mu2 - mu2_0), s1, s2, cum1, cum2, 0.0 if withdrawn else 1.0]
        for k, v in zip(KEYS, vals):
            hist[k].append(float(v))

        decay = 1.0 if p["efficacy_halflife"] is None else 0.5 ** (t / p["efficacy_halflife"])
        mu_h = (1 - p2) * mu1 + p2 * mu2
        ev = tpr if p["evasion"] == "tpr" else (0.0 if withdrawn else p["evasion_rate"])
        muA = muA - decay * p["eta"] * p["q"] * ev * w_hat + p["rho"] * (mu_h - muA)

        # pressure on each group is its FPR relative to the operating point
        g1, g2 = f1 / fpr0, f2 / fpr0
        if p["response_fn"] == "threshold":
            g1, g2 = (g if g >= p["resp_threshold"] else 0.0 for g in (g1, g2))
        elif p["response_fn"] == "saturating":
            g1, g2 = (p["resp_cap"] * np.tanh(g / p["resp_cap"]) for g in (g1, g2))
        past.append((g1, g2, w_hat))
        if p["open_loop"]:
            if frozen is None:
                frozen = past[0]
            g1, g2, w_resp = frozen
        elif p["delay"] > 0:
            if t >= p["delay"]:
                g1, g2, w_resp = past[t - p["delay"]]
            else:
                g1, g2, w_resp = 0.0, 0.0, w_hat
        else:
            w_resp = w_hat
        u = response_direction(w_resp)
        step1 = 0.05 * p["alpha"] * g1 * decay
        step2 = 0.05 * p["alpha"] * g2 * decay * p["resp_g2"]
        if p["avoid_budget"] is not None:
            step1 = min(step1, max(0.0, p["avoid_budget"] - used1))
            step2 = min(step2, max(0.0, p["avoid_budget"] - used2))
            used1 += step1; used2 += step2
        mu1 = mu1 - step1 * u + p["kappa"] * (muA - mu1)
        mu2 = mu2 - step2 * u + p["kappa"] * (muA - mu2)
        s1 = max(0.3, s1 * (1 - 0.1 * p["beta"] * g1))
        s2 = max(0.3, s2 * (1 - 0.1 * p["beta"] * g2 * p["resp_g2"]))
    return hist


def multi(seeds, **kw):
    runs = [run(seed=s, **kw) for s in range(seeds)]
    out = {k: (np.mean([r[k] for r in runs], 0).tolist(),
               np.std([r[k] for r in runs], 0).tolist()) for k in KEYS}
    # per-seed values after the last round, for paired comparisons
    out["final_by_seed"] = {k: [r[k][-1] for r in runs] for k in KEYS}
    out["fpr_ratio0_by_seed"] = [r["fpr_g2"][0] / r["fpr_g1"][0] if r["fpr_g1"][0] > 0 else None
                                 for r in runs]
    return out


MAIN = {
    "baseline": dict(),
    "q05": dict(q=0.5),
    "q0": dict(q=0.0),
    "no_human_adapt": dict(alpha=0.0, beta=0.0),
    "no_imitation": dict(rho=0.0),
    "no_assimilation": dict(kappa=0.0),
    "static": dict(q=0.0, alpha=0.0, beta=0.0, rho=0.0, kappa=0.0),
}

SENSITIVITY = {
    "alpha": [0.25, 1.0],
    "eta": [0.3, 1.2],
    "rho": [0.05],
    "kappa": [0.03],
    "m2": [0.5, 1.5],
    "a0": [2.0, 4.0],
    "target_fpr": [0.001, 0.05],
}

SWEEP = {"alpha": [0.0, 0.25, 0.5, 1.0], "eta": [0.15, 0.3, 0.6, 1.2]}

# Literature-anchored parameters. Under an equal-variance binormal score model,
# a change in flag rate from p to p' corresponds to a shift of
# Phi^-1(1-p) - Phi^-1(1-p') score standard deviations, whatever the threshold.
#   Liang et al. (2023): non-native 61.22% vs native 5.19% flagged -> m2 = 1.91;
#     GPT-4 polishing of non-native essays, 61.22% -> 11.77% -> delta_h = 1.47.
#   Krishna et al. (2023): DIPPER, 70.3% -> 4.6% detected at 1% FPR -> delta_A = 2.22;
#     70.3% detected at 1% FPR is matched by a0 = 4.0 in this model.
# Evasion: every caught machine text is paraphrased once per round, so the mean
# shift is TPR * delta_A, i.e. eta = delta_A with q = 1.
# Avoidance: a fraction r of flagged writers rewrite once per round, so the mean
# shift is r * FPR_g * delta_h, i.e. alpha_h = r * delta_h * target_fpr / 0.05.
DELTA_H, DELTA_A = 1.472, 2.218
CALIBRATED_BASE = dict(m2=1.912, a0=4.0, eta=DELTA_A, q=1.0)


def calibrated_alpha(r, target_fpr=0.01):
    return r * DELTA_H * target_fpr / 0.05


CALIBRATED = {f"r={r}": {**CALIBRATED_BASE, "alpha": calibrated_alpha(r),
                         **({"beta": 0.0} if r == 0 else {})}
              for r in [0.0, 0.25, 0.5, 1.0]}
CALIBRATED["static"] = {**CALIBRATED_BASE, "q": 0.0, "alpha": 0.0, "beta": 0.0,
                        "rho": 0.0, "kappa": 0.0}

# Robustness of the feedback to sampling noise: exact (analytic) group rates in
# place of sample estimates, and a five-fold larger evaluation sample.
ROBUSTNESS = {
    "default_analytic": dict(feedback="analytic"),
    "default_n10000": dict(n=10000),
    "anchored_r0_analytic": {**CALIBRATED["r=0.0"], "feedback": "analytic"},
    "anchored_r1_analytic": {**CALIBRATED["r=1.0"], "feedback": "analytic"},
}

# ---- Extensions ---------------------------------------------------------
# Ablations that separate the human mechanisms, and an open-loop control in
# which each group's pressure and the response direction stay at their
# round-0 values, so that only feedback-free consequences of the response
# rules remain.
ABLATIONS = {
    "no_avoidance": dict(alpha=0.0),
    "no_homogenisation": dict(beta=0.0),
    "no_human_adapt": dict(alpha=0.0, beta=0.0),
    "no_human_change": dict(alpha=0.0, beta=0.0, kappa=0.0),
    "open_loop": dict(open_loop=True),
}

# Less idealised human responses (default parameters otherwise).
RESPONSES = {
    "noisy_1": dict(response="noisy", dir_noise=1.0),
    "noisy_3": dict(response="noisy", dir_noise=3.0),
    "misdirected_45": dict(response="misdirected", misdirect_deg=45.0),
    "misdirected_90": dict(response="misdirected", misdirect_deg=90.0),
    "misdirected_135": dict(response="misdirected", misdirect_deg=135.0),
    "bounded_0.5": dict(avoid_budget=0.5),
    "bounded_1.0": dict(avoid_budget=1.0),
    "delayed_3": dict(delay=3),
    "delayed_10": dict(delay=10),
    "g2_half_response": dict(resp_g2=0.5),
    "g2_double_response": dict(resp_g2=2.0),
    "anchored_r1_misdirected_90": {**CALIBRATED["r=1.0"], "response": "misdirected", "misdirect_deg": 90.0},
    "anchored_r1_misdirected_135": {**CALIBRATED["r=1.0"], "response": "misdirected", "misdirect_deg": 135.0},
}

# Deployment policies other than recalibrating to a fixed FPR every round.
POLICIES = {
    "default_fixed_threshold": dict(policy="fixed_threshold"),
    "default_withdraw": dict(policy="withdraw"),
    "anchored_r0.5_fixed_threshold": {**CALIBRATED["r=0.5"], "policy": "fixed_threshold"},
    "anchored_r0.5_withdraw": {**CALIBRATED["r=0.5"], "policy": "withdraw"},
}


# Robustness of the policy comparison: a fixed threshold when the training
# prior is not 1/2 or drifts, and withdrawal under other rules, a small audit
# sample, other assumed prevalences and a drifting prevalence.
POLICY_ROBUSTNESS = {
    "fixed_threshold_prior_0.8": dict(policy="fixed_threshold", train_machine_frac=0.8),
    "fixed_threshold_prior_drift": dict(policy="fixed_threshold", train_machine_frac=(0.5, 0.8)),
    "fixed_fpr_prior_drift": dict(train_machine_frac=(0.5, 0.8)),
    "fixed_threshold_prior_drift_0.98": dict(policy="fixed_threshold", train_machine_frac=(0.5, 0.98)),
    "withdraw_pi_0.05": dict(policy="withdraw", withdraw_prevalence=0.05),
    "withdraw_pi_0.25": dict(policy="withdraw", withdraw_prevalence=0.25),
    "withdraw_pi_drift": dict(policy="withdraw", withdraw_prevalence=(0.05, 0.3)),
    "withdraw_auroc_0.75": dict(policy="withdraw", withdraw_rule="auroc"),
    "withdraw_audit_200": dict(policy="withdraw", audit_n=200),
    "anchored_fixed_threshold_prior_drift": {**CALIBRATED["r=0.5"], "policy": "fixed_threshold",
                                             "train_machine_frac": (0.5, 0.8)},
    "anchored_fixed_threshold_prior_drift_0.98": {**CALIBRATED["r=0.5"], "policy": "fixed_threshold",
                                                  "train_machine_frac": (0.5, 0.98)},
    "anchored_withdraw_audit_200": {**CALIBRATED["r=0.5"], "policy": "withdraw", "audit_n": 200},
    "anchored_withdraw_auroc_0.75": {**CALIBRATED["r=0.5"], "policy": "withdraw",
                                     "withdraw_rule": "auroc"},
}


def anchor_configs(path="anchors.json"):
    """Anchors from each commercial detector (anchors.py), plus variants of
    the Krishna-based anchors: the machine-score shift expressed in machine
    rather than human standard deviations (machine style s.d. is 0.8), and
    rewriting efficacy that decays over time."""
    cfgs = {}
    try:
        per_detector = json.load(open(path))
    except FileNotFoundError:
        per_detector = {}
    for det, a in per_detector.items():
        for r in [0.0, 1.0]:
            cfg = dict(m2=a["m2"], a0=4.0, eta=a["delta_A"], q=1.0,
                       alpha=r * a["delta_h"] * 0.01 / 0.05)
            if r == 0.0:
                cfg["beta"] = 0.0
            cfgs[f"{det}_r={r}"] = cfg
    cfgs["krishna_machine_sd_r=0.5"] = {**CALIBRATED["r=0.5"], "eta": 0.8 * DELTA_A}
    cfgs["krishna_halflife10_r=0.5"] = {**CALIBRATED["r=0.5"], "efficacy_halflife": 10}
    cfgs["krishna_halflife30_r=0.5"] = {**CALIBRATED["r=0.5"], "efficacy_halflife": 30}
    return cfgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sens-seeds", type=int, default=10)
    ap.add_argument("--sweep-only", action="store_true",
                    help="run only the two-dimensional sweep")
    ap.add_argument("--calibrated-only", action="store_true",
                    help="run only the literature-anchored configurations")
    ap.add_argument("--robustness-only", action="store_true",
                    help="run only the feedback-noise robustness configurations")
    ap.add_argument("--policy-robustness-only", action="store_true",
                    help="run only the policy-robustness configurations")
    ap.add_argument("--extensions-only", action="store_true",
                    help="run only the ablation, response, policy and anchor extensions")
    args = ap.parse_args()

    if args.policy_robustness_only:
        pr = {k: multi(args.seeds, **c) for k, c in POLICY_ROBUSTNESS.items()}
        json.dump(pr, open("policy_robustness.json", "w"))
        raise SystemExit

    extensions = {
        "ablations": {k: multi(args.seeds, **c) for k, c in ABLATIONS.items()},
        "responses": {k: multi(args.seeds, **c) for k, c in RESPONSES.items()},
        "policies": {k: multi(args.seeds, **c) for k, c in POLICIES.items()},
        "anchors": {k: multi(args.seeds, **c) for k, c in anchor_configs().items()},
    }
    json.dump(extensions, open("extensions.json", "w"))
    if args.extensions_only:
        raise SystemExit

    pr = {k: multi(args.seeds, **c) for k, c in POLICY_ROBUSTNESS.items()}
    json.dump(pr, open("policy_robustness.json", "w"))

    robustness = {name: multi(args.seeds, **cfg) for name, cfg in ROBUSTNESS.items()}
    json.dump(robustness, open("robustness.json", "w"))
    if args.robustness_only:
        raise SystemExit

    calibrated = {name: multi(args.seeds, **cfg) for name, cfg in CALIBRATED.items()}
    json.dump(calibrated, open("calibrated.json", "w"))
    if args.calibrated_only:
        raise SystemExit

    sweep = {}
    for a in SWEEP["alpha"]:
        for e in SWEEP["eta"]:
            kw = dict(alpha=a, eta=e)
            if a == 0.0:
                kw["beta"] = 0.0
            sweep[f"alpha={a},eta={e}"] = multi(args.sens_seeds, **kw)
    json.dump(sweep, open("sweep.json", "w"))
    if args.sweep_only:
        raise SystemExit

    main = {name: multi(args.seeds, **cfg) for name, cfg in MAIN.items()}
    json.dump(main, open("results.json", "w"))

    sens = {}
    for param, values in SENSITIVITY.items():
        for v in values:
            sens[f"{param}={v}"] = multi(args.sens_seeds, **{param: v})
    json.dump(sens, open("sensitivity.json", "w"))
