"""Real-text check of the detection loop.

One round consists of:
  1. Fit a detector to human and machine texts and evaluate it on held-out
     data over repeated random splits (AUROC, TPR, pooled and per-group FPR).
  2. Deploy a cross-fitted detector: five fold models, each text scored by
     the model not trained on it, and a threshold at the target pooled human
     false-positive rate of these out-of-fold scores. Flag texts with it.
  3. Rewrite flagged human texts (a fraction --response-rate of them) and,
     unless --no-evasion, caught machine texts.
  4. Measure how far rewritten texts move in the deployed detector's
     standardized feature space and whether the same deployed detector still
     flags them.
  5. Replace the texts and start the next round, so the detector is retrained
     on the rewritten corpus. Each later round also reports the previous
     round's deployed detector, frozen, on the new corpus, separating the
     effect of the rewrites from the effect of retraining.

Held-out intervals are the 2.5th and 97.5th percentiles across repeated
random splits. Splits reuse the same texts, so these are split-variability
intervals, not confidence intervals for population performance.

Example (published rewrites from Liang et al., stylometric detector):
  python run_experiment.py --liang-root ChatGPT-Detector-Bias --freq-list en_50k.txt
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_csv, load_liang
from features import Featurizer, FrequencyTable, LMScorer
from rewriters import APIRewriter, HFRewriter, PublishedRewriter


def make_detector():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def human_groups(groups):
    return sorted(set(groups) - {"ai"})


def evaluate(X, y, groups, target_fpr, n_splits, seed):
    """Held-out metrics over repeated train / calibration / evaluation splits."""
    rng = np.random.RandomState(seed)
    hg = human_groups(groups)
    keys = ["auroc", "tpr", "fpr"] + [f"fpr_{g}" for g in hg] + [f"auroc_{g}" for g in hg]
    out = {k: [] for k in keys}
    idx = np.arange(len(y))
    for _ in range(n_splits):
        s = rng.randint(1 << 30)
        tr, rest = train_test_split(idx, test_size=0.5, stratify=groups, random_state=s)
        cal, ev = train_test_split(rest, test_size=0.5, stratify=groups[rest], random_state=s + 1)
        det = make_detector().fit(X[tr], y[tr])
        cal_h = cal[y[cal] == 0]
        thr = np.quantile(det.decision_function(X[cal_h]), 1 - target_fpr)
        sc = det.decision_function(X[ev])
        yv, gv = y[ev], groups[ev]
        flag = sc > thr
        out["auroc"].append(roc_auc_score(yv, sc))
        out["tpr"].append(flag[yv == 1].mean())
        out["fpr"].append(flag[yv == 0].mean())
        for g in hg:
            m = gv == g
            out[f"fpr_{g}"].append(flag[m].mean())
            sel = m | (yv == 1)
            out[f"auroc_{g}"].append(roc_auc_score(yv[sel], sc[sel]))
    return {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                    hi=float(np.percentile(v, 97.5))) for k, v in out.items()}


class CrossFittedDetector:
    """Deployed detector built by 5-fold cross-fitting.

    Each text is scored only by the fold model that was not trained on it.
    The threshold is set at the target quantile of these out-of-fold scores
    on human texts. Flag selection, the movement of rewritten texts and the
    frozen comparison in the next round all use these same fold models and
    this same threshold, so every deployed decision about a text comes from a
    model that never saw that text (or the version it was rewritten from).
    """

    def __init__(self, X, y, groups, target_fpr, seed, n_folds=5):
        self.fold_of = np.zeros(len(y), dtype=int)
        self.models = []
        folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, groups)
        for k, (tr, te) in enumerate(folds):
            self.models.append(make_detector().fit(X[tr], y[tr]))
            self.fold_of[te] = k
        self.oof = self.score(X, np.arange(len(y)))
        self.thr = float(np.quantile(self.oof[y == 0], 1 - target_fpr))
        self.flags = self.oof > self.thr

    def score(self, X, idx):
        """Scores rows X, where row r is the current version of text idx[r]."""
        out = np.empty(len(idx))
        for k, m in enumerate(self.models):
            sel = self.fold_of[idx] == k
            if sel.any():
                out[sel] = m.decision_function(X[sel])
        return out


def unit_direction(model):
    w = model[-1].coef_.ravel()
    return w / np.linalg.norm(w)


def movement(det, idx, X_before, X_after):
    """How rewritten texts move relative to the fold model that scores them."""
    proj, dist = np.empty(len(idx)), np.empty(len(idx))
    for r, i in enumerate(idx):
        m = det.models[det.fold_of[i]]
        zb, za = m[0].transform(X_before[r:r + 1]), m[0].transform(X_after[r:r + 1])
        proj[r] = float(((za - zb) @ unit_direction(m))[0])
        dist[r] = float(np.linalg.norm(za - zb))
    sb, sa = det.score(X_before, idx), det.score(X_after, idx)
    return dict(n=len(idx),
                mean_delta_projection=float(proj.mean()),
                mean_distance=float(dist.mean()),
                mean_delta_score=float((sa - sb).mean()),
                share_still_flagged=float((sa > det.thr).mean()))


def frozen_metrics(det, X, y, groups):
    """The previous round's deployed detector, unchanged, on the current corpus."""
    flag = det.score(X, np.arange(len(y))) > det.thr
    res = dict(tpr=float(flag[y == 1].mean()), fpr=float(flag[y == 0].mean()))
    for g in human_groups(groups):
        res[f"fpr_{g}"] = float(flag[groups == g].mean())
    return res


def drift(X, groups, ref_scaler, ref_means):
    Z = ref_scaler.transform(X)
    return {g: float(np.linalg.norm(Z[groups == g].mean(0) - ref_means[g]))
            for g in ref_means}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--liang-root", help="clone of Weixin-Liang/ChatGPT-Detector-Bias")
    src.add_argument("--csv", help="corpus CSV with columns text, label, group")
    ap.add_argument("--native", default="hewlett,college,cs224n",
                    help="native sub-corpora from the Liang data (comma-separated)")
    ap.add_argument("--freq-list", help="word frequency list; omit to use the wordfreq package")
    ap.add_argument("--lm-model", help="causal LM for log-probability features, e.g. gpt2")
    ap.add_argument("--rewriter", choices=["published", "hf", "api"], default="published")
    ap.add_argument("--rewrite-model", help="model name for the hf or api rewriter")
    ap.add_argument("--api-base", help="base URL of an OpenAI-compatible endpoint")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--target-fpr", type=float, default=0.10,
                    help="pooled human false-positive rate; small corpora need a rate well above 1%%")
    ap.add_argument("--response-rate", type=float, default=1.0,
                    help="fraction of flagged human texts that are rewritten")
    ap.add_argument("--no-evasion", action="store_true", help="do not rewrite caught machine texts")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiment_results.json")
    args = ap.parse_args()

    if args.liang_root:
        corpus = load_liang(args.liang_root, native=tuple(args.native.split(",")))
    else:
        corpus = load_csv(args.csv)
    corpus = corpus.reset_index(drop=True)

    if args.rewriter == "published":
        rewriter = PublishedRewriter(corpus)
        if args.rounds > 1:
            print("The published rewriter has one rewrite per text; using --rounds 1.")
            args.rounds = 1
    elif args.rewriter == "hf":
        rewriter = HFRewriter(args.rewrite_model)
    else:
        rewriter = APIRewriter(args.rewrite_model, base_url=args.api_base)

    lm = LMScorer(args.lm_model) if args.lm_model else None
    feat = Featurizer(FrequencyTable(args.freq_list), lm)
    rng = np.random.RandomState(args.seed)
    y = corpus.label.to_numpy()
    groups = corpus.group.to_numpy()

    X0 = feat(corpus.text)
    ref_scaler = StandardScaler().fit(X0)
    Z0 = ref_scaler.transform(X0)
    ref_means = {g: Z0[groups == g].mean(0) for g in sorted(set(groups))}

    results = dict(config=vars(args), features=feat.names,
                   interval_note=("lo and hi are the 2.5th and 97.5th percentiles across random "
                                  "splits: split-variability intervals, not confidence intervals"),
                   corpus={g: int((groups == g).sum()) for g in sorted(set(groups))},
                   rounds=[])
    prev = None
    for t in range(args.rounds + 1):
        X = feat(corpus.text)
        rec = dict(round=t, heldout=evaluate(X, y, groups, args.target_fpr, args.splits, args.seed + t),
                   drift_from_round0=drift(X, groups, ref_scaler, ref_means))
        if prev is not None:
            rec["previous_detector_frozen"] = frozen_metrics(prev, X, y, groups)
        det = CrossFittedDetector(X, y, groups, args.target_fpr, args.seed + t)
        flags = det.flags
        rec["deployed_threshold"] = det.thr
        rec["deployed_pooled_human_fpr"] = float(flags[y == 0].mean())
        rec["deployed"] = frozen_metrics(det, X, y, groups)
        if t < args.rounds:
            prev = det
            moves, rec["flagged"] = {}, {}
            for g in sorted(set(groups)):
                kind = "ai" if g == "ai" else "human"
                if kind == "ai" and args.no_evasion:
                    continue
                cand = np.where(flags & (groups == g))[0]
                if kind == "human":
                    cand = cand[rng.random_sample(len(cand)) < args.response_rate]
                new = rewriter(list(corpus.text.iloc[cand]), kind)
                done = [(i, r) for i, r in zip(cand, new) if r]
                rec["flagged"][g] = dict(flagged=int((flags & (groups == g)).sum()),
                                         selected=int(len(cand)), rewritten=len(done))
                if done:
                    ii = [i for i, _ in done]
                    moves[g] = movement(det, np.array(ii), X[ii], feat([r for _, r in done]))
                    corpus.loc[ii, "text"] = [r for _, r in done]
            rec["movement"] = moves
        results["rounds"].append(rec)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    report(results)


def report(res):
    print(f"Corpus: {res['corpus']}   target pooled FPR: {res['config']['target_fpr']:.0%}")
    for rec in res["rounds"]:
        h = rec["heldout"]
        print(f"\nRound {rec['round']} (retrained, held-out: mean [95% split-variability interval])")
        for k, v in h.items():
            print(f"  {k:18s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        print("  drift from round 0: " + ", ".join(f"{g} {d:.2f}" for g, d in rec["drift_from_round0"].items()))
        if "deployed" in rec:
            print("  deployed (cross-fitted) detector: " + ", ".join(
                f"{k} {v:.3f}" for k, v in rec["deployed"].items()))
        if "previous_detector_frozen" in rec:
            print("  previous detector, frozen: " + ", ".join(
                f"{k} {v:.3f}" for k, v in rec["previous_detector_frozen"].items()))
        for g, f in rec.get("flagged", {}).items():
            print(f"  {g}: flagged {f['flagged']}, selected {f['selected']}, rewritten {f['rewritten']}")
        for g, m in rec.get("movement", {}).items():
            print(f"  move {g}: n {m['n']}, d(projection) {m['mean_delta_projection']:+.2f}, "
                  f"distance {m['mean_distance']:.2f}, d(score) {m['mean_delta_score']:+.2f}, "
                  f"still flagged {m['share_still_flagged']:.0%}")


if __name__ == "__main__":
    main()
