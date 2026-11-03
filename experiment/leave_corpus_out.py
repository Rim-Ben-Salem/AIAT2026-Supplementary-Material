"""Leave-corpus-out evaluation of the stylometric detector.

Human and machine texts in the corpus differ in provenance, genre and
prompt, so a detector's in-domain accuracy may partly reflect corpus
differences rather than authorship. Here each detector is trained with one
sub-corpus held out entirely, its threshold is set at a 10% false-positive
rate on its own training human texts, and it is then evaluated only on the
held-out sub-corpus and on the TOEFL essays, none of which it has seen:

  hold out college   train: Hewlett + CS224N human vs CS224N GPT-3
                     test:  college human vs college GPT-3
  hold out cs224n    train: Hewlett + college human vs college GPT-3
                     test:  CS224N human vs CS224N GPT-3
  hold out hewlett   train: college + CS224N human vs both GPT-3 sets
                     test:  false-positive rate on Hewlett essays

For each design we report the in-domain cross-validated AUROC, the
held-out AUROC where held-out machine text exists, the false-positive rates
on held-out native texts and on TOEFL essays, and the effect of GPT-4
polishing on the TOEFL essays' scores and flags.

Usage:
  python leave_corpus_out.py --liang-root ChatGPT-Detector-Bias --freq-list en_50k.txt \
      --out results/leave_corpus_out.json
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data import _docs, LIANG_HUMAN, LIANG_AI
from features import Featurizer, FrequencyTable
from run_experiment import make_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liang-root", required=True)
    ap.add_argument("--freq-list", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/leave_corpus_out.json")
    args = ap.parse_args()

    feat = Featurizer(FrequencyTable(args.freq_list))
    root = args.liang_root
    H = {k: feat(_docs(root, v[0])) for k, v in LIANG_HUMAN.items()}
    A = {k: feat(_docs(root, v[0])) for k, v in LIANG_AI.items()}
    toefl, toefl_pol = H["toefl"], feat(_docs(root, LIANG_HUMAN["toefl"][2]))

    designs = {
        "hold_out_college": dict(train_h=["hewlett", "cs224n"], train_a=["cs224n_gpt3"],
                                 test_h="college", test_a="college_gpt3"),
        "hold_out_cs224n": dict(train_h=["hewlett", "college"], train_a=["college_gpt3"],
                                test_h="cs224n", test_a="cs224n_gpt3"),
        "hold_out_hewlett": dict(train_h=["college", "cs224n"], train_a=["college_gpt3", "cs224n_gpt3"],
                                 test_h="hewlett", test_a=None),
    }
    out = {}
    for name, d in designs.items():
        Xh = np.vstack([H[k] for k in d["train_h"]])
        Xa = np.vstack([A[k] for k in d["train_a"]])
        X = np.vstack([Xh, Xa]); y = np.r_[np.zeros(len(Xh)), np.ones(len(Xa))]
        oof = np.empty(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=args.seed).split(X, y):
            oof[te] = make_detector().fit(X[tr], y[tr]).decision_function(X[te])
        det = make_detector().fit(X, y)
        sc_train_h = det.decision_function(Xh)
        thr = np.quantile(sc_train_h, 0.9)
        unit = sc_train_h.std(ddof=1)
        sh = det.decision_function(H[d["test_h"]])
        s0, s1 = det.decision_function(toefl), det.decision_function(toefl_pol)
        res = dict(n_train_human=len(Xh), n_train_machine=len(Xa),
                   cv_auroc_in_domain=float(roc_auc_score(y, oof)),
                   fpr_heldout_native=float((sh > thr).mean()),
                   fpr_toefl=float((s0 > thr).mean()),
                   toefl_flagged_after_polishing=float((s1 > thr).mean()),
                   toefl_mean_score_change_sd=float(((s1 - s0) / unit).mean()),
                   toefl_share_score_up=float((s1 > s0).mean()))
        if d["test_a"] is not None:
            sa = det.decision_function(A[d["test_a"]])
            res["auroc_heldout"] = float(roc_auc_score(np.r_[np.zeros(len(sh)), np.ones(len(sa))],
                                                       np.r_[sh, sa]))
            res["tpr_heldout_machine"] = float((sa > thr).mean())
        out[name] = res
        print(f"{name}: CV AUROC {res['cv_auroc_in_domain']:.3f}"
              + (f", held-out AUROC {res['auroc_heldout']:.3f}, held-out TPR {res['tpr_heldout_machine']:.0%}"
                 if "auroc_heldout" in res else "")
              + f"; FPR held-out native {res['fpr_heldout_native']:.0%}, TOEFL {res['fpr_toefl']:.0%}"
              + f" -> {res['toefl_flagged_after_polishing']:.0%} after polishing"
              + f" (mean change {res['toefl_mean_score_change_sd']:+.2f} sd, {res['toefl_share_score_up']:.0%} up)")
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
