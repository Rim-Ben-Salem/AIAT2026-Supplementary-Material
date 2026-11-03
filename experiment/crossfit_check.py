"""Sensitivity of the cross-fitted deployed detector to pooled thresholds.

The deployed detector in run_experiment.py pools the out-of-fold decision
scores of five fold models and sets one threshold at the target rate of the
pooled human scores. The fold models are fitted separately, so their score
scales need not coincide. This script repeats round 0 of the reported runs
(same corpus, features, seeds and folds) and compares the pooled threshold
with fold-specific thresholds, each set at the target rate of that fold's own
out-of-fold human scores. It reports the spread of fold score scales, the
agreement between the two flag sets, the group false-positive rates, and
how many flagged non-native essays remain flagged after GPT-4 polishing.

Usage:
  python crossfit_check.py --liang-root ChatGPT-Detector-Bias \
      --freq-list FrequencyWords/content/2018/en/en_50k.txt --out results/crossfit_check.json
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liang-root", required=True)
    ap.add_argument("--freq-list", required=True)
    ap.add_argument("--target-fpr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/crossfit_check.json")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_liang
    from features import Featurizer, FrequencyTable
    from run_experiment import CrossFittedDetector

    feat = Featurizer(FrequencyTable(args.freq_list))
    out = {}
    for name, native in [("all_native", ("hewlett", "college", "cs224n")), ("hewlett_native", ("hewlett",))]:
        corpus = load_liang(args.liang_root, native=native).reset_index(drop=True)
        X = feat(corpus.text); y = corpus.label.to_numpy(); g = corpus.group.to_numpy()
        # round 0 of run_experiment.py uses seed + 0 for the folds
        det = CrossFittedDetector(X, y, g, args.target_fpr, args.seed)
        oof, fold = det.oof, det.fold_of
        human = y == 0
        thr_f = np.array([np.quantile(oof[human & (fold == k)], 1 - args.target_fpr) for k in range(5)])
        flags_pooled = det.flags
        flags_fold = oof > thr_f[fold]
        nn = np.where(g == "nonnative")[0]
        pol = feat(corpus.published_rewrite.iloc[nn])
        s_pol = det.score(pol, nn)
        res = dict(
            fold_human_score_mean=[float(oof[human & (fold == k)].mean()) for k in range(5)],
            fold_human_score_sd=[float(oof[human & (fold == k)].std(ddof=1)) for k in range(5)],
            pooled_threshold=float(det.thr), fold_thresholds=thr_f.tolist(),
            flag_agreement=float((flags_pooled == flags_fold).mean()),
            human_flags_changed=int((flags_pooled != flags_fold)[human].sum()), n_human=int(human.sum()))
        for lab, fl, thr_nn in [("pooled", flags_pooled, np.full(len(nn), det.thr)),
                                ("fold_specific", flags_fold, thr_f[fold[nn]])]:
            flagged_nn = fl[nn]
            res[lab] = dict(
                fpr_native=float(fl[g == "native"].mean()), fpr_nonnative=float(fl[g == "nonnative"].mean()),
                pooled_human_fpr=float(fl[human].mean()), tpr=float(fl[y == 1].mean()),
                nonnative_flagged=int(flagged_nn.sum()),
                still_flagged_after_polishing=int((s_pol[flagged_nn] > thr_nn[flagged_nn]).sum()))
        out[name] = res
        print(f"{name}: fold human-score s.d. {np.round(res['fold_human_score_sd'], 2)}, "
              f"thresholds pooled {res['pooled_threshold']:.2f} vs folds {np.round(thr_f, 2)}; "
              f"agreement {res['flag_agreement']:.1%} ({res['human_flags_changed']} of {res['n_human']} human flags change)")
        for lab in ["pooled", "fold_specific"]:
            r = res[lab]
            print(f"   {lab:13s} FPR native {r['fpr_native']:.1%} non-native {r['fpr_nonnative']:.1%} "
                  f"pooled {r['pooled_human_fpr']:.1%} TPR {r['tpr']:.1%}; still flagged after polishing "
                  f"{r['still_flagged_after_polishing']}/{r['nonnative_flagged']}")
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
