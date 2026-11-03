"""Separating rewrite effects from genre and prompt effects in the real-text check.

The direction of the polishing effect is measured within essays (each TOEFL
essay before and after GPT-4 polishing), so it is not confounded by
differences between essays. What genre and prompt differences can confound
is the direction a stylometric detector learns. This script therefore:

  1. Describes what polishing does to the 15 features, without any detector
     (paired change per feature, in units of the TOEFL essays' s.d.).
  2. Trains stylometric detectors on corpora that never contain a TOEFL essay,
     including prompt-matched ones (human and GPT-3 texts on the same prompt
     type), and scores each TOEFL essay before and after polishing. Because
     the score is linear in standardised features, the change decomposes
     exactly into per-feature contributions w_j * dz_j.
  3. Bootstraps the training corpus to give intervals for the mean change.
  4. Compares, essay by essay, with the seven commercial detectors' scores.

Usage:
  python matched_check.py --liang-root ChatGPT-Detector-Bias --freq-list en_50k.txt \
      --code-dir ../code/experiment --out matched_check.json
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
    ap.add_argument("--code-dir", default=".", help="the paper's experiment/ directory")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="matched_check.json")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.code_dir))
    from data import _docs, LIANG_HUMAN, LIANG_AI
    from features import Featurizer, FrequencyTable, STYLE_FEATURES
    from published_detectors import DETECTORS, scores
    from run_experiment import make_detector
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    root = args.liang_root
    feat = Featurizer(FrequencyTable(args.freq_list))
    H = {k: feat(_docs(root, v[0])) for k, v in LIANG_HUMAN.items()}
    toefl_pol = feat(_docs(root, LIANG_HUMAN["toefl"][2]))
    A = {k: feat(_docs(root, v[0])) for k, v in LIANG_AI.items()}
    A_eng = {k: feat(_docs(root, v[1])) for k, v in LIANG_AI.items()}
    toefl = H["toefl"]
    names = STYLE_FEATURES

    # 1. detector-free description of what polishing does
    d = toefl_pol - toefl
    sd = toefl.std(0, ddof=1)
    feat_change = {}
    for j, nm in enumerate(names):
        p = wilcoxon(d[:, j]).pvalue if np.any(d[:, j] != 0) else 1.0
        feat_change[nm] = dict(mean_change_sd=float(d[:, j].mean() / sd[j]),
                               share_increase=float((d[:, j] > 0).mean()),
                               wilcoxon_p=float(p))

    # 2. detectors trained without any TOEFL essay
    designs = {
        "prompt_matched_college": ([H["college"]], [A["college_gpt3"]]),
        "prompt_matched_college_incl_rewrites": ([H["college"]], [A["college_gpt3"], A_eng["college_gpt3"]]),
        "prompt_matched_abstracts": ([H["cs224n"]], [A["cs224n_gpt3"]]),
        "all_native_vs_all_gpt3": ([H["hewlett"], H["college"], H["cs224n"]],
                                   [A["college_gpt3"], A["cs224n_gpt3"]]),
    }
    rng = np.random.default_rng(args.seed)
    out = dict(n_toefl=len(toefl), features=names, feature_change=feat_change, detectors={})
    for name, (hs, as_) in designs.items():
        Xh, Xa = np.vstack(hs), np.vstack(as_)
        X = np.vstack([Xh, Xa]); y = np.r_[np.zeros(len(Xh)), np.ones(len(Xa))]
        # in-domain cross-validated AUROC, to show the detector is informative
        oof = np.empty(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=args.seed).split(X, y):
            oof[te] = make_detector().fit(X[tr], y[tr]).decision_function(X[te])
        det = make_detector().fit(X, y)
        sc_h = det.decision_function(Xh)
        unit = sc_h.std(ddof=1)          # score s.d. of the training human texts
        s0, s1 = det.decision_function(toefl), det.decision_function(toefl_pol)
        ds = (s1 - s0) / unit
        thr = np.quantile(sc_h, 0.9)     # 10% FPR on the detector's own human texts
        scaler, lr = det[0], det[-1]
        contrib = ((scaler.transform(toefl_pol) - scaler.transform(toefl)) * lr.coef_.ravel()).mean(0) / unit
        top = np.argsort(-np.abs(contrib))[:4]
        boot = []
        for _ in range(args.boot):
            ih = rng.integers(0, len(Xh), len(Xh)); ia = rng.integers(0, len(Xa), len(Xa))
            Xb = np.vstack([Xh[ih], Xa[ia]])
            m = make_detector().fit(Xb, y)
            u = m.decision_function(Xh[ih]).std(ddof=1)
            boot.append(((m.decision_function(toefl_pol) - m.decision_function(toefl)) / u).mean())
        boot = np.array(boot)
        out["detectors"][name] = dict(
            n_human=len(Xh), n_machine=len(Xa),
            cv_auroc=float(roc_auc_score(y, oof)),
            auroc_toefl_vs_machine=float(roc_auc_score(np.r_[np.zeros(len(toefl)), np.ones(len(Xa))],
                                                       np.r_[s0, det.decision_function(Xa)])),
            mean_score_change_sd=float(ds.mean()),
            boot_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            share_boot_positive=float((boot > 0).mean()),
            share_essays_score_up=float((ds > 0).mean()),
            wilcoxon_p=float(wilcoxon(ds).pvalue),
            flagged_before=float((s0 > thr).mean()), flagged_after=float((s1 > thr).mean()),
            top_contributions={names[j]: float(contrib[j]) for j in top},
        )

    # 3. essay-by-essay comparison with the commercial detectors
    com0 = np.vstack([scores(root, LIANG_HUMAN["toefl"][0], k) for k in DETECTORS])
    com1 = np.vstack([scores(root, LIANG_HUMAN["toefl"][2], k) for k in DETECTORS])
    down_commercial = (com1.mean(0) < com0.mean(0))
    det = make_detector().fit(np.vstack([H["college"], A["college_gpt3"]]),
                              np.r_[np.zeros(len(H["college"])), np.ones(len(A["college_gpt3"]))])
    up_matched = det.decision_function(toefl_pol) > det.decision_function(toefl)
    out["essay_level"] = dict(
        share_commercial_mean_down=float(down_commercial.mean()),
        share_commercial_down_and_matched_up=float((down_commercial & up_matched).mean()),
        share_each_commercial_down={k: float((com1[i] < com0[i]).mean()) for i, k in enumerate(DETECTORS)},
    )
    json.dump(out, open(args.out, "w"), indent=2)

    print("Polishing, detector-free (mean paired change in TOEFL s.d.; share of essays increasing):")
    for nm, v in feat_change.items():
        print(f"  {nm:28s} {v['mean_change_sd']:+.2f}  {v['share_increase']:.0%}  p={v['wilcoxon_p']:.1g}")
    for nm, r in out["detectors"].items():
        print(f"\n{nm}: {r['n_human']} human vs {r['n_machine']} machine; CV AUROC {r['cv_auroc']:.3f}; "
              f"AUROC TOEFL vs machine {r['auroc_toefl_vs_machine']:.3f}")
        print(f"  mean score change {r['mean_score_change_sd']:+.2f} sd [{r['boot_ci'][0]:+.2f}, {r['boot_ci'][1]:+.2f}], "
              f"boot>0 {r['share_boot_positive']:.1%}, essays up {r['share_essays_score_up']:.0%}, p={r['wilcoxon_p']:.1g}")
        print(f"  flagged at 10% FPR: before {r['flagged_before']:.0%}, after {r['flagged_after']:.0%}")
        print("  top contributions: " + ", ".join(f"{k} {v:+.2f}" for k, v in r["top_contributions"].items()))
    print("\nEssay level:", out["essay_level"])


if __name__ == "__main__":
    main()
