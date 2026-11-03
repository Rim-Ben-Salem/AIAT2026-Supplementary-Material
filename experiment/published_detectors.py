"""Behaviour of seven commercial detectors on the Liang et al. (2023) corpus.

The corpus release stores each detector's score (probability of being
machine-generated) for every text, including the rewritten versions, so the
following can be measured without querying any detector:
  * group false-positive rates on non-native (TOEFL) and native (US 8th-grade,
    Hewlett) essays at the vendors' 0.5 threshold;
  * how disparity changes as the threshold is tightened to fixed pooled
    false-positive rates (Proposition 4.6);
  * how far GPT-4 polishing moves each non-native essay's score, and how many
    flagged essays are no longer flagged afterwards (human-side avoidance);
  * how prompt-engineered rewriting of GPT-3 texts changes detection
    (generator-side evasion).

Usage: python published_detectors.py --liang-root ChatGPT-Detector-Bias
"""
import argparse
import json
import os

import numpy as np

DETECTORS = ["GPTZero", "OriginalityAI", "HFOpenAI", "Crossplag", "Quil", "Sapling", "ZeroGPT"]


def scores(root, rel, det):
    with open(os.path.join(root, "Data_and_Results", rel, f"{det}.json"), encoding="utf-8") as fh:
        return np.array([d["score"] for d in json.load(fh)])


def threshold_for_pooled_fpr(a, b, rate):
    """Smallest threshold at which the share of a and b above it is at most rate."""
    pooled = np.sort(np.r_[a, b])[::-1]
    k = int(np.floor(rate * len(pooled)))
    return pooled[k] if k < len(pooled) else -np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liang-root", required=True)
    ap.add_argument("--out", default="published_detectors.json")
    args = ap.parse_args()
    root = args.liang_root

    res = {}
    for det in DETECTORS:
        toefl = scores(root, "Human_Data/TOEFL_real_91", det)
        polished = scores(root, "Human_Data/TOEFL_gpt4polished_91", det)
        native = scores(root, "Human_Data/HewlettStudentEssay_real_88", det)
        ai = np.r_[scores(root, "GPT_Data/CS224N_gpt3_145", det),
                   scores(root, "GPT_Data/CollegeEssay_gpt3_31", det)]
        ai_eng = np.r_[scores(root, "GPT_Data/CS224N_gpt3PromptEng_145", det),
                       scores(root, "GPT_Data/CollegeEssay_gpt3PromptEng_31", det)]
        flagged = toefl > 0.5
        r = dict(
            fpr_native=float((native > 0.5).mean()),
            fpr_nonnative=float((toefl > 0.5).mean()),
            fpr_nonnative_polished=float((polished > 0.5).mean()),
            n_nonnative_flagged=int(flagged.sum()),
            share_flagged_unflagged_by_polishing=float((polished[flagged] <= 0.5).mean()) if flagged.any() else None,
            mean_score_change_polishing=float((polished - toefl).mean()),
            share_score_decreased=float((polished < toefl).mean()),
            tpr_ai=float((ai > 0.5).mean()),
            tpr_ai_prompt_engineered=float((ai_eng > 0.5).mean()),
            pooled={},
        )
        for rate in [0.20, 0.10, 0.05]:
            thr = threshold_for_pooled_fpr(toefl, native, rate)
            nn, nt = int((native > thr).sum()), int((toefl > thr).sum())
            r["pooled"][f"{rate:.0%}"] = dict(
                realised_pooled_fpr=(nn + nt) / (len(native) + len(toefl)),
                fpr_native=nn / len(native), fpr_nonnative=nt / len(toefl),
                nonnative_share=nt / (nn + nt) if nn + nt else None)
        res[det] = r

    keys = ["fpr_native", "fpr_nonnative", "fpr_nonnative_polished", "share_flagged_unflagged_by_polishing",
            "share_score_decreased", "tpr_ai", "tpr_ai_prompt_engineered"]
    res["mean_over_detectors"] = {k: float(np.mean([res[d][k] for d in DETECTORS if res[d][k] is not None]))
                                  for k in keys}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"{'detector':14s} FPR nat  FPR non  polished  unflagged  score down  TPR AI  TPR AI eng")
    for d in DETECTORS:
        r = res[d]
        u = r["share_flagged_unflagged_by_polishing"]
        print(f"{d:14s} {r['fpr_native']:7.0%} {r['fpr_nonnative']:8.0%} {r['fpr_nonnative_polished']:9.0%} "
              f"{(u if u is not None else float('nan')):10.0%} {r['share_score_decreased']:11.0%} "
              f"{r['tpr_ai']:7.0%} {r['tpr_ai_prompt_engineered']:11.0%}")
    m = res["mean_over_detectors"]
    print(f"{'mean':14s} {m['fpr_native']:7.0%} {m['fpr_nonnative']:8.0%} {m['fpr_nonnative_polished']:9.0%} "
          f"{m['share_flagged_unflagged_by_polishing']:10.0%} {m['share_score_decreased']:11.0%} "
          f"{m['tpr_ai']:7.0%} {m['tpr_ai_prompt_engineered']:11.0%}")
    print("\nTightening the threshold (pooled native + non-native FPR):")
    for d in DETECTORS:
        cells = []
        for k, v in res[d]["pooled"].items():
            share = "no flags (tied scores)" if v["nonnative_share"] is None else f"non-native share {v['nonnative_share']:.0%}"
            cells.append(f"{k} (realised {v['realised_pooled_fpr']:.1%}): nat {v['fpr_native']:.1%} "
                         f"non {v['fpr_nonnative']:.1%}, {share}")
        print(f"  {d:14s} " + "  ".join(cells))
    print(f"\nNon-native essays are {91 / 179:.0%} of the essays in this comparison.")


if __name__ == "__main__":
    main()
