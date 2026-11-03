"""Per-detector anchors from the commercial-detector rates in the Liang et al. corpus.

Reads ../experiment/results/published_detectors.json (written by
../experiment/published_detectors.py) and writes anchors.json.

Under an equal-variance binormal model with z(p) = Phi^-1(1 - p), a group
flagged at rate p sits z(p_ref) - z(p) score standard deviations above a
group flagged at rate p_ref, and a transformation that lowers a flag rate
from p to p' moves texts by z(p') - z(p) > 0 standard deviations away from
the flagged side. For each detector, from the same detector and corpus:
  m2      = z(FPR native)              - z(FPR non-native)
  delta_h = z(FPR non-native, polished) - z(FPR non-native)
  delta_A = z(TPR GPT-3, rewritten)    - z(TPR GPT-3)
Rates use a continuity correction, (k + 0.5) / (n + 1), because several
detectors flag no native essays.
"""
import json

import numpy as np
from scipy.stats import norm

N = dict(native=88, nonnative=91, ai=176)


def z(rate, n):
    k = round(rate * n)
    return norm.ppf(1 - (k + 0.5) / (n + 1))


def main():
    pub = json.load(open("../experiment/results/published_detectors.json"))
    out = {}
    for det, r in pub.items():
        if det == "mean_over_detectors":
            continue
        out[det] = dict(
            m2=float(z(r["fpr_native"], N["native"]) - z(r["fpr_nonnative"], N["nonnative"])),
            delta_h=float(z(r["fpr_nonnative_polished"], N["nonnative"]) - z(r["fpr_nonnative"], N["nonnative"])),
            delta_A=float(z(r["tpr_ai_prompt_engineered"], N["ai"]) - z(r["tpr_ai"], N["ai"])),
        )
    json.dump(out, open("anchors.json", "w"), indent=2)
    for det, a in out.items():
        print(f"{det:14s} m2 {a['m2']:.2f}  delta_h {a['delta_h']:.2f}  delta_A {a['delta_A']:.2f}")
    for k in ["m2", "delta_h", "delta_A"]:
        v = [a[k] for a in out.values()]
        print(f"{k:8s} range {min(v):.2f} to {max(v):.2f}, median {np.median(v):.2f}")


if __name__ == "__main__":
    main()
