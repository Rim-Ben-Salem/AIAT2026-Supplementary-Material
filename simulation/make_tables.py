"""Prints the values in Tables 5, 7, 8 and 10 of the paper, Tables S1-S6 of the supplement and Section S1.3.

Reads results.json, sensitivity.json, calibrated.json, robustness.json and
extensions.json and policy_robustness.json, written by arms_race_sim.py, and anchors.json, written by
anchors.py. Tables 3 and 4 are printed by make_figures.py; the real-text
tables by the scripts in ../experiment.

Paired differences use the per-seed final values: configurations run with the
same seeds share their random draws, so differences are computed seed by seed
and summarised by the mean and a 95% t-interval over seeds.
"""
import json

import numpy as np
from scipy.stats import t as tdist


def first_below(v, level=0.75):
    return next((t for t, a in enumerate(v["auroc"][0]) if a < level), None)


def last(v, k, i=-1):
    return v[k][0][i]


def row(v):
    return dict(
        initial_disparity=(last(v, "fpr_g2", 0) / last(v, "fpr_g1", 0)
                           if last(v, "fpr_g1", 0) > 0 else None),
        auroc=last(v, "auroc"), tpr=last(v, "tpr"),
        disparity=last(v, "fpr_g2") / max(last(v, "fpr_g1"), 1e-12),
        below_075=first_below(v),
        drift_g1=last(v, "drift_g1"), drift_g2=last(v, "drift_g2"),
        rounds_deployed=sum(v["deployed"][0]) if "deployed" in v else None,
        mean_fpr_pct=100 * float(np.mean(v["fpr"][0])),
        sd_g1=last(v, "std_g1"), sd_g2=last(v, "std_g2"),
        false_flags_g1=last(v, "cum_ff_g1"), false_flags_g2=last(v, "cum_ff_g2"),
    )


def show(title, data, cols):
    print(f"\n{title}")
    print(f"  {'configuration':24s}" + "".join(f"{c:>16s}" for c in cols))
    for name, v in data.items():
        r = row(v)
        cells = []
        for c in cols:
            x = r[c]
            cells.append(f"{'--' if x is None else (str(x) if isinstance(x, int) else f'{x:.2f}'):>16s}")
        print(f"  {name:24s}" + "".join(cells))


def paired(a, b, k):
    """Mean and 95% t-interval of a minus b over seeds, for final values of k."""
    x = np.array(a["final_by_seed"][k]) - np.array(b["final_by_seed"][k])
    h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def within(v, k1, k2):
    """G2 minus G1 within one configuration, paired by seed."""
    x = np.array(v["final_by_seed"][k2]) - np.array(v["final_by_seed"][k1])
    h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h


def show_paired(title, rows, ref, keys):
    print(f"\n{title}")
    for name, v in rows.items():
        cells = []
        for k in keys:
            m, lo, hi = paired(v, ref, k)
            cells.append(f"{k} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {name:30s} " + "  ".join(cells))


if __name__ == "__main__":
    main = json.load(open("results.json"))
    show("Table 5 (ablations)", main,
         ["auroc", "tpr", "disparity", "drift_g1", "drift_g2", "false_flags_g1", "false_flags_g2"])
    sens = {"default": main["baseline"], **json.load(open("sensitivity.json"))}
    show("Table S1 (one-at-a-time sensitivity)", sens,
         ["initial_disparity", "auroc", "below_075", "drift_g1", "drift_g2",
          "false_flags_g1", "false_flags_g2"])
    show("Table 7 (literature-anchored parameters)", json.load(open("calibrated.json")),
         ["auroc", "below_075", "drift_g1", "drift_g2", "false_flags_g1", "false_flags_g2"])
    show("Feedback robustness (Section S1.3)", json.load(open("robustness.json")),
         ["initial_disparity", "auroc", "below_075", "drift_g1", "drift_g2",
          "false_flags_g1", "false_flags_g2"])

    ext = json.load(open("extensions.json"))
    calib = json.load(open("calibrated.json"))
    base = main["baseline"]
    keys = ["auroc", "drift_g1", "drift_g2", "cum_ff_g1", "cum_ff_g2"]
    cols = ["auroc", "below_075", "drift_g1", "drift_g2", "false_flags_g1", "false_flags_g2"]

    show_paired("Table S2 (paired differences from the full loop)", ext["ablations"], base, keys)
    abl = ext["ablations"]
    show_paired("Section 5.3: avoidance only and homogenisation only, minus neither",
                {"avoidance only": abl["no_homogenisation"],
                 "homogenisation only": abl["no_avoidance"]}, abl["no_human_adapt"], ["auroc"])
    show_paired("Section 5.3: assimilation, minus none of the human mechanisms",
                {"assimilation only": abl["no_human_adapt"]}, abl["no_human_change"], ["auroc"])
    print("  cumulative TPR summed over rounds: " + ", ".join(
        f"{k} {sum(v['tpr'][0]):.2f}" for k, v in abl.items()))

    show("Tables 8 and S3 (less idealised responses)", {"full loop": base, "anchored r=1, aligned": calib["r=1.0"],
                                                 **ext["responses"]}, cols)
    show_paired("Table S3, paired differences from the full loop",
                {k: v for k, v in ext["responses"].items() if not k.startswith("anchored")}, base, keys)

    show("Table 10 (deployment policies)",
         {"default_fixed_fpr": base, **{k: v for k, v in ext["policies"].items() if k.startswith("default")},
          "anchored_r0.5_fixed_fpr": calib["r=0.5"],
          **{k: v for k, v in ext["policies"].items() if k.startswith("anchored")}},
         ["auroc", "rounds_deployed", "mean_fpr_pct", "drift_g1", "drift_g2", "false_flags_g1", "false_flags_g2"])

    anchors = json.load(open("anchors.json"))
    print("\nTable S5 (per-detector anchors)")
    for det, a in anchors.items():
        r0, r1 = ext["anchors"][f"{det}_r=0.0"], ext["anchors"][f"{det}_r=1.0"]
        x0, x1 = row(r0), row(r1)
        m, lo, hi = paired(r1, r0, "auroc")
        print(f"  {det:14s} m2 {a['m2']:.2f} dh {a['delta_h']:.2f} dA {a['delta_A']:.2f} | "
              f"AUROC r0 {x0['auroc']:.2f} (below 0.75 at {x0['below_075']}) r1 {x1['auroc']:.2f} "
              f"(at {x1['below_075']}) | r1-r0 {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | r1 drift "
              f"{x1['drift_g1']:.2f}/{x1['drift_g2']:.2f} flags {x1['false_flags_g1']:.2f}/{x1['false_flags_g2']:.2f}"
              f" | r0 flags {x0['false_flags_g1']:.2f}/{x0['false_flags_g2']:.2f}")
    show("Anchor variants (Krishna-based, r=0.5)", {"krishna_r=0.5": calib["r=0.5"],
         **{k: v for k, v in ext["anchors"].items() if k.startswith("krishna")}}, cols)

    print("\nG2 minus G1 within each configuration (mean +/- 95% half-width)")
    allc = {"full loop": base, **ext["ablations"], **ext["responses"], **ext["policies"],
            **ext["anchors"], **{f"anchored {k}": v for k, v in calib.items()}}
    for name, v in allc.items():
        dm, dh = within(v, "drift_g1", "drift_g2")
        fm, fh = within(v, "cum_ff_g1", "cum_ff_g2")
        print(f"  {name:32s} drift {dm:+.2f} +/- {dh:.2f}   false flags {fm:+.2f} +/- {fh:.2f}")

    pr = json.load(open("policy_robustness.json"))
    ref = {"default fixed FPR": base, "default fixed threshold": ext["policies"]["default_fixed_threshold"],
           "default withdrawal (pi=0.1)": ext["policies"]["default_withdraw"],
           "anchored fixed FPR": calib["r=0.5"],
           "anchored withdrawal (pi=0.1)": ext["policies"]["anchored_r0.5_withdraw"]}
    show("Table S6 (robustness of the policy comparison)", {**ref, **pr},
         ["rounds_deployed", "mean_fpr_pct", "false_flags_g1", "false_flags_g2"])
    print("  reduction in cumulative false flags relative to fixed FPR (same parameters):")
    for name, v in pr.items():
        r0 = calib["r=0.5"] if name.startswith("anchored") else base
        if "withdraw" in name or "fixed_threshold" in name:
            red = [1 - last(v, k) / last(r0, k) for k in ("cum_ff_g1", "cum_ff_g2")]
            print(f"    {name:42s} G1 {100*red[0]:6.1f}%  G2 {100*red[1]:6.1f}%")
    for name in ["fixed_threshold_prior_drift_0.98", "anchored_fixed_threshold_prior_drift_0.98"]:
        print(f"  {name} FPR (%) every 4 rounds: " +
              " ".join(f"{100*x:.2f}" for x in pr[name]["fpr"][0][::4]))

