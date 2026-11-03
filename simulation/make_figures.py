import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

C = ["#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b"]
LS = ["-", "--", "-.", ":"]  # line styles keep series distinguishable in greyscale
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "lines.linewidth": 2, "legend.frameon": False})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
r = json.load(open("results.json"))
T = np.arange(len(r["baseline"]["auroc"][0]))

def band(ax, cfg, key, i, label):
    m, s = map(np.array, r[cfg][key])
    ax.plot(T, m, color=C[i], ls=LS[i], label=label)
    ax.fill_between(T, m - s, m + s, color=C[i], alpha=0.15, lw=0)

# Figure 3: AUROC decay vs query access
fig, ax = plt.subplots(figsize=(4.2, 2.8))
for i, (cfg, lab) in enumerate([("baseline", "full query access (q=1)"), ("q05", "partial (q=0.5)"),
                                ("q0", "no query access (q=0)"), ("static", "static world")]):
    band(ax, cfg, "auroc", i, lab)
ax.axhline(0.5, color="#999", lw=0.8); ax.set_ylim(0.45, 1.02)
ax.set_xlabel("round"); ax.set_ylabel("detector AUROC (retrained each round)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_auroc.pdf"))

# Figure 4: benefit vanishes, cost stays
fig, axs = plt.subplots(1, 2, figsize=(7, 2.7))
band(axs[0], "baseline", "tpr", 0, "AI text caught (TPR)")
band(axs[0], "static", "tpr", 3, "TPR, static world")
axs[0].set_xlabel("round"); axs[0].set_ylabel("rate at 1% pooled FPR"); axs[0].legend(fontsize=7)
band(axs[1], "baseline", "fpr_g1", 0, "majority group G1")
band(axs[1], "baseline", "fpr_g2", 1, "minority group G2")
axs[1].set_xlabel("round"); axs[1].set_ylabel("false-positive rate"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cost_benefit.pdf"))

# Figure 5: human style drift and homogenisation
fig, axs = plt.subplots(1, 2, figsize=(7, 2.7))
band(axs[0], "baseline", "drift_g1", 0, "G1"); band(axs[0], "baseline", "drift_g2", 1, "G2")
axs[0].set_xlabel("round"); axs[0].set_ylabel("style drift from baseline"); axs[0].legend(fontsize=7)
band(axs[1], "baseline", "std_g1", 0, "G1"); band(axs[1], "baseline", "std_g2", 1, "G2")
axs[1].set_xlabel("round"); axs[1].set_ylabel("within-group style s.d."); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_human.pdf"))

# Figure 2: analytic precision (PPV) vs prevalence, binormal equal-variance model
pi = np.logspace(-3, -0.05, 200); fpr = 0.01
fig, ax = plt.subplots(figsize=(4.2, 2.8))
for i, auc in enumerate([0.99, 0.9, 0.75, 0.6]):
    dp = np.sqrt(2) * norm.ppf(auc); tpr = norm.cdf(dp - norm.ppf(1 - fpr))
    ax.plot(pi, tpr * pi / (tpr * pi + fpr * (1 - pi)), color=C[i], ls=LS[i], label=f"AUROC {auc}")
ax.set_xscale("log"); ax.set_xlabel("prevalence of AI text $\\pi$"); ax.set_ylabel("precision of a flag")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ppv.pdf"))

# Table 4: precision of a flag at several prevalences, start and end of the full loop
b = r["baseline"]
fpr = float(np.mean(b["fpr"][0]))
print(f"mean realised pooled FPR {fpr:.4f}")
for p in [0.01, 0.05, 0.1, 0.25, 0.5]:
    ppv = lambda t: t * p / (t * p + fpr * (1 - p))
    print(f"pi={p:<5} PPV start {ppv(b['tpr'][0][0]):.2f}  end {ppv(b['tpr'][0][-1]):.2f}")

# Figure S1: two-dimensional sweep, human avoidance strength vs evasion step
from matplotlib.colors import LinearSegmentedColormap
sw = json.load(open("sweep.json"))
A, E = [0.0, 0.25, 0.5, 1.0], [0.15, 0.3, 0.6, 1.2]
blues = LinearSegmentedColormap.from_list("seq", ["#eaf2fc", "#2a78d6", "#0b2e5c"])
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
for ax, key, title in [(axs[0], "auroc", "final AUROC"), (axs[1], "drift_g2", "G2 style drift")]:
    M = np.array([[sw[f"alpha={a},eta={e}"][key][0][-1] for e in E] for a in A])
    im = ax.imshow(M, cmap=blues, origin="lower", aspect="auto")
    lo, hi = M.min(), M.max()
    for i in range(len(A)):
        for j in range(len(E)):
            dark = (M[i, j] - lo) / (hi - lo) > 0.55
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if dark else "#1a1a1a")
    ax.set_xticks(range(len(E)), [str(e) for e in E]); ax.set_yticks(range(len(A)), [str(a) for a in A])
    ax.set_xlabel("evasion step $\\eta$"); ax.set_ylabel("human avoidance $\\alpha_h$")
    ax.set_title(title, fontsize=9); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sweep.pdf"))

# Table 3: worked example of threshold-amplified disparity
from scipy.optimize import brentq
print("Table 3: pooled FPR, FPR_1, FPR_2, ratio, G2 share, G2 flags per 1,000")
for pooled in [0.10, 0.05, 0.01, 0.001]:
    z = brentq(lambda t: 0.8 * norm.sf(t) + 0.2 * norm.sf(t - 1) - pooled, -5, 10)
    f1, f2 = norm.sf(z), norm.sf(z - 1)
    print(f"{pooled:<6} {100*f1:.3f}% {100*f2:.2f}% {f2/f1:.1f} {0.2*f2/pooled:.0%} {1000*f2:.1f}")
