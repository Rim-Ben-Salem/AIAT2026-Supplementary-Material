# Simulation code and results

Supplementary material for *Who Bears the Burden of Detecting AI with AI? A Model of Collateral Performative Prediction in the Detection Arms Race*. Table and section numbers refer to the main paper; numbers with an "S" prefix refer to the supplementary material.

## Contents

| File | Purpose |
|---|---|
| `arms_race_sim.py` | Population-level simulation of the detection loop (Section 5), including the response variants, deployment policies (recalibration, fixed threshold, FPR cap, fixed alert budget, withdrawal), audit options and evasion rules |
| `anchors.py` | Derives per-detector anchors from `../experiment/results/published_detectors.json` and writes `anchors.json` (Table S5) |
| `make_figures.py` | Produces Figures 2–5 and S1 and prints the values in Tables 3 and 4 |
| `make_tables.py` | Prints the values in Tables 5, 7, 8 and 10, Tables S1–S6, the feedback-robustness results (Section S1.3) and all paired differences with 95% intervals |
| `results.json` | Main configurations and ablations, 20 seeds (Figures 3–5, Table 5, default rows of Tables 8 and S1) |
| `sensitivity.json` | One-at-a-time sensitivity analysis, 10 seeds (Table S1, rows of Table 8) |
| `sweep.json` | Joint sweep over human avoidance and evasion step, 10 seeds (Figure S1) |
| `calibrated.json` | Configurations anchored to published detector results (Liang et al., 2023; Krishna et al., 2023), 20 seeds (Tables 6–7, anchored rows of Tables 10 and S3) |
| `robustness.json` | Default and anchored settings with exact (analytic) feedback rates, and the default with a five-fold larger sample, 20 seeds (Section S1.3) |
| `extensions.json` | Separate ablations and the open-loop control (Table S2), less idealised responses (Tables 8 and S3), deployment policies (Table 10) and per-detector anchors (Table S5), 20 seeds |
| `anchors.json` | Per-detector anchors (Table S5) |
| `policy_robustness.json` | Fixed threshold under training-prior shift and withdrawal under other rules, audit sizes and prevalences, 20 seeds (Table S6) |
| `fixed_threshold_check.py` | Fixed raw-score cut-off versus a cut-off fixed on Platt-recalibrated probabilities, under a stable and a drifting training prior; the exact-posterior check of Proposition 4.4(2); and the calibration error of the deployed probabilities against the equal-prior posterior. Writes `fixed_threshold_check.json`, 20 seeds (Section S1.8, calibrated-threshold rows of Table S6) |
| `nonlinear_sim.py` | Neural-network detector with responses along local gradients, and a second generator; writes `nonlinear.json`, 20 seeds (Section S1.6, Table S4) |
| `prevalence.py` | Withdrawal under a misspecified or estimated prevalence; writes `prevalence.json`, 20 seeds (Corollary 4.5, Section S1.9, Table S7) |
| `audit_evasion_checks.py` | Withdrawal with an independent audit sample compared with an audit that reuses the evaluation sample, and the effect of human adaptation under a constant evasion step; writes `audit_evasion_checks.json`, 20 seeds (Sections S1.11–S1.12, Tables S9–S10) |
| `response_calibration_checks.py` | Threshold, saturating and partial-awareness response functions, and recalibration on small, unrepresentative or stale calibration samples, with default and anchored parameters; writes `response_calibration_checks.json`, 20 seeds (Table 8, Section 5.6, Sections S1.13–S1.14, Tables S11–S12) |
| `policy_revision_runs.py` | FPR cap, fixed alert budget and withdrawal with a confidence-bound audit (Clopper–Pearson, 200/200 and 2,000/200 texts, per-round and simultaneous error levels), with default and anchored parameters; writes `policy_revision.json`, 20 seeds, and prints the values (Table 10; Section 5.6, paragraphs "Caps and alert budgets" and "Confidence-bound audits") |

Every result file from `arms_race_sim.py` stores, for each configuration and metric, the mean and standard deviation across seeds for every round, and the per-seed values after the last round (`final_by_seed`), which `make_tables.py` uses for paired comparisons. Configurations run with the same seeds share their random draws, so differences are computed seed by seed.

Figure 2 and Table 3 are analytical; `make_figures.py` also computes them.

## Requirements

Python 3.10 or later; `pip install -r requirements.txt` installs the versions used for the reported results (Python 3.13, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2).

## Reproducing the results

```
python anchors.py           # writes anchors.json (needs ../experiment/results/published_detectors.json)
python arms_race_sim.py     # writes extensions.json, policy_robustness.json, robustness.json, calibrated.json, results.json, sensitivity.json, sweep.json (about 20 minutes)
python make_figures.py      # writes the figures to ../figures and prints the Table 3 and 4 values
python make_tables.py       # prints the table values, the robustness results and paired differences
python fixed_threshold_check.py   # about 6 minutes on 2 cores
python prevalence.py --sim-dir .  # under a minute
python audit_evasion_checks.py    # about 5 minutes
python response_calibration_checks.py   # about 15 minutes
python policy_revision_runs.py    # about 2 minutes
python nonlinear_sim.py --sim-dir . --seeds 20   # about 3 hours on 2 cores
```

All runs are seeded, so the JSON files are reproduced exactly with the pinned package versions. `../checksums.sha256` lists the SHA-256 hash of every stored result file; `sha256sum -c checksums.sha256`, run from the top-level folder, compares regenerated files with them. Each JSON file maps a configuration name to a dictionary of metrics; each metric holds a pair of lists, the mean and the standard deviation across seeds, with one entry per round.

## Literature-anchored parameters

Under an equal-variance binormal score model, with z(p) = Φ⁻¹(1−p), a transformation that lowers a group's flag rate from p to p' < p moves it z(p') − z(p) > 0 score standard deviations away from the flagged side, whatever the threshold. This converts published rates into model units:

| Published result | Model parameter |
|---|---|
| Non-native 61.22% vs native 5.19% flagged (Liang et al., mean of 7 detectors) | group offset m2 = 1.91 |
| GPT-4 polishing of non-native essays: 61.22% → 11.77% flagged | shift per human rewrite δh = 1.47 |
| DIPPER paraphrasing: DetectGPT 70.3% → 4.6% detected at 1% FPR (Krishna et al.) | shift per machine paraphrase δA = 2.22 |
| 70.3% detected at 1% FPR | initial separation a0 = 4.0 |

Evasion assumes every caught machine text is paraphrased once per round (η = δA). Avoidance assumes a fraction r of flagged writers rewrite once per round, giving λh = r · δh · α / 0.05; the configurations use r = 0, 0.25, 0.5 and 1. In the code, the avoidance strength λh is the parameter `alpha`.

## Notes on the policies and options

- **Fixed threshold.** `policy="fixed_threshold"` stores the round-0 cut-off on the logistic regression's decision function (its log-odds) and reuses that number after each retraining. This is a fixed cut-off on the model's own predicted probability, not on the equal-prior posterior that Proposition 4.4(2) assumes. `fixed_threshold_check.py` reproduces this policy exactly with `threshold="raw"` (checked at start-up) and compares it with a cut-off fixed on probabilities recalibrated each round by Platt scaling on balanced labelled data.
- **FPR cap.** `policy="cap"` keeps the round-0 cut-off unless the threshold recalibrated to the target FPR is higher, so the threshold is raised only when the realised rate would exceed the cap and is never lowered below its round-0 value.
- **Fixed alert budget.** `policy="budget"` sets the threshold each round so that the target rate of all submissions is flagged, on a calibration sample of human texts plus machine texts in the share `budget_prevalence` (default 0.1).
- **Confidence-bound audit.** With `policy="withdraw"` and `withdraw_rule="cb"`, each round's audit draws `audit_nh` human and `audit_na` machine texts and computes exact Clopper–Pearson bounds, a lower bound on the TPR and an upper bound on the FPR, each at level `audit_delta / 2`. The FPR bound is 1 − (`audit_delta` / 2)^(1/`audit_nh`) when no human text is flagged. The detector is withdrawn once the ratio of the bounds falls below (1 − π̂)/π̂. `audit_delta = 0.05 / 60` gives a guarantee over all 60 rounds.
- **Precision of a flag.** `ppv_true` records, each round, the precision of a flag at 10% prevalence computed from the evaluation sample at the deployed threshold, whether or not the detector has been withdrawn.
- **Withdrawal audit.** With `policy="withdraw"`, the default `audit_source="independent"` audits the detector each round on a fresh labelled sample (`audit_n` texts per class, or the evaluation-sample size when `audit_n` is `None`) drawn from the current populations with its own random stream, so that the trajectory before withdrawal does not depend on the audit. `audit_source="evaluation"` reuses the evaluation sample instead; `audit_evasion_checks.py` compares the two.
- **Evasion rule.** The default `evasion="tpr"` scales the evasion step with the true-positive rate. `evasion="constant"` holds it at `evasion_rate`, a constant TPR-equivalent rate.
- **Response function.** The default `response_fn="linear"` makes each group's avoidance and homogenisation steps proportional to its pressure g = FPR_g / target FPR. `response_fn="threshold"` sets the pressure to zero while it is below `resp_threshold`, and `response_fn="saturating"` replaces it by `resp_cap * tanh(g / resp_cap)`.
- **Calibration sample.** Under recalibration to a fixed FPR, the threshold is the empirical quantile of a sample of human texts. By default it is a fresh sample of n texts from the current populations, in proportion to their shares. `cal_n` sets its size, `cal_p2` the share of G2 in it, and `cal_stale` draws it from the populations that many rounds earlier.
- **Exact feedback.** `feedback="analytic"` solves a new threshold at the target pooled FPR each round, so it applies to recalibration to a fixed FPR only, and `run()` raises an error if it is combined with `policy="fixed_threshold"`.
- **Ablations with no avoidance.** In the joint sweep (`sweep.json`) and in the anchored configuration with r = 0, setting the avoidance strength to zero also switches homogenisation off, so these runs are the no-adaptation ablation. The avoidance-only ablation `no_avoidance` in `extensions.json` keeps homogenisation.
- **Seed comparisons.** The intervals printed by `make_tables.py`, `audit_evasion_checks.py` and `response_calibration_checks.py` are paired *t*-intervals over simulated seeds. They describe variability across simulation runs, not uncertainty about real populations.
- **Reusing the recalibration trajectory.** `prevalence.py` relies on the fact that, before withdrawal, a detector under the withdrawal rule follows the recalibration trajectory exactly, and treats the audit as exact, as Corollary 4.5 assumes.
- **Nonlinear variant.** With `detector="logistic", n_gen=1`, `nonlinear_sim.py` reproduces `arms_race_sim.run()` exactly, and checks this at start-up.
