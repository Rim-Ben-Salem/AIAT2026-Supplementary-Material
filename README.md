# Supplementary material

Code, stored results, figures and supplementary material for *Who Bears the Burden of Detecting AI with AI? A Model of Collateral Performative Prediction in the Detection Arms Race*.

## Contents

| Folder or file | What it contains | Paper |
|---|---|---|
| `supplementary_material.pdf` | Supplementary material: additional simulation results and robustness checks (S1), real-text protocol and results (S2), and a reproducibility map linking each figure and table to its script, configuration and result file (S3) | Supplement S1–S3 |
| `experiment/` | Real-text analyses on the public corpus of Liang et al. (2023): commercial-detector scores, the one-round stylometric loop, prompt-matched detectors, the leave-corpus-out evaluation and the cross-fitting check | Section 6, Tables 11–12; Supplement S2 |
| `simulation/` | Population-level simulation of the detection loop: main runs, ablations, sensitivity analyses, literature-anchored parameters, response variants, nonlinear detector, deployment policies (including an FPR cap, a fixed alert budget and confidence-bound audits), fixed-threshold calibration check, prevalence analysis, audit-sample comparison, constant evasion step, alternative response functions and imperfect recalibration | Sections 4–5, Figures 2–5, Tables 3–10; Supplement S1 |
| `figures/` | The figure files used in the paper, as written by `simulation/make_figures.py` | Figures 2–5 and S1 |
| `checksums.sha256` | SHA-256 hashes of every stored result file | |

Table and section numbers refer to the main paper; numbers with an "S" prefix refer to `supplementary_material.pdf`. Each folder has its own README. It lists every script and output file and the table, figure or section that each one supports.

## Requirements

- Python 3.13.
- The package versions in `simulation/requirements.txt` and `experiment/requirements.txt`, which are those used for the reported results.
- Everything runs on a CPU; no GPU is needed.

```
pip install -r simulation/requirements.txt -r experiment/requirements.txt
```

## Data

No external data are redistributed here. `run_all.sh` fetches two public resources at fixed commits:

| Data | Source | Commit |
|---|---|---|
| Corpus and stored detector scores (Liang et al., 2023) | https://github.com/Weixin-Liang/ChatGPT-Detector-Bias | `1d05ed8242d4694a0ed7b73a79c52be8864470f9` |
| English word-frequency list | https://github.com/hermitdave/FrequencyWords, `content/2018/en/en_50k.txt` | `525f9b560de45753a5ea01069454e72e9aa541c6` |

Both remain under their original licences. The simulation needs no external data.

## Reproducing everything

```
bash run_all.sh            # everything, about 4 hours on a 2-core CPU
bash run_all.sh --quick    # skips the two long runs (main simulation and nonlinear detector), about 10 minutes
```

The order matters in one place. The per-detector anchors in `simulation/anchors.py` are derived from `experiment/results/published_detectors.json`, so the real-text analyses run first.

Approximate runtimes on a 2-core CPU:

| Step | Time |
|---|---|
| Real-text analyses | about 5 minutes |
| `simulation/arms_race_sim.py` (all main configurations) | about 20 minutes |
| `simulation/fixed_threshold_check.py` | about 6 minutes |
| `simulation/prevalence.py` | under 1 minute |
| `simulation/audit_evasion_checks.py` | about 5 minutes |
| `simulation/response_calibration_checks.py` | about 15 minutes |
| `simulation/policy_revision_runs.py` | about 2 minutes |
| `simulation/nonlinear_sim.py` (20 seeds) | about 3 hours |
| Figures and table values | under 1 minute |

All runs are seeded and deterministic, so with the pinned package versions they regenerate the stored JSON files byte for byte. The last step of `run_all.sh` checks this with `sha256sum -c checksums.sha256`, which compares every regenerated result file with the hash stored in `checksums.sha256`. `make_tables.py` and `make_figures.py` print the values reported in the tables.

## Licence

The code is released under the MIT licence (`LICENSE`).
