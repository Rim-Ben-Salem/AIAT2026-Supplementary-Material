#!/usr/bin/env bash
# Reproduces every reported result. Run from this directory.
#   bash run_all.sh           everything (about 4 hours on a 2-core CPU)
#   bash run_all.sh --quick   skips arms_race_sim.py and nonlinear_sim.py and
#                             uses their stored outputs (about 10 minutes)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="$ROOT/data"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# 1. Public data at fixed commits
mkdir -p "$DATA"
if [ ! -d "$DATA/ChatGPT-Detector-Bias" ]; then
  git clone https://github.com/Weixin-Liang/ChatGPT-Detector-Bias "$DATA/ChatGPT-Detector-Bias"
fi
git -C "$DATA/ChatGPT-Detector-Bias" checkout -q 1d05ed8242d4694a0ed7b73a79c52be8864470f9
if [ ! -d "$DATA/FrequencyWords" ]; then
  git clone --filter=blob:none --sparse https://github.com/hermitdave/FrequencyWords "$DATA/FrequencyWords"
fi
git -C "$DATA/FrequencyWords" checkout -q 525f9b560de45753a5ea01069454e72e9aa541c6
git -C "$DATA/FrequencyWords" sparse-checkout set content/2018/en
LIANG="$DATA/ChatGPT-Detector-Bias"
FREQ="$DATA/FrequencyWords/content/2018/en/en_50k.txt"

# 2. Real-text analyses (Section 6)
cd "$ROOT/experiment"
python published_detectors.py --liang-root "$LIANG" --out results/published_detectors.json
python run_experiment.py --liang-root "$LIANG" --freq-list "$FREQ" --out results/stylometric_all_native.json
python run_experiment.py --liang-root "$LIANG" --freq-list "$FREQ" --native hewlett \
    --out results/stylometric_hewlett_native.json
python matched_check.py --liang-root "$LIANG" --freq-list "$FREQ" --code-dir . --out results/matched_check.json
python crossfit_check.py --liang-root "$LIANG" --freq-list "$FREQ" --out results/crossfit_check.json
python leave_corpus_out.py --liang-root "$LIANG" --freq-list "$FREQ" --out results/leave_corpus_out.json

# 3. Simulation (Sections 4-5)
cd "$ROOT/simulation"
python anchors.py
if [ "$QUICK" -eq 0 ]; then
  python arms_race_sim.py
fi
python fixed_threshold_check.py
python prevalence.py --sim-dir . --out prevalence.json
python audit_evasion_checks.py --seeds 20 --out audit_evasion_checks.json
python response_calibration_checks.py --seeds 20 --out response_calibration_checks.json
python policy_revision_runs.py --seeds 20 --out policy_revision.json
if [ "$QUICK" -eq 0 ]; then
  python nonlinear_sim.py --sim-dir . --seeds 20 --out nonlinear.json
fi

# 4. Figures and the values printed for the tables
python make_figures.py
python make_tables.py

# 5. Compare every regenerated result file with the stored checksums
cd "$ROOT"
sha256sum -c checksums.sha256
echo "Done."
