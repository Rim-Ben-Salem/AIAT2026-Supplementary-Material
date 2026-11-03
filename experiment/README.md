# Real-text check of the detection loop

Supplementary material for Section 6 and Supplement S2 of *Who Bears the Burden of Detecting AI with AI? A Model of Collateral Performative Prediction in the Detection Arms Race*.

These scripts connect the model to real detector behaviour. Neither involves human participants; both use the public corpus released by Liang et al. (2023), which contains TOEFL essays by non-native writers, essays by native writers, GPT-4-polished versions of the TOEFL essays, GPT-3 texts, prompt-engineered rewrites of those texts, and the scores that seven commercial detectors assigned to every text.

## Reproducing the reported results

The reported results were produced with Python 3.13, NumPy 2.5.3, pandas 3.0.5 and scikit-learn 1.9.1 (pinned in `requirements.txt`), from the following data:

| Data | Source | Version |
|---|---|---|
| Corpus and stored detector scores | https://github.com/Weixin-Liang/ChatGPT-Detector-Bias | commit `1d05ed8242d4694a0ed7b73a79c52be8864470f9` |
| Word frequency list | `content/2018/en/en_50k.txt` from https://github.com/hermitdave/FrequencyWords | commit `525f9b560de45753a5ea01069454e72e9aa541c6` (SHA-256 of the file begins `5351ff405b1126ef`) |

```
git clone https://github.com/Weixin-Liang/ChatGPT-Detector-Bias
git -C ChatGPT-Detector-Bias checkout 1d05ed8242d4694a0ed7b73a79c52be8864470f9

git clone --filter=blob:none --sparse https://github.com/hermitdave/FrequencyWords
git -C FrequencyWords checkout 525f9b560de45753a5ea01069454e72e9aa541c6
git -C FrequencyWords sparse-checkout set content/2018/en

pip install -r requirements.txt

# Table 11 and the threshold-tightening results (a few seconds)
python published_detectors.py --liang-root ChatGPT-Detector-Bias \
    --out results/published_detectors.json

# Table S13, first column: all native sub-corpora (under a minute)
python run_experiment.py --liang-root ChatGPT-Detector-Bias \
    --freq-list FrequencyWords/content/2018/en/en_50k.txt \
    --out results/stylometric_all_native.json

# Table S13, second column: genre-matched comparison
python run_experiment.py --liang-root ChatGPT-Detector-Bias \
    --freq-list FrequencyWords/content/2018/en/en_50k.txt --native hewlett \
    --out results/stylometric_hewlett_native.json

# Table 12: detectors trained without TOEFL essays, incl. prompt-matched (about a minute)
python matched_check.py --liang-root ChatGPT-Detector-Bias \
    --freq-list FrequencyWords/content/2018/en/en_50k.txt \
    --code-dir . --out results/matched_check.json

# Supplement S2: pooled versus fold-specific thresholds for the cross-fitted detector
python crossfit_check.py --liang-root ChatGPT-Detector-Bias \
    --freq-list FrequencyWords/content/2018/en/en_50k.txt \
    --out results/crossfit_check.json

# Section 6.3 and Table S14: leave-corpus-out evaluation of the stylometric detector
python leave_corpus_out.py --liang-root ChatGPT-Detector-Bias \
    --freq-list FrequencyWords/content/2018/en/en_50k.txt \
    --out results/leave_corpus_out.json
```

All runs are seeded and deterministic. The outputs of these commands are included in `results/`.

## 1. Commercial detectors: `published_detectors.py`

Uses only the stored detector scores. At the vendors' 0.5 threshold it reproduces the published rates exactly (61.22% of non-native and 5.19% of native essays flagged; 11.77% of polished non-native essays). It reports, per detector:

- false-positive rates for native and non-native essays;
- the share of false flags falling on non-native essays as the threshold is tightened to pooled rates of 20%, 10% and 5%;
- for each flagged non-native essay, whether GPT-4 polishing removes the flag, and how often the score falls;
- detection of GPT-3 texts before and after prompt-engineered rewriting.

## 2. Retraining loop: `run_experiment.py`

Each round:

1. **Held-out description.** Over repeated random 50/25/25 train/calibration/evaluation splits (200 by default), fit a detector, set the threshold on the calibration set, and measure AUROC, TPR and per-group FPR on the evaluation set. Intervals are the 2.5th and 97.5th percentiles across splits. Because the splits reuse the same texts, these are split-variability intervals, not confidence intervals for population performance.
2. **Deployment.** Build a cross-fitted detector: five fold models, each text scored by the model not trained on it, and a threshold at the target rate of these out-of-fold scores on human texts. Flag selection, the movement of rewritten texts and the next round's frozen comparison all use these same fold models and this same threshold.
3. **Rewriting.** Rewrite flagged human texts (a fraction `--response-rate` of them) and, unless `--no-evasion`, caught machine texts. Measure how the rewritten texts move in the deployed detector's standardized feature space and whether the deployed detector still flags them.
4. **Next round.** Replace the texts. The next round fits a new cross-fitted detector (retrained) and also applies the previous round's deployed detector unchanged (frozen), under the same protocol.

The default operating point is a 10% pooled false-positive rate; with a corpus this small, a 1% rate would rest on one or two texts.

### Features

- Stylometric and word-frequency features (always on). Word frequencies come from `wordfreq` if installed, or from a plain `word count` list passed with `--freq-list`.
- Language-model features (`--lm-model gpt2`, for example): mean and spread of token log-probabilities, mean log-rank and mean entropy, the signals perplexity-based detectors use. These need `torch` and `transformers`.

### Rewriters

- `--rewriter published` (default): the rewrites already in the corpus. GPT-4 polishing is available for non-native essays and prompt-engineered rewrites for GPT-3 texts; flagged native essays have no rewrite and stay unchanged. One round only.
- `--rewriter hf --rewrite-model Qwen/Qwen2.5-1.5B-Instruct`: a local instruction-tuned model. Flagged human texts are rewritten with the prompt Liang et al. used; caught machine texts are paraphrased. Any number of rounds; a GPU is advisable.
- `--rewriter api --rewrite-model <model>`: any OpenAI-compatible endpoint (`OPENAI_API_KEY`; `--api-base` for other providers or a local server).

The language-model detector and LLM-rewriting options are provided for further work; they were not used for the reported results. Example:

```
pip install torch transformers
python run_experiment.py --liang-root ChatGPT-Detector-Bias --lm-model gpt2 \
    --rewriter hf --rewrite-model Qwen/Qwen2.5-1.5B-Instruct --rounds 3 --response-rate 0.5
```

### Using another corpus

`--csv corpus.csv` with columns `text`, `label` (0 human, 1 machine) and `group` (writer group for human texts), and optionally `published_rewrite`.

## Caveats

- Published rewrites exist only for the non-native essays and the GPT-3 texts, so the two human groups do not have equal opportunities to adapt. The check measures how one group's rewrites move under different detectors; it does not compare adaptation between groups.
- The texts differ in genre, source and prompt as well as in authorship. The stylometric detector's high AUROC can partly reflect these differences and should not be read as typical of task-matched texts. `--native hewlett` gives a genre-matched human comparison; the machine texts still come from different prompts.
- An LLM rewrite of a flagged essay is a proxy for a writer's response, not an observation of one.
- The direction in which a rewrite moves a text depends on the detector's features. A rewrite that lowers the score of a perplexity-based detector can raise the score of a detector that relies on lexical sophistication.

## Additional checks

- **`matched_check.py`** separates the effect of polishing from genre and prompt differences. It first describes the change in the 15 features without any detector. It then trains four stylometric detectors on corpora that contain no TOEFL essay, two of them prompt-matched (human against GPT-3 college essays, human against GPT-3 abstracts), and decomposes each essay's score change into per-feature contributions. Intervals come from 1,000 bootstrap resamples of the training corpus. They describe sensitivity to the training sample, not population uncertainty.
- **`leave_corpus_out.py`** trains the stylometric detector with one sub-corpus held out entirely (college essays, abstracts or Hewlett essays), sets its threshold at a 10% false-positive rate on its own training human texts, and evaluates it only on the held-out sub-corpus and the TOEFL essays. It reports in-domain cross-validated AUROC, held-out AUROC and true-positive rate where held-out machine text exists, false-positive rates on held-out native texts and TOEFL essays, and the effect of polishing on the TOEFL essays.
- **`crossfit_check.py`** repeats round 0 of the reported runs with the same folds and replaces the pooled out-of-fold threshold with fold-specific thresholds. With pooling it reproduces the stored round-0 results. The two flag sets agree on 97.7% and 97.5% of texts.
- **Operating point.** The real-text analyses use a 10% pooled false-positive rate because the corpus has only 179 to 394 human texts. The simulation uses 1%. Group rates and precision should not be compared numerically between the two.
