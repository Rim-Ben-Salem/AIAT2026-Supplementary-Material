"""Text features for the detector.

Two families:
  * Stylometric and lexical-frequency features. These need only a word
    frequency list and run on any machine.
  * Optional language-model features (token log-probability statistics), the
    signal that perplexity-based detectors rely on. These need torch and
    transformers and are enabled with --lm-model.
"""
import math
import re

import numpy as np

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"[^.!?]+[.!?]*")

FUNCTION_WORDS = set("""
a an the and or but if because as of at by for from in into on onto to with
without about over under between through during before after above below is
are was were be been being am do does did have has had i you he she it we they
me him her us them my your his its our their this that these those there here
not no so than too very can could will would shall should may might must
""".split())

STYLE_FEATURES = [
    "mean_zipf", "sd_zipf", "frac_rare", "frac_very_common", "mattr",
    "mean_word_len", "sd_word_len", "frac_long_words", "mean_sent_len",
    "sd_sent_len", "commas_per_word", "semicolons_colons_per_word",
    "function_word_ratio", "first_person_rate", "contraction_rate",
]
LM_FEATURES = ["lm_mean_logprob", "lm_sd_logprob", "lm_mean_logrank", "lm_mean_entropy"]


class FrequencyTable:
    """Zipf frequencies (log10 occurrences per billion words).

    Uses the `wordfreq` package if installed; otherwise reads a plain-text list
    with one `word count` pair per line, such as the FrequencyWords lists
    (https://github.com/hermitdave/FrequencyWords).
    """

    def __init__(self, path=None):
        self._wf = None
        self._table = None
        if path is None:
            try:
                import wordfreq
                self._wf = wordfreq
                return
            except ImportError as err:
                raise SystemExit("Install `wordfreq` or pass --freq-list PATH") from err
        counts = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    counts[parts[0].lower()] = int(parts[1])
        total = sum(counts.values())
        self._table = {w: math.log10(c / total * 1e9) for w, c in counts.items()}
        self._floor = math.log10(1 / total * 1e9)

    def zipf(self, word):
        w = word.lower()
        if self._wf is not None:
            return self._wf.zipf_frequency(w, "en")
        return self._table.get(w, self._floor)


def _mattr(tokens, window=50):
    """Moving-average type-token ratio, which is insensitive to text length."""
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    ratios = [len(set(tokens[i:i + window])) / window
              for i in range(len(tokens) - window + 1)]
    return float(np.mean(ratios))


def style_features(text, freq):
    words = WORD.findall(text)
    lower = [w.lower() for w in words]
    n = max(len(words), 1)
    zipfs = np.array([freq.zipf(w) for w in lower]) if words else np.zeros(1)
    lens = np.array([len(w) for w in words]) if words else np.zeros(1)
    sents = [s for s in SENT.findall(text) if WORD.search(s)]
    slens = np.array([len(WORD.findall(s)) for s in sents]) if sents else np.array([n])
    return {
        "mean_zipf": float(zipfs.mean()),
        "sd_zipf": float(zipfs.std()),
        "frac_rare": float((zipfs < 3.0).mean()),
        "frac_very_common": float((zipfs >= 6.0).mean()),
        "mattr": _mattr(lower),
        "mean_word_len": float(lens.mean()),
        "sd_word_len": float(lens.std()),
        "frac_long_words": float((lens >= 7).mean()),
        "mean_sent_len": float(slens.mean()),
        "sd_sent_len": float(slens.std()),
        "commas_per_word": text.count(",") / n,
        "semicolons_colons_per_word": (text.count(";") + text.count(":")) / n,
        "function_word_ratio": sum(w in FUNCTION_WORDS for w in lower) / n,
        "first_person_rate": sum(w in {"i", "me", "my", "mine"} for w in lower) / n,
        "contraction_rate": sum("'" in w for w in words) / n,
    }


class LMScorer:
    """Token-level statistics under a causal language model."""

    def __init__(self, model_name, device=None, max_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.max_tokens = max_tokens

    def __call__(self, text):
        torch = self.torch
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_tokens).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return dict.fromkeys(LM_FEATURES, 0.0)
        with torch.no_grad():
            logits = self.model(ids).logits[0, :-1].float()
        target = ids[0, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(1, target[:, None]).squeeze(1)
        rank = (logits > logits.gather(1, target[:, None])).sum(1) + 1
        entropy = -(logp.exp() * logp).sum(1)
        return {
            "lm_mean_logprob": tok_logp.mean().item(),
            "lm_sd_logprob": tok_logp.std().item(),
            "lm_mean_logrank": torch.log(rank.float()).mean().item(),
            "lm_mean_entropy": entropy.mean().item(),
        }


class Featurizer:
    """Computes and caches feature vectors for texts."""

    def __init__(self, freq, lm=None):
        self.freq = freq
        self.lm = lm
        self.names = STYLE_FEATURES + (LM_FEATURES if lm is not None else [])
        self._cache = {}

    def __call__(self, texts):
        rows = []
        for t in texts:
            if t not in self._cache:
                f = style_features(t, self.freq)
                if self.lm is not None:
                    f.update(self.lm(t))
                self._cache[t] = np.array([f[k] for k in self.names])
            rows.append(self._cache[t])
        return np.vstack(rows)
