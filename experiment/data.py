"""Corpus loading.

A corpus is a pandas DataFrame with one row per text and columns
  text     the text itself
  label    0 for human-written, 1 for machine-generated
  group    writer group for human texts (e.g. "native", "nonnative"); "ai" otherwise
  source   name of the sub-corpus the text came from
and an optional `published_rewrite` column holding an existing rewrite of
the same text, used by the `published` rewriter.
"""
import json
import os

import pandas as pd

LIANG_HUMAN = {
    "toefl": ("Human_Data/TOEFL_real_91", "nonnative", "Human_Data/TOEFL_gpt4polished_91"),
    "hewlett": ("Human_Data/HewlettStudentEssay_real_88", "native", None),
    "college": ("Human_Data/CollegeEssay_real_70", "native", None),
    "cs224n": ("Human_Data/CS224N_real_145", "native", None),
}
LIANG_AI = {
    "cs224n_gpt3": ("GPT_Data/CS224N_gpt3_145", "GPT_Data/CS224N_gpt3PromptEng_145"),
    "college_gpt3": ("GPT_Data/CollegeEssay_gpt3_31", "GPT_Data/CollegeEssay_gpt3PromptEng_31"),
}


def _docs(root, rel):
    with open(os.path.join(root, "Data_and_Results", rel, "data.json"), encoding="utf-8") as fh:
        return [d["document"].strip() for d in json.load(fh)]


def load_liang(root, native=("hewlett", "college", "cs224n")):
    """Load the corpus released by Liang et al. (2023).

    `root` is a clone of https://github.com/Weixin-Liang/ChatGPT-Detector-Bias.
    Non-native writers are the TOEFL essays; `native` selects which native
    sub-corpora to include. Published rewrites are the GPT-4-polished TOEFL
    essays (prompt: enhance word choices to sound more like a native speaker)
    and the prompt-engineered GPT-3 texts (prompt: elevate the text with
    literary language), paired with their originals by position.
    """
    rows = []
    for key in ["toefl", *native]:
        rel, group, rewrite_rel = LIANG_HUMAN[key]
        docs = _docs(root, rel)
        rewrites = _docs(root, rewrite_rel) if rewrite_rel else [None] * len(docs)
        assert len(docs) == len(rewrites)
        rows += [dict(text=d, label=0, group=group, source=key, published_rewrite=r)
                 for d, r in zip(docs, rewrites)]
    for key, (rel, rewrite_rel) in LIANG_AI.items():
        docs, rewrites = _docs(root, rel), _docs(root, rewrite_rel)
        assert len(docs) == len(rewrites)
        rows += [dict(text=d, label=1, group="ai", source=key, published_rewrite=r)
                 for d, r in zip(docs, rewrites)]
    return pd.DataFrame(rows)


def load_csv(path):
    """Load a corpus from a CSV with columns text, label, group[, source, published_rewrite]."""
    df = pd.read_csv(path)
    missing = {"text", "label", "group"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    if "source" not in df:
        df["source"] = "csv"
    if "published_rewrite" not in df:
        df["published_rewrite"] = None
    df.loc[df.label == 1, "group"] = "ai"
    return df
