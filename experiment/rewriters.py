"""Rewriters used to move texts after a detector flags them.

Each rewriter maps a list of texts to a list of rewritten texts, or None
where no rewrite is available. `kind` is "human" for flagged human texts
(avoidance) and "ai" for caught machine texts (evasion).
"""

PROMPTS = {
    # The prompt Liang et al. (2023) used to polish non-native essays.
    "human": "Enhance the word choices to sound more like that of a native speaker. "
             "Return only the revised text.\n\n{text}",
    # A prompt-based stand-in for a dedicated paraphraser such as DIPPER.
    "ai": "Paraphrase the following text so that it reads naturally, keeping its meaning "
          "and length. Return only the paraphrased text.\n\n{text}",
}


class PublishedRewriter:
    """Uses rewrites already present in the corpus (`published_rewrite` column).

    Each text has at most one published rewrite, so this rewriter supports a
    single round of moves.
    """

    def __init__(self, corpus):
        self.lookup = {t: r for t, r in zip(corpus.text, corpus.published_rewrite)
                       if isinstance(r, str) and r.strip()}

    def __call__(self, texts, kind):
        return [self.lookup.get(t) for t in texts]


class HFRewriter:
    """Rewrites with a local instruction-tuned model via transformers."""

    def __init__(self, model_name, device=None, max_new_tokens=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto").to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def _one(self, text, kind):
        messages = [{"role": "user", "content": PROMPTS[kind].format(text=text)}]
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt").to(self.device)
        n_in = len(self.tok(text).input_ids)
        with self.torch.no_grad():
            out = self.model.generate(ids, do_sample=False, pad_token_id=self.tok.eos_token_id,
                                      max_new_tokens=min(self.max_new_tokens, 2 * n_in + 64))
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def __call__(self, texts, kind):
        return [self._one(t, kind) for t in texts]


class APIRewriter:
    """Rewrites through any OpenAI-compatible chat-completions endpoint.

    The API key is read from the OPENAI_API_KEY environment variable; pass
    `base_url` to use another provider or a local server.
    """

    def __init__(self, model_name, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model_name

    def __call__(self, texts, kind):
        out = []
        for t in texts:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "user", "content": PROMPTS[kind].format(text=t)}])
            out.append(resp.choices[0].message.content.strip())
        return out
