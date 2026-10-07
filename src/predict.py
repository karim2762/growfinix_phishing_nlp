"""
src/predict.py - load the trained model and score new text (used by app.py).

Also contains a simple explanation method ("occlusion"): hide one word at a
time and see how much the phishing probability drops. Words whose removal
lowers the score the most are the ones the model finds suspicious.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import torch

from src.model import PhishingBiLSTM
from src.preprocess import UNK_IDX, Vocab, preprocess_text

MODELS = Path(__file__).resolve().parents[1] / "models"


class Detector:
    def __init__(self):
        cfg = json.loads((MODELS / "config.json").read_text())
        self.cfg = cfg
        self.threshold = cfg["threshold"]
        self.vocab = Vocab.load(MODELS / "vocab.json")
        self.model = PhishingBiLSTM(cfg["vocab_size"], cfg["embed_dim"], cfg["hidden_dim"], cfg["dropout"])
        self.model.load_state_dict(torch.load(MODELS / "lstm.pt", map_location="cpu", weights_only=True))
        self.model.eval()

    def _proba(self, token_lists, has_url: int) -> np.ndarray:
        ids, lengths = self.vocab.encode_many(token_lists, self.cfg["max_len"])
        url = torch.full((len(token_lists),), float(has_url))
        return self.model.predict_proba(torch.from_numpy(ids), torch.from_numpy(lengths), url).numpy()

    def predict(self, text: str) -> dict:
        tokens, has_url = preprocess_text(text)
        p = float(self._proba([tokens], has_url)[0])
        return {"probability": p, "is_phishing": p >= self.threshold, "has_url": bool(has_url),
                "tokens": tokens}

    def suspicious_words(self, text: str, top_k: int = 8, min_drop: float = 0.02) -> dict[str, float]:
        """Return {word: importance}; importance = how much P(phishing) falls without it."""
        tokens, has_url = preprocess_text(text)
        tokens = tokens[: self.cfg["max_len"]]
        if not tokens:
            return {}
        # Row 0 = the original text. Row i+1 = the text with token i replaced by <unk>.
        # (We replace instead of delete so every row keeps the same length.)
        variants = [tokens] + [tokens[:i] + ["<unk>"] + tokens[i + 1:] for i in range(len(tokens))]
        probs = self._proba(variants, has_url)
        drops = probs[0] - probs[1:]
        scores: dict[str, float] = {}
        for tok, d in zip(tokens, drops):
            scores[tok] = max(scores.get(tok, 0.0), float(d))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return {w: s for w, s in ranked if s >= min_drop}


def highlight_html(text: str, scores: dict[str, float]) -> str:
    """Wrap suspicious words of the ORIGINAL text in a yellow-to-red <mark> tag."""
    if not scores:
        return html.escape(text).replace("\n", "<br>")
    top = max(scores.values())
    out, last = [], 0
    for m in re.finditer(r"[A-Za-z0-9']+", text):
        out.append(html.escape(text[last:m.start()]))
        s = scores.get(m.group(0).lower())
        if s:
            alpha = 0.25 + 0.6 * (s / top)  # stronger evidence -> stronger colour
            out.append(f"<mark style='background:rgba(255,80,60,{alpha:.2f});padding:0 2px;"
                       f"border-radius:3px' title='importance {s:.2f}'>{html.escape(m.group(0))}</mark>")
        else:
            out.append(html.escape(m.group(0)))
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")
