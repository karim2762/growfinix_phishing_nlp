"""
src/preprocess.py - turn raw text into numbers a neural network can read.

Pipeline (and WHY each step exists):
    1. unescape + strip HTML   -> tags like <div> are noise, not language
    2. detect URLs FIRST       -> "contains a link" is a strong phishing signal,
                                  so we record it as a has_url feature before
                                  the URL text is removed
    3. remove URLs             -> every URL is unique, so they would bloat the
                                  vocabulary with words the model sees only once
    4. lowercase + tokenize    -> "FREE" and "free" should be the same word
    5. drop stopwords          -> "the", "and"... carry little signal
    6. vocabulary + padding    -> networks need fixed-size integer matrices

Special case: if the whole input IS a URL (the URL part of the dataset, or a
link pasted into the app) there is no surrounding text to read. Then we keep
the URL's pieces ("paypal", "secure", "login"...) as the tokens instead.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

SEED = 42
PAD, UNK = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1  # index 0 is reserved for padding, 1 for unknown words

# ---- regexes ---------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"']+", re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
# WHY RegexpTokenizer: it's an NLTK tokenizer that needs no extra data download
# (unlike word_tokenize, which needs the 'punkt' model), so setup stays simple.
TOKENIZER = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?|\d+")


def _load_stopwords() -> set[str]:
    """NLTK's English stopword list (downloads it once if missing)."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception:
            print("[warn] could not get NLTK stopwords (offline?) - using a tiny built-in list")
            return set("a an the and or but if of to in on at for with is are was were be been "
                       "it this that i you he she we they my your our their me him her us them "
                       "as by from so do does did not no".split())


STOPWORDS = _load_stopwords()


BARE_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?[\w-]+(?:\.[\w-]+)+(?:[/:?#]\S*)?$", re.IGNORECASE)


def looks_like_bare_url(text: str) -> bool:
    """True when the whole input is one URL / domain (e.g. 'paypal-login.xyz/verify').

    WHY a strict pattern: a plain word with a full stop ("Thanks.") must NOT count.
    """
    t = text.strip()
    return len(t) < 2000 and bool(BARE_URL_RE.match(t))


def preprocess_text(raw: str) -> tuple[list[str], int]:
    """Return (tokens, has_url) for one raw email / SMS / URL."""
    text = html.unescape(str(raw))
    has_url = int(bool(URL_RE.search(text)) or looks_like_bare_url(text))

    if looks_like_bare_url(text):
        # A lone URL: split it into words at every non-alphanumeric character.
        pieces = re.split(r"[^a-z0-9]+", text.lower())
        tokens = [p for p in pieces if p and p not in {"http", "https", "www"}]
        return tokens, has_url

    text = SCRIPT_STYLE_RE.sub(" ", text)  # drop <script>/<style> blocks entirely
    text = TAG_RE.sub(" ", text)  # drop remaining HTML tags
    text = URL_RE.sub(" ", text)  # drop URLs (has_url already recorded)
    tokens = TOKENIZER.tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens, has_url


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'tokens' (list[str]) and 'has_url' (0/1) columns to a dataframe."""
    out = df.copy()
    results = [preprocess_text(t) for t in out["text"]]
    out["tokens"] = [r[0] for r in results]
    out["has_url"] = [r[1] for r in results]
    return out


def split_data(df: pd.DataFrame, seed: int = SEED):
    """70% train / 15% val / 15% test, stratified so each split keeps the class ratio.

    WHY three splits? Train = learn weights, val = early stopping + choosing the
    decision threshold, test = one honest final score the model never touched.
    """
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---- vocabulary --------------------------------------------------------------
class Vocab:
    """Maps word <-> integer id. Build it from TRAIN data only (no test leakage)."""

    def __init__(self, word2idx: dict[str, int]):
        self.word2idx = word2idx

    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int = 20000) -> "Vocab":
        counts = Counter(t for toks in token_lists for t in toks)
        # min_freq drops typos / one-off words; max_size keeps the embedding table small
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        word2idx = {PAD: PAD_IDX, UNK: UNK_IDX}
        for w in words:
            word2idx[w] = len(word2idx)
        return cls(word2idx)

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, tokens: list[str], max_len: int) -> tuple[list[int], int]:
        """Words -> ids, truncated to max_len and padded with PAD_IDX."""
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens][:max_len]  # truncate
        if not ids:  # empty text -> a single <unk> so the LSTM always has >=1 step
            ids = [UNK_IDX]
        length = len(ids)  # real length, so the LSTM can ignore the padding
        ids = ids + [PAD_IDX] * (max_len - length)  # pad
        return ids, length

    def encode_many(self, token_lists, max_len: int):
        enc = [self.encode(t, max_len) for t in token_lists]
        ids = np.array([e[0] for e in enc], dtype=np.int64)
        lengths = np.array([e[1] for e in enc], dtype=np.int64)
        return ids, lengths

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.word2idx), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
