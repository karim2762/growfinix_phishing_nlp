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
PAD_IDX, UNK_IDX = 0, 1

URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"']+", re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
TOKENIZER = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?|\d+")


def _load_stopwords() -> set[str]:
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
    t = text.strip()
    return len(t) < 2000 and bool(BARE_URL_RE.match(t))


def preprocess_text(raw: str) -> tuple[list[str], int]:
    text = html.unescape(str(raw))
    has_url = int(bool(URL_RE.search(text)) or looks_like_bare_url(text))

    if looks_like_bare_url(text):
        pieces = re.split(r"[^a-z0-9]+", text.lower())
        tokens = [p for p in pieces if p and p not in {"http", "https", "www"}]
        return tokens, has_url

    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    tokens = TOKENIZER.tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens, has_url


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    results = [preprocess_text(t) for t in out["text"]]
    out["tokens"] = [r[0] for r in results]
    out["has_url"] = [r[1] for r in results]
    return out


def split_data(df: pd.DataFrame, seed: int = SEED):
    # 70 / 15 / 15, stratified on label
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class Vocab:
    def __init__(self, word2idx: dict[str, int]):
        self.word2idx = word2idx

    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int = 20000) -> "Vocab":
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        word2idx = {PAD: PAD_IDX, UNK: UNK_IDX}
        for w in words:
            word2idx[w] = len(word2idx)
        return cls(word2idx)

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, tokens: list[str], max_len: int) -> tuple[list[int], int]:
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens][:max_len]
        if not ids:
            ids = [UNK_IDX]
        length = len(ids)
        ids = ids + [PAD_IDX] * (max_len - length)
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
