from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

SEED = 42
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "clean.csv"

HF_BASE = "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/"
HF_FILES = {"texts": "texts.json", "urls": "urls.json"}
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

MAX_CHARS = 5000


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return dest
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def load_hf_json(path: Path, source: str) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    except json.JSONDecodeError:
        df = pd.read_json(path, lines=True)
    df = df[["text", "label"]].copy()
    df["source"] = source
    return df


def load_huggingface() -> pd.DataFrame:
    frames = []
    for source, fname in HF_FILES.items():
        path = download(HF_BASE + fname, RAW_DIR / fname)
        frames.append(load_hf_json(path, source))
    return pd.concat(frames, ignore_index=True)


def load_sms() -> pd.DataFrame:
    path = download(UCI_URL, RAW_DIR / "sms_spam_collection.zip")
    with zipfile.ZipFile(path) as z:
        with z.open("SMSSpamCollection") as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), sep="\t",
                             header=None, names=["label", "text"])
    df["label"] = (df["label"] == "spam").astype(int)
    df["source"] = "sms"
    return df[["text", "label", "source"]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip().str.slice(0, MAX_CHARS)
    df["label"] = df["label"].astype(int)
    df = df[(df["text"].str.len() >= 3) & df["label"].isin([0, 1])]

    # same text with different labels -> drop all of them
    conflict = df.groupby("text")["label"].transform("nunique") > 1
    df = df[~conflict]
    df = df.drop_duplicates(subset="text")
    print(f"[clean] {n0} -> {len(df)} rows")
    return df


def cap(df: pd.DataFrame, limits: dict[str, int]) -> pd.DataFrame:
    parts = []
    for source, g in df.groupby("source"):
        limit = limits.get(source)
        if limit and len(g) > limit:
            g = g.sample(n=limit, random_state=SEED)
        parts.append(g)
    return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "sms"], default="hf")
    ap.add_argument("--max-texts", type=int, default=12000, help="cap on email/SMS rows")
    ap.add_argument("--max-urls", type=int, default=6000, help="cap on URL rows")
    args = ap.parse_args()

    try:
        df = load_huggingface() if args.source == "hf" else load_sms()
    except Exception as e:
        if args.source == "hf":
            print(f"[warn] Hugging Face download failed ({e}).\n"
                  f"       Falling back to the UCI SMS Spam Collection.")
            df = load_sms()
        else:
            raise

    df = clean(df)
    df = cap(df, {"texts": args.max_texts, "urls": args.max_urls})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    print("\n(label 1 = phishing/spam, 0 = genuine)")


if __name__ == "__main__":
    main()
