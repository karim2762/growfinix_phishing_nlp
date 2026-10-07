"""
train.py - train the BiLSTM and the TF-IDF baseline, compare them, save plots.

Run:  python train.py
Everything is seeded and capped (--time-budget-min) to finish on a laptop CPU
in well under 15 minutes.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed: we only save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, fbeta_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import PhishingBiLSTM
from src.preprocess import SEED, Vocab, add_features, split_data

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "clean.csv"
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"


def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator -> same results on every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- metrics ----
def pick_threshold(y_true, proba) -> float:
    """Choose the decision threshold on the VALIDATION set by maximising F2.

    WHY F2? It weighs recall twice as much as precision. Missing a phishing email
    (false negative) is worse than flagging a safe one (false positive), so we
    deliberately lean towards catching more phishing.
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y_true, proba >= t, beta=2, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "threshold": thr,
    }


# ------------------------------------------------------------------ LSTM ----
def make_loader(ids, lengths, has_url, labels, batch_size, shuffle) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(ids), torch.from_numpy(lengths),
                       torch.tensor(has_url, dtype=torch.float32),
                       torch.tensor(labels, dtype=torch.float32))
    g = torch.Generator().manual_seed(SEED)  # reproducible shuffling
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()  # turns Dropout off
    probas = []
    for ids, lengths, url, _ in loader:
        probas.append(model.predict_proba(ids, lengths, url).numpy())
    return np.concatenate(probas)


def train_lstm(model, train_loader, val_loader, y_val, pos_weight, args):
    # CLASS IMBALANCE: pos_weight > 1 makes mistakes on the rarer class cost more.
    # (pos_weight = n_negative / n_positive; if phishing is rare, it's > 1.)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc, best_state, bad_epochs = -1.0, None, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ids, lengths, url, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths, url), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops exploding gradients in RNNs
            opt.step()
            total += loss.item() * len(y)

        val_auc = roc_auc_score(y_val, predict_loader(model, val_loader))
        print(f"epoch {epoch:2d} | train loss {total / len(train_loader.dataset):.4f} "
              f"| val ROC-AUC {val_auc:.4f} | {time.time() - start:.0f}s")

        # EARLY STOPPING: stop when validation score hasn't improved for `patience`
        # epochs, and go back to the best weights. WHY: after a point the network
        # just memorises the training set (overfits).
        if val_auc > best_auc:
            best_auc, best_state, bad_epochs = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break
        if (time.time() - start) / 60 > args.time_budget_min:
            print(f"time budget of {args.time_budget_min} min reached - stopping")
            break
    model.load_state_dict(best_state)
    return model


# ----------------------------------------------------------------- plots ----
def save_confusion(y_true, proba, thr, title, path):
    cm = confusion_matrix(y_true, (proba >= thr).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Phishing"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc(y_true, curves: dict, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, proba in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--time-budget-min", type=float, default=10.0,
                    help="stop LSTM training after this many minutes (keeps the run < 15 min)")
    args = ap.parse_args()

    set_seed()
    ASSETS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()

    if not DATA_CSV.exists():
        raise SystemExit("data/clean.csv not found - run `python download_data.py` first.")

    print("Preprocessing...")
    df = add_features(pd.read_csv(DATA_CSV))
    train_df, val_df, test_df = split_data(df)
    print(f"rows: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"| phishing share (train): {train_df['label'].mean():.1%}")

    # Vocabulary from TRAIN only, so the test set stays truly unseen.
    vocab = Vocab.build(train_df["tokens"])
    print(f"vocabulary size: {len(vocab)}")

    def encode(d):
        ids, lengths = vocab.encode_many(d["tokens"], args.max_len)
        return ids, lengths, d["has_url"].to_numpy(), d["label"].to_numpy()

    tr, va, te = encode(train_df), encode(val_df), encode(test_df)
    train_loader = make_loader(*tr, args.batch_size, shuffle=True)
    val_loader = make_loader(*va, 256, shuffle=False)
    test_loader = make_loader(*te, 256, shuffle=False)
    y_val, y_test = va[3], te[3]

    # ---------------- BiLSTM ----------------
    n_pos = int(train_df["label"].sum())
    pos_weight = (len(train_df) - n_pos) / max(n_pos, 1)
    model = PhishingBiLSTM(len(vocab), args.embed_dim, args.hidden_dim, args.dropout)
    print("\nTraining BiLSTM...")
    t = time.time()
    model = train_lstm(model, train_loader, val_loader, y_val, pos_weight, args)
    lstm_time = time.time() - t

    lstm_val = predict_loader(model, val_loader)
    lstm_test = predict_loader(model, test_loader)
    lstm_thr = pick_threshold(y_val, lstm_val)
    lstm_metrics = evaluate(y_test, lstm_test, lstm_thr)

    # ---------------- Baseline: TF-IDF + Logistic Regression ----------------
    # WHY a baseline? If a tiny linear model matches the LSTM, the LSTM isn't
    # earning its complexity. Comparing against it keeps us honest.
    def to_text(d):  # tokens -> one string; has_url becomes a pseudo-word
        return [" ".join(t) + (" hasurlflag" if u else "") for t, u in zip(d["tokens"], d["has_url"])]

    print("\nTraining TF-IDF + Logistic Regression baseline...")
    t = time.time()
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ]).fit(to_text(train_df), train_df["label"])
    base_time = time.time() - t
    base_val = baseline.predict_proba(to_text(val_df))[:, 1]
    base_test = baseline.predict_proba(to_text(test_df))[:, 1]
    base_thr = pick_threshold(y_val, base_val)
    base_metrics = evaluate(y_test, base_test, base_thr)

    # ---------------- save artifacts ----------------
    torch.save(model.state_dict(), MODELS / "lstm.pt")
    vocab.save(MODELS / "vocab.json")
    joblib.dump(baseline, MODELS / "baseline.joblib")
    config = {"max_len": args.max_len, "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
              "dropout": args.dropout, "threshold": lstm_thr, "vocab_size": len(vocab)}
    (MODELS / "config.json").write_text(json.dumps(config, indent=2))

    save_confusion(y_test, lstm_test, lstm_thr, "BiLSTM - confusion matrix (test)", ASSETS / "confusion_matrix_lstm.png")
    save_confusion(y_test, base_test, base_thr, "TF-IDF + LR - confusion matrix (test)", ASSETS / "confusion_matrix_baseline.png")
    save_roc(y_test, {"BiLSTM": lstm_test, "TF-IDF + LR": base_test}, ASSETS / "roc_curve.png")

    # ---------------- comparison table ----------------
    rows = [("TF-IDF + Logistic Regression", base_metrics, base_time),
            ("BiLSTM (PyTorch)", lstm_metrics, lstm_time)]
    lines = ["| Model | Precision | Recall | F1 | ROC-AUC | Threshold | Train time |",
             "|---|---|---|---|---|---|---|"]
    for name, m, secs in rows:
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                     f"{m['roc_auc']:.3f} | {m['threshold']:.2f} | {secs:.0f}s |")
    table = "\n".join(lines)
    (ASSETS / "comparison.md").write_text(table + "\n")
    (ASSETS / "metrics.json").write_text(json.dumps(
        {"lstm": lstm_metrics, "baseline": base_metrics}, indent=2))

    print("\n=== TEST SET RESULTS (recall = share of phishing we catch) ===")
    print(table)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min. Plots saved in {ASSETS}")


if __name__ == "__main__":
    main()
