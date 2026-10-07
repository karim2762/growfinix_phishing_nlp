# task2_phishing_nlp

Phishing / spam detection for emails, SMS and URLs using a BiLSTM (PyTorch + NLTK), compared against a TF-IDF + Logistic Regression baseline.

**Disclaimer:** this is an educational project, not a security tool.
- It's trained on limited public data, so it will miss real phishing and flag some legit messages.
- A "Safe" result is not a guarantee. Don't use it to decide whether to click a link or share credentials.
- The datasets may contain real personal data. Don't try to identify anyone and don't paste private messages into the app.
- Don't use it to tune phishing messages until they pass.
- The model can be biased by its data (language, region, writing style).

## Setup

Run these from the `task2_phishing_nlp/` folder (Python 3.10-3.12).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python download_data.py              # use --source sms for the small UCI SMS dataset
python train.py
streamlit run app.py
```

Optional: `pip install jupyter && jupyter notebook notebooks/exploration.ipynb`

## Layout

```
download_data.py     download, merge and clean data -> data/clean.csv
src/preprocess.py    HTML/URL removal, has_url flag, tokenizing, stopwords, Vocab, padding, split
src/model.py         BiLSTM model
src/predict.py       load model, score text, suspicious word explanation
train.py             training, early stopping, metrics, baseline, plots
app.py               Streamlit UI
assets/              plots and metrics written by train.py
notebooks/           exploration.ipynb
```

## Design choices

| Choice | Reason |
|---|---|
| HF `ealvaradob/phishing-dataset` (emails/SMS + URLs), SMS Spam as fallback | Open data, covers both text and links |
| Capped to ~18k rows, 200 tokens, 64-d BiLSTM | Runs in under 15 min on a CPU |
| URLs removed from text, `has_url` kept as a feature | URLs are unique and bloat the vocab, but having a link is a real signal |
| Pure URL inputs split into words | Otherwise they'd be empty after URL removal |
| `BCEWithLogitsLoss(pos_weight)` / `class_weight="balanced"` | Handles class imbalance |
| Stratified 70/15/15 split, vocab built on train only | No leakage into the test score |
| Early stopping on validation ROC-AUC | Stops before it memorises the training set |
| Threshold tuned for F2 on validation | A missed phishing mail costs more than a false alarm |
| Seed 42, pathlib, pinned requirements | Reproducible and cross-platform |

## Results

`python train.py` prints a comparison table and saves it to `assets/comparison.md`. Numbers depend on the data sample, so paste yours here after training.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | | | | |
| BiLSTM (PyTorch) | | | | |

Plots: `assets/confusion_matrix_lstm.png`, `assets/confusion_matrix_baseline.png`, `assets/roc_curve.png`.

## Limitations

- The dataset mixes emails, SMS and URLs, which look very different, so the model may partly learn which source a row came from. Check per-source scores.
- It's word-level, so misspelled brands ("paypa1") and character tricks in URLs are only partly handled. A character-level model or URL features would help.
- The suspicious words come from hiding one word at a time, which only approximates what the model uses.
- No adversarial testing.
