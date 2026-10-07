# task2_phishing_nlp - Phishing / Spam detection with an LSTM

Classify an email, SMS or URL as **Phishing/Spam** or **Safe** using deep learning on text
(Python, NLTK, PyTorch BiLSTM), compared against a TF-IDF + Logistic Regression baseline.

> ## ⚠️ Ethics & disclaimer
> **This is an educational project, not a production security tool.**
> - It is trained on limited public datasets; attackers constantly change tactics, so it will miss
>   real phishing and flag legitimate messages.
> - A "Safe" result is **not** a guarantee. Never use it to decide whether to click a link, open an
>   attachment or share credentials.
> - Datasets may contain real personal data (names, addresses, numbers). Don't try to re-identify
>   anyone, and don't paste confidential/personal messages into the app.
> - Don't use this to build or optimise phishing messages (e.g. by tuning text until it "passes").
> - Models can be biased by their data (language, region, writing style). Evaluate before any real use.

## Quick start (from the `task2_phishing_nlp/` folder)

```bash
# 1. install (Python 3.10-3.12; a virtual env keeps your monorepo clean)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. download + clean the data -> data/clean.csv
python download_data.py              # add `--source sms` for the small UCI SMS dataset

# 3. train both models, write metrics + plots to assets/ and the model to models/
python train.py

# 4. launch the web app
streamlit run app.py
```

Optional: `pip install jupyter && jupyter notebook notebooks/exploration.ipynb`

## Project layout

```
download_data.py     download + merge + clean -> data/clean.csv
src/preprocess.py    HTML/URL removal, has_url flag, NLTK tokenising/stopwords, Vocab, padding, split
src/model.py         PyTorch BiLSTM (every layer explained in comments)
src/predict.py       load model, score text, "suspicious word" explanation
train.py             training, early stopping, metrics, baseline, plots
app.py               Streamlit UI
assets/              confusion matrices, ROC curve, comparison.md, metrics.json (created by train.py)
notebooks/           exploration.ipynb
```

## Design choices (the "why")

| Choice | Reason |
|---|---|
| Data: HF `ealvaradob/phishing-dataset` (emails/SMS + URLs), SMS Spam as fallback | Open, covers both text and links |
| Capped to ~18k rows, 200 tokens, small BiLSTM (64-d) | Whole run fits in <15 min on a CPU; `--time-budget-min` enforces it |
| URLs removed from text, `has_url` kept as a feature | URLs are unique strings that bloat the vocabulary, but "has a link" is a real signal |
| Pure-URL inputs are split into words (`paypal`, `login`...) | Otherwise they'd be empty after URL removal |
| `BCEWithLogitsLoss(pos_weight)` / `class_weight="balanced"` | Handles class imbalance without throwing data away |
| Stratified 70/15/15 split, vocabulary built on train only | Honest test score, no leakage |
| Early stopping on validation ROC-AUC | Stops before the network memorises the training set |
| Decision threshold tuned for **F2** on the validation set | Recall matters most: a missed phishing mail costs more than a false alarm |
| Seeds fixed (42), `pathlib` everywhere, pinned `requirements.txt` | Reproducible and cross-platform |

## Results

`python train.py` prints the table below and saves it to `assets/comparison.md`
(paste your own numbers here after training - they depend on the data sample):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | _run train.py_ | | | |
| BiLSTM (PyTorch) | _run train.py_ | | | |

Plots: `assets/confusion_matrix_lstm.png`, `assets/confusion_matrix_baseline.png`, `assets/roc_curve.png`.

## Known limitations

- The dataset mixes sources (emails, SMS, URLs) that differ a lot in style, so the model may partly learn
  "which source is this?" instead of "is this phishing?". Check per-source scores before trusting it.
- Word-level model: misspelled brands ("paypa1") or character tricks in URLs are only partly handled.
  A character-level model or URL features (length, digits, subdomains) would help.
- "Suspicious words" come from occlusion (hiding one word at a time) - an approximation of what the model
  uses, not a proof of why it decided.
- No adversarial testing.
