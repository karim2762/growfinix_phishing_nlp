"""
src/model.py - the bidirectional LSTM classifier.

Data flow (B = batch size, L = sequence length):

    word ids (B, L) -> Embedding -> Dropout -> BiLSTM -> Dropout
                    -> (+ has_url flag) -> Linear -> sigmoid -> P(phishing)
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class PhishingBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64,
                 dropout: float = 0.4, pad_idx: int = 0):
        super().__init__()

        # EMBEDDING: a lookup table that turns each word id into a learnable vector
        # of `embed_dim` numbers. Similar words (e.g. "verify", "confirm") end up
        # with similar vectors after training. padding_idx keeps the <pad> vector
        # at zero so padding never contributes anything.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # DROPOUT #1: randomly zeroes some embedding values during training.
        # WHY: forces the network not to rely on any single word -> less overfitting.
        self.embed_dropout = nn.Dropout(dropout)

        # BIDIRECTIONAL LSTM: an LSTM reads the sequence word by word and keeps a
        # "memory" of what it has seen. Bidirectional = two LSTMs, one reading left->right
        # and one right->left, so each word is understood with context from both sides.
        # WHY LSTM and not a plain RNN: its gates let it remember long-range cues
        # and avoid the vanishing-gradient problem.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # DROPOUT #2: same idea, applied to the sentence summary before the classifier.
        self.hidden_dropout = nn.Dropout(dropout)

        # LINEAR OUTPUT: combines the summary (2*hidden_dim: forward + backward)
        # with 1 extra number, the has_url flag, and produces ONE score (a "logit").
        self.fc = nn.Linear(hidden_dim * 2 + 1, 1)

        # SIGMOID: squashes the logit into a probability between 0 and 1.
        # It's applied in predict_proba(); during training we use
        # BCEWithLogitsLoss, which applies the sigmoid internally (more numerically
        # stable than sigmoid + BCELoss and supports class weights via pos_weight).
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, lengths, has_url):
        """Return raw logits, shape (B,). Positive = leans phishing."""
        x = self.embed_dropout(self.embedding(ids))  # (B, L, embed_dim)

        # Packing tells the LSTM each sequence's real length, so it stops at the
        # end of the text instead of "reading" the padding.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (2, B, hidden_dim)

        # h_n[0] = forward LSTM's final state, h_n[1] = backward LSTM's final state.
        summary = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*hidden_dim)
        summary = self.hidden_dropout(summary)

        features = torch.cat([summary, has_url.unsqueeze(1)], dim=1)
        return self.fc(features).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, ids, lengths, has_url):
        """Probability of phishing in [0, 1] (call model.eval() first)."""
        return self.sigmoid(self.forward(ids, lengths, has_url))
