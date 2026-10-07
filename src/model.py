import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class PhishingBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64,
                 dropout: float = 0.4, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_dropout = nn.Dropout(dropout)
        # 2 * hidden_dim from both directions + 1 for the has_url flag
        self.fc = nn.Linear(hidden_dim * 2 + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, lengths, has_url):
        x = self.embed_dropout(self.embedding(ids))

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)

        summary = torch.cat([h_n[-2], h_n[-1]], dim=1)
        summary = self.hidden_dropout(summary)

        features = torch.cat([summary, has_url.unsqueeze(1)], dim=1)
        return self.fc(features).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, ids, lengths, has_url):
        return self.sigmoid(self.forward(ids, lengths, has_url))
