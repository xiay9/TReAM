"""Sequence encoding and biaffine scoring layers used by TReAM."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LockedDropout(nn.Module):
    """Share a feature-dropout mask across the turns of each dialogue."""

    def __init__(self, probability: float = 0.1) -> None:
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("dropout probability must lie in [0, 1)")
        self.probability = probability

    def forward(self, features: Tensor) -> Tensor:
        if not self.training or self.probability == 0:
            return features
        keep = 1 - self.probability
        mask = features.new_empty(features.size(0), 1, features.size(2))
        return features * mask.bernoulli_(keep).div_(keep)


class PackedBiLSTM(nn.Module):
    """Bidirectional dialogue encoding with padding excluded from recurrence."""

    def __init__(
        self, input_size: int, hidden_size: int = 400, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or hidden_size % 2:
            raise ValueError("hidden_size must be a positive even integer")
        self.input_dropout = LockedDropout(dropout)
        self.output_dropout = LockedDropout(dropout)
        self.recurrent = nn.LSTM(
            input_size, hidden_size // 2, batch_first=True, bidirectional=True,
        )

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(
            self.input_dropout(features), lengths.detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        encoded, _ = self.recurrent(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=features.size(1),
        )
        return self.output_dropout(encoded)


class BiaffinePairScorer(nn.Module):
    """Score (emotion turn i, cause turn j) using KS representations."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.tail = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.weight = nn.Parameter(torch.empty(2, hidden_size + 1, hidden_size + 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: Tensor) -> Tensor:
        """Return logits [B, T, T, 2] for non-pair and pair classes."""
        head = F.pad(self.head(features), (0, 1), value=1.0)
        tail = F.pad(self.tail(features), (0, 1), value=1.0)
        return torch.einsum("bix,oxy,bjy->bijo", head, self.weight, tail)
