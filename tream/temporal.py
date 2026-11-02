"""Dual-path temporal injection: kernel smoothing and speaker interaction."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import PackedBiLSTM


class KernelSmoothing(nn.Module):
    """Encode one modality and aggregate its complete preceding history."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 400,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.encoder = PackedBiLSTM(input_size, hidden_size, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Tensor, lengths: Tensor, decay: Tensor) -> Tensor:
        """Accept features [B, T, D], lengths [B], and ATDG decays [B]."""
        steps = features.size(1)
        encoded = self.encoder(features, lengths)

        positions = torch.arange(steps, device=features.device)
        distance = positions[:, None] - positions[None, :]
        valid = positions[None, :] < lengths.to(features.device)[:, None]
        support = (
            (distance >= 0)[None, :, :]
            & valid[:, :, None]
            & valid[:, None, :]
        )
        weights = torch.exp(
            -self.temperature * decay.float()[:, None, None] * distance.clamp_min(0)
        ) * support
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        history = torch.bmm(weights.to(encoded.dtype), encoded)
        output = self.fusion(torch.cat([encoded, history], dim=-1))
        return output.masked_fill(~valid[:, :, None], 0)


class SpeakerGraph(nn.Module):
    """Refine KS node features through local cross-speaker messages."""

    def __init__(
        self,
        hidden_size: int,
        num_speakers: int,
        window: int = 6,
        strength: float = 0.02,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window = window
        self.strength = strength
        self.susceptibility = nn.Embedding(num_speakers, 1)
        nn.init.uniform_(self.susceptibility.weight, 0.3, 0.8)
        self.compatibility = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_size),
            nn.Tanh(),
        )

    def forward(
        self, features: Tensor, speaker_ids: Tensor, mask: Tensor, decay: Tensor,
    ) -> Tensor:
        """Accept KS features [B, T, H], a valid-turn mask, and decay [B, 1]."""
        safe_ids = speaker_ids.masked_fill(~mask, 0)
        susceptibility = self.susceptibility(safe_ids).sigmoid().float()
        values = self.message(features).float()
        numerator = torch.zeros_like(values)
        denominator = torch.zeros_like(values[..., :1])

        for distance in range(1, min(self.window + 1, features.size(1))):
            source, target = features[:, :-distance], features[:, distance:]
            edges = (
                mask[:, :-distance] & mask[:, distance:]
                & (safe_ids[:, :-distance] != safe_ids[:, distance:])
            )
            compatibility = self.compatibility(torch.cat([source, target], -1)).sigmoid().float()
            weights = (
                compatibility * susceptibility[:, distance:]
                * torch.exp(-decay.float()[:, None, :] * distance)
                * edges[:, :, None]
            )
            numerator = numerator + F.pad(
                weights * values[:, :-distance], (0, 0, distance, 0),
            )
            denominator = denominator + F.pad(weights, (0, 0, distance, 0))

        messages = (numerator / (denominator + 1e-8)).to(features.dtype)
        output = features + self.strength * messages
        return output.masked_fill(~mask[:, :, None], 0)
