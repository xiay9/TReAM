"""Dialogue pace statistics and the Adaptive Temporal Decay Generator."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DialoguePaceStatistics(nn.Module):
    """Build the eight-dimensional, label-free dialogue descriptor psi(D)."""

    def forward(self, speaker_ids: Tensor, lengths: Tensor) -> Tensor:
        """Accept speaker IDs [B, T] and positive dialogue lengths [B]."""
        rows = []
        for batch_index, length in enumerate(lengths.detach().cpu().tolist()):
            speakers = speaker_ids[batch_index, :length]
            _, counts = torch.unique(speakers, return_counts=True)
            counts = counts.float()
            probabilities = counts / length
            speaker_count = counts.numel()
            switches = (speakers[1:] != speakers[:-1]).float().sum()
            switch_rate = switches / max(length - 1, 1)
            entropy = -(probabilities * probabilities.log()).sum()

            _, run_lengths = torch.unique_consecutive(speakers, return_counts=True)
            run_lengths = run_lengths.float()
            ranks = torch.arange(1, speaker_count + 1, device=speakers.device)
            imbalance = (
                (2 * ranks - speaker_count - 1) * counts.sort().values
            ).sum() / (speaker_count * length)

            rows.append(torch.stack([
                counts.new_tensor(length).log(),
                counts.new_tensor(speaker_count),
                switch_rate,
                entropy,
                run_lengths.mean(),
                run_lengths.std(unbiased=False),
                run_lengths.max(),
                imbalance,
            ]))
        return torch.stack(rows)


class ATDG(nn.Module):
    """Generate module-specific positive decays from a shared latent factor."""

    def __init__(
        self,
        num_modalities: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.statistics = DialoguePaceStatistics()
        self.latent = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.sequence_projection = nn.Linear(latent_dim, num_modalities)
        self.graph_projection = nn.Linear(latent_dim, 1)

    def forward(self, speaker_ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Return sequential decays [B, M] and graph decays [B, 1]."""
        descriptor = self.statistics(speaker_ids, lengths)
        latent = self.latent(descriptor.to(self.sequence_projection.weight.dtype))
        return (
            F.softplus(self.sequence_projection(latent)),
            F.softplus(self.graph_projection(latent)),
        )
