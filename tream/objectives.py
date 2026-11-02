"""Temporal-precedence targets and the joint extraction objective."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def build_pair_targets(
    pairs: Tensor, pair_counts: Tensor, utterance_mask: Tensor,
) -> Tensor:
    """Build binary targets [B, T, T] from padded zero-based pairs [B, P, 2].

    The pair axes are (emotion turn, cause turn). pair_counts identifies real
    annotations in each row. Only links satisfying padding validity and j <= i
    enter the pair target matrix; self-links are admissible.
    """
    batch_size, steps = utterance_mask.shape
    lengths = utterance_mask.long().sum(dim=1)
    targets = torch.zeros(
        batch_size, steps, steps, device=utterance_mask.device, dtype=torch.long,
    )
    pairs = pairs.to(utterance_mask.device)
    emotion, cause = pairs.unbind(dim=-1)
    slots = torch.arange(pairs.size(1), device=pairs.device)
    active = slots[None, :] < pair_counts.to(pairs.device)[:, None]
    active = (
        active & (emotion >= 0) & (cause >= 0)
        & (emotion < lengths[:, None]) & (cause < lengths[:, None])
        & (cause <= emotion)
    )
    batch = torch.arange(batch_size, device=pairs.device)[:, None].expand_as(emotion)
    targets[batch[active], emotion[active], cause[active]] = 1
    return targets


def masked_cross_entropy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Average cross-entropy over admissible items, excluding padding."""
    active_logits = logits[mask]
    active_labels = labels.to(logits.device)[mask].long()
    if active_labels.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(active_logits, active_labels)


class JointExtractionLoss(nn.Module):
    """L_pair + alpha * L_emotion + beta * L_cause."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, output: Mapping[str, Tensor], targets: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Accept dense targets keyed by 'pair', 'emotion', and 'cause'.

        Pair targets are binary [B, T, T]; emotion and cause labels are [B, T].
        Padded labels are excluded by the model's masks. Emotion supervision
        uses the supplied category IDs and cause supervision uses {0, 1}.
        """
        pair = masked_cross_entropy(
            output["pair_logits"], targets["pair"], output["pair_mask"],
        )
        emotion = masked_cross_entropy(
            output["emotion_logits"], targets["emotion"], output["utterance_mask"],
        )
        cause = masked_cross_entropy(
            output["cause_logits"], targets["cause"], output["utterance_mask"],
        )
        return {
            "loss": pair + self.alpha * emotion + self.beta * cause,
            "pair_loss": pair,
            "emotion_loss": emotion,
            "cause_loss": cause,
        }
