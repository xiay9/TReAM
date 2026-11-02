"""Distance-adjusted pair probabilities and auxiliary-gated decoding."""

from collections.abc import Mapping

import torch
from torch import Tensor


@torch.no_grad()
def estimate_distance_prior(
    pair_targets: Tensor, pair_mask: Tensor, smoothing: float = 1.0,
) -> Tensor:
    """Estimate pi(d) from admissible training annotations with additive smoothing.

    pair_targets and pair_mask have shape [N, T, T] for training dialogues.
    Validation and test annotations are excluded from this estimate.
    """
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    steps = pair_targets.size(1)
    positions = torch.arange(steps, device=pair_targets.device)
    distances = (positions[:, None] - positions[None, :]).expand_as(pair_targets)
    valid = pair_mask & (distances >= 0) & pair_targets.bool()
    counts = torch.bincount(distances[valid], minlength=steps).float()
    counts = counts + smoothing
    return counts / counts.sum()


@torch.no_grad()
def decode_pairs(
    output: Mapping[str, Tensor],
    *,
    distance_prior: Tensor,
    distance_weight: float,
    pair_threshold: float,
    emotion_threshold: float,
    cause_threshold: float,
    neutral_id: int,
) -> dict[str, Tensor]:
    """Decode with externally supplied, fixed validation-selected parameters.

    The distance prior adjusts log-odds before thresholding. Auxiliary gates
    use non-neutral emotion probability at i and cause probability at j.
    Raw pair probabilities are returned separately for ranking evaluation.
    """
    if not all(0 <= value <= 1 for value in (
        pair_threshold, emotion_threshold, cause_threshold,
    )):
        raise ValueError("decoding thresholds must lie in [0, 1]")
    logits = output["pair_logits"].float()
    steps = logits.size(1)
    prior = distance_prior.to(device=logits.device, dtype=torch.float32)
    if prior.ndim != 1 or prior.numel() < steps:
        raise ValueError("distance_prior must cover every candidate distance")
    if not torch.isfinite(prior).all() or (prior <= 0).any():
        raise ValueError("distance_prior entries must be finite and positive")
    prior = prior / prior.sum()

    positions = torch.arange(steps, device=logits.device)
    distances = (positions[:, None] - positions[None, :]).clamp_min(0)
    log_odds = logits[..., 1] - logits[..., 0]
    adjusted = (log_odds + distance_weight * prior[distances].log()).sigmoid()
    emotion_probabilities = output["emotion_logits"].float().softmax(dim=-1)
    cause_probabilities = output["cause_logits"].float().softmax(dim=-1)
    non_neutral = 1 - emotion_probabilities[..., neutral_id]
    cause = cause_probabilities[..., 1]
    predictions = (
        output["pair_mask"]
        & (adjusted >= pair_threshold)
        & (non_neutral[:, :, None] >= emotion_threshold)
        & (cause[:, None, :] >= cause_threshold)
    )
    return {
        "pair_predictions": predictions,
        "pair_probabilities": logits.softmax(dim=-1)[..., 1],
        "adjusted_pair_probabilities": adjusted,
        "emotion_predictions": emotion_probabilities.argmax(dim=-1),
        "cause_predictions": cause_probabilities.argmax(dim=-1),
    }


def pair_lists(predictions: Tensor) -> list[list[tuple[int, int]]]:
    """Convert a decoded boolean matrix into zero-based (emotion, cause) links."""
    return [
        [tuple(pair) for pair in dialogue.nonzero(as_tuple=False).cpu().tolist()]
        for dialogue in predictions
    ]
