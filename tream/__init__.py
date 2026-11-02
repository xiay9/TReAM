"""Core modules for Temporal Recency Alignment across Modules."""

from .atdg import ATDG, DialoguePaceStatistics
from .decoding import decode_pairs, estimate_distance_prior, pair_lists
from .encoders import MultimodalUtteranceEncoder, TextUtteranceEncoder
from .model import TReAM, TReAMModel
from .objectives import JointExtractionLoss, build_pair_targets
from .temporal import KernelSmoothing, SpeakerGraph

__all__ = [
    "ATDG",
    "DialoguePaceStatistics",
    "JointExtractionLoss",
    "KernelSmoothing",
    "MultimodalUtteranceEncoder",
    "SpeakerGraph",
    "TReAM",
    "TReAMModel",
    "TextUtteranceEncoder",
    "build_pair_targets",
    "decode_pairs",
    "estimate_distance_prior",
    "pair_lists",
]
