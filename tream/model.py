"""TReAM modality fusion and emotion, cause, and biaffine pair heads."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .atdg import ATDG
from .encoders import MultimodalUtteranceEncoder
from .layers import BiaffinePairScorer
from .objectives import JointExtractionLoss
from .temporal import KernelSmoothing, SpeakerGraph


class TReAM(nn.Module):
    """Core network over pre-encoded utterance-level modality features."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        num_speakers: int,
        num_emotions: int = 7,
        hidden_size: int = 400,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")
        self.modalities = tuple(input_dims)
        self.atdg = ATDG(len(self.modalities), dropout=dropout)
        self.kernel_paths = nn.ModuleDict({
            name: KernelSmoothing(size, hidden_size, dropout)
            for name, size in input_dims.items()
        })
        self.fusion_logits = nn.Parameter(torch.ones(len(self.modalities)))
        self.speaker_graph = SpeakerGraph(hidden_size, num_speakers, dropout=dropout)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_emotions),
        )
        self.cause_head = nn.Linear(hidden_size, 2)
        self.pair_head = BiaffinePairScorer(hidden_size, dropout)

    def forward(
        self,
        features: Mapping[str, Tensor],
        speaker_ids: Tensor,
        lengths: Tensor,
    ) -> dict[str, Tensor]:
        """Return logits and masks for a padded batch.

        Each modality has shape [B, T, input_dims[name]]. Speaker IDs [B, T]
        are zero-based on valid turns; padding is ignored. lengths [B] must
        lie in [1, T]. All features and speaker IDs share the model's device.
        Pair logits use axes (batch, emotion turn, cause turn, class).
        """
        sequence_decay, graph_decay = self.atdg(speaker_ids, lengths)
        modality_states = torch.stack([
            self.kernel_paths[name](features[name], lengths, sequence_decay[:, index])
            for index, name in enumerate(self.modalities)
        ], dim=2)
        ks_states = torch.einsum(
            "btmh,m->bth", modality_states, self.fusion_logits.softmax(dim=0),
        )
        positions = torch.arange(ks_states.size(1), device=ks_states.device)
        mask = positions[None, :] < lengths.to(ks_states.device)[:, None]
        sg_states = self.speaker_graph(ks_states, speaker_ids, mask, graph_decay)
        pair_mask = (
            mask[:, :, None] & mask[:, None, :]
            & (positions[None, :] <= positions[:, None])[None, :, :]
        )
        return {
            "pair_logits": self.pair_head(ks_states),
            "emotion_logits": self.emotion_head(sg_states),
            "cause_logits": self.cause_head(sg_states),
            "utterance_mask": mask,
            "pair_mask": pair_mask,
        }


class TReAMModel(nn.Module):
    """Compose utterance encoding, dual-path reasoning, and joint supervision."""

    def __init__(
        self,
        encoder: MultimodalUtteranceEncoder,
        num_speakers: int,
        num_emotions: int = 7,
        hidden_size: int = 400,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.reasoner = TReAM(
            encoder.output_dims, num_speakers, num_emotions, hidden_size, dropout,
        )
        self.objective = JointExtractionLoss()

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        spans: Tensor,
        lengths: Tensor,
        speaker_ids: Tensor,
        modality_features: Mapping[str, Tensor],
        targets: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Run the core model on an aligned tensor batch.

        Text is packed into input_ids [N, L] with attention_mask [N, L].
        spans [B, T, 3] maps utterances to (chunk, start, end) token intervals.
        lengths [B] excludes padded utterances. Non-text features and speaker
        IDs share the [B, T] utterance axes. Optional dense supervision uses
        the keys 'emotion', 'cause', and 'pair'.
        """
        features = self.encoder(
            input_ids, attention_mask, spans, lengths, modality_features,
        )
        output = self.reasoner(features, speaker_ids, lengths)
        if targets is not None:
            output.update(self.objective(output, targets))
        return output
