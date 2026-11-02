"""Text-to-utterance aggregation and acoustic/visual feature projections."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class TextUtteranceEncoder(nn.Module):
    """Adapt a Transformers text backbone to packed dialogue utterances."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(backbone.config.hidden_size)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        cache_dir: str,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "TextUtteranceEncoder":
        """Load an explicitly selected backbone using the official AutoModel API."""
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(backbone)

    @staticmethod
    def pool_utterances(token_states: Tensor, spans: Tensor, mask: Tensor) -> Tensor:
        """Combine each utterance's last content token and its chunk's CLS state.

        token_states is [N, L, D] for N packed text chunks. spans is [B, T, 3]
        containing (chunk index, start, end) with a nonempty [start, end)
        content-token interval. Special separator tokens are outside that
        interval. Chunk indices refer to the flattened chunk batch.
        """
        safe_spans = spans.masked_fill(~mask[:, :, None], 0)
        chunk, start, end = safe_spans.unbind(dim=-1)
        last = (end - 1).clamp_min(0)
        active_chunks = chunk[mask]
        active_start, active_end = start[mask], end[mask]
        invalid = (
            (active_chunks < 0) | (active_chunks >= token_states.size(0))
            | (active_start < 0) | (active_start >= active_end)
            | (active_end > token_states.size(1))
        )
        if invalid.any():
            raise ValueError("valid utterance spans must address nonempty token intervals")
        utterances = token_states[chunk, last] + token_states[chunk, 0]
        return utterances.masked_fill(~mask[:, :, None], 0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        spans: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Encode text chunks and return padded utterance states [B, T, D]."""
        output = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
        )
        positions = torch.arange(spans.size(1), device=spans.device)
        mask = positions[None, :] < lengths.to(spans.device)[:, None]
        return self.pool_utterances(output.last_hidden_state, spans, mask)


class MultimodalUtteranceEncoder(nn.Module):
    """Align cached modality features with the text representation dimension."""

    def __init__(
        self,
        text_encoder: TextUtteranceEncoder,
        feature_dims: Mapping[str, int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if "text" in feature_dims:
            raise ValueError("feature_dims describes non-text modalities")
        self.text_encoder = text_encoder
        hidden_size = text_encoder.hidden_size
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )
            for name, size in feature_dims.items()
        })
        self.output_dims = {
            "text": hidden_size,
            **{name: hidden_size for name in feature_dims},
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        spans: Tensor,
        lengths: Tensor,
        modality_features: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Accept cached features [B, T, D_m] aligned to text utterances."""
        text = self.text_encoder(input_ids, attention_mask, spans, lengths)
        positions = torch.arange(text.size(1), device=text.device)
        mask = positions[None, :] < lengths.to(text.device)[:, None]
        features = {"text": text}
        for name, projection in self.projections.items():
            values = modality_features[name]
            if values.shape[:2] != text.shape[:2]:
                raise ValueError(f"{name} features must align with text utterances")
            features[name] = projection(values).masked_fill(~mask[:, :, None], 0)
        return features
