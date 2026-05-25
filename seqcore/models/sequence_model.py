"""Composed sequence unit."""
from __future__ import annotations

import torch
import torch.nn as nn

from seqcore.models.feature_encoder import FeatureAdapter
from seqcore.models.prediction_head import make_projection_head
from seqcore.models.token_layout import add_group_offset, add_time_offset
from seqcore.models.query_objective import query_group_error
from seqcore.models.sequence_encoder import SequenceStack
from seqcore.models.masking import local_span_mask, suffix_mask


class SequenceUnit(nn.Module):
    """Composed shared stack with three abstract output paths."""

    def __init__(self, input_dim: int = 33, embed_dim: int = 128, depth: int = 8, heads: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.token_emb = FeatureAdapter(input_dim, embed_dim)
        self.time_pos = nn.Parameter(torch.randn(512, embed_dim))
        self.row_type_emb = nn.Parameter(torch.randn(2, embed_dim))
        self.encoder = SequenceStack(depth, embed_dim, heads)
        self.decoder = make_projection_head(embed_dim, input_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        z = self.token_emb(x)
        return add_time_offset(z, self.time_pos, x.size(1))

    def _patchify_table(self, rows: torch.Tensor, row_type: torch.Tensor) -> torch.Tensor:
        z = self.token_emb(rows)
        z = add_time_offset(z, self.time_pos, rows.size(1))
        return add_group_offset(z, self.row_type_emb, row_type)

    @staticmethod
    def _local_mask(x: torch.Tensor, p_max: float = 0.4, c_max: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
        return local_span_mask(x, p_max=p_max, c_max=c_max)

    @staticmethod
    def _suffix_mask(x: torch.Tensor, past_len: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        return suffix_mask(x, past_len=past_len)

    def forward_masked(self, full: torch.Tensor, mask_ratio: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
        xm, mask = self._local_mask(full, mask_ratio)
        rec = self.decoder(self.encoder(self._patchify(xm)))
        return rec, mask

    def forward_projection(self, full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xm, fut_mask = self._suffix_mask(full)
        rec = self.decoder(self.encoder(self._patchify(xm)))
        return rec[:, 50:], fut_mask[:, 50:]

    def forward_table(self, table_masked: torch.Tensor, row_type: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self._patchify_table(table_masked, row_type))
        rec = self.decoder(z)
        return query_group_error(rec, table_masked, row_type, query_mask)
