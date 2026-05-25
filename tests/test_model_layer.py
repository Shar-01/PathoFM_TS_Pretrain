from __future__ import annotations

import torch

from seqcore.models import SequenceStack, SequenceUnit, FeatureAdapter, OffsetEncoding
from seqcore.models.prediction_head import make_projection_head
from seqcore.models.query_objective import query_group_error
from seqcore.models.masking import local_span_mask, suffix_mask
from seqcore.models.token_layout import add_group_offset, add_time_offset


def test_offset_encoding_preserves_shape_and_is_nonpersistent_buffer() -> None:
    pe = OffsetEncoding(d_model=6, max_len=12)
    x = torch.zeros(2, 5, 6)
    out = pe(x)
    assert out.shape == x.shape
    assert "pe" in dict(pe.named_buffers())
    assert "pe" not in pe.state_dict()


def test_adapter_stack_head_components_preserve_expected_shapes() -> None:
    x = torch.randn(2, 100, 4)
    embedding = FeatureAdapter(in_dim=4, embed_dim=8)
    z = embedding(x)
    assert z.shape == (2, 100, 8)

    encoder = SequenceStack(depth=1, d_model=8, n_heads=2)
    encoded = encoder(z)
    assert encoded.shape == (2, 100, 8)

    decoder = make_projection_head(embed_dim=8, output_dim=4)
    decoded = decoder(encoded)
    assert decoded.shape == x.shape


def test_transform_functions_apply_expected_masks() -> None:
    torch.manual_seed(0)
    x = torch.ones(2, 100, 4)
    xm, mask = local_span_mask(x, p_max=0.7, c_max=6)
    assert xm.shape == x.shape
    assert mask.shape == x.shape
    assert mask.dtype is torch.bool
    assert mask.any()
    assert torch.equal(xm[mask], torch.zeros_like(xm[mask]))
    assert torch.equal(xm[~mask], x[~mask])

    fut_xm, fut_mask = suffix_mask(x, past_len=50)
    assert not fut_mask[:, :50].any()
    assert fut_mask[:, 50:].all()
    assert torch.equal(fut_xm[:, :50], x[:, :50])
    assert torch.equal(fut_xm[:, 50:], torch.zeros_like(fut_xm[:, 50:]))


def test_layout_helpers_add_offsets_in_place_semantics() -> None:
    z = torch.zeros(2, 3, 4)
    time_pos = torch.ones(5, 4)
    z_with_time = add_time_offset(z, time_pos, 3)
    assert z_with_time is z
    assert torch.equal(z_with_time, torch.ones_like(z_with_time))

    row_type_emb = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    rows = torch.zeros(2, 3, 4)
    row_type = torch.tensor([0, 1])
    rows_with_type = add_group_offset(rows, row_type_emb, row_type)
    assert rows_with_type is rows
    assert torch.equal(rows_with_type[0, 0], row_type_emb[0])
    assert torch.equal(rows_with_type[1, 0], row_type_emb[1])


def test_unit_forward_paths_and_static_transforms() -> None:
    torch.manual_seed(123)
    model = SequenceUnit(input_dim=4, embed_dim=8, depth=1, heads=2)
    full = torch.randn(3, 100, 4)

    rec, masked_view = model.forward_masked(full, mask_ratio=0.7)
    assert rec.shape == full.shape
    assert masked_view.shape == full.shape
    assert masked_view.dtype is torch.bool

    pred_future, suffix_mask_tensor = model.forward_projection(full)
    assert pred_future.shape == (3, 50, 4)
    assert suffix_mask_tensor.shape == (3, 50, 4)
    assert suffix_mask_tensor.all()

    table = full.clone()
    row_type = torch.tensor([0, 1, 0])
    query_mask = torch.zeros_like(table, dtype=torch.bool)
    query_mask[1] = True
    table[1] = 0.0
    table_loss = model.forward_table(table, row_type, query_mask)
    assert table_loss.ndim == 0
    assert torch.isfinite(table_loss)

    xm_static, mask_static = SequenceUnit._suffix_mask(full)
    assert xm_static.shape == full.shape
    assert mask_static[:, 50:].all()


def test_query_group_error_matches_manual_mse() -> None:
    rec = torch.tensor([[[1.0, 2.0]], [[5.0, 7.0]]])
    target = torch.tensor([[[0.0, 0.0]], [[2.0, 3.0]]])
    row_type = torch.tensor([0, 1])
    query_mask = torch.zeros_like(target, dtype=torch.bool)
    query_mask[1] = True
    loss = query_group_error(rec, target, row_type, query_mask)
    expected = torch.mean(torch.tensor([(5.0 - 2.0) ** 2, (7.0 - 3.0) ** 2]))
    assert torch.equal(loss, expected)
