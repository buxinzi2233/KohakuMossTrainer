"""Per-sample storage views preserve CPU values, offsets and gradients without compiling."""

import pytest
import torch

from tests._util.negative import must_fail

blockops = pytest.importorskip("kohakumoss_kernels.backends.tilelang.blocks._blockops")
pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("columns", (1, 2))
def test_storage_view_preserves_strided_modulation(columns):
    mod = torch.arange(3 * 6 * 16, dtype=torch.float32).reshape(3, 6, 16)
    mod.requires_grad_()
    view = mod[:, 2, ::columns]
    flat, rs, cs = blockops.per_sample(view, *view.shape, view.device)
    assert flat.untyped_storage().data_ptr() == mod.untyped_storage().data_ptr()
    assert flat.storage_offset() == view.storage_offset()
    offsets = torch.arange(3)[:, None] * rs + torch.arange(view.shape[1])[None, :] * cs
    got = flat[offsets]
    assert torch.equal(got, view)
    actual = torch.autograd.grad(got.square().sum(), mod, retain_graph=True)[0]
    expected = torch.autograd.grad(view.square().sum(), mod)[0]
    assert torch.equal(actual, expected)

    def check_dense_rows():
        dense = (
            torch.arange(3)[:, None] * view.shape[1]
            + torch.arange(view.shape[1])[None, :]
        )
        assert torch.equal(flat[dense], view)

    must_fail(check_dense_rows, "row and column strides ignored")
