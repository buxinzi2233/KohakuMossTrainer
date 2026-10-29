"""CPU fault injection of the attention host wrapper; no device kernel is compiled."""

import pytest
import torch

from kohakumoss_kernels import registry, status
from kohakumoss_kernels.backends.eager.attention import attention_varlen as eager
from kohakumoss_kernels.errors import BackendUnavailable
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

host = pytest.importorskip("kohakumoss_kernels.backends.tilelang.attention.varlen")
pytestmark = pytest.mark.cpu
OP = "test_host_attention_failure"
ERROR = "injected compiler failure"


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(status, "_CHECKS", {})
    monkeypatch.setitem(registry._STATE.impls, OP, {})
    monkeypatch.setitem(registry._STATE.priority, "cpu", ("injected", "eager"))
    monkeypatch.setattr(host, "_BOUND", {})
    status.clear_failures()
    yield
    status.clear_failures()


@pytest.mark.parametrize("mode", ("off", "table"))
@pytest.mark.parametrize("phase", ("forward", "backward"))
def test_host_failure_reaches_registry_before_autograd(
    mode, phase, isolated, monkeypatch
):
    calls = []

    def forward(q, k, v, cu, length, scale):
        calls.append("forward")
        if phase == "forward":
            raise RuntimeError(ERROR)
        out = eager(q, k, v, cu, length, scale)
        stat = q.new_zeros(q.shape[1], q.shape[0])
        return out, stat, stat

    def backward(*args):
        calls.append("backward")
        raise RuntimeError(ERROR)

    monkeypatch.setattr(host, "attn_varlen_tilelang", forward)
    monkeypatch.setattr(host, "attn_varlen_tilelang_bwd", backward)
    tuner.set_mode(mode)
    registry.register(OP, "injected", host.attention_varlen)
    registry.register(OP, "eager", eager)
    fn = registry.resolve(OP, "cpu", torch.float32)
    inputs = [torch.randn(7, 2, 4, requires_grad=True) for _ in range(3)]
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    out = fn(*inputs, cu, 4, 0.5)
    assert calls == (["forward"] if phase == "forward" else ["forward", "backward"])
    assert fn.backend == "eager"
    assert any(ERROR in text for text in registry.failures().values())
    grads = torch.autograd.grad(out.sum(), inputs)
    ref = eager(*inputs, cu, 4, 0.5)
    expected = torch.autograd.grad(ref.sum(), inputs)
    assert torch.equal(out, ref)
    for got, want in zip(grads, expected):
        assert torch.equal(got, want)

    def no_wrapper():
        try:
            host.attention_varlen(*inputs, cu, 4, 0.5)
        except BackendUnavailable as err:
            raise AssertionError("raw host wrapper did not fall back") from err

    must_fail(no_wrapper, "no registry fallback")
