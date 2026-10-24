"""A refused registration still degrades into pinned chunks when the runtime's last error
cannot be cleared (a build without a shared libcudart): one WARNING names the reason, no
exception escapes, and every view follows the data."""

import logging

import pytest
import torch

from kohakumoss.offload.host import arena as ha
from kohakumoss.offload.host import cudart
from tests._util.logs import collecting

pytestmark = pytest.mark.cpu


def test_reset_failure_keeps_the_fallback(monkeypatch):
    def refuse(ptr, n):
        return "cudaErrorHostMemoryAlreadyRegistered: injected"

    def unavailable():
        raise RuntimeError("no shared libcudart")

    monkeypatch.setattr(cudart, "register", refuse)
    monkeypatch.setattr(cudart, "reset_last_error", unavailable)
    monkeypatch.setattr(ha, "pinned_empty", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(ha, "arena_kind", lambda pinned, device: "registered")
    src = torch.randn(8, 4)
    arena = ha.HostArena([(src.shape, src.dtype)], True, "cuda")
    view = arena.fill(0, src)
    owner = torch.empty(0)
    owner.data = view
    arena.track(0, owner)
    with collecting(logging.WARNING) as logs:
        arena.finish()
    assert arena.kind == "chunks" and arena.pinned
    assert torch.equal(arena.view(0), src), "the values did not survive the move"
    assert owner.data_ptr() == arena.view(0).data_ptr(), "a tracked tensor left behind"
    text = "\n".join(logs.messages)
    assert "no shared libcudart" in text and "cudaHostRegister refused" in text, text
