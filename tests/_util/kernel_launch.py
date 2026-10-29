"""Require an explicit backend's forward to launch its own kernel."""

import sys
from functools import wraps


def require_kernel_launch(fn, backend, monkeypatch):
    """Wrap `fn` with a real-launch check; eager fallback cannot pass as the backend."""
    if backend == "tilelang":
        cls = sys.modules[
            "kohakumoss_kernels.backends.tilelang.blocks._blockops"
        ].OnDevice
        method = "__call__"
    else:
        cls = sys.modules["kohakumoss_kernels.backends.triton._common"].Launcher
        method = "_launch"
    original = getattr(cls, method)
    calls = []

    def launch(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(None)
        return result

    monkeypatch.setattr(cls, method, launch)

    @wraps(fn)
    def checked(*args, **kwargs):
        before = len(calls)
        result = fn(*args, **kwargs)
        assert len(calls) > before, f"{backend} returned without a kernel launch"
        return result

    return checked
