"""A real-backend check rejects an eager result with no backend launch."""

import sys
from types import SimpleNamespace

import pytest

from tests._util.kernel_launch import require_kernel_launch
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class Launcher:
    def _launch(self, value):
        return value + 1


def test_check_requires_a_launch_on_every_call(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "kohakumoss_kernels.backends.triton._common",
        SimpleNamespace(Launcher=Launcher),
    )
    launch = Launcher()
    checked = require_kernel_launch(launch._launch, "triton", monkeypatch)
    must_fail(lambda: checked(1), "a bound method bypasses instrumentation")
    checked = require_kernel_launch(lambda x: launch._launch(x), "triton", monkeypatch)
    assert checked(1) == 2
    fallback = require_kernel_launch(lambda x: x + 1, "triton", monkeypatch)
    must_fail(lambda: fallback(1), "an eager result without a backend launch")
    assert checked(2) == 3
