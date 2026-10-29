"""TileLang fused linears: a fused config that fails to compile is dropped and the
problem runs the next candidate, under TUNE=off as well."""

import pytest

from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytest.importorskip("tilelang")
fused_linear = pytest.importorskip(
    "kohakumoss_kernels.backends.tilelang.blocks.fused_linear"
)
blockops = pytest.importorskip("kohakumoss_kernels.backends.tilelang.blocks._blockops")

pytestmark = pytest.mark.cpu
CFGS = [
    {"path": "fused", "bm": 64, "bn": 64, "bk": 32, "threads": 128, "stages": 1},
    {"path": "split"},
]


def _fused(cfg, x):
    raise blockops.CompileError("does not lower")


def _split(cfg, x):
    return ("split", x)


PATHS = {"fused": _fused, "split": _split}


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(blockops, "_DEAD", set())
    monkeypatch.setattr(fused_linear.FUSED, "bound", {})
    tuner.reset()
    yield
    tuner.reset()


def _no_retry(name, device, key, cfgs, paths, args):
    return fused_linear.FUSED._bound(name, device, key, cfgs, paths, args)(*args)


def _check_falls_back(run, name: str) -> None:
    for _ in range(2):
        try:
            got = run(name, "cpu", "k", CFGS, PATHS, (3,))
        except blockops.CompileError as err:
            raise AssertionError(f"re-raised: {err}") from err
        assert got == ("split", 3)


@pytest.mark.parametrize(
    "second_device,second_key", [("cpu", "other"), ("cpu:1", "first")]
)
def test_failed_config_does_not_poison_another_problem(
    fresh, monkeypatch, second_device, second_key
):
    monkeypatch.setattr(blockops, "tuned", lambda *args: args[3][0])
    configs = [{"rows": 4}, {"rows": 8}]

    def fails_first(cfg):
        if cfg == configs[0]:
            raise blockops.CompileError("shape-specific lowering failure")
        return cfg

    assert (
        blockops.tuned_run("local", "cpu", "first", configs, fails_first) == configs[1]
    )
    assert (
        blockops.tuned_run("local", second_device, second_key, configs, lambda c: c)
        == configs[0]
    )


def test_fused_failure_does_not_poison_another_problem(fresh):
    def fused(cfg, x):
        if x == 3:
            raise blockops.CompileError("shape-specific lowering failure")
        return ("fused", x)

    paths = {"fused": fused, "split": _split}
    run = fused_linear.FUSED.run
    assert run("local_fused", "cpu", "first", CFGS, paths, (3,)) == ("split", 3)
    assert run("local_fused", "cpu", "other", CFGS, paths, (4,)) == ("fused", 4)


def test_cached_compile_error_keeps_the_diagnostic_tail(monkeypatch):
    monkeypatch.setattr(blockops, "_CACHE", {})
    detail = "generated code\n" * 400 + "Compilation error: invalid device instruction"
    calls = []

    def build():
        calls.append(None)
        raise RuntimeError(detail)

    for _ in range(2):
        with pytest.raises(blockops.CompileError) as caught:
            blockops.compile_once(("diagnostic",), build, blockops.CompileError)
        assert detail in str(caught.value)
    assert len(calls) == 1


def test_failing_default_falls_back_to_split(fresh):
    _check_falls_back(fused_linear.FUSED.run, "t_dead_a")
    assert fused_linear.FUSED.bound[("t_dead_a", "cpu", "k")].args[0] == CFGS[1]
    must_fail(
        lambda: _check_falls_back(_no_retry, "t_dead_b"),
        "fused failure re-raised on every call",
    )
