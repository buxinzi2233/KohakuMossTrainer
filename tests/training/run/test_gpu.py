"""MFU peaks: known cards use their nominal tensor-core peak; unknown cards and
non-CUDA devices return None. Desktop peaks never stand in for laptop GPUs."""

import pytest
import torch

from kohakumoss.model import flops
from kohakumoss.model.flops import PEAK_TFLOPS, peak_tflops
from tests._util.negative import must_fail

CASES = {
    "NVIDIA GeForce RTX 4090": 165.2,
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA H100 80GB HBM3": 989.0,
    "Tesla V100-SXM2-16GB": 125.0,
    "NVIDIA GeForce RTX 3060": 25.6,
    "Tesla T10": 68.8,
    "Some Future GPU": None,
    "NVIDIA GeForce RTX 4070 Laptop GPU": None,
    "NVIDIA GeForce RTX 4090 Laptop GPU": None,
    "NVIDIA GeForce RTX 5090 Laptop GPU": None,
    "NVIDIA GeForce RTX 3060 Laptop GPU": None,
}


def check_peaks(peak, monkeypatch) -> None:
    for name, want in CASES.items():
        monkeypatch.setattr(flops.torch.cuda, "get_device_name", lambda d, n=name: n)
        assert peak(torch.device("cuda")) == want, (name, want)
    assert peak(torch.device("cpu")) is None


def _last_match(device):
    name = torch.cuda.get_device_name(device)
    return next((v for k, v in reversed(PEAK_TFLOPS) if k in name), None)


@pytest.mark.cpu
def test_peak_by_name(monkeypatch):
    check_peaks(peak_tflops, monkeypatch)
    must_fail(lambda: check_peaks(_last_match, monkeypatch), "the last key matched")


@pytest.mark.cuda
def test_peak_known():
    name = torch.cuda.get_device_name(torch.device("cuda"))
    matches = [
        v
        for k, v in PEAK_TFLOPS
        if k in name and ("Laptop" not in name or "Laptop" in k)
    ]
    actual = peak_tflops(torch.device("cuda"))
    if matches:
        assert actual == matches[0] > 0, name
    else:
        assert actual is None, f"invented PEAK_TFLOPS for {name}: {actual}"
