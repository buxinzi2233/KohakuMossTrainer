"""The shared host region's directory: KOHAKUMOSS_SHM_DIR first, then /dev/shm, then the
system temp directory (macOS); and a Windows process is refused before any rank waits on
a broadcast (no shared memory is created here)."""

import os
import tempfile

import pytest

from kohakumoss.offload.weights import shared

pytestmark = pytest.mark.cpu


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHAKUMOSS_SHM_DIR", str(tmp_path))
    assert shared._shm_dir() == str(tmp_path)


def test_dev_shm_or_temp_dir(monkeypatch):
    monkeypatch.delenv("KOHAKUMOSS_SHM_DIR", raising=False)
    expected = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    assert shared._shm_dir() == expected


def test_temp_dir_without_dev_shm(monkeypatch):
    monkeypatch.delenv("KOHAKUMOSS_SHM_DIR", raising=False)
    monkeypatch.setattr(shared.os.path, "isdir", lambda path: False)
    assert shared._shm_dir() == tempfile.gettempdir()


def test_windows_is_refused_before_shared_memory(monkeypatch):
    monkeypatch.setattr(shared.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="POSIX shared memory"):
        shared.SharedHost(4096)
