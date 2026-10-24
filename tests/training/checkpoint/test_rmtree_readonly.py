"""A checkpoint directory left with read-only files (a Windows run) still deletes: the
rmtree onerror clears the bit and retries."""

import os
import shutil
import stat

import pytest

from kohakumoss.training.checkpoint import manager

pytestmark = pytest.mark.cpu


def test_rmtree_clears_readonly_files(tmp_path, monkeypatch):
    final = tmp_path / "step_00000007"
    (final / "sub").mkdir(parents=True)
    state = final / "sub" / "state.json"
    state.write_text("{}", encoding="utf-8")
    os.chmod(state, 0o444)
    real_unlink = os.unlink
    refused = []

    def windows_unlink(path, *args, **kwargs):
        """Windows refuses to unlink a file without its write bit."""
        mode = os.stat(path, dir_fd=kwargs.get("dir_fd")).st_mode
        if mode & stat.S_IWRITE:
            return real_unlink(path, *args, **kwargs)
        refused.append(path)
        raise PermissionError(13, "read-only file", str(path))

    monkeypatch.setattr(os, "unlink", windows_unlink)
    shutil.rmtree(final, onerror=manager._retry_readonly)
    assert refused == [os.path.basename(str(state))]
    assert not final.exists()
