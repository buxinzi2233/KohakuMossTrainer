"""Windows file-in-use semantics: a delete / replace another process holds open is
retried; a lock release or a shard merge degrades with a warning instead of failing, and
a delete that never succeeds names the file."""

import logging
import os
from unittest import mock

import pytest
import torch

from kohakumoss_common import io as io_mod
from kohakumoss_data.cache.store import TensorStore, merge_shards, shard_path
from kohakumoss_data.format import catalog as catalog_mod
from kohakumoss_data.format.catalog import read_catalog
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan import lock, manage, paths
from kohakumoss_data.scan.runner import scan_catalog
from tests.data import fixtures

pytestmark = pytest.mark.cpu


def _exists(path: str) -> str:
    """An empty file (a stand-in vault another process has open)."""
    with open(path, "wb"):
        pass
    return path


def _in_use() -> PermissionError:
    return PermissionError(13, "file is in use by another process")


def test_remove_vault_retries_a_file_held_open(tmp_path, monkeypatch):
    path = _exists(str(tmp_path / "c.db"))
    real = os.remove
    tried = []

    def flaky(target):
        tried.append(target)
        if len(tried) == 1:
            raise _in_use()
        return real(target)

    monkeypatch.setattr(catalog_mod, "REMOVE_WAIT_S", 0)
    with mock.patch("os.remove", side_effect=flaky):
        catalog_mod.remove_vault(path)
    assert tried == [path, path]
    assert not os.path.exists(path)


def test_remove_vault_names_a_file_that_stays_open(tmp_path, monkeypatch):
    path = _exists(str(tmp_path / "busy.db"))
    monkeypatch.setattr(catalog_mod, "REMOVE_WAIT_S", 0)
    with mock.patch("os.remove", side_effect=_in_use()):
        with pytest.raises(PermissionError) as caught:
            catalog_mod.remove_vault(path)
    assert path in str(caught.value)


def test_release_leaves_a_lock_it_cannot_delete(tmp_path, monkeypatch, caplog):
    path = str(tmp_path / "c.db")
    assert lock.acquire(path)
    monkeypatch.setattr(catalog_mod, "REMOVE_WAIT_S", 0)
    with caplog.at_level(logging.WARNING, logger="kohakumoss_data.scan.lock"):
        with mock.patch("os.remove", side_effect=_in_use()):
            lock.release(path)
    assert os.path.exists(path + lock.LOCK)
    assert "left in place" in caplog.text


def test_merge_shards_leaves_a_shard_still_open(tmp_path, monkeypatch, caplog):
    main = str(tmp_path / "text.db")
    writer = TensorStore(main, writes=shard_path(main, 0))
    writer.put("a" * 32, torch.zeros(4, 8))
    writer.flush()
    del writer
    shard = shard_path(main, 0)
    monkeypatch.setattr(catalog_mod, "REMOVE_TRIES", 2)
    monkeypatch.setattr(catalog_mod, "REMOVE_WAIT_S", 0)
    with caplog.at_level(logging.WARNING, logger="kohakumoss_data.cache.store"):
        with mock.patch("os.remove", side_effect=_in_use()):
            assert merge_shards(main) == 1
    assert os.path.exists(shard)
    assert "stop training before merging" in caplog.text


def _scanned(tmp_path):
    """A kohya tree and its catalog file, cold-scanned."""
    root = str(tmp_path / "d")
    fixtures.make_tree(root, 1, 2)
    source = SOURCES.get("kohya")(root)
    path = paths.scan_path(str(tmp_path / "cache"), {"source": "kohya", "path": root})
    scan_catalog(source, path, workers=1)
    return source, path


def test_compact_retries_a_target_held_open(tmp_path, monkeypatch):
    source, path = _scanned(tmp_path)
    real = os.replace
    replaced = []

    def flaky(src, dst):
        replaced.append(str(dst))
        if len(replaced) == 1:
            raise _in_use()
        return real(src, dst)

    monkeypatch.setattr(io_mod, "REPLACE_WAIT_S", 0)
    with mock.patch("os.replace", side_effect=flaky):
        assert manage.compact(source, path, workers=1) == 2
    assert len(replaced) >= 2
    assert read_catalog(path).live.all()
