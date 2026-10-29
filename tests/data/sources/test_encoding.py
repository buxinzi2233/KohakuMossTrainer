"""Sidecar encodings: a UTF-8 BOM written by a Windows editor, and a caption saved in the
locale's legacy code page (Windows ANSI); both must read cleanly, the BOM mark must never
reach a tag."""

import json
import locale
import logging

import pytest

from kohakumoss_data.registry import SOURCES
from tests.data import fixtures

pytestmark = pytest.mark.cpu
BOM = "\ufeff"


def _bom(path: str, text: str) -> None:
    """Write `text` as UTF-8 with a BOM, like a Windows editor does."""
    fixtures.write(path, (BOM + text).encode("utf-8"))


def test_bom_in_sidecar_json_and_jsonl_manifest(tmp_path):
    root = str(tmp_path / "f")
    fixtures.write(f"{root}/a.png", fixtures.png_bytes(16, 32, 1))
    _bom(f"{root}/a.json", json.dumps({"tags": ["bom_a", "bom_b"], "artist": "w"}))
    rows = list(SOURCES.get("folder")(root).scan())
    assert rows[0].tags == ["bom_a", "bom_b"] and rows[0].artist == "w"

    fixtures.write(str(tmp_path / "img" / "p.png"), fixtures.png_bytes(16, 16, 2))
    path = str(tmp_path / "meta.jsonl")
    _bom(path, json.dumps({"image": "img/p.png", "tags": "x, y"}) + "\n")
    rows = list(SOURCES.get("jsonl")(path).scan())
    assert rows[0].tags == ["x", "y"]


def test_bom_caption_keeps_tags_clean(tmp_path):
    root = str(tmp_path / "f")
    fixtures.write(f"{root}/a.png", fixtures.png_bytes(16, 16, 3))
    _bom(f"{root}/a.txt", "first_tag, second_tag")
    rows = list(SOURCES.get("folder")(root).scan())
    assert rows[0].tags == ["first_tag", "second_tag"]
    assert BOM not in "".join(rows[0].tags)


def test_legacy_caption_falls_back_to_the_locale_code_page(
    tmp_path, monkeypatch, caplog
):
    root = str(tmp_path / "f")
    fixtures.write(f"{root}/a.png", fixtures.png_bytes(16, 16, 4))
    caption = str(tmp_path / "f" / "a.caption")
    fixtures.write(caption, "\u6c49\u670d, \u5fae\u7b11".encode("gbk"))
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *_: "gbk")
    with caplog.at_level(logging.WARNING, logger="kohakumoss_data.sources.base"):
        rows = list(SOURCES.get("folder")(root).scan())
    assert rows[0].caption_nl == "\u6c49\u670d, \u5fae\u7b11"
    assert "\ufffd" not in rows[0].caption_nl
    assert caption in caplog.text
