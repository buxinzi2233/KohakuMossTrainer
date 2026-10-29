"""vcvars64's `set` output is captured as bytes and decoded with UTF-8, then the console's
code page, then the ANSI code page: a clean capture feeds the delta (non-ASCII user paths
included) and an unclean one is dropped instead of writing replacement characters into
`os.environ`. Timeouts, return codes and the capture's flags are reported, not raised.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kohakumoss_kernels.hardware import toolchain

pytestmark = pytest.mark.cpu
SCRIPT = Path("C:/VS/vcvars64.bat")
# `set` output as a Chinese Windows writes it: cp936 bytes for U+5F20 in a user path.
CP936 = (
    b"PATH=C:\\Users\\\xd5\xc5\\AppData\\Local\\Temp\r\n"
    b"INCLUDE=C:\\inc\r\n"
    b"LIB=C:\\lib\r\n"
)


@pytest.fixture
def cp936_locale(monkeypatch):
    """This console's code page is cp936; the test process's own locale is not."""
    monkeypatch.setattr(
        toolchain.locale, "getpreferredencoding", lambda do_setlocale=True: "cp936"
    )


def _environ_text() -> str:
    """Every key and value of `os.environ`, joined (the capture must not have written)."""
    return "\n".join(f"{k}={v}" for k, v in toolchain.os.environ.items())


def test_cp936_output_decodes_cleanly(cp936_locale, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("INCLUDE", raising=False)
    monkeypatch.delenv("LIB", raising=False)

    def run(cmd, **kwargs):
        assert "vcvars64.bat" in str(cmd)
        return SimpleNamespace(returncode=0, stdout=CP936, stderr=b"")

    monkeypatch.setattr(toolchain.subprocess, "run", run)
    delta, detail = toolchain._run_vcvars(SCRIPT)
    assert detail == ""
    assert delta["PATH"] == ["prepend", "C:\\Users\\\u5f20\\AppData\\Local\\Temp"]
    # INCLUDE / LIB are path lists too, so they are prepended, not assigned.
    assert delta["INCLUDE"] == ["prepend", "C:\\inc"]
    assert delta["LIB"] == ["prepend", "C:\\lib"]


def test_no_mojibake_reaches_the_environment(cp936_locale, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("INCLUDE", "C:\\existing\\inc")
    parent = _environ_text()
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout=CP936, stderr=b""),
    )
    toolchain._run_vcvars(SCRIPT)
    assert _environ_text() == parent, "the capture wrote into os.environ"
    assert toolchain.os.environ["PATH"] == "/usr/bin"


def test_capture_uses_bytes_timeout_and_no_window(monkeypatch):
    calls: list[dict] = []

    def run(cmd, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"FOO=bar\r\n", stderr=b"")

    monkeypatch.setattr(toolchain.subprocess, "run", run)
    toolchain._run_vcvars(SCRIPT)
    assert "text" not in calls[0] and "encoding" not in calls[0]
    assert calls[0]["capture_output"] is True
    assert calls[0]["timeout"] == toolchain.VCVARS_TIMEOUT


def test_timeout_is_a_detail_not_an_exception(cp936_locale, monkeypatch):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(toolchain.subprocess, "run", run)
    delta, detail = toolchain._run_vcvars(SCRIPT)
    assert delta == {} and "timed out" in detail


def test_nonzero_return_code_records_stderr(cp936_locale, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("INCLUDE", raising=False)
    monkeypatch.delenv("LIB", raising=False)
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=1, stdout=CP936, stderr=b"error: no SDK\r\n"
        ),
    )
    delta, detail = toolchain._run_vcvars(SCRIPT)
    assert delta["INCLUDE"] == ["prepend", "C:\\inc"], "a partial capture is not kept"
    assert "exited 1" in detail and "error: no SDK" in detail


def test_applied_environment_is_marked_as_an_sdk(monkeypatch):
    monkeypatch.setenv("INCLUDE", "C:\\inc")
    monkeypatch.setenv("LIB", "C:\\lib")
    monkeypatch.delenv("DISTUTILS_USE_SDK", raising=False)
    monkeypatch.delenv("MSSdk", raising=False)
    toolchain._apply({})
    assert toolchain.os.environ["DISTUTILS_USE_SDK"] == "1"
    assert toolchain.os.environ["MSSdk"] == "1"


def test_no_sdk_marker_without_include_lib(monkeypatch):
    monkeypatch.delenv("INCLUDE", raising=False)
    monkeypatch.delenv("LIB", raising=False)
    monkeypatch.delenv("DISTUTILS_USE_SDK", raising=False)
    monkeypatch.delenv("MSSdk", raising=False)
    toolchain._apply({})
    assert "DISTUTILS_USE_SDK" not in toolchain.os.environ
    assert "MSSdk" not in toolchain.os.environ


def test_windows_sdk_probe_reads_program_files(monkeypatch, tmp_path):
    stdio = tmp_path / "Windows Kits" / "10" / "Include" / "10.0.26100.0" / "ucrt"
    stdio.mkdir(parents=True)
    (stdio / "stdio.h").write_text("")
    monkeypatch.delenv("WindowsSdkDir", raising=False)
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert toolchain.windows_sdk_present(), "an SDK include tree was not seen"
    (stdio / "stdio.h").unlink()
    assert not toolchain.windows_sdk_present(), "an empty tree passed as an SDK"


def test_undecodable_output_is_dropped(cp936_locale, monkeypatch):
    monkeypatch.setattr(
        toolchain.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=0, stdout=b"PATH=\xff\xfe\r\n", stderr=b""
        ),
    )
    delta, detail = toolchain._run_vcvars(SCRIPT)
    assert delta == {}, "mojibake was applied"
    assert "decodable" in detail
