"""Toolchain discovery and the host-compiler setup every JIT backend runs on import
(`ensure_toolchain`: the vcvars64 environment and `CXX` on Windows, a PATH report on
Linux). See docs/guides/install-windows.md#compiler-setup.
"""

import glob
import importlib.metadata
import importlib.util
import locale
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import torch

from kohakumoss_kernels.hardware.device import cache_dir, read_json, save_json
from kohakumoss_kernels.hardware.xpu_triton import utf8_sources

try:
    from triton.runtime.build import _find_compiler
except ImportError:
    _find_compiler = None

VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
VS_ROOTS = (
    r"C:\Program Files\Microsoft Visual Studio",
    r"C:\Program Files (x86)\Microsoft Visual Studio",
)
VC_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
VCVARS = Path("VC", "Auxiliary", "Build", "vcvars64.bat")
CL_GLOB = r"*\*\VC\Tools\MSVC\*\bin\Host*\**\cl.exe"
HOST_ARCH = {
    "AMD64": ["x64"],
    "X86_64": ["x64"],
    "ARM64": ["x64", "arm64"],
    "X86": ["x64", "x86"],
}
PATH_LISTS = ("PATH", "INCLUDE", "LIB", "LIBPATH")
MSVC_NAME = "cl.EXE"
NO_MSVC = "MSVC (x64 C++ tools) not found; install VS Build Tools with 'Desktop C++'"
VCVARS_TIMEOUT = 120
# A console-less (GUI) process must not flash a cmd window per vcvars call.
CREATIONFLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


@dataclass(frozen=True)
class Status:
    """Whether a component is usable, and why not when it is not."""

    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Toolchain:
    """The host compiler the JIT backends use. `source`: dev-shell | vcvars |
    vcvars-cache | env | posix | missing; `cxx` is the compiler path or None."""

    ok: bool
    source: str
    cxx: str | None
    msvc_version: str | None = None
    detail: str = ""
    sources: str = ""


def installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def version(dist: str) -> str | None:
    """Installed version of distribution `dist`, or None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


@cache
def _vswhere(*args: str) -> list[str]:
    """vswhere's output lines for `args`: once per process, and across processes from
    `<cache>/toolchain/vswhere.json` while every listed path (a Visual Studio install)
    still exists."""
    if not VSWHERE.exists():
        return []
    store = cache_dir() / "toolchain" / "vswhere.json"
    key = " ".join(args)
    saved = (read_json(store) or {}).get(key)
    if saved and all(Path(p).exists() for p in saved):
        return list(saved)
    out = subprocess.run(
        [str(VSWHERE), "-prerelease", "-products", "*", *args],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if lines:
        save_json(store, {**(read_json(store) or {}), key: lines})
    return lines


def vswhere(*args: str) -> list[str]:
    """vswhere's output lines for `args` (cached, `_vswhere`); empty without vswhere."""
    return _vswhere(*args)


def host_preference() -> list[str]:
    """This machine's MSVC `bin/Host*` directories, best first: x64 everywhere (x64
    emulation on ARM64), arm64 only as the fallback of an ARM64 machine."""
    return HOST_ARCH.get(platform.machine().upper(), ["x64"])


def _host_rank(path: str) -> tuple[int, str]:
    """Sort key of a `cl.exe`: this machine's host directory first, then the path."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    host = next((p[4:].lower() for p in parts if p.lower().startswith("host")), "")
    order = host_preference()
    return (order.index(host) if host in order else len(order), path.lower())


def msvc_compilers() -> list[str]:
    """Every MSVC `cl.exe` of any host architecture: under the installs vswhere lists,
    then under the Program Files roots (VS_ROOTS); this machine's host directory first."""
    found: list[str] = []
    for root in _vswhere("-all", "-property", "installationPath"):
        found += glob.glob(
            os.path.join(
                root, "VC", "Tools", "MSVC", "*", "bin", "Host*", "**", "cl.exe"
            ),
            recursive=True,
        )
    for root in VS_ROOTS:
        found += glob.glob(os.path.join(root, CL_GLOB), recursive=True)
    return sorted(dict.fromkeys(found), key=_host_rank)


@cache
def vcvars_script() -> Path | None:
    """`vcvars64.bat` of the newest Visual Studio with the x64 C++ tools, or None."""
    roots = _vswhere(
        "-latest", "-requires", VC_COMPONENT, "-property", "installationPath"
    )
    for root in roots:
        if (Path(root) / VCVARS).exists():
            return Path(root) / VCVARS
    found = [
        p for r in VS_ROOTS for p in glob.glob(os.path.join(r, "*", "*", str(VCVARS)))
    ]
    return Path(max(found)) if found else None


def _decode_env(out: bytes) -> str | None:
    """`out` decoded with the first of UTF-8, the locale code page and the Windows ANSI
    code page that yields no replacement character; None when none of them does (the
    capture is then dropped instead of polluting the environment with mojibake)."""
    for encoding in ("utf-8", locale.getpreferredencoding(False), "mbcs"):
        try:
            text = out.decode(encoding, errors="replace")
        except LookupError:
            continue
        if "\ufffd" not in text:
            return text
    return None


def _run_vcvars(script: Path) -> tuple[dict[str, list[str]], str]:
    """The environment change `script` makes — {VAR: ["prepend" | "set", value]} — and a
    short reason when the capture timed out, failed or could not be decoded ("" else).

    `cmd` writes `set` in the console's OEM/ANSI code page, so the output is captured as
    bytes and decoded by `_decode_env` rather than with the locale's text mode."""
    command = f'cmd /d /s /c ""{script}" >nul 2>&1 && set"'
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=VCVARS_TIMEOUT,
            creationflags=CREATIONFLAGS,
        )
    except subprocess.TimeoutExpired:
        return {}, f"vcvars64 timed out after {VCVARS_TIMEOUT} s"
    except OSError as err:
        return {}, f"vcvars64 could not run: {err}"
    out = _decode_env(proc.stdout or b"")
    if out is None:
        return (
            {},
            "vcvars64 output is not decodable in UTF-8, the locale or ANSI code page",
        )
    detail = ""
    if proc.returncode != 0:
        why = _decode_env(proc.stderr or b"") or ""
        line = next((ln.strip() for ln in why.splitlines() if ln.strip()), "")
        detail = f"vcvars64 exited {proc.returncode}" + (
            f": {line[:200]}" if line else ""
        )
    delta: dict[str, list[str]] = {}
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key or key.startswith("="):
            continue
        before = os.environ.get(key)
        if before == value:
            continue
        if key.upper() in PATH_LISTS:
            known = {p.lower() for p in (before or "").split(os.pathsep)}
            added = [p for p in value.split(os.pathsep) if p and p.lower() not in known]
            delta[key.upper()] = ["prepend", os.pathsep.join(added)]
        else:
            delta[key.upper()] = ["set", value]
    if not out.strip() and not detail:
        detail = "vcvars64 set no environment variables"
    return delta, detail


def _dirs_exist(delta: dict[str, list[str]]) -> bool:
    """Every directory a cached delta adds to a path list still exists."""
    for key in PATH_LISTS:
        mode_value = delta.get(key)
        if mode_value is None:
            continue
        for part in mode_value[1].split(os.pathsep):
            if part and not os.path.isdir(part):
                return False
    return "INCLUDE" in delta and "LIB" in delta


def _apply(delta: dict[str, list[str]]) -> None:
    for key, (mode, value) in delta.items():
        if mode == "set":
            os.environ[key] = value
            continue
        current = os.environ.get(key, "")
        known = {p.lower() for p in current.split(os.pathsep)}
        added = [p for p in value.split(os.pathsep) if p and p.lower() not in known]
        if added:
            os.environ[key] = os.pathsep.join(added + ([current] if current else []))
    if os.environ.get("INCLUDE") and os.environ.get("LIB"):
        # setuptools / torch's cpp_extension then trust this environment instead of
        # querying the registry for the SDK.
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")
        os.environ.setdefault("MSSdk", "1")


def windows_sdk_present() -> bool:
    """True when a Windows 10/11 SDK include tree is here. Kept local on purpose:
    `kohakumoss.cli.missing` imports this module, so importing it back would cycle."""
    if os.environ.get("WindowsSdkDir"):
        return True
    roots = [os.environ.get(name) for name in ("ProgramFiles(x86)", "ProgramFiles")]
    pattern = os.path.join("Windows Kits", "10", "Include", "*", "ucrt", "stdio.h")
    return any(root and glob.glob(os.path.join(root, pattern)) for root in roots)


def _msvc_cxx(cl: str) -> str:
    """`cl` spelled `...\\cl.EXE` (see docs/guides/install-windows.md)."""
    return str(Path(cl).with_name(MSVC_NAME))


def _windows_toolchain() -> Toolchain:
    source = "dev-shell"
    detail = ""
    if not (
        os.environ.get("VCINSTALLDIR")
        and os.environ.get("INCLUDE")
        and os.environ.get("LIB")
        and shutil.which("cl")
    ):
        script = vcvars_script()
        if script is None:
            return Toolchain(False, "missing", os.environ.get("CXX"), None, NO_MSVC)
        key = f"{script}|{script.stat().st_mtime_ns}"
        stored = read_json(cache_dir() / "toolchain" / "vcvars.json") or {}
        delta = stored.get(key)
        source = "vcvars-cache"
        if delta is None or not _dirs_exist(delta):
            delta, detail = _run_vcvars(script)
            source = "vcvars"
            if _dirs_exist(delta):
                save_json(cache_dir() / "toolchain" / "vcvars.json", {key: delta})
        _apply(delta)
    cl = shutil.which("cl")
    if cl is None:
        return Toolchain(
            False, "missing", os.environ.get("CXX"), None, detail or NO_MSVC
        )
    cxx = os.environ.get("CXX")
    if not cxx or Path(cxx).name.lower() == "cl.exe":
        os.environ["CXX"] = _msvc_cxx(cxx if cxx and os.path.isfile(cxx) else cl)
    elif shutil.which(cxx) is not None:
        source = "env"
    # A TileLang compiler query made before this point re-reads CXX on its next call.
    tilelang_cc = sys.modules.get("tilelang.contrib.cc")
    if tilelang_cc is not None:
        tilelang_cc.get_cplus_compiler.cache_clear()
    if not (os.environ.get("INCLUDE") and os.environ.get("LIB")):
        return Toolchain(
            False,
            source,
            os.environ["CXX"],
            None,
            detail or "INCLUDE/LIB not set (kernel JIT finds no C/C++ headers or libs)",
        )
    if not windows_sdk_present():
        return Toolchain(
            False,
            source,
            os.environ["CXX"],
            None,
            detail or "no Windows SDK (Windows Kits\\10\\Include\\*\\ucrt\\stdio.h)",
        )
    return Toolchain(
        True, source, os.environ["CXX"], os.environ.get("VCToolsVersion"), detail
    )


@cache
def ensure_toolchain() -> Toolchain:
    """Set up the host compiler for every JIT backend, once per process, and make
    Intel Triton's source I/O UTF-8 on any console code page."""
    sources = utf8_sources()
    if sys.platform == "win32":
        return replace(_windows_toolchain(), sources=sources)
    cxx = os.environ.get("CXX")
    found = shutil.which(cxx) if cxx else shutil.which("g++") or shutil.which("clang++")
    detail = "" if found else "no g++ / clang++ on PATH"
    return Toolchain(
        found is not None, "env" if cxx else "posix", found, None, detail, sources
    )


def ensure_host_compiler() -> Status:
    """`ensure_toolchain()` as a Status (the host compiler nvcc and the JITs use)."""
    tc = ensure_toolchain()
    return Status(tc.ok, f"CXX={tc.cxx}" if tc.ok else tc.detail)


def _flavour(path: str | None) -> str:
    """A compiler as `name [msvc <version>]`, no directories (the doctor holds none)."""
    if not path:
        return "none"
    name = Path(path).name
    tools = os.environ.get("VCToolsVersion")
    return f"{name} (msvc {tools})" if name.lower() == "cl.exe" and tools else name


def _triton_compiler() -> str:
    """The launcher compiler Triton will use on this machine."""
    if not installed("triton"):
        return "triton not installed"
    if installed("triton.backends.intel") and torch.xpu.is_available():
        cxx = shutil.which(os.environ.get("CXX", "")) if os.environ.get("CXX") else None
        cc = (
            cxx
            or shutil.which("icpx")
            or shutil.which("dpclang++")
            or shutil.which("cl")
        )
        style = "msvc flags" if cc and MSVC_NAME in cc else "gcc flags"
        return f"{_flavour(cc)}, {style}" if cc else "none"
    if _find_compiler is None:
        return "unknown (triton has no runtime.build._find_compiler)"
    try:
        return _flavour(_find_compiler("c"))
    except RuntimeError as err:
        return f"none ({err})"


def compilers() -> dict[str, str]:
    """The host compiler each JIT backend uses here (names and versions, no paths)."""
    tc = ensure_toolchain()
    out = {"toolchain": f"{tc.source}: {_flavour(tc.cxx)}" if tc.ok else tc.detail}
    out["triton"] = _triton_compiler()
    out["triton.sources"] = tc.sources
    out["tilelang"] = (
        f"nvcc {nvcc_version()} host {_flavour(tc.cxx)}"
        if installed("tilelang")
        else "tilelang not installed"
    )
    out["compile"] = f"inductor host {_flavour(tc.cxx)}, gpu via triton"
    return out


def _prepend_path(directory: str) -> None:
    """Put `directory` first on this process's PATH, once (idempotent): the JITs spawn
    nvcc's siblings (ptxas, cicc) by name."""
    known = {p.lower() for p in os.environ.get("PATH", "").split(os.pathsep) if p}
    if directory.lower() in known:
        return
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = directory + (os.pathsep + current if current else "")


@cache
def nvcc_version() -> str | None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    exe = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    nvcc = shutil.which("nvcc") or (
        str(Path(home, "bin", exe))
        if home and Path(home, "bin", exe).is_file()
        else None
    )
    if nvcc is None:
        return None
    if sys.platform == "win32":
        _prepend_path(str(Path(nvcc).parent))
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return None
    line = next((ln for ln in out.splitlines() if "release" in ln), "")
    return line.split("release")[-1].strip() or None


def triton_status() -> Status:
    if not installed("triton"):
        hint = "triton-windows" if sys.platform == "win32" else "triton"
        return Status(False, f"triton not installed (pip install {hint})")
    return Status(
        True,
        version("triton") or version("triton-windows") or version("triton-xpu") or "?",
    )


def tilelang_status() -> Status:
    if not installed("tilelang"):
        return Status(False, "tilelang not installed")
    host = ensure_host_compiler()
    if not host.ok:
        return host
    return Status(True, version("tilelang") or "?")


def report() -> dict[str, str]:
    """Versions and component status, one line each (the doctor's view)."""
    out = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch.cuda": str(torch.version.cuda),
        "nvcc": str(nvcc_version()),
        "cuda.available": str(torch.cuda.is_available()),
        "xpu.available": str(hasattr(torch, "xpu") and torch.xpu.is_available()),
    }
    for name, status in (("triton", triton_status()), ("tilelang", tilelang_status())):
        out[name] = ("ok " if status.ok else "missing: ") + status.detail
    out.update({f"compiler.{k}": v for k, v in compilers().items()})
    return out
