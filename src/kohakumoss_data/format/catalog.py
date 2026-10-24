"""Catalog files: `write_catalog` writes the catalog layout (kohakumoss_data.catalog);
`read_catalog` opens one lazily. Only this layout is read: the scan cache rebuilds a
file of another layout by a rescan; a prebuilt one (a moss_shard dataset's catalog.db)
is an error naming the import command (REIMPORT)."""

import os
import time
from collections.abc import Iterable

from kohakumoss_data.catalog.table import Catalog, open_catalog
from kohakumoss_data.catalog.vault import NO_STAMP, CatalogVault, is_catalog_vault
from kohakumoss_data.types import CatalogRow

__all__ = [
    "REIMPORT",
    "Catalog",
    "read_catalog",
    "remove_file",
    "remove_vault",
    "write_catalog",
]
CHUNK = 4096
REMOVE_TRIES = 5
REMOVE_WAIT_S = 0.05
REIMPORT = (
    "import the source again with scripts/data/catalog_build.py (import_to in its "
    "dataset entry)"
)


def remove_file(path: str) -> None:
    """`os.remove(path)`, retried while another process holds the file open (Windows
    raises PermissionError where POSIX unlinks anyway: an extra attempt changes nothing
    there); the last failure is a PermissionError naming the file."""
    for attempt in range(REMOVE_TRIES):
        try:
            os.remove(path)
            return
        except PermissionError as err:
            if attempt == REMOVE_TRIES - 1:
                raise PermissionError(
                    f"{path}: still open in another process, cannot delete it"
                ) from err
            time.sleep(REMOVE_WAIT_S * 2**attempt)


def remove_vault(path: str) -> None:
    """Delete a vault file and its SQLite sidecars (`remove_file`: a reader holding one
    open is retried, then reported with its name)."""
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            remove_file(path + suffix)


def write_catalog(path: str, rows: Iterable[CatalogRow]) -> int:
    """Write rows to a fresh catalog file at `path` (single writer); returns the count."""
    remove_vault(path)
    vault = CatalogVault(path, write=True)
    count, chunk = 0, []
    for row in rows:
        chunk.append(row)
        if len(chunk) == CHUNK:
            count += _append(vault, chunk)
            chunk = []
    count += _append(vault, chunk)
    vault.commit(1, "catalog")
    return count


def _append(vault: CatalogVault, rows: list[CatalogRow]) -> int:
    if rows:
        vault.append(rows, [r.locator for r in rows], [NO_STAMP] * len(rows))
    return len(rows)


def read_catalog(path: str) -> Catalog:
    """The catalog file at `path`, opened lazily; FileNotFoundError when it is absent,
    ValueError naming REIMPORT when it has another layout."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no catalog file {path}")
    if not is_catalog_vault(path):
        raise ValueError(f"{path} is not a catalog file of this layout; {REIMPORT}")
    return open_catalog(path)
