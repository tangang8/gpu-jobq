"""Crash-safe atomic writes and a file lock, plus the shared-permissions switch.

Queue state is written atomically (temp file in the same directory + ``os.replace``), so a
reader on any machine sees either the old complete file or the new one, never a torn mix.

Permissions follow the user's umask by default. When the policy of this machine sets
``shared_perms``, :func:`set_shared_perms` is called before the first write into the folder
(see :func:`jobq.store.apply_folder_perms`); from then on this process creates queue
state world-writable, which is what lets one person reach the same queue folder from
machines where their account has a different user id.
"""

from __future__ import annotations

import contextlib
import fcntl
import itertools
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

_SHARED_PERMS = False


def set_shared_perms(enabled: bool) -> None:
    """Turn world-writable creation of queue state on or off for this process."""
    global _SHARED_PERMS
    _SHARED_PERMS = bool(enabled)


def shared_perms() -> bool:
    """Whether this process creates queue state world-writable."""
    return _SHARED_PERMS


def lock_file_mode() -> int:
    """Creation mode for lock files: world-writable only under ``shared_perms``."""
    return 0o666 if _SHARED_PERMS else 0o644


_WRITER_SEQ = itertools.count()


def _writer_token() -> str:
    """A temp-file suffix unique to (process, thread, call).

    The pid alone is not enough: two threads of one process writing the same destination
    would pick the same temp name, and the second ``os.replace`` would fail after the first
    renamed the shared file away.
    """
    return f"{os.getpid()}.{threading.get_ident()}.{next(_WRITER_SEQ)}"


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp in the same dir + ``os.replace``).

    Parent directories are created if missing. The content is fsynced before the replace, so
    a crash right after the rename cannot leave the new name pointing at zero-length data;
    the directory fsync that persists the rename is best-effort, since not every filesystem
    allows opening a directory for fsync.

    Args:
        path: Destination file.
        text: Full file contents.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{_writer_token()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    if _SHARED_PERMS:
        with contextlib.suppress(OSError):  # another uid's file is not ours to chmod
            os.chmod(tmp, 0o666)
    os.replace(tmp, path)
    with contextlib.suppress(OSError):
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def atomic_write_json(
    path: str | Path,
    obj: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    trailing_newline: bool = True,
) -> None:
    """Serialize ``obj`` as JSON and write it atomically (see :func:`atomic_write_text`).

    Args:
        path: Destination file.
        obj: JSON-serializable object.
        indent: ``json.dumps`` indent.
        sort_keys: ``json.dumps`` key sorting.
        trailing_newline: Append a final newline after the JSON body.
    """
    body = json.dumps(obj, indent=indent, sort_keys=sort_keys)
    atomic_write_text(path, body + "\n" if trailing_newline else body)


@contextlib.contextmanager
def file_lock(path: str | Path, mode: int | None = None) -> Iterator[None]:
    """Hold an exclusive ``fcntl.flock`` on ``path`` for the duration of the block.

    Serializes a read-modify-write of queue state across processes and machines. The kernel
    drops the lock when the holding process dies, so a crash can never leave a stale lock
    behind. The lock file is separate from the data file (which is replaced, not modified,
    by :func:`atomic_write_text`) and its contents are never read or written.

    Args:
        path: Lock file (created if missing).
        mode: Creation mode; defaults to :func:`lock_file_mode`.
    """
    mode = lock_file_mode() if mode is None else mode
    fd = os.open(path, os.O_CREAT | os.O_RDWR, mode)
    try:
        if _SHARED_PERMS:
            with contextlib.suppress(OSError):  # another uid's file is not ours to chmod
                os.fchmod(fd, mode)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
