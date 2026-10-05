"""Where the queue folder comes from: one settings file the person writes.

A person who always works on the same folder should not have to name it in every shell,
and two ways of naming it would let two commands act on different folders. jobq therefore
reads one key, ``queue_folder``, from a small TOML file named ``jobq_paths.toml`` in the
current directory or in a directory above it, and uses the nearest such file. That file is
the only thing that names the folder, so a command run anywhere inside a project acts on
the folder the project names.

Unknown keys are ignored, which leaves room for later settings without breaking an older
copy of jobq. A file that cannot be parsed is an error naming the file: silently acting on
a different queue folder than the file asks for is worse than refusing to act.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

FILE_NAME = "jobq_paths.toml"
QUEUE_FOLDER_KEY = "queue_folder"


class SettingsError(Exception):
    """A settings file exists but cannot be used; the message names the file."""


def find_settings_file(start: Path | None = None) -> Path | None:
    """The nearest ``jobq_paths.toml`` in ``start`` or a directory above it, if there is one."""
    here = Path(start) if start is not None else Path.cwd()
    here = here.resolve()
    for directory in [here, *here.parents]:
        candidate = directory / FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def read_settings(path: Path) -> dict:
    """The settings in one file, or a ``SettingsError`` naming it."""
    try:
        text = Path(path).read_bytes()
    except OSError as exc:
        raise SettingsError(f"could not read the settings file {path}: {exc}") from exc
    try:
        return tomllib.loads(text.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise SettingsError(f"could not parse the settings file {path}: {exc}") from exc


def queue_folder_in_file(path: Path) -> Path | None:
    """The queue folder a settings file names, or ``None`` when it names none.

    A relative path is taken from the directory of the file that holds it, so a file can
    name a folder beside itself and stay correct whatever directory inside the project a
    command is run from. A leading ``~`` is expanded.
    """
    value = read_settings(path).get(QUEUE_FOLDER_KEY)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SettingsError(
            f"the settings file {path} sets {QUEUE_FOLDER_KEY} to {value!r}; it takes the "
            "queue folder as a string"
        )
    folder = Path(value).expanduser()
    if not folder.is_absolute():
        folder = Path(path).resolve().parent / folder
    return folder


def queue_folder_source(start: Path | None = None) -> tuple[Path, Path] | None:
    """The queue folder and the file naming it, or ``None`` when no file names one."""
    path = find_settings_file(start)
    if path is None:
        return None
    folder = queue_folder_in_file(path)
    return None if folder is None else (folder, path)
