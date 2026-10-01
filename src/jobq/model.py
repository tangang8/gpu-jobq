"""The pure job model: the ``Job`` dataclass, filesystem-safe keys, jobs-file expansion.

No I/O and no GPU here, so expansion is unit-testable. A ``Job``'s ``key`` is the
human-readable id; its ``jobkey`` is that key sanitized into a single path segment used for
claim dirs / result files / per-job logs.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
from dataclasses import dataclass, field

# Anything that would fork a path or read as whitespace collapses to a single "__".
# Keep well under the 255-byte filename limit: logs prepend a stamp and append .<host>.log.
MAX_JOBKEY_BYTES = 180

# What a filesystem allows in one path component. A name is capped in bytes, which is what
# the kernel counts, not in characters: one character outside ASCII takes several bytes.
NAME_MAX_BYTES = 255

# The longest names built around a jobkey, beside its own: the per-job log adds a stamp,
# an underscore, a dot, this machine's name and ``.log``; a temporary file written beside a
# result or an attempts record adds the extension, the temp marker and a writer token
# (process, thread and call counter).
_LOG_STAMP_BYTES = len("20260101T000000Z_")
_LOG_SUFFIX_BYTES = len(".") + len(".log")
TEMP_NAME_OVERHEAD_BYTES = len(".json") + len(".tmp.") + 48

_UNSAFE = re.compile(r"[|/\\\s]+")


def _clip_bytes(text: str, limit: int) -> str:
    """The longest prefix of ``text`` whose encoding fits ``limit`` bytes.

    Cutting the encoded form and dropping an incomplete tail keeps whole characters: half a
    character is neither a name the reader recognises nor valid text.
    """
    encoded = text.encode()
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode(errors="ignore")


def sanitize_jobkey(key: str) -> str:
    """Collapse ``|``, ``/``, ``\\`` and whitespace runs in ``key`` into ``__``.

    Idempotent: sanitizing an already-sanitized key is a no-op. This is what turns an
    ``A|B|C`` style key into one filesystem segment.
    """
    jk = _UNSAFE.sub("__", key.strip())
    if len(jk.encode()) > MAX_JOBKEY_BYTES:
        # The jobkey becomes a filename (claims/<jobkey>, results/<jobkey>.json) and common
        # filesystems cap names at 255 bytes, so a longer one would make claiming raise.
        # Truncate + content hash keeps it unique and idempotent (the result is short).
        digest = hashlib.sha1(jk.encode()).hexdigest()[:12]
        jk = f"{_clip_bytes(jk, MAX_JOBKEY_BYTES - 13)}-{digest}"
    return jk


def unusable_name_reason(jobkey: str, hostname: str | None = None) -> str | None:
    """Why the names built from ``jobkey`` do not fit one path component, or ``None``.

    The jobkey itself is capped by :func:`sanitize_jobkey`, so what can still overflow is a
    name built around it: the per-job log on a machine with a long name, and the temporary
    file an atomic write puts beside a result or an attempts record.
    """
    host = socket.gethostname() if hostname is None else hostname
    size = len(jobkey.encode())
    log_size = size + _LOG_STAMP_BYTES + _LOG_SUFFIX_BYTES + len(host.encode())
    if log_size > NAME_MAX_BYTES:
        return (
            f"its log file name would be {log_size} bytes on {host}, and one file name "
            f"holds at most {NAME_MAX_BYTES}"
        )
    temp_size = size + TEMP_NAME_OVERHEAD_BYTES
    if temp_size > NAME_MAX_BYTES:
        return (
            f"the temporary file written beside its result would be {temp_size} bytes, "
            f"and one file name holds at most {NAME_MAX_BYTES}"
        )
    return None


@dataclass(frozen=True)
class Job:
    """One unit of work. ``cmd`` runs via ``bash -c``; the rest are optional overrides.

    Queue-level ``defaults`` (from ``meta.json``) fill in ``env``/``mem_mib``/``cwd`` when a
    per-job value is absent — that merge happens in the worker, not here.
    """

    key: str
    cmd: str
    env: dict[str, str] = field(default_factory=dict)
    mem_mib: int | None = None
    slots: int | None = None
    cwd: str | None = None

    @property
    def jobkey(self) -> str:
        """This job's ``key`` sanitized into a single filesystem path segment."""
        return sanitize_jobkey(self.key)

    def to_dict(self) -> dict:
        """Serialize for one ``jobs.jsonl`` line (drops empty/None optionals)."""
        d: dict = {"key": self.key, "cmd": self.cmd}
        if self.env:
            d["env"] = dict(self.env)
        if self.mem_mib is not None:
            d["mem_mib"] = self.mem_mib
        if self.slots is not None:
            d["slots"] = self.slots
        if self.cwd is not None:
            d["cwd"] = self.cwd
        return d

    @classmethod
    def from_dict(cls, d: dict, *, default_key: str | None = None) -> Job:
        """Rebuild a ``Job`` from a ``jobs.jsonl`` / JSONL-submit dict.

        ``key`` falls back to ``default_key`` (the raw submit line) when absent.
        """
        key = d.get("key") or default_key
        if not key:
            raise ValueError(f"job dict has no 'key' and no default: {d!r}")
        if "cmd" not in d:
            raise ValueError(f"job {key!r} has no 'cmd'")
        return cls(
            key=str(key),
            cmd=str(d["cmd"]),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            mem_mib=None if d.get("mem_mib") is None else int(d["mem_mib"]),
            slots=None if d.get("slots") is None else int(d["slots"]),
            cwd=None if d.get("cwd") is None else str(d["cwd"]),
        )


def parse_jobs_file(text: str, *, template: str | None = None) -> list[Job]:
    """Expand a jobs-file into ``Job``s.

    Each non-blank, non-``#`` line is either:

    - **JSONL** (starts with ``{``): parsed as a job dict; ``key`` defaults to the raw line.
    - **plain** (needs ``template``): the whole line becomes the ``key`` and the template is
      substituted with ``{line}`` = the whole line and ``{0},{1},...`` = the ``|``-split
      fields to build ``cmd``.

    Raises ``ValueError`` if a plain line appears without a ``template``, if a line that
    opens with ``{`` is not valid JSON, or if the template carries a brace that is not one
    of the placeholders above. Each message names the offending line.
    """
    jobs: list[Job] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                # A line opening with a brace is read as JSON, so a plain line that merely
                # starts with one lands here: name the line and both line shapes rather
                # than let a parser error out.
                raise ValueError(
                    f"jobs-file line opens with '{{' and is therefore read as JSON, but it "
                    f"is not valid JSON ({exc.msg}, column {exc.colno}): {line!r}. A plain "
                    "line (one expanded through --template) must not start with a brace."
                ) from exc
            jobs.append(Job.from_dict(d, default_key=line))
            continue
        if template is None:
            raise ValueError(
                f"plain jobs-file line needs --template to build cmd: {line!r}"
            )
        fields = line.split("|")
        try:
            cmd = template.format(*fields, line=line)
        except IndexError as exc:
            raise ValueError(
                f"template placeholder {exc} not satisfied by jobs-file line with "
                f"{len(fields)} '|'-field(s): {line!r} (valid: {{line}}, {{0}}..{{{len(fields) - 1}}})"
            ) from exc
        except (KeyError, ValueError) as exc:
            # Either a name that is not a placeholder or an unbalanced brace: a shell brace
            # expansion, a JSON fragment, an awk program. Both braces have to be doubled to
            # reach the command literally.
            raise ValueError(
                f"--template has a brace that is not a placeholder ({exc}) while expanding "
                f"jobs-file line {line!r}: {template!r}. Write a literal brace as "
                "'{{' or '}}'; the placeholders are {line} and {0},{1},..."
            ) from exc
        jobs.append(Job(key=line, cmd=cmd))
    return jobs
