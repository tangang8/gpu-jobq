"""On-disk queue state: meta, jobs, claims, results, and the derived job states.

State layout under the queue folder ``<root>``::

    <root>/<queue>/meta.json                    queue config + defaults
    <root>/<queue>/jobs.jsonl                   one Job per line
    <root>/<queue>/claims/<jobkey>/owner.json   claim = mkdir claims/<jobkey> (atomic)
    <root>/<queue>/results/<jobkey>.json        terminal outcome (rc, gpu, log, attempt)
    <root>/<queue>/attempts/<jobkey>.json       requeue counter (survives result deletion)
    <root>/<queue>/failures.json                the queue's run of consecutive job failures
    <root>/<queue>/failures.unreadable.<stamp>.json  a failure record that could not be read
    <root>/<queue>/paused.json                  present while the queue is paused
    <root>/<queue>/mem_learned.json             what this queue's jobs were measured to need
    <root>/<queue>/complete.state.json          the cached "every job is terminal" verdict
    <root>/<queue>/logs/<stamp>_<jobkey>.a<attempt>.<host>.log
    <root>/<queue>/.jobs.lock                   serializes appends to jobs.jsonl
    <root>/<queue>/.attempts.lock               serializes attempts-sidecar rewrites
    <root>/<queue>/.failures.lock               serializes the failure run and the pause
    <root>/<queue>/.claims.lock                 serializes recovery of this queue's claims
    <root>/<queue>/.mem_learned.lock            serializes mem_learned.json rewrites
    <root>/.submit.lock                         serializes submissions and settings changes
    <root>/worker.<host>.{lock,pid,json}        one pool per machine, its pid and its log
    <root>/.worker.<host>.lock.rmw              serializes rewrites of worker.<host>.json
    <root>/stop.<host>                          present while this machine's pool is asked to stop
    <root>/stop.now.<host>                      present while that stop is asked to end running jobs
    <root>/yielded.<host>.json                  the claims this machine's pool parked when it yielded
    <root>/gpu_policy.<host>.json               this machine's scheduling policy

Job states are never stored — they are derived from the presence of a claim dir and a
result file: pending (neither) / running (claim, no result) / done (result rc==0) /
failed (result rc!=0). A result always wins: a claim left beside a result is terminal.

``jobs.jsonl`` is grown by an atomic full-file rewrite (read-all, append, replace), which
keeps the "all writes atomic" invariant at the cost of rewriting a file holding one line
per job. The rewrite is serialized on ``<queue>/.jobs.lock`` so concurrent submits cannot
lose a batch.

Machine safety (several machines share ``<root>``): a claim that names an owner is only
ever recovered automatically when that owner is this hostname and its process is gone. A
pid recorded by another machine means nothing here, so no other machine's claim is
touched except via ``force_release``. The one exception is a claim directory carrying no
owner record at all: ownership was never written, so it can never finish on its own, and
any machine may reclaim it once it is older than the grace and this process has seen it
without an owner on an earlier pass (see :func:`steal_stale`).
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import signal
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from jobq import io
from jobq.io import atomic_write_json, atomic_write_text, file_lock
from jobq.model import Job, sanitize_jobkey, unusable_name_reason

# Written into every new queue's meta.json. A meta without the key reads as 1.
FORMAT_VERSION = 1

# An owner-less claim dir older than this is crash debris and may be reclaimed by any machine.
ORPHAN_CLAIM_GRACE_S = 120.0


def utc_stamp() -> str:
    """A UTC stamp as year, month, day, ``T``, hour, minute, second, ``Z``, all digits."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (for log lines and result timestamps)."""
    return datetime.now(UTC).isoformat()


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process on this machine.

    The ``os.kill(pid, 0)`` probe also answers for a process that has exited but whose
    parent has not collected it, so a pid that passes the probe is read from the process
    table as well and a zombie counts as not alive. That entry is world-readable, so the
    check holds for a pid owned by another user too; when there is no entry to read, the
    probe's answer stands.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists but owned by another user
    except OSError:
        return False
    entry = read_proc_stat(pid)
    return entry is None or entry[0] != "Z"


BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"


def boot_id() -> str | None:
    """This machine's boot identifier, or ``None`` when the kernel does not offer one.

    A pid only means something within one boot: after a restart the same number belongs to
    an unrelated process, and a record carrying the boot it was written in can say so.
    """
    try:
        return Path(BOOT_ID_PATH).read_text().strip() or None
    except OSError:
        return None


def read_proc_stat(pid: int) -> tuple[str, int, int] | None:
    """``(run state, process group, start time in ticks)`` for ``pid``, or ``None``.

    The single reader of ``/proc/<pid>/stat``: everything anything here wants to know
    about a process id comes from this one line, so it is parsed in one place. The
    command name sits in parentheses and may hold spaces, so the fields are counted from
    the closing parenthesis: the run state is the first after it, the process group the
    third and the start time the twentieth.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        after = stat[stat.rindex(")") + 2 :].split()
        return after[0], int(after[2]), int(after[19])
    except (ValueError, IndexError):
        return None


def pid_start_ticks(pid: int) -> int | None:
    """When ``pid`` started, in clock ticks since boot, or ``None`` if it cannot be read.

    Together with the pid this identifies one process: the pair is not reused while the
    machine is up, so a recycled pid is recognisable as a different process.
    """
    entry = read_proc_stat(pid)
    return None if entry is None else entry[2]


def process_identity() -> dict:
    """The keys that pin a record to this process on this boot, for an owner or pid file."""
    out: dict = {}
    bid = boot_id()
    if bid is not None:
        out["boot_id"] = bid
    start = pid_start_ticks(os.getpid())
    if start is not None:
        out["pid_start"] = start
    return out


def process_gone(pid: int, record: dict | None) -> bool:
    """Whether the process a record describes is certainly gone from this machine.

    A record written before a restart names a different boot, so its pid means nothing and
    the process is gone. A record from this boot whose pid is alive but started at another
    time belongs to an unrelated process that reused the number. A process that has
    exited but whose parent has not collected it is gone as well. Without either key this
    falls back to the plain pid check, so records that do not carry them still work.
    """
    rec = record or {}
    recorded_boot = rec.get("boot_id")
    here = boot_id()
    if recorded_boot and here and recorded_boot != here:
        return True
    if not pid_alive(pid):
        return True
    recorded_start = rec.get("pid_start")
    if isinstance(recorded_start, int) and not isinstance(recorded_start, bool):
        now_start = pid_start_ticks(pid)
        if now_start is not None and now_start != recorded_start:
            return True
    return False


def read_json_dict(path: Path) -> tuple[str, dict]:
    """Tri-state read of a JSON object: ``("missing"|"ok"|"unreadable", record)``.

    JSON of the wrong shape (a list, a string, a number, null) counts as unreadable, so no
    caller has to guard against a record that parses but has no keys.
    """
    import json

    try:
        rec = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return "missing", {}
    except (OSError, json.JSONDecodeError):
        return "unreadable", {}
    if not isinstance(rec, dict):
        return "unreadable", {}
    return "ok", rec


# --------------------------- path helpers ---------------------------


def ensure_dir(path: Path) -> Path:
    """``mkdir -p`` the directory, honouring the user's umask.

    Under ``shared_perms`` (see :mod:`jobq.io`) the leaf is additionally chmod'ed
    world-writable, best-effort: the same account may hold a different numeric user id on
    each machine, and cross-machine access then rides on the 'other' bits. A directory
    another machine created may refuse the chmod, which is fine as long as it created it
    open too.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if io.shared_perms():
        try:
            os.chmod(path, 0o777)
        except OSError:
            pass
    return path


# A queue is one directory directly inside the queue folder, so its name is one path
# component: letters, digits, underscore, hyphen and dot, not opening with a dot, and short
# enough to leave room for the files written inside it.
MAX_QUEUE_NAME_LEN = 64

_QUEUE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


class InvalidQueueName(ValueError):
    """A queue name is not one plain path component inside the queue folder."""


def queue_name_reason(name: object) -> str | None:
    """Why ``name`` cannot be a queue name, or ``None`` when it can be one."""
    if not isinstance(name, str) or not name:
        return "a queue name must be a non-empty name"
    if len(name) > MAX_QUEUE_NAME_LEN:
        return f"a queue name holds at most {MAX_QUEUE_NAME_LEN} characters"
    if name.startswith("."):
        return "a queue name must not start with a dot"
    if not _QUEUE_NAME_RE.fullmatch(name):
        return (
            "a queue name is one path component of letters, digits, underscores, hyphens "
            "and dots"
        )
    return None


def valid_queue_name(name: object) -> bool:
    """Whether ``name`` is a usable queue name."""
    return queue_name_reason(name) is None


def check_queue_name(name: object) -> str:
    """Return ``name`` when it is a usable queue name; otherwise raise.

    Every path inside the queue folder is built from a queue name, so the name is checked
    here, before anything is created, read or removed.

    Raises:
        InvalidQueueName: naming the value and the rule it breaks.
    """
    reason = queue_name_reason(name)
    if reason is not None:
        raise InvalidQueueName(f"queue name {name!r} cannot be used: {reason}")
    return str(name)


class QueuePathUnsafe(ValueError):
    """A queue directory, or one of the directories jobq keeps inside it, is a link."""


# The directories jobq creates, writes into and removes entries from inside a queue. Each
# must be a directory of the queue itself: a link here would point claim removal, result
# writing and recovery at whatever the link names, anywhere on the machine.
QUEUE_OWN_DIRS = ("claims", "results", "attempts", "logs")


def queue_path_reason(root: Path, name: str) -> str | None:
    """Why the queue's own directories cannot be used, or ``None`` when they can be.

    The queue folder itself, and any directory above it, may be reached through links: the
    check starts at the queue directory. From there down the names jobq owns must be real
    directories, so that an entry built from a jobkey stays inside the queue folder.
    """
    qdir = Path(root) / name
    if qdir.is_symlink():
        return f"{qdir} is a link, and a queue directory must be a directory"
    for sub in QUEUE_OWN_DIRS:
        p = qdir / sub
        if p.is_symlink():
            return f"{p} is a link, and a queue's {sub} must be a directory"
    return None


def queue_dir(root: Path, name: str) -> Path:
    """The queue's directory, once its name and its own directories can be used.

    Raises:
        InvalidQueueName: the name is not one usable path component.
        QueuePathUnsafe: the queue directory or one of the directories jobq keeps inside
            it is a link, naming the path.
    """
    qdir = Path(root) / check_queue_name(name)
    reason = queue_path_reason(root, str(name))
    if reason is not None:
        raise QueuePathUnsafe(f"queue {name!r} cannot be used: {reason}")
    return qdir


def _meta_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "meta.json"


def _jobs_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "jobs.jsonl"


def _claims_dir(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "claims"


class InvalidJobName(ValueError):
    """A job's file name is not a plain entry directly inside the directory it belongs in."""


def check_job_file_name(jobkey: object) -> str:
    """Return ``jobkey`` when it names one entry inside a queue's directories.

    A jobkey is turned into a claim directory, a result file, an attempts record and a log,
    so a value that walks the tree (empty, a dot, a double dot, anything holding a
    separator) names something other than this job's own file and is refused here, before
    any of those paths is built.

    Raises:
        InvalidJobName: naming the value and the rule it breaks.
    """
    if not isinstance(jobkey, str) or not jobkey:
        raise InvalidJobName("a job file name must be a non-empty name")
    if jobkey in (".", ".."):
        raise InvalidJobName(
            f"job file name {jobkey!r} names a directory, not one job's claim"
        )
    if "/" in jobkey or os.sep in jobkey or "\0" in jobkey:
        raise InvalidJobName(
            f"job file name {jobkey!r} holds a path separator, so it names an entry "
            "outside this queue's directory"
        )
    return jobkey


def _child_of(parent: Path, jobkey: str) -> Path:
    """``parent/jobkey``, once that is a direct child of ``parent`` on this filesystem.

    The name check alone cannot see a symbolic link, which points a plain name at any place
    on the machine, so the resolved path's own parent is compared with the resolved
    directory it is supposed to sit in.

    Raises:
        InvalidJobName: the path resolves outside ``parent``.
    """
    check_job_file_name(jobkey)
    path = parent / jobkey
    resolved = Path(os.path.realpath(path))
    if path.is_symlink() or resolved.parent != Path(os.path.realpath(parent)):
        raise InvalidJobName(
            f"{path} does not sit directly inside {parent}, so it is not this job's own "
            "entry"
        )
    return path


def _claim_dir(root: Path, name: str, jobkey: str) -> Path:
    return _child_of(_claims_dir(root, name), jobkey)


def _owner_path(root: Path, name: str, jobkey: str) -> Path:
    return _claim_dir(root, name, jobkey) / "owner.json"


def _results_dir(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "results"


def _result_path(root: Path, name: str, jobkey: str) -> Path:
    return _child_of(_results_dir(root, name), f"{check_job_file_name(jobkey)}.json")


def _attempt_path(root: Path, name: str, jobkey: str) -> Path:
    return _child_of(
        queue_dir(root, name) / "attempts", f"{check_job_file_name(jobkey)}.json"
    )


def queue_logs_dir(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "logs"


# --------------------------- meta / queues ---------------------------


class InvalidMachineTie(ValueError):
    """A machine a queue is tied to, or a GPU list on it, that cannot be used."""


_MACHINE_NAME_RE = re.compile(r"[^\s:/\\]+")


@dataclass(frozen=True)
class MachineTie:
    """One entry of a queue's machine tie: a machine, and the GPUs of it the queue may use.

    ``machine`` is ``None`` for a tie that names GPU numbers and no machine, which is
    read as those GPU numbers on whichever machine runs the queue. Empty ``gpus`` means
    every GPU the machine's own policy allows.
    """

    machine: str | None = None
    gpus: tuple[int, ...] = ()

    @classmethod
    def parse(cls, text: str) -> MachineTie:
        """One tie written as ``machine``, ``machine:0,1`` or ``machine:4-7``.

        Raises:
            InvalidMachineTie: the machine name or the GPU list cannot be used. The
                message quotes what was written, since these come from a command line.
        """
        raw = str(text).strip()
        name, sep, gpus = raw.partition(":")
        if not _MACHINE_NAME_RE.fullmatch(name):
            raise InvalidMachineTie(
                f"{raw!r} does not name a machine: a machine name is one plain name, "
                "without spaces, colons or path separators"
            )
        return cls(machine=name, gpus=parse_gpu_list(gpus) if sep else ())

    @property
    def text(self) -> str:
        """The tie as it is written on a command line and shown in the overview."""
        head = self.machine or "any machine"
        return f"{head}:{gpu_list_text(self.gpus)}" if self.gpus else head

    def to_dict(self) -> dict:
        return {"machine": self.machine, "gpus": list(self.gpus)}

    @classmethod
    def from_dict(cls, d: object) -> MachineTie:
        """One stored tie; a record that cannot be read ties nothing on no machine."""
        if not isinstance(d, dict):
            return cls()
        machine = d.get("machine")
        gpus = d.get("gpus")
        return cls(
            machine=machine if isinstance(machine, str) and machine else None,
            gpus=tuple(
                sorted(
                    g
                    for g in (gpus or [])
                    if isinstance(g, int) and not isinstance(g, bool) and g >= 0
                )
            ),
        )


def ties_text(ties) -> str:
    """A machine tie written as it is typed, or ``any machine`` when it ties nothing."""
    return " ".join(t.text for t in ties) or "any machine"


def parse_gpu_list(text: str) -> tuple[int, ...]:
    """GPU numbers written as single numbers and ranges: ``0,1`` or ``4-7`` or ``0,4-7``.

    Raises:
        InvalidMachineTie: the list is empty, holds something that is not a GPU number,
            or holds a range that counts backwards.
    """
    out: set[int] = set()
    parts = [p.strip() for p in str(text).split(",")]
    if not any(parts):
        raise InvalidMachineTie(
            f"{text!r} names no GPU: write the GPUs as numbers and ranges, such as 0,1 or 4-7"
        )
    for part in parts:
        low, dash, high = part.partition("-")
        try:
            first = int(low)
            last = int(high) if dash else first
        except ValueError as exc:
            raise InvalidMachineTie(
                f"{part!r} is not a GPU number or a range of them, such as 0 or 4-7"
            ) from exc
        if first < 0 or last < 0:
            raise InvalidMachineTie(f"{part!r} is not a GPU number: they start at 0")
        if last < first:
            raise InvalidMachineTie(
                f"{part!r} counts backwards: write a range as the lower number first"
            )
        out.update(range(first, last + 1))
    return tuple(sorted(out))


def gpu_list_text(gpus: tuple[int, ...]) -> str:
    """GPU numbers written back as they are typed: runs of three or more become ranges."""
    parts: list[str] = []
    run: list[int] = []

    def _flush() -> None:
        if not run:
            return
        parts.append(f"{run[0]}-{run[-1]}" if len(run) > 2 else ",".join(str(g) for g in run))

    for g in sorted(gpus):
        if run and g == run[-1] + 1:
            run.append(g)
            continue
        _flush()
        run = [g]
    _flush()
    return ",".join(parts)


@dataclass
class QueueMeta:
    """Parsed ``meta.json``: queue config + the defaults that fill in per-job gaps.

    ``format_version`` stamps the on-disk layout; a meta written without the key reads as 1.

    The machine tie lives in two fields. ``node`` is the single machine name, which is
    what a queue tied to one machine with no GPU list is stored as. ``machines`` is the
    fuller form — several machines, or GPUs named on them — and when it is there it is
    the one that counts. :attr:`ties` answers with whichever applies.
    """

    name: str
    created_utc: str
    depends_on: list[str] = field(default_factory=list)
    node: str | None = None
    machines: tuple[MachineTie, ...] = ()
    priority: int = 0
    strict_deps: bool = False
    defaults: dict = field(default_factory=dict)
    format_version: int = FORMAT_VERSION
    # Set when the stored ``format_version`` is not a number. The layout stamp says
    # nothing about the queue's jobs, so a value that cannot be read is reported rather
    # than making the whole queue unreadable; the queue is then treated as the version a
    # meta without the key has.
    format_version_problem: str | None = None
    # Set aside by ``jobq park``: no pool claims from this queue until it is unparked. A
    # meta written without the two keys reads as a queue that is not parked.
    parked: bool = False
    parked_reason: str | None = None

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "created_utc": self.created_utc,
            "depends_on": list(self.depends_on),
            "node": self.node,
            "priority": self.priority,
            "strict_deps": self.strict_deps,
            "defaults": dict(self.defaults),
            "format_version": self.format_version,
        }
        if self.parked:
            d["parked"] = True
            if self.parked_reason:
                d["parked_reason"] = self.parked_reason
        if self.machines:
            d["machines"] = [t.to_dict() for t in self.machines]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> QueueMeta:
        version, version_problem = _format_version(d)
        node = d.get("node")
        stored = d.get("machines")
        machines = (
            tuple(MachineTie.from_dict(t) for t in stored)
            if isinstance(stored, list) and stored
            else ()
        )
        return cls(
            name=d["name"],
            created_utc=d["created_utc"],
            depends_on=list(d.get("depends_on") or []),
            node=node,
            machines=machines,
            priority=int(d.get("priority", 0)),
            strict_deps=bool(d.get("strict_deps", False)),
            defaults=dict(d.get("defaults") or {}),
            format_version=version,
            format_version_problem=version_problem,
            parked=bool(d.get("parked", False)) or _parked_tie_reason(node) is not None,
            parked_reason=(
                str(d.get("parked_reason")) if d.get("parked_reason") else None
            )
            or _parked_tie_reason(node),
        )

    @property
    def machine(self) -> str | None:
        """The single machine this queue is tied to, or ``None``.

        ``None`` covers three cases that are alike to a reader of one name: no tie at
        all, a tie naming several machines (see :attr:`ties`), and a ``PARKED...`` tie,
        which marks a queue as set aside rather than naming a machine.
        """
        if self.machines:
            named = {t.machine for t in self.machines}
            return named.pop() if len(named) == 1 else None
        return None if _parked_tie_reason(self.node) is not None else self.node

    @property
    def ties(self) -> tuple[MachineTie, ...]:
        """The machines this queue may run on, empty when any machine may take it."""
        if self.machines:
            return self.machines
        machine = self.machine
        return (MachineTie(machine=machine),) if machine else ()

    @property
    def runs_on_text(self) -> str:
        """The tie as it is written: ``any machine``, or the entries separated by spaces."""
        return ties_text(self.ties)

    def allows_machine(self, hostname: str) -> bool:
        """Whether a pool on ``hostname`` may claim from this queue at all.

        The GPUs a tie names are a separate question, answered by :meth:`gpus_on`: a
        machine named in the tie may still have none of them, and a job that uses no GPU
        runs there regardless.
        """
        return not self.ties or any(t.machine in (None, hostname) for t in self.ties)

    def gpus_on(self, hostname: str) -> tuple[int, ...] | None:
        """The GPUs of ``hostname`` this queue may use; ``None`` when it may use any.

        An empty tuple means the tie names this machine but no GPU of it that the queue
        may use, which only happens when every entry for the machine names GPUs.
        """
        gpus: set[int] = set()
        named = False
        for tie in self.ties:
            if tie.machine not in (None, hostname):
                continue
            if not tie.gpus:
                return None  # an entry for this machine with no GPU list: any GPU here
            named = True
            gpus.update(tie.gpus)
        return tuple(sorted(gpus)) if named else None


def _format_version(d: dict) -> tuple[int, str | None]:
    """The stored layout stamp, and why it could not be read when it could not be.

    A stamp that is not a number is reported rather than refused: it says which layout
    the file was written in and nothing about the jobs, so a queue carrying a broken one
    is still a queue whose work can be drained.
    """
    raw = d.get("format_version", 1)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return 1, f"format_version {raw!r} is not a number; reading it as 1"
    try:
        return int(raw), None
    except ValueError:
        return 1, f"format_version {raw!r} is not a number; reading it as 1"


# A second way a queue says it is set aside: its machine tie spells PARKED, and whatever
# follows the word is the reason.
PARKED_TIE_PREFIX = "PARKED"


def _parked_tie_reason(node: object) -> str | None:
    """The reason in a ``PARKED...`` machine tie, or ``None`` when the tie is a machine.

    A tie with nothing after the word answers an empty reason, which is still a reason
    that the queue is parked, so callers check for ``None`` rather than for truth.
    """
    if not isinstance(node, str) or not node.startswith(PARKED_TIE_PREFIX):
        return None
    return node[len(PARKED_TIE_PREFIX) :].strip(" :-")


def queue_order_key(meta: QueueMeta) -> tuple[int, str]:
    """Sort key putting the queue a pool takes from first: priority down, then creation time.

    The one place this order is written. The worker sorts its ready queues with it and
    :mod:`jobq.summary` orders its table with it, so the first row of the table is the
    queue the next claim comes from.
    """
    return (-meta.priority, meta.created_utc)


def queue_exists(root: Path, name: str) -> bool:
    return _meta_path(root, name).exists()


class QueueMetaUnreadable(RuntimeError):
    """A queue's ``meta.json`` is missing, unparsable, or not a JSON object."""


# Queue settings whose layout stamp could not be read, so the line saying so is written
# once per file in this process rather than on every pass over the queue.
_META_VERSION_WARNED: set[str] = set()
_META_WARN_LOCK = threading.Lock()


def read_meta(root: Path, name: str) -> QueueMeta:
    """A queue's parsed ``meta.json``.

    Raises:
        QueueMetaUnreadable: the file cannot be read, is not JSON, is not a JSON object,
            or does not carry the keys that make a queue a queue.
    """
    path = _meta_path(root, name)
    state, rec = read_json_dict(path)
    if state == "missing":
        raise QueueMetaUnreadable(f"queue settings {path} are missing")
    if state == "unreadable":
        raise QueueMetaUnreadable(f"queue settings {path} cannot be read as a JSON object")
    try:
        meta = QueueMeta.from_dict(rec)
    except (KeyError, TypeError, ValueError) as exc:
        raise QueueMetaUnreadable(f"queue settings {path} are not usable: {exc}") from exc
    if meta.format_version_problem is not None:
        with _META_WARN_LOCK:
            warn = str(path) not in _META_VERSION_WARNED
            _META_VERSION_WARNED.add(str(path))
        if warn:
            logger.warning("queue settings {}: {}", path, meta.format_version_problem)
    return meta


def _queue_dir_names(root: Path) -> list[str]:
    """Names of every directory under ``root`` holding a ``meta.json``, sorted."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()
    )


def list_queues(root: Path) -> list[str]:
    """Names of every queue under ``root`` (any dir holding a ``meta.json``), sorted.

    A directory whose name is not a usable queue name is left out, so nothing claims from
    it or writes into it; :func:`list_invalid_queue_dirs` reports those for ``jobq status``.
    One whose own directories are links is left out for the same reason, and
    :func:`list_unsafe_queues` reports it.
    """
    return [
        n
        for n in _queue_dir_names(root)
        if valid_queue_name(n) and queue_path_reason(root, n) is None
    ]


def queues_with_claims(root: Path) -> list[str]:
    """Every queue directory under ``root`` that holds a claims directory, sorted.

    Recovery of claims left by a pool that is gone asks this rather than
    :func:`list_queues`: a claim has to be handed back whatever the queue's settings say
    and whether they can be read at all, and the claim directory is readable either way.
    A name jobq cannot use, or a queue whose own directories are links, is still left
    out — nothing is removed from a path jobq refuses to build.
    """
    root = Path(root)
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    return [
        p.name
        for p in entries
        if (p / "claims").is_dir()
        and valid_queue_name(p.name)
        and queue_path_reason(root, p.name) is None
    ]


def list_invalid_queue_dirs(root: Path) -> list[str]:
    """Names of directories under ``root`` that hold a ``meta.json`` but cannot be queues."""
    return [n for n in _queue_dir_names(root) if not valid_queue_name(n)]


def list_unsafe_queues(root: Path) -> list[tuple[str, str]]:
    """``(name, reason)`` for each queue whose own directories jobq refuses to use."""
    out = []
    for n in _queue_dir_names(root):
        if not valid_queue_name(n):
            continue
        reason = queue_path_reason(root, n)
        if reason is not None:
            out.append((n, reason))
    return out


def create_queue(
    root: Path,
    name: str,
    *,
    depends_on: list[str] | None = None,
    node: str | None = None,
    ties: tuple[MachineTie, ...] | None = None,
    priority: int = 0,
    strict_deps: bool = False,
    defaults: dict | None = None,
    exist_ok: bool = False,
) -> QueueMeta:
    """Write ``meta.json`` for a new queue.

    ``node`` names one machine and ``ties`` the fuller form (several machines, or GPUs
    named on them); ``ties`` wins when it is given, and how each is stored is
    :func:`tie_fields`.

    Errors if the queue exists unless ``exist_ok`` — in which case the existing meta is
    kept untouched (append semantics: more jobs may be submitted, meta is set once).
    """
    if queue_exists(root, name):
        if not exist_ok:
            raise FileExistsError(f"queue {name!r} already exists under {root}")
        return read_meta(root, name)
    tie_keys = (
        tie_fields(ties) if ties is not None else {"node": node, "machines": ()}
    )
    meta = QueueMeta(
        name=name,
        created_utc=now_iso(),
        depends_on=list(depends_on or []),
        **tie_keys,
        priority=priority,
        strict_deps=strict_deps,
        defaults=dict(defaults or {}),
    )
    ensure_dir(Path(root))
    ensure_dir(queue_dir(root, name))
    atomic_write_json(_meta_path(root, name), meta.to_dict())
    return meta


# --------------------------- jobs ---------------------------


def load_jobs(root: Path, name: str) -> list[Job]:
    """All jobs of a queue in submission (file) order."""
    import json

    p = _jobs_path(root, name)
    if not p.exists():
        return []
    return [
        Job.from_dict(json.loads(line))
        for line in p.read_text().splitlines()
        if line.strip()
    ]


# Job keys that cannot become one file name inside the queue: nothing to name the file
# after, the queue directory itself, its parent, or a name the shell and most listings
# hide.
_REFUSED_JOBKEYS = ("", ".", "..")


def check_job_names(jobs: list[Job]) -> None:
    """Refuse a job key that cannot be turned into a usable file name.

    Every job owns a claim directory, a result file and a log named after its key, so a
    key that sanitizes to nothing, to a directory entry that already means something, or
    to a hidden name has nowhere to live. Raises ``ValueError`` naming the key.
    """
    for j in jobs:
        jk = j.jobkey
        if jk in _REFUSED_JOBKEYS or jk.startswith("."):
            raise ValueError(
                f"job key {j.key!r} cannot be used: it becomes the file name {jk!r}, and a "
                "job's key must not be empty, a single or double dot, or start with a dot"
            )
        too_long = unusable_name_reason(jk)
        if too_long is not None:
            raise ValueError(f"job key {j.key!r} cannot be used: {too_long}")


def check_new_jobs(root: Path, name: str, jobs: list[Job]) -> list[Job]:
    """Validate an incoming batch against a queue's existing jobs; return the existing jobs.

    Refuses a job key that cannot become a file name, and duplicate ``jobkey``s — both
    against already-submitted jobs and within the incoming batch. Callers that create the
    queue meta should validate first so a rejected submit never leaves a meta-only
    (zero-job) queue behind to wedge dependents.
    """
    check_job_names(jobs)
    existing = load_jobs(root, name)
    seen = {j.jobkey for j in existing}
    for j in jobs:
        if j.jobkey in seen:
            raise ValueError(
                f"duplicate job key {j.key!r} (jobkey {j.jobkey!r}) in queue {name!r}"
            )
        seen.add(j.jobkey)
    return existing


def _jobs_lock_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / ".jobs.lock"


def submit_lock_path(root: Path) -> Path:
    """The one lock a submission holds across the whole queue folder.

    It lives beside the queues rather than inside one of them because a submission may
    create the queue directory, and because the check that queues sharing a cap group
    declare one cap reads the other queues.
    """
    return Path(root) / ".submit.lock"


class QueueSettingsConflict(ValueError):
    """A submission asks for settings that differ from the ones the queue was created with.

    ``diffs`` names each option, the stored value and the requested one.
    """

    def __init__(self, diffs: list[str]) -> None:
        super().__init__("; ".join(diffs))
        self.diffs = list(diffs)


def submit_jobs(
    root: Path,
    name: str,
    jobs: list[Job],
    *,
    depends_on: list[str] | None = None,
    node: str | None = None,
    ties: tuple[MachineTie, ...] | None = None,
    priority: int = 0,
    strict_deps: bool = False,
    defaults: dict | None = None,
    differing_options=None,
    precheck=None,
) -> QueueMeta:
    """Create the queue if it is new, check the submission against it, and append the jobs.

    All three happen while this queue folder's submission lock is held, so two first
    submissions carrying different settings cannot both decide the queue is new: one
    writes ``meta.json`` and appends its jobs, and the other sees the settings the first
    wrote and is refused, having written nothing.

    ``differing_options(meta)`` returns the options of an existing queue this submission
    asks to change, and ``precheck(root)`` runs whatever a caller must check against the
    other queues in the folder (the one cap per cap group). Both run under the same lock,
    so the state they read is the state the append lands on.

    Raises:
        QueueSettingsConflict: the queue exists with other settings; nothing is written.
        ValueError: a job key cannot be used, or is already in the queue.
    """
    check_queue_name(name)
    ensure_dir(Path(root))
    with file_lock(submit_lock_path(root)):
        if queue_exists(root, name):
            meta = read_meta(root, name)
            diffs = list(differing_options(meta)) if differing_options else []
            if diffs:
                raise QueueSettingsConflict(diffs)
        else:
            meta = None
        if precheck is not None:
            precheck(root)
        if meta is None:
            meta = create_queue(
                root,
                name,
                depends_on=depends_on,
                node=node,
                ties=ties,
                priority=priority,
                strict_deps=strict_deps,
                defaults=defaults,
            )
        append_jobs(root, name, jobs)
    return meta


def append_jobs(root: Path, name: str, jobs: list[Job]) -> list[Job]:
    """Append ``jobs`` to ``jobs.jsonl`` (atomic full-file rewrite); validates duplicates.

    The read-validate-rewrite is serialized on ``<queue>/.jobs.lock``: without it two
    concurrent ``jobq submit`` calls both read the pre-existing file and the second rewrite silently drops the first batch. The
    duplicate check is re-run inside the lock, so an earlier unlocked check by a caller is
    only an early-rejection convenience.
    """
    import json

    ensure_dir(queue_dir(root, name))
    with file_lock(_jobs_lock_path(root, name)):
        existing = check_new_jobs(root, name, jobs)
        body = "".join(json.dumps(j.to_dict()) + "\n" for j in existing + jobs)
        atomic_write_text(_jobs_path(root, name), body)
    return existing + jobs


# --------------------------- changing a live queue ---------------------------


def _update_meta(root: Path, name: str, mutate) -> tuple[QueueMeta, QueueMeta]:
    """Read a queue's settings, let ``mutate`` change one of them, write them back.

    Under the lock a submission holds, so a change and an append cannot interleave and
    lose each other. ``mutate`` receives the parsed settings and returns the new ones;
    every other setting is written back exactly as it was read. Returns the settings
    before and after.

    Raises:
        QueueMetaUnreadable: there is no queue by that name, or its settings cannot be read.
    """
    check_queue_name(name)
    with file_lock(submit_lock_path(root)):
        before = read_meta(root, name)
        after = mutate(before)
        atomic_write_json(_meta_path(root, name), after.to_dict())
    return before, after


def set_queue_priority(root: Path, name: str, priority: int) -> tuple[int, int]:
    """Set a queue's priority; returns the old and the new value.

    Pools re-read the settings of every queue on each pass through their ready list, so
    the new priority decides the next claim rather than the queue being restarted.
    """
    before, after = _update_meta(
        root, name, lambda m: replace(m, priority=int(priority))
    )
    return before.priority, after.priority


def tie_fields(ties: tuple[MachineTie, ...]) -> dict:
    """How a tie is stored: the single-machine field alone whenever that says it all.

    A queue tied to exactly one machine with no GPU list is stored in ``node`` and
    nothing else, so a reader of that one field sees the whole tie. Anything fuller is
    stored in ``machines`` as well, and ``node`` then carries the one machine name when
    the tie names one, so a reader of that field alone still sees the machine the queue
    is confined to.
    """
    ties = tuple(ties)
    if not ties:
        return {"node": None, "machines": ()}
    if len(ties) == 1 and ties[0].machine and not ties[0].gpus:
        return {"node": ties[0].machine, "machines": ()}
    named = {t.machine for t in ties}
    return {
        "node": named.pop() if len(named) == 1 and None not in named else None,
        "machines": ties,
    }


def set_queue_ties(
    root: Path, name: str, ties: tuple[MachineTie, ...]
) -> tuple[str, str]:
    """Tie a queue to machines and their GPUs, or untie it with an empty tie.

    Returns the tie before and after, written as it is typed.
    """
    before, after = _update_meta(
        root, name, lambda m: replace(m, **tie_fields(ties))
    )
    return before.runs_on_text, after.runs_on_text


def add_queue_tie(
    root: Path, name: str, tie: MachineTie
) -> tuple[str, str]:
    """Add one machine to a queue's tie, keeping the rest; returns the tie before and after.

    The tie is read, changed and written inside the one lock, so two of these at the same
    moment both land: the second reads what the first wrote. A machine already in the tie
    is replaced, which is how the GPUs named for it are changed.
    """
    def _mutate(meta: QueueMeta) -> QueueMeta:
        kept = tuple(t for t in meta.ties if t.machine != tie.machine)
        return replace(meta, **tie_fields(kept + (tie,)))

    before, after = _update_meta(root, name, _mutate)
    return before.runs_on_text, after.runs_on_text


def remove_queue_tie(root: Path, name: str, machine: str) -> tuple[str, str]:
    """Remove one machine from a queue's tie, keeping the rest, under the one lock."""

    def _mutate(meta: QueueMeta) -> QueueMeta:
        kept = tuple(t for t in meta.ties if t.machine != machine)
        return replace(meta, **tie_fields(kept))

    before, after = _update_meta(root, name, _mutate)
    return before.runs_on_text, after.runs_on_text


def set_queue_parked(
    root: Path, name: str, parked: bool, *, reason: str | None = None
) -> tuple[bool, bool]:
    """Set a queue aside, or bring it back; returns whether it was parked, and whether it is.

    Unparking also removes a machine tie that spells the queue's parked state, since that
    tie is what sets such a queue aside.
    """

    def _mutate(meta: QueueMeta) -> QueueMeta:
        if parked:
            return replace(meta, parked=True, parked_reason=reason or meta.parked_reason)
        node = None if _parked_tie_reason(meta.node) is not None else meta.node
        return replace(meta, parked=False, parked_reason=None, node=node)

    before, after = _update_meta(root, name, _mutate)
    return before.parked, after.parked


# --------------------------- the learned memory request ---------------------------


def learned_mem_path(root: Path, name: str) -> Path:
    """Where a queue's learned memory request lives."""
    return queue_dir(root, name) / "mem_learned.json"


def _learned_mem_lock_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / ".mem_learned.lock"


# How much room the learned request leaves above the largest peak seen, as a percentage
# of it, and the step it is rounded up to, so the figure moves in readable numbers. The
# percentage is an integer and the arithmetic below is integer arithmetic, so a peak of
# 12000 MiB earns exactly 13200 rather than the next step above it.
LEARNED_MEM_PERCENT = 110
LEARNED_MEM_STEP_MIB = 100


def learned_mem_request_mib(peak_mib: int) -> int:
    """The request a peak of ``peak_mib`` earns: a tenth more, rounded up to 100 MiB."""
    with_room = -(-int(peak_mib) * LEARNED_MEM_PERCENT // 100)
    return -(-with_room // LEARNED_MEM_STEP_MIB) * LEARNED_MEM_STEP_MIB


def read_learned_mem(root: Path, name: str) -> dict | None:
    """A queue's learned memory request, or ``None`` when it has none.

    The record holds ``request_mib`` (what a job of the queue asks for when neither it nor
    the queue names an amount), ``peak_mib`` (the largest peak behind it), ``jobs`` (how
    many peaks it has seen) and ``reported``/``measured`` (how those peaks were known).
    One small file read on the admission path, never a scan of the results.
    """
    state, rec = read_json_dict(learned_mem_path(root, name))
    if state != "ok":
        return None
    try:
        request = int(rec.get("request_mib", 0))
    except (TypeError, ValueError):
        return None
    return rec if request > 0 else None


def learned_mem_mib(root: Path, name: str) -> int | None:
    """A queue's learned memory request in MiB, or ``None`` when it has none."""
    rec = read_learned_mem(root, name)
    return None if rec is None else int(rec["request_mib"])


# How many shared measurements must agree, and how far apart the largest and the
# smallest of them may be, before a queue learns from them. One shared figure says
# little: the split between jobs granted together is proportional to what they asked
# for, so a single one can be wrong in either direction. Several that land on the same
# value are evidence that the split matched what the jobs really took.
SHARED_MEM_AGREEING = 3
SHARED_MEM_SPREAD = 1.2
# How many shared measurements are kept waiting for company.
SHARED_MEM_KEPT = 12


def agreeing_shared_mem(values: list[int]) -> int | None:
    """The largest of a group of shared measurements that agree, or ``None``.

    Agreement is :data:`SHARED_MEM_AGREEING` or more values within
    :data:`SHARED_MEM_SPREAD` of the smallest of them. The largest agreeing value is the
    one returned, so the queue asks for what the group's biggest job needed.
    """
    ordered = sorted(int(v) for v in values if isinstance(v, int) and v > 0)
    best: int | None = None
    for i, low in enumerate(ordered):
        group = [v for v in ordered[i:] if v <= low * SHARED_MEM_SPREAD]
        if len(group) >= SHARED_MEM_AGREEING:
            best = max(best or 0, group[-1])
    return best


def note_peak_mem(
    root: Path, name: str, peak_mib: int, *, measured: bool, shared: bool = False
) -> dict | None:
    """Fold one job's peak into its queue's learned memory request; returns the record.

    A larger peak raises the request and a smaller one leaves it where it is, so a job
    that died early — or one that ran out of memory and so never reached its true
    appetite — cannot teach the queue to ask for less than the work needs.

    ``measured`` says the figure is a measured footprint rather than a peak the job
    reported, which is what the counts in the record keep apart. ``shared`` says the
    footprint was divided among jobs that were granted on one GPU within one start-up
    window (see :meth:`jobq.gpu.GpuManager.measure_footprint`). A shared figure is kept
    aside in ``shared_pending`` and only raises the request once
    :func:`agreeing_shared_mem` finds enough of them agreeing, so one bad split cannot
    teach the queue anything. A record written without those keys reads as a queue with
    none pending, and behaves exactly as a queue that has only reported peaks.
    """
    if peak_mib is None or int(peak_mib) <= 0:
        return None
    peak = int(peak_mib)
    ensure_dir(queue_dir(root, name))
    with file_lock(_learned_mem_lock_path(root, name)):
        state, rec = read_json_dict(learned_mem_path(root, name))
        if state != "ok":
            rec = {}
        pending = [
            int(v)
            for v in (rec.get("shared_pending") or [])
            if isinstance(v, int) and not isinstance(v, bool) and v > 0
        ]
        best = _as_int(rec.get("peak_mib"), 0)
        if shared:
            pending = (pending + [peak])[-SHARED_MEM_KEPT:]
            agreed = agreeing_shared_mem(pending)
            peak = 0 if agreed is None else agreed
        # Edited in place, so every other key of the record survives — the note a machine
        # wrote about having to cut this request down to what its GPUs grant
        # (``capped_to_mib``/``capped_by``) is the one ``jobq status`` reads.
        out = dict(rec)
        out.update(
            {
                "peak_mib": max(best, peak),
                "jobs": _as_int(rec.get("jobs"), 0) + 1,
                "reported": _as_int(rec.get("reported"), 0) + (0 if measured else 1),
                "measured": _as_int(rec.get("measured"), 0) + (1 if measured else 0),
                "shared": _as_int(rec.get("shared"), 0) + (1 if shared else 0),
                "shared_pending": pending,
                "updated_utc": now_iso(),
            }
        )
        out["request_mib"] = learned_mem_request_mib(out["peak_mib"])
        atomic_write_json(learned_mem_path(root, name), out)
    return out


def note_learned_mem_capped(root: Path, name: str, to_mib: int, host: str) -> None:
    """Record that a machine cut a queue's learned request down to what its GPUs grant.

    Written on the queue's learned-memory record so ``jobq status`` can say that the
    queue asks for less than its jobs were measured to need. The tightest cap any
    machine has had to apply is the one kept, so machines with GPUs of different sizes
    do not write over each other on every job. A record without these keys is one whose
    request no machine has had to cut down.
    """
    path = learned_mem_path(root, name)
    with file_lock(_learned_mem_lock_path(root, name)):
        state, rec = read_json_dict(path)
        if state != "ok":
            return
        held = _as_int(rec.get("capped_to_mib"), 0)
        if held and held <= int(to_mib):
            return
        rec["capped_to_mib"] = int(to_mib)
        rec["capped_by"] = host
        atomic_write_json(path, rec)


# --------------------------- claims ---------------------------


def has_result(root: Path, name: str, jobkey: str) -> bool:
    return _result_path(root, name, jobkey).exists()


def claim_next(
    root: Path, name: str, node: str, pid: int, accept=None
) -> Job | None:
    """Atomically claim the first pending job of ``name`` (file order), or ``None``.

    A result always wins (terminal, skipped even if a stray claim dir remains). The claim
    itself is ``os.mkdir(claims/<jobkey>)`` — the sole atomic primitive; ``EEXIST`` means a
    peer already owns it, so we move on. ``owner.json`` is written after a successful mkdir.
    A job deferred by a tempfail requeue (``not_before`` in the future) is skipped here.

    The owner record carries ``run_id``, an identifier of this run of the job, written
    before the job's process exists and put into that process's environment as
    ``JOBQ_RUN_ID``, which its children inherit. It is what lets the next pool on this
    machine find whatever the run left behind. A claim whose owner record cannot be
    written is given back here and the job is not handed to the caller, because a run
    nobody can identify is one nobody can end.

    ``accept`` lets the caller pass over jobs it cannot run itself: it is asked about
    each pending job and the job is left for another pool when it says no. The default
    takes every pending job.
    """
    ensure_dir(_claims_dir(root, name))
    for job in load_jobs(root, name):
        jk = job.jobkey
        if has_result(root, name, jk):
            continue
        if accept is not None and not accept(job):
            continue
        if read_not_before(root, name, jk) > time.time():
            # Deferred after an EX_TEMPFAIL (rc=75) exit: still pending and still counted
            # as outstanding work, just not claimable yet.
            continue
        cdir = _claim_dir(root, name, jk)
        try:
            os.mkdir(cdir)
        except FileExistsError:
            continue
        if io.shared_perms():
            try:  # other machine must be able to grace-reclaim / force-release this claim
                os.chmod(cdir, 0o777)
            except OSError:
                pass
        if has_result(root, name, jk):
            # TOCTOU guard: a finishing worker recorded the result and removed its claim
            # between our has_result check and our mkdir — don't re-run a finished job.
            remove_claim(root, name, jk)
            continue
        try:
            atomic_write_json(
                _owner_path(root, name, jk),
                {
                    "node": node,
                    "pid": pid,
                    "gpu": None,
                    "start_utc": now_iso(),
                    "run_id": new_run_id(),
                    **process_identity(),
                },
            )
        except OSError as exc:
            logger.warning(
                "claim {}/{}: its owner record could not be written ({}), so the run has "
                "no identifier and the claim is given back",
                name, jk, exc,
            )
            remove_claim(root, name, jk)
            continue
        return job
    return None


def new_run_id() -> str:
    """An identifier for one run of one job, unique across machines and boots."""
    return uuid.uuid4().hex


def read_run_id(root: Path, name: str, jobkey: str) -> str | None:
    """The identifier of the run a claim describes, or ``None`` if it carries none."""
    value = (read_owner(root, name, jobkey) or {}).get("run_id")
    return value if isinstance(value, str) and value else None


def _owner_state(root: Path, name: str, jobkey: str) -> tuple[str, dict | None]:
    """Tri-state owner read: ``("owner", dict)`` / ``("missing", None)`` / ``("unreadable", None)``.

    The distinction exists for `steal_stale`: only a provably absent owner record may
    enter the ownerless-reclaim path. A record that is present-but-unreadable right now
    (an I/O or quota error), corrupt, or of the wrong shape must never be conflated with
    debris, because reclaiming a live peer's claim is duplicate execution. Every other
    caller may collapse the last two states to ``None`` (they all fail safe on it).
    """
    state, rec = read_json_dict(_owner_path(root, name, jobkey))
    return ("owner", rec) if state == "ok" else (state, None)


def read_owner(root: Path, name: str, jobkey: str) -> dict | None:
    """A claim's owner record, or ``None``.

    The record's fields:

    - ``node``: the machine whose pool made the claim.
    - ``pid``: that pool's process id on that machine.
    - ``boot_id`` / ``pid_start``: which boot and which process that number is, so a pid
      an unrelated process reused does not read as a live pool.
    - ``gpu``: the GPU the job was granted, ``null`` until it is and for a job that
      takes none.
    - ``start_utc``: when the claim was made.
    - ``run_id``: the identifier of this run of the job, exported to the job's process as
      ``JOBQ_RUN_ID`` and inherited by everything it starts.
    - ``job_pid`` / ``job_pgid`` / ``job_pid_start`` / ``job_boot_id``: the job's own
      process, added once it exists, which recovery uses to end a run that outlived its
      pool.

    The single owner reader. ``None`` covers both "no claim" and "owner not yet written or
    unreadable"; only :func:`steal_stale`, which must tell those apart, reads the
    tri-state :func:`_owner_state` directly.
    """
    return _owner_state(root, name, jobkey)[1]


def update_owner_gpu(root: Path, name: str, jobkey: str, gpu: int | None) -> None:
    """Stamp the acquired physical GPU into an existing claim's owner.json (best-effort).

    The claim is made before GPU acquisition, so ``gpu`` starts as null; this makes the live
    GPU visible to ``jobq status`` while the job runs. Stays ``None`` for a declared
    CPU-only job (``slots: 0``), which reserves no GPU at all.
    """
    owner = read_owner(root, name, jobkey)
    if owner is None:
        return
    owner["gpu"] = gpu
    try:
        atomic_write_json(_owner_path(root, name, jobkey), owner)
    except OSError:
        pass  # claim raced away; the result file will carry the gpu regardless


def remove_claim(root: Path, name: str, jobkey: str) -> None:
    """Remove a claim dir (best-effort; a claim is transient, a result is the record).

    Raises:
        InvalidJobName: ``jobkey`` names something other than a direct child of the
            queue's claims directory; nothing is removed.
    """
    shutil.rmtree(_claim_dir(root, name, jobkey), ignore_errors=True)


def record_job_process(
    root: Path, name: str, jobkey: str, *, pid: int, pgid: int
) -> None:
    """Add the running job's process to its claim (best-effort).

    The claim already names the pool that made it. A pool can be killed while the job it
    started keeps running in its own session, and then the pool's pid alone says nothing
    about the job, so the claim additionally carries the job's process, its process group,
    when that process started and the boot it belongs to. Another pool on this machine
    uses those to find and end the surviving job before handing the job back. It is the
    fast path, and what ``jobq status`` reads; the claim's ``run_id`` is what makes a
    survivor findable when this record is not there.
    """
    owner = read_owner(root, name, jobkey)
    if owner is None:
        logger.warning(
            "claim {}/{}: its owner record could not be read, so the job's process is not "
            "recorded on it; the run identifier is what recovery uses",
            name, jobkey,
        )
        return
    owner["job_pid"] = int(pid)
    owner["job_pgid"] = int(pgid)
    start = pid_start_ticks(pid)
    if start is not None:
        owner["job_pid_start"] = start
    bid = boot_id()
    if bid is not None:
        owner["job_boot_id"] = bid
    try:
        atomic_write_json(_owner_path(root, name, jobkey), owner)
    except OSError as exc:
        logger.warning(
            "claim {}/{}: the job's process could not be recorded on it ({}); recovery "
            "finds the job by its run identifier instead",
            name, jobkey, exc,
        )


def _pid_running(pid: int, *, start: int | None = None, pgid: int | None = None) -> bool:
    """Whether ``pid`` is a running process of this process group that started at ``start``.

    A process that has exited but whose parent has not collected it still answers a
    signal, and a pool that was killed never collects the jobs it started, so the remains
    would read as work that is still going. The process table entry says which it is, and
    reading one entry costs one small file rather than a walk of every process.
    """
    entry = read_proc_stat(pid)
    if entry is None:
        return False
    state, group, started = entry
    if state == "Z":
        return False
    if pgid is not None and group != pgid:
        return False
    return start is None or started == start


def _pid_identity(pid: int) -> tuple[int, int] | None:
    """The start time and process group of a live ``pid``, or ``None`` when it is not one.

    A process that has exited but not been collected counts as not one: its entry answers
    signals while the work behind it is over.
    """
    entry = read_proc_stat(pid)
    if entry is None:
        return None
    state, group, started = entry
    return None if state == "Z" else (started, group)


def job_group_alive(owner: dict | None) -> bool:
    """Whether the job a claim describes is still running on this machine.

    The job's own process is the one that is looked at: it is the leader of the process
    group the job and its children share, and it is the process a pool waits on. A process
    group number only means something within one boot and only on the machine that wrote
    it, so a record from another boot, or one without the job's process, reads as a job
    that is not running.
    """
    rec = owner or {}
    pgid = rec.get("job_pgid")
    pid = rec.get("job_pid")
    for value in (pgid, pid):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 1:
            return False
    recorded_boot = rec.get("job_boot_id")
    here = boot_id()
    if recorded_boot and here and recorded_boot != here:
        return False
    start = rec.get("job_pid_start")
    return _pid_running(
        int(pid),
        start=start if isinstance(start, int) and not isinstance(start, bool) else None,
        pgid=int(pgid),
    )


RUN_ID_ENV = "JOBQ_RUN_ID"


# One pass over the process table is reused for this long, so recovering several claims
# of one dead pool costs one pass rather than one each. A process of a run that has been
# going since before the pass cannot be missed by reusing it, and a process started after
# it belongs to a claim of the pool doing the recovering.
RUN_ID_SCAN_TTL_S = 2.0
_run_id_scan: tuple[float, dict[str, list[int]]] | None = None
_RUN_ID_SCAN_LOCK = threading.Lock()


def scan_run_id_processes() -> dict[str, list[int]]:
    """Every run identifier carried by a process of this user here, with its process ids.

    The job's process, the shell above it and every child it started inherit the variable,
    so this finds a whole run however its processes were rearranged: a child that outlived
    the shell the pool started, one that moved to another process group, one that moved to
    its own session. The process table is read directly, because only the environment says
    which run a process belongs to.

    The table can hold several hundred thousand entries, nearly all of them processes that
    have ended and were never collected by their parent. An entry is passed over on the
    first syscall that fails for those — reading the link to a process's working directory,
    which also fails for a process of another user — and only what is left is stat'ed for
    its owner and has its environment read. The result is kept for
    :data:`RUN_ID_SCAN_TTL_S`. This pass belongs to recovery of a claim whose pool is gone;
    nothing on the path of starting or finishing a job calls it.
    """
    global _run_id_scan
    with _RUN_ID_SCAN_LOCK:
        cached = _run_id_scan
        if cached is not None and time.monotonic() - cached[0] < RUN_ID_SCAN_TTL_S:
            return cached[1]
    prefix = f"{RUN_ID_ENV}=".encode()
    uid = os.getuid()
    me = os.getpid()
    found: dict[str, list[int]] = {}
    try:
        entries = os.scandir("/proc")
    except OSError:
        return found
    with entries as it:
        for entry in it:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == me:
                continue
            try:
                os.readlink(f"{entry.path}/cwd")
            except OSError:
                continue  # ended and not collected, or a process this user cannot inspect
            try:
                if entry.stat().st_uid != uid:
                    continue
                with open(f"{entry.path}/environ", "rb") as fh:
                    env = fh.read()
            except OSError:
                continue
            for item in env.split(b"\0"):
                if item.startswith(prefix):
                    found.setdefault(
                        item[len(prefix) :].decode("utf-8", "replace"), []
                    ).append(pid)
                    break
    with _RUN_ID_SCAN_LOCK:
        _run_id_scan = (time.monotonic(), found)
    return found


def find_run_id_processes(run_id: str) -> list[int]:
    """Process ids on this machine whose environment carries ``run_id``."""
    return list(scan_run_id_processes().get(run_id, ()))


def _any_running(watched: list[tuple[int, int | None, int | None]]) -> bool:
    """Whether any watched process is still the process it was when it was found.

    Each entry is a process id with the start time and process group that identify it. A
    number whose process has ended and been handed to unrelated work does not match,
    so it reads as gone rather than as work still running.
    """
    return any(_pid_running(pid, start=start, pgid=pgid) for pid, start, pgid in watched)


def end_job_survivors(
    owner: dict | None,
    *,
    grace_s: float = 10.0,
    poll_s: float = 0.2,
    should_stop=None,
) -> str:
    """End whatever a claim's run left running here; ``absent``, ``ended`` or ``alive``.

    Two things are ended. Every process carrying the claim's run identifier
    (:func:`find_run_id_processes`), which covers a child that outlived its parent, one
    that left the group or the session, and a claim whose job process was never recorded.
    And the process the pool started, together with its process group, since the work is
    usually a child of that process.

    A process id is only a number, and the machine hands it to unrelated work once the
    process behind it has ended, so the claim's ``job_pid`` is signalled only when the
    process there is the one the claim describes: the same boot, the start time the claim
    recorded, and the process group it recorded. A claim written without a start time is
    matched on the group alone, and one that matches on neither is left to the run
    identifier, which names the run itself rather than a number. The group is signalled
    only once something in it has been recognised as this run's, for the same reason.

    Each gets the termination signal, ``grace_s`` to leave, then the kill signal. ``absent``
    means the run left nothing running, ``ended`` that nothing of it is running now, and
    ``alive`` that something still is, which is the one case where the claim must stay
    where it is.

    A signal a process refused (it belongs to another user, or its entry could not be
    read) is not by itself a reason to answer ``alive``: the answer is what is verifiably
    running after the grace and the kill, and a refusal with nothing running is logged
    and the run counts as ended.

    ``should_stop()`` is asked while waiting for the processes to leave; when it says yes
    the wait is cut short, which can only make the answer ``alive`` and leave the claim
    where it is.
    """
    rec = owner or {}
    recorded_boot = rec.get("job_boot_id")
    here = boot_id()
    same_boot = not (recorded_boot and here and recorded_boot != here)
    pgid = rec.get("job_pgid")
    if not (same_boot and isinstance(pgid, int) and not isinstance(pgid, bool) and pgid > 1):
        pgid = None
    run_id = rec.get("run_id")
    watched: list[tuple[int, int | None, int | None]] = []
    seen: set[int] = set()
    groups: set[int] = set()
    if isinstance(run_id, str) and run_id:
        for pid in find_run_id_processes(run_id):
            ident = _pid_identity(pid)
            if ident is None:
                continue  # it ended between the scan and now
            watched.append((pid, ident[0], ident[1]))
            seen.add(pid)
            groups.add(ident[1])
    job_pid = rec.get("job_pid")
    if (
        same_boot
        and isinstance(job_pid, int)
        and not isinstance(job_pid, bool)
        and job_pid > 1
        and job_pid not in seen
    ):
        ident = _pid_identity(job_pid)
        recorded_start = rec.get("job_pid_start")
        if isinstance(recorded_start, bool) or not isinstance(recorded_start, int):
            recorded_start = None
        if ident is not None and (
            recorded_start == ident[0]
            if recorded_start is not None
            else pgid is not None and pgid == ident[1]
        ):
            watched.append((job_pid, ident[0], ident[1]))
            groups.add(ident[1])
    if pgid is not None and pgid not in groups:
        pgid = None  # nothing of this run was recognised in that group
    if pgid is None and not watched:
        return "absent"
    refused = not _signal_survivors(pgid, watched, signal.SIGTERM)
    deadline = time.time() + grace_s
    while _any_running(watched):
        if time.time() >= deadline or (should_stop is not None and should_stop()):
            break
        time.sleep(poll_s)
    if _any_running(watched):
        refused = not _signal_survivors(pgid, watched, signal.SIGKILL) or refused
        for _ in range(10):
            if not _any_running(watched) or (should_stop is not None and should_stop()):
                break
            time.sleep(poll_s)
    if _any_running(watched):
        return "alive"
    if refused:
        logger.warning(
            "a signal to the processes of run {} was refused, but nothing of that run is "
            "running here any more, so its claim is recovered",
            run_id if isinstance(run_id, str) and run_id else "with no identifier",
        )
    return "ended"


def _signal_survivors(
    pgid: int | None, watched: list[tuple[int, int | None, int | None]], sig: int
) -> bool:
    """Send ``sig`` to the group and to each process; False if one refused the signal.

    A process that is already gone is nothing to refuse. A signal that fails for any other
    reason (a process of another user, say) means this run cannot be ended from here, and
    the caller leaves its claim alone rather than starting the job beside it. A process
    whose identity does not match the one it was found with is skipped: the number
    belongs to unrelated work by then.
    """
    ok = True
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            ok = False
    for pid, start, group in watched:
        if not _pid_running(pid, start=start, pgid=group):
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            ok = False
    return ok


# Orphan (owner-less) claim reclaim needs two sightings at least this far apart on this
# process's own monotonic clock, on top of the cross-machine mtime grace: the mtime is written by the
# other machine's clock, so skew > ORPHAN_CLAIM_GRACE_S could otherwise make a peer's
# just-created mid-claim dir look reclaimable on first sight (duplicate execution). A
# fresh mid-claim gains owner.json within seconds, clearing its registry entry before a
# second sighting can reclaim it. A module attribute, so a caller can shorten it.
ORPHAN_RESIGHT_MIN_S = 5.0
_orphan_first_seen: dict[tuple[str, str, str], float] = {}
_ORPHAN_LOCK = threading.Lock()


# Claims recovery looked at and left where they are, with the reasons already reported,
# so each reason is said once per claim in this process rather than on every pass over
# the queue. Pruned with the sighting registry when the claim itself is gone.
_SURVIVOR_WARNED: dict[tuple[str, str, str], set[str]] = {}

# Why a claim was left in place. Recovery never passes over a claim without saying one of
# these, so nothing about a claim is skipped in silence.
KEPT_SURVIVOR = "alive"
KEPT_OWNER_UNREADABLE = "owner"
KEPT_ATTEMPTS_UNREADABLE = "attempts"


def _clear_orphan_registry() -> None:
    """Drop the orphan-claim sighting registry, so nothing counts as sighted before."""
    global _run_id_scan
    with _ORPHAN_LOCK:
        _orphan_first_seen.clear()
        _SURVIVOR_WARNED.clear()
    with _RUN_ID_SCAN_LOCK:
        _run_id_scan = None


def _note_kept(key: tuple[str, str, str], reason: str) -> bool:
    """Record that a claim was left in place for ``reason``; True the first time."""
    with _ORPHAN_LOCK:
        reasons = _SURVIVOR_WARNED.setdefault(key, set())
        first = reason not in reasons
        reasons.add(reason)
    return first


def steal_stale(
    root: Path,
    name: str,
    node: str,
    *,
    grace_s: float | None = None,
    kill_grace_s: float = 10.0,
    on_event=None,
    should_stop=None,
) -> list[str]:
    """Remove claims owned by this machine whose owning pid is dead; return their jobkeys.

    Never touches a claim stamped with a different ``node`` (cross-machine pids are
    meaningless here). A claim that already has a result is just cleaned up as terminal.
    ``grace_s`` (default :data:`ORPHAN_CLAIM_GRACE_S`) is how old an owner-less claim must
    be before any machine may reclaim it.

    A claim whose pool is gone may still have the job itself running: the job has its own
    session, so it outlives the pool that started it. Such a job is ended first (see
    :func:`end_job_survivors`, which looks for the recorded process group and for every
    process carrying the claim's run identifier) and its attempt counter is raised, and
    only then does the job go back into the queue; a job that cannot be ended keeps its
    claim and is not handed to anyone. ``kill_grace_s`` is the wait between the two
    signals. A claim written without a run identifier and without the job's process keeps
    the plain behaviour of removing it.

    ``on_event(kind, jobkey, detail)`` is called for a surviving job that was ended
    (``"ended"``), and once per claim per process for each reason a claim was left where
    it is: ``"alive"`` (its surviving job could not be ended), ``"owner"`` (its owner
    record is present but unreadable) and ``"attempts"`` (its attempts record is present
    but unreadable). Every claim this pass looks at and does not recover therefore
    reaches the caller's log under one of those.

    ``should_stop()`` is asked between claims and while waiting for a surviving job to
    leave: when it says yes the pass returns what it has recovered so far.
    """
    grace_s = ORPHAN_CLAIM_GRACE_S if grace_s is None else grace_s
    if not _claims_dir(root, name).exists():
        return []
    # One recovery per claim, whatever the number of worker threads or passes looking at
    # the queue at the same moment: two passes reading the same claim would each end the
    # job, each raise its attempt counter and each report it. Inside the lock a claim is
    # read and removed once, so a pass arriving second finds nothing left to recover.
    with file_lock(_claims_lock_path(root, name)):
        return _steal_stale_locked(
            root, name, node, grace_s=grace_s, kill_grace_s=kill_grace_s,
            on_event=on_event, should_stop=should_stop,
        )


def _claims_lock_path(root: Path, name: str) -> Path:
    """The lock serializing recovery of a queue's claims (``file_lock``; never read)."""
    return queue_dir(root, name) / ".claims.lock"


def _steal_stale_locked(
    root: Path,
    name: str,
    node: str,
    *,
    grace_s: float,
    kill_grace_s: float,
    on_event=None,
    should_stop=None,
) -> list[str]:
    """One pass of :func:`steal_stale` over a queue's claims, with its lock held."""
    cdir = _claims_dir(root, name)
    stolen: list[str] = []
    seen_ownerless: set[tuple[str, str, str]] = set()
    seen_claims: set[tuple[str, str, str]] = set()

    def _kept(key, reason: str, detail) -> None:
        if _note_kept(key, reason) and on_event is not None:
            on_event(reason, key[2], detail)

    for entry in cdir.iterdir():
        if not entry.is_dir():
            continue
        if should_stop is not None and should_stop():
            break
        jk = entry.name
        seen_claims.add((str(root), name, jk))
        if has_result(root, name, jk):
            remove_claim(root, name, jk)
            continue
        state, owner = _owner_state(root, name, jk)
        key = (str(root), name, jk)
        if state == "unreadable":
            # Owner record present (or unknown) but not readable right now — a transient
            # I/O or quota error, or a corrupt record. Never treat it as ownerless debris: the
            # claim may belong to a live peer job, and reclaiming it is duplicate
            # execution. Skip with no registry mutation; the prune below then resets any
            # earlier ownerless sighting (conservative — delays a real reclaim, never
            # enables a wrong one). A permanently corrupt owner wedges the job until a
            # manual `jobq release --force` — the fail-safe trade.
            logger.warning("claim {}/{}: owner.json unreadable — not reclaiming", name, jk)
            _kept(key, KEPT_OWNER_UNREADABLE, str(_owner_path(root, name, jk)))
            continue
        if state == "missing":
            # Claim dir without owner.json: either a peer mid-claim (sub-second window)
            # or debris of a crash between mkdir and the owner write. After a grace
            # period any machine may reclaim it — ownership was never recorded, so the machine
            # guard doesn't apply, and an owner-less claim can never finish on its own.
            # The reclaim additionally requires a prior sighting >= ORPHAN_RESIGHT_MIN_S
            # ago on this process's monotonic clock (see the registry note above).
            try:
                age_s = time.time() - entry.stat().st_mtime
            except OSError:
                continue
            now = time.monotonic()
            with _ORPHAN_LOCK:
                first = _orphan_first_seen.setdefault(key, now)
            seen_ownerless.add(key)
            if age_s > grace_s and now - first >= ORPHAN_RESIGHT_MIN_S:
                remove_claim(root, name, jk)
                stolen.append(jk)
                seen_ownerless.discard(key)
                with _ORPHAN_LOCK:
                    _orphan_first_seen.pop(key, None)
            continue
        with _ORPHAN_LOCK:
            _orphan_first_seen.pop(key, None)  # owner landed: not an orphan after all
        if owner.get("node") != node:
            continue  # never steal another machine's claim
        pid = owner.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and process_gone(pid, owner):
            outcome = end_job_survivors(
                owner, grace_s=kill_grace_s, should_stop=should_stop
            )
            if outcome == "alive":
                if _note_kept(key, KEPT_SURVIVOR):
                    logger.warning(
                        "claim {}/{}: the pool that made it is gone but its job process "
                        "group {} is still running and could not be ended — leaving the "
                        "claim in place",
                        name, jk, owner.get("job_pgid"),
                    )
                    if on_event is not None:
                        on_event(KEPT_SURVIVOR, jk, owner.get("job_pgid"))
                continue
            if outcome == "ended":
                try:
                    bump_attempt(root, name, jk)
                except AttemptRecordUnreadable:
                    # Already warned by the sidecar reader. The job is ended but its
                    # counters cannot be written, so the claim stays where it is rather
                    # than going back with a retry budget nobody can see.
                    _kept(
                        key,
                        KEPT_ATTEMPTS_UNREADABLE,
                        str(_attempt_path(root, name, jk)),
                    )
                    continue
                if on_event is not None:
                    on_event("ended", jk, owner.get("job_pgid"))
            remove_claim(root, name, jk)
            seen_claims.discard(key)
            stolen.append(jk)
    # Registry hygiene: entries for this queue whose claim was not seen ownerless on this
    # pass (claim vanished, or gained an owner) are stale sightings, and a claim that is
    # gone altogether has nothing left to report about. Both registries are therefore
    # pruned down to what this pass actually saw.
    with _ORPHAN_LOCK:
        for k in [k for k in _orphan_first_seen
                  if k[0] == str(root) and k[1] == name and k not in seen_ownerless]:
            _orphan_first_seen.pop(k)
        for k in [k for k in _SURVIVOR_WARNED
                  if k[0] == str(root) and k[1] == name and k not in seen_claims]:
            _SURVIVOR_WARNED.pop(k)
    return stolen


def force_release(root: Path, name: str, key: str) -> bool:
    """Remove one submitted job's claim, ignoring the machine guard (``release --force``).

    ``key`` may be the human key or an already-sanitized jobkey, and it must belong to a
    job submitted to this queue: a claim directory is removed with everything under it, so
    the target is confined to one job's own entry inside the queue's claims directory.
    Returns whether a claim existed. Never called automatically.

    Raises:
        InvalidJobName: the key names no job of this queue, or resolves to something other
            than a direct child of the claims directory.
    """
    jk = sanitize_jobkey(key)
    check_job_file_name(jk)
    if jk not in {job.jobkey for job in load_jobs(root, name)}:
        raise InvalidJobName(
            f"{key!r} is not a job of queue {name!r} (its file name would be {jk!r}), so "
            "there is no claim of its own to release"
        )
    cdir = _claim_dir(root, name, jk)
    existed = cdir.exists()
    remove_claim(root, name, jk)
    return existed


# --------------------------- results / attempts ---------------------------


class AttemptRecordUnreadable(RuntimeError):
    """An attempts sidecar exists but could not be read, so it must not be rewritten.

    Raised by every writer instead of clobbering a record whose real counters (attempt,
    ``oom_requeues``, ``tempfails``, ``mem_mib_floor``) we cannot see: overwriting it
    would silently hand a wedged job its whole retry budget back.
    """


def _attempts_lock_path(root: Path, name: str) -> Path:
    """The single lock guarding a queue's attempts sidecars (``file_lock``, like ``.jobs.lock``).

    A separate file, never read: a sidecar is replaced by `atomic_write_json`, so the lock
    cannot live on it. Every read-modify-write of any sidecar in the queue — from any
    thread, process or machine — is serialized here, which is what makes two concurrent
    counter bumps both land instead of one overwriting the other's read. One file per
    queue, not per job: a queue folder may hold a great many jobs, and one lock file per
    job would be a file per job, while the write is short enough that waiting for the one
    lock costs nothing worth measuring.
    """
    return queue_dir(root, name) / ".attempts.lock"


# How long an unreadable sidecar defers its job's next claim. The record may hold a live
# ``not_before``, so a job whose deferral we cannot read must not become claimable early;
# bounded (rather than "never") so a transient error delays the job instead of wedging it.
ATTEMPT_UNREADABLE_DEFER_S = 60.0

# Last successfully-read sidecar per (root, name, jobkey). A read that fails serves this
# instead of defaults, so an unreadable record can never lower a memory floor or reset a
# backstop counter within a process's lifetime. Bounded by the jobs this pool has touched.
_ATTEMPT_LAST_GOOD: dict[tuple[str, str, str], dict] = {}
_ATTEMPT_WARNED: set[tuple[str, str, str]] = set()
_ATTEMPT_MEMO_LOCK = threading.Lock()


def _clear_attempt_cache() -> None:
    """Drop the last-known-good attempts-record cache, so the next read goes to disk."""
    with _ATTEMPT_MEMO_LOCK:
        _ATTEMPT_LAST_GOOD.clear()
        _ATTEMPT_WARNED.clear()


def _attempt_state(root: Path, name: str, jobkey: str) -> tuple[str, dict]:
    """Tri-state sidecar read: ``("missing"|"ok"|"unreadable", record)``.

    Mirrors :func:`_owner_state`: "absent" (defaults are correct) must never be conflated
    with "present but unreadable right now" (an I/O or quota error, a file owned by
    another uid, or a corrupt record), where the defaults are a silent counter reset.
    """
    return read_json_dict(_attempt_path(root, name, jobkey))


def _read_attempt_record(root: Path, name: str, jobkey: str) -> dict:
    """The sidecar dict for reading; an unreadable one serves the last known-good record.

    Absent stays ``{}`` (the defaults are correct there). Unreadable warns once per job
    and falls back to whatever this process last read successfully — never to defaults,
    which would lower the memory floor and reset the retry counters.
    """
    key = (str(root), name, jobkey)
    state, rec = _attempt_state(root, name, jobkey)
    if state == "ok":
        with _ATTEMPT_MEMO_LOCK:
            _ATTEMPT_LAST_GOOD[key] = rec
            _ATTEMPT_WARNED.discard(key)
        return rec
    if state == "missing":
        return {}
    with _ATTEMPT_MEMO_LOCK:
        warn = key not in _ATTEMPT_WARNED
        _ATTEMPT_WARNED.add(key)
        fallback = dict(_ATTEMPT_LAST_GOOD.get(key, {}))
    if warn:
        logger.warning(
            "attempts sidecar {} is present but unreadable — using the last known-good "
            "record; no writer will overwrite it",
            _attempt_path(root, name, jobkey),
        )
    return fallback


def _update_attempt_record(root: Path, name: str, jobkey: str, mutate) -> dict:
    """Read-modify-write the sidecar under its lock; ``mutate(rec)`` edits it in place.

    The lock is the only thing that makes concurrent bumps of different keys both land:
    each writer reads, edits and rewrites the whole record, so without it the later write
    drops the earlier one. An unreadable record raises
    :class:`AttemptRecordUnreadable` rather than being clobbered.
    """
    path = _attempt_path(root, name, jobkey)
    ensure_dir(path.parent)
    with file_lock(_attempts_lock_path(root, name)):
        state, rec = _attempt_state(root, name, jobkey)
        if state == "unreadable":
            # One retry: a transient I/O error here is usually a blip, and refusing is expensive
            # (it fails the job's requeue path).
            time.sleep(0.05)
            state, rec = _attempt_state(root, name, jobkey)
        if state == "unreadable":
            logger.warning("attempts sidecar {} unreadable — refusing to overwrite it", path)
            raise AttemptRecordUnreadable(str(path))
        rec = dict(rec)
        mutate(rec)
        atomic_write_json(path, rec)
    with _ATTEMPT_MEMO_LOCK:
        _ATTEMPT_LAST_GOOD[(str(root), name, jobkey)] = dict(rec)
    return rec


def _as_int(value, default: int = 0) -> int:
    """A counter read from a record another process wrote, or ``default``.

    A boolean is refused like every other reader here: it is an integer subclass, so
    ``true`` in a record would silently read as the count 1.
    """
    if isinstance(value, bool):
        return default
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def read_attempt(root: Path, name: str, jobkey: str) -> int:
    return _as_int(_read_attempt_record(root, name, jobkey).get("attempt", 0))


def read_mem_floor(root: Path, name: str, jobkey: str) -> int:
    """Persisted per-job ``mem_mib`` floor (OOM escalation), or ``0`` when never escalated.

    Optional: a record for a job that has never been escalated carries no such key.
    """
    return _as_int(_read_attempt_record(root, name, jobkey).get("mem_mib_floor", 0))


def write_mem_floor(root: Path, name: str, jobkey: str, mem_mib: int) -> None:
    """Persist a job's escalated ``mem_mib`` floor in the attempts sidecar.

    Lives here rather than in ``jobs.jsonl``, which is append-only and shared cross-machine.
    """
    _update_attempt_record(
        root, name, jobkey, lambda rec: rec.__setitem__("mem_mib_floor", int(mem_mib))
    )


def read_oom_requeues(root: Path, name: str, jobkey: str) -> int:
    """How many times this job has been OOM-requeued (attempts sidecar; 0 when absent)."""
    return _as_int(_read_attempt_record(root, name, jobkey).get("oom_requeues", 0))


def _bump_key(rec: dict, key: str) -> None:
    rec[key] = _as_int(rec.get(key, 0)) + 1


def bump_oom_requeues(root: Path, name: str, jobkey: str) -> int:
    """Increment (and persist) the job's OOM-requeue counter; return the new value.

    Lives in the attempts sidecar like ``mem_mib_floor``.
    """
    rec = _update_attempt_record(root, name, jobkey, lambda r: _bump_key(r, "oom_requeues"))
    return _as_int(rec["oom_requeues"])


def note_oom_requeue(
    root: Path, name: str, jobkey: str, *, mem_mib_floor: int | None
) -> tuple[int, int]:
    """Record an out-of-memory requeue in one locked update; return (oom requeues, attempt).

    The escalated memory floor, the out-of-memory counter and the attempt counter all live
    in the same record, so they are written together: three separate updates would each
    read and rewrite the whole record, and a failure between them would leave a job
    counted but not escalated, or escalated without being counted.
    """

    def _mutate(rec: dict) -> None:
        if mem_mib_floor is not None:
            rec["mem_mib_floor"] = int(mem_mib_floor)
        _bump_key(rec, "oom_requeues")
        _bump_key(rec, "attempt")

    rec = _update_attempt_record(root, name, jobkey, _mutate)
    return _as_int(rec["oom_requeues"]), _as_int(rec["attempt"])


def read_tempfails(root: Path, name: str, jobkey: str) -> int:
    """How many times this job exited 75 (EX_TEMPFAIL) and was requeued; 0 when absent."""
    return _as_int(_read_attempt_record(root, name, jobkey).get("tempfails", 0))


def read_not_before(root: Path, name: str, jobkey: str) -> float:
    """Epoch second before which this job must not be claimed again; 0.0 when absent.

    An unreadable record defers the job by ``ATTEMPT_UNREADABLE_DEFER_S`` instead of
    reading as 0.0: the claim path must never make a deferred job claimable early just
    because its sidecar could not be parsed.
    """
    state, rec = _attempt_state(root, name, jobkey)
    if state == "unreadable":
        _read_attempt_record(root, name, jobkey)  # warn once, refresh the last-good cache
        return time.time() + ATTEMPT_UNREADABLE_DEFER_S
    try:
        return float(rec.get("not_before", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def bump_tempfails(root: Path, name: str, jobkey: str, *, not_before: float) -> int:
    """Increment the job's tempfail counter and defer its next claim; return the new count.

    Both keys live in the attempts sidecar like ``oom_requeues``.
    """

    def _mutate(rec: dict) -> None:
        _bump_key(rec, "tempfails")
        rec["not_before"] = float(not_before)

    rec = _update_attempt_record(root, name, jobkey, _mutate)
    return _as_int(rec["tempfails"])


def read_mem_floors(
    root: Path,
    name: str,
    *,
    jobkeys: list[str] | None = None,
    above_mib: int | None = None,
) -> list[tuple[str, int]]:
    """The escalated floors :func:`reset_mem_floors` would clear, as (jobkey, MiB).

    Read-only, so a caller can say what it is about to clear before clearing it.
    """
    adir = queue_dir(root, name) / "attempts"
    if not adir.is_dir():
        return []
    if jobkeys is None:
        try:
            keys = sorted(p.stem for p in adir.glob("*.json"))
        except OSError:
            return []
    else:
        keys = list(jobkeys)
    out: list[tuple[str, int]] = []
    for jk in keys:
        if not _attempt_path(root, name, jk).exists():
            continue
        old = read_mem_floor(root, name, jk)
        if old <= 0 or (above_mib is not None and old <= above_mib):
            continue
        out.append((jk, old))
    return out


def reset_mem_floors(
    root: Path,
    name: str,
    *,
    jobkeys: list[str] | None = None,
    above_mib: int | None = None,
) -> list[tuple[str, int]]:
    """Drop the escalated ``mem_mib_floor`` from attempts sidecars; return (jobkey, old).

    The operator un-wedge for a floor no GPU can satisfy: a co-tenant OOM can escalate one
    past what any GPU has, leaving the job pending forever. Unlike :func:`requeue` this works in
    any state and touches neither results nor claims — clearing a running job's floor only
    affects its next claim. ``jobkeys`` restricts to named jobs; ``above_mib`` to floors
    strictly above that many MiB. Every other sidecar key is preserved.
    """
    cleared: list[tuple[str, int]] = []
    for jk, old in read_mem_floors(
        root, name, jobkeys=jobkeys, above_mib=above_mib
    ):
        path = _attempt_path(root, name, jk)
        try:
            _update_attempt_record(root, name, jk, lambda rec: rec.pop("mem_mib_floor", None))
        except AttemptRecordUnreadable:
            continue  # already warned; never clobber a record we cannot read
        except PermissionError:
            # A sidecar owned by another uid: report it and keep going.
            logger.warning("attempts sidecar {} not writable (permission denied) — skipping", path)
            continue
        cleared.append((jk, old))
    return cleared


def _write_attempt(
    root: Path,
    name: str,
    jobkey: str,
    attempt: int,
    *,
    reset_oom: bool = False,
) -> None:
    """Persist the attempt counter, preserving any other keys (e.g. ``mem_mib_floor``).

    ``reset_oom`` is the operator escape hatch (``jobq requeue --reset-oom``): it drops the
    OOM-requeue counter instead of carrying it over.
    """

    def _mutate(rec: dict) -> None:
        rec["attempt"] = attempt
        if reset_oom:
            rec.pop("oom_requeues", None)

    _update_attempt_record(root, name, jobkey, _mutate)


def bump_attempt(root: Path, name: str, jobkey: str) -> int:
    """Increment (and persist) a job's attempt counter; return the new value.

    Used when a running job is requeued without a result (e.g. yield-kill): the retry
    should record ``attempt = prev + 1``, mirroring :func:`requeue`. The read and the
    write happen inside one sidecar lock, so two concurrent bumps cannot both read the
    same value.
    """
    rec = _update_attempt_record(root, name, jobkey, lambda r: _bump_key(r, "attempt"))
    return _as_int(rec["attempt"])


def record_result(
    root: Path,
    name: str,
    job: Job,
    *,
    rc: int,
    node: str,
    gpu: int | None,
    start_utc: str,
    end_utc: str,
    log: str,
    attempt: int,
    peak_mem_mib: int | None = None,
    peak_mem_alloc_mib: int | None = None,
    measured_mem_mib: int | None = None,
    not_failure_reason: str | None = None,
) -> None:
    """Write a job's terminal result (atomic) and persist its attempt counter.

    ``peak_mem_mib`` / ``peak_mem_alloc_mib`` (the high-water marks a job reported in its
    log, see :mod:`jobq.report`) are optional: they are written only when measured, so a
    result file without them stays readable. ``measured_mem_mib`` is the separate figure
    the pool read from the GPU itself while the job was starting up (see
    :meth:`jobq.gpu.GpuManager.measure_footprint`); it has its own key because it says
    less than a peak the job reported.

    ``not_failure_reason`` records why a non-zero result is not the job's own failure (the
    pool could not run it here), which is what keeps it out of the queue's run of
    consecutive failures. Written only when there is one; ``rc`` keeps its meaning.

    ``log`` is written only when the caller has a log to name. A job the pool never
    managed to start has none, and a record pointing at a file that is not there is
    worse than a record with no path in it.
    """
    jk = job.jobkey
    ensure_dir(_results_dir(root, name))
    ensure_dir(_attempt_path(root, name, jk).parent)
    rec = {
        "key": job.key,
        "rc": rc,
        "node": node,
        "gpu": gpu,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "attempt": attempt,
    }
    # Written only when there is a log to point at: a job the pool could not start
    # leaves none, and a key naming a file that is not there reads as one that is.
    if log:
        rec["log"] = log
    if peak_mem_mib is not None:
        rec["peak_mem_mib"] = int(peak_mem_mib)
    if peak_mem_alloc_mib is not None:
        rec["peak_mem_alloc_mib"] = int(peak_mem_alloc_mib)
    if measured_mem_mib is not None:
        rec["measured_mem_mib"] = int(measured_mem_mib)
    if not_failure_reason:
        rec["not_failure_reason"] = str(not_failure_reason)
    atomic_write_json(_result_path(root, name, jk), rec)
    _write_attempt(root, name, jk, attempt)


def load_results(root: Path, name: str) -> list[dict]:
    """Every terminal result record of a queue as parsed dicts (unreadable ones skipped).

    Each dict carries the fields written by :func:`record_result` (``rc``, ``start_utc``,
    ``end_utc``, ...). Used by the yield watchdog's median-duration progress estimate.
    """
    rdir = _results_dir(root, name)
    if not rdir.exists():
        return []
    out: list[dict] = []
    for rp in sorted(rdir.glob("*.json")):
        state, rec = read_json_dict(rp)
        if state == "ok":
            out.append(rec)
    return out


def requeue_targets(
    root: Path, name: str, *, failed_only: bool = True
) -> list[tuple[str, int]]:
    """The jobs :func:`requeue` would hand back, with the attempt each result records.

    Read-only, so a caller can say what a requeue is about to throw away before it does.
    """
    rdir = _results_dir(root, name)
    if not rdir.exists():
        return []
    out: list[tuple[str, int]] = []
    for rp in sorted(rdir.glob("*.json")):
        state, res = read_json_dict(rp)
        if state == "missing":
            continue
        # A result that cannot be read counts as a failure, the same way the job's state
        # does: the job ran, and nothing here can say it succeeded.
        rc = res.get("rc", 1) if state == "unreadable" else res.get("rc", 0)
        if failed_only and rc == 0:
            continue
        out.append((rp.stem, _as_int(res.get("attempt", 0))))
    return out


def requeue(
    root: Path,
    name: str,
    *,
    failed_only: bool = True,
    reset_oom: bool = False,
) -> list[str]:
    """Make terminal jobs pending again by deleting their results (+ stray claims).

    ``failed_only`` (default) touches only rc!=0 results. Bumps each affected job's attempt
    counter so its next run records ``attempt = prev + 1``. Returns the requeued jobkeys.
    Also resumes the queue (:func:`resume_queue`): the jobs it paused on are the jobs this
    call is handing back to the pools.

    The attempts sidecar's OOM state survives a requeue by default, so a job that exhausted
    the worker's OOM-requeue backstop stays terminal on its next OOM. ``reset_oom`` clears
    that counter (the explicit operator reset). The escalated ``mem_mib`` floor is always
    kept — ``jobq reset-mem-floor`` is the one way to clear it.

    Idempotent per job: the new attempt counter is the result file's attempt + 1 (an absolute
    value derived from the file being deleted, never a read-modify-write of the sidecar),
    and the result file is unlinked last. A crash after any step therefore converges on a
    re-run — while the result still exists the same absolute value is rewritten, and once
    it is gone the job is already pending and is skipped. Removing the claim before the
    result is safe because ``claim_next`` skips any job that has a result.
    """
    resume_queue(root, name)
    requeued: list[str] = []
    for jk, attempt in requeue_targets(root, name, failed_only=failed_only):
        rp = _result_path(root, name, jk)
        try:
            _write_attempt(
                root,
                name,
                jk,
                attempt + 1,
                reset_oom=reset_oom,
            )
        except AttemptRecordUnreadable:
            continue  # already warned; leave the job terminal rather than lose its counters
        remove_claim(root, name, jk)
        rp.unlink(missing_ok=True)
        requeued.append(jk)
    if requeued:
        drop_complete_cache(root, name)
    return requeued


# --------------------------- failure pause ---------------------------
#
# A queue whose jobs fail one after another for reasons of their own (a bad config, a
# missing input) stops being claimed, so one mistake cannot burn the whole queue. The state
# is two small files in the queue directory, not pool memory, so every machine reads the same
# verdict and a pool restart changes nothing:
#
#   failures.json   {"streak": n, "keys": [...]}   the run of consecutive job failures
#   paused.json     {"since_utc": ..., "keys": [...], "limit": n}   present while paused
#
# Both are written under ``<queue>/.failures.lock``, so two pools finishing failing jobs at
# the same moment cannot lose a count or disagree about whether the queue is paused. A pool
# deciding whether it may claim only stats paused.json.

# Consecutive job failures that pause a queue when its meta names no other number.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


def failures_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "failures.json"


def paused_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "paused.json"


def _failures_lock_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / ".failures.lock"


def max_consecutive_failures(meta: QueueMeta) -> int:
    """A queue's pause threshold from its meta defaults; 0 means the queue never pauses."""
    raw = meta.defaults.get("max_consecutive_failures")
    if raw is None:
        return DEFAULT_MAX_CONSECUTIVE_FAILURES
    return max(0, _as_int(raw, DEFAULT_MAX_CONSECUTIVE_FAILURES))


def queue_paused(root: Path, name: str) -> bool:
    """Whether the queue is paused — one stat, which is what the claim loop calls."""
    return paused_path(root, name).exists()


def read_pause(root: Path, name: str) -> dict | None:
    """The pause record (``since_utc``, ``keys``, ``limit``), or None when not paused.

    An unreadable record still reports a pause (with empty detail): the file's presence is
    the decision, and failing open would hand the queue back to the pools.
    """
    state, rec = read_json_dict(paused_path(root, name))
    return None if state == "missing" else rec


def _empty_streak() -> dict:
    return {"streak": 0, "keys": []}


def read_failure_streak(root: Path, name: str) -> dict:
    """The current run of consecutive job failures: ``{"streak": n, "keys": [...]}``.

    A record that cannot be read reports no run. Nothing is written here; the writer moves
    such a record aside first (see :func:`note_job_failure`), so a count is never quietly
    replaced by a lower one.
    """
    state, rec = read_json_dict(failures_path(root, name))
    if state != "ok":
        return _empty_streak()
    keys = [str(k) for k in (rec.get("keys") or [])] if isinstance(rec.get("keys") or [], list) else []
    return {"streak": _as_int(rec.get("streak", 0)), "keys": keys}


def _set_aside_unreadable_failures(root: Path, name: str) -> None:
    """Move an unreadable failure record aside, so counting can restart from this failure.

    The count of consecutive failures is what pauses a queue, and overwriting a record we
    cannot read would silently discard however many failures it held. The file is renamed
    to one that says it is unreadable and names the moment, and the name is logged so the
    old content can be looked at.
    """
    path = failures_path(root, name)
    aside = path.with_name(f"failures.unreadable.{utc_stamp()}.json")
    try:
        os.replace(path, aside)
    except OSError as exc:
        logger.warning("could not move the unreadable failure record {} aside: {}", path, exc)
        return
    logger.warning(
        "failure record {} could not be read, so it was moved to {} and counting restarts "
        "from this failure",
        path,
        aside,
    )


def note_job_failure(root: Path, name: str, jobkey: str, limit: int) -> bool:
    """Extend the run of job failures; return whether this call paused the queue.

    ``limit`` of 0 disables the feature for the queue, and nothing is written. Only a job
    process that exited non-zero on its own belongs here — see the worker for the exits
    that are the pool's fault or a retry rather than the job's failure.
    """
    if limit <= 0:
        return False
    ensure_dir(queue_dir(root, name))
    with file_lock(_failures_lock_path(root, name)):
        state, _raw = read_json_dict(failures_path(root, name))
        if state == "unreadable":
            _set_aside_unreadable_failures(root, name)
        rec = read_failure_streak(root, name)
        keys = [*rec["keys"], jobkey][-limit:]
        streak = rec["streak"] + 1
        atomic_write_json(failures_path(root, name), {"streak": streak, "keys": keys})
        if streak < limit or paused_path(root, name).exists():
            return False
        atomic_write_json(
            paused_path(root, name),
            {"since_utc": now_iso(), "keys": keys, "limit": limit},
        )
        return True


def note_job_success(root: Path, name: str) -> None:
    """Reset the run of job failures. Never resumes a paused queue (only ``resume`` does).

    The record is looked at only under the failures lock, so a job finishing here and a
    job failing on another machine cannot interleave a read of one with a write of the
    other.
    """
    if not queue_dir(root, name).is_dir():
        return
    with file_lock(_failures_lock_path(root, name)):
        failures_path(root, name).unlink(missing_ok=True)


def resume_queue(root: Path, name: str) -> bool:
    """Clear a queue's pause and its run of failures; return whether it was paused.

    The only way out of a pause: nothing here runs on a timer, on a new submission or on a
    later success.
    """
    if not queue_dir(root, name).is_dir():
        return False
    with file_lock(_failures_lock_path(root, name)):
        was_paused = paused_path(root, name).exists()
        paused_path(root, name).unlink(missing_ok=True)
        failures_path(root, name).unlink(missing_ok=True)
    return was_paused


# --------------------------- derived state ---------------------------

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


def job_state(root: Path, name: str, job: Job) -> str:
    """Derive one job's state from claim/result presence (a result always wins)."""
    jk = job.jobkey
    rp = _result_path(root, name, jk)
    state, rec = read_json_dict(rp)
    if state != "missing":
        # A result that cannot be read is still terminal, and reads as a failure: the job
        # ran, and nothing here can say it succeeded.
        try:
            rc = int(rec.get("rc", 1))
        except (TypeError, ValueError):
            rc = 1
        return DONE if rc == 0 else FAILED
    if _claim_dir(root, name, jk).exists():
        return RUNNING
    return PENDING


def claim_liveness(owner: dict, *, this_node: str) -> dict:
    """What can be said here about the pool and the job behind a claim.

    ``pool_gone`` is ``True`` when the pool that made the claim is not running, ``False``
    when it is, and ``None`` for a claim another machine made, whose pid means nothing
    here. It uses the rule recovery uses, so a claim reported as left behind is one the
    next pool on that machine hands back. ``job_running`` says whether the job's own
    process is still running here, which a claim written without it reads as ``False``.
    """
    if owner.get("node") != this_node:
        return {"pool_gone": None, "job_running": False}
    pid = owner.get("pid")
    gone = not (isinstance(pid, int) and not isinstance(pid, bool)) or process_gone(
        pid, owner
    )
    return {"pool_gone": gone, "job_running": job_group_alive(owner)}


def queue_status(root: Path, name: str, *, with_deferred: bool = False) -> dict:
    """Counts by derived state + a ``running`` list carrying each owner's node/gpu/key.

    ``with_deferred`` (off by default) additionally returns ``deferred`` — the pending jobs
    a tempfail requeue has parked behind a ``not_before`` in the future, work the pool is
    deliberately refusing to claim, which otherwise looks like a stalled queue — and
    ``deferred_until``, the earliest epoch second one becomes claimable (or None).
    ``pending`` keeps its meaning and still includes them. It is opt-in because it costs a
    sidecar read per pending job and the worker calls this function on its claim-loop hot
    path (the strict-deps gate) once per queue in the queue folder; only ``jobq status``
    asks for it.
    """
    counts = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0}
    running: list[dict] = []
    check_deferred = with_deferred and (queue_dir(root, name) / "attempts").is_dir()
    now = time.time()
    deferred = 0
    deferred_until: float | None = None
    for job in load_jobs(root, name):
        st = job_state(root, name, job)
        counts[st] += 1
        if st == PENDING and check_deferred:
            nb = read_not_before(root, name, job.jobkey)
            if nb > now:
                deferred += 1
                deferred_until = nb if deferred_until is None else min(deferred_until, nb)
        if st == RUNNING:
            owner = read_owner(root, name, job.jobkey) or {}
            running.append(
                {
                    "key": job.key,
                    "node": owner.get("node"),
                    "pid": owner.get("pid"),
                    "gpu": owner.get("gpu"),
                    "start_utc": owner.get("start_utc"),
                    **claim_liveness(owner, this_node=this_host()),
                }
            )
    out = {**counts, "total": sum(counts.values()), "running_jobs": running}
    if with_deferred:
        out.update(deferred=deferred, deferred_until=deferred_until)
    return out


def complete_state_path(root: Path, name: str) -> Path:
    return queue_dir(root, name) / "complete.state.json"


def drop_complete_cache(root: Path, name: str) -> None:
    """Invalidate a queue's completion cache, so the next check re-derives it."""
    complete_state_path(root, name).unlink(missing_ok=True)


def _complete_witness(root: Path, name: str) -> tuple[int, int] | None:
    """``(jobs.jsonl size, number of result files)`` right now, or None if unreadable.

    The two content-derived quantities that a change to a settled queue must move.
    ``jobs.jsonl`` is only ever grown (append-only, atomic full-file rewrite), so its byte
    size strictly increases when a job is submitted; a requeue (or any external deletion)
    lowers the result-file count. Neither reads an mtime, so nothing depends on the clock
    of the machine that wrote the file.
    """
    try:
        size = _jobs_path(root, name).stat().st_size
        with os.scandir(_results_dir(root, name)) as it:
            n_results = sum(1 for e in it if e.name.endswith(".json"))
    except OSError:
        return None
    return size, n_results


def complete_cache_valid(root: Path, name: str) -> bool:
    """Whether the completion cache marks ``name`` complete and still matches the queue.

    One file read, one stat and one directory scan, no per-job reads, and nothing written:
    a reader that must not touch the queue folder can ask this and fall back to its own
    scan when the answer is ``False``.
    """
    _state, cached = read_json_dict(complete_state_path(root, name))
    try:
        witness = (int(cached["jobs_size"]), int(cached["n_results"]))
    except (ValueError, TypeError, KeyError):
        return False
    return witness == _complete_witness(root, name)


def queue_complete(root: Path, name: str) -> bool:
    """Whether every job of ``name`` is terminal (has a result — done or failed).

    Cached on disk: the first full check that finds the queue complete writes
    ``complete.state.json`` (the witness below), and later calls trust the cache only
    while the witness still matches. Every worker thread re-derives the
    ready list before each claim, and on a root of thousands of settled queues the per-job
    ``has_result`` stats dominate the claim loop. The fast path costs one read + one stat +
    one directory scan per settled queue instead of N+1 stats; an incomplete queue still
    takes the full path.

    The witness is ``(jobs.jsonl size, number of result files)`` rather than an mtime
    ordering: on a shared network filesystem a job appended by another machine can carry
    an mtime older than the cache, and the queue would then be reported complete with
    pending work in it.
    """
    if not queue_exists(root, name):
        return False
    if complete_cache_valid(root, name):
        return True
    before = _complete_witness(root, name)
    jobs = load_jobs(root, name)
    done = bool(jobs) and all(has_result(root, name, j.jobkey) for j in jobs)
    if done:
        # Cache only a verdict the queue did not move under: taking the witness after the
        # scan alone would pair a job list read before an append with the post-append size,
        # caching "complete" for a queue that has pending work. Unequal -> cache nothing and
        # return this scan's answer; the next call re-checks against the settled state.
        after = _complete_witness(root, name)
        try:
            if before is not None and after == before:
                atomic_write_json(
                    complete_state_path(root, name),
                    {"jobs_size": before[0], "n_results": before[1], "written_utc": now_iso()},
                )
        except OSError:  # read-only / other-uid queue dir: cache is optional
            pass
    return done


# --------------------------- node files ---------------------------


def worker_pid_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"worker.{hostname}.pid"


def worker_lock_path(root: Path, hostname: str) -> Path:
    """The file a pool locks for its whole life, so this machine runs one pool per folder.

    The lock is the decision; the pid and info files beside it are the human-readable
    record. The file itself is never removed or replaced, because a pool may be holding a
    lock on it.
    """
    return Path(root) / f"worker.{hostname}.lock"


def worker_info_path(root: Path, hostname: str) -> Path:
    """Sidecar beside the pid file holding this machine's pool pid and current log path."""
    return Path(root) / f"worker.{hostname}.json"


def worker_info_lock_path(root: Path, hostname: str) -> Path:
    """The one lock every rewrite of ``worker.<host>.json`` is taken under.

    Three writers touch that sidecar — the pool recording itself, the supervisor's
    heartbeat and the note that a signal asked the pool to drain — and each reads the
    record, edits one key and writes the whole file back. Without one lock the later
    write drops the earlier one, so a heartbeat can erase the drain note or the log path.
    A separate file, never read, like the other read-modify-write locks here.
    """
    return Path(root) / f".worker.{hostname}.lock.rmw"


def _update_worker_info(root: Path, hostname: str, mutate) -> None:
    """Read the pool sidecar, let ``mutate`` edit it, write it back, under its lock."""
    ensure_dir(Path(root))
    try:
        with file_lock(worker_info_lock_path(root, hostname)):
            info = read_worker_info(root, hostname) or {"hostname": hostname}
            if mutate(info) is False:
                return
            atomic_write_json(worker_info_path(root, hostname), info)
    except OSError as exc:
        logger.warning(
            "could not update {}: {}", worker_info_path(root, hostname), exc
        )


def write_worker_info(root: Path, hostname: str, *, pid: int, log: str) -> None:
    """Record this machine's running pool for ``jobq status``.

    Carries the pid, the log the pool is writing, and the boot identifier and process
    start time that say which process that pid is, so a pid reused after a restart is not
    mistaken for a live pool. The heartbeat starts here and is refreshed on every
    supervisor tick (:func:`write_heartbeat`).
    """

    def _mutate(info: dict) -> None:
        info.clear()
        info.update(
            {
                "hostname": hostname,
                "pid": pid,
                "log": log,
                "started_utc": now_iso(),
                "heartbeat_utc": now_iso(),
                **process_identity(),
            }
        )

    _update_worker_info(root, hostname, _mutate)


def write_heartbeat(root: Path, hostname: str) -> None:
    """Stamp this machine's pool sidecar with the current time, keeping its other fields.

    A machine can only check a pid on itself, so the stamp is how a reader anywhere tells
    a pool that is still going from one that stopped saying anything. It is information
    only: the claims of a machine that falls silent are still recovered by that machine's
    own pool. Best-effort, like the rest of the sidecar.
    """
    _update_worker_info(
        root, hostname, lambda info: info.__setitem__("heartbeat_utc", now_iso())
    )


def heartbeat_state(root: Path, hostname: str, stale_s: float) -> dict:
    """What a machine's heartbeat says: its stamp, its age in seconds, and whether it is fresh.

    ``heartbeat_utc`` is ``None`` for a machine that has never written one, which is also
    a machine not heard from.
    """
    stamp = (read_worker_info(root, hostname) or {}).get("heartbeat_utc")
    age = None
    if isinstance(stamp, str):
        try:
            beat = datetime.fromisoformat(stamp)
        except ValueError:
            stamp = None
        else:
            if beat.tzinfo is None:
                beat = beat.replace(tzinfo=UTC)
            age = max(0.0, (datetime.now(UTC) - beat).total_seconds())
    return {
        "heartbeat_utc": stamp if isinstance(stamp, str) else None,
        "age_s": age,
        "fresh": age is not None and age <= stale_s,
    }


def mark_pool_draining(root: Path, hostname: str) -> None:
    """Record on this machine's pool sidecar that a signal asked the pool to drain.

    A signal reaches the pool's own terminal, so a status run in another terminal has
    nothing else to read it from. Best-effort: the note is a convenience for the reader,
    not something the pool's behaviour depends on.
    """
    def _mutate(info: dict) -> bool | None:
        if not worker_info_path(root, hostname).exists():
            return False  # no pool has recorded itself here; nothing to annotate
        info["draining_after_signal"] = True
        return None

    _update_worker_info(root, hostname, _mutate)


def pool_draining_after_signal(root: Path, hostname: str) -> bool:
    """Whether this machine's pool has recorded that a signal asked it to drain."""
    return bool((read_worker_info(root, hostname) or {}).get("draining_after_signal"))


def pool_pid_is_live(root: Path, hostname: str, pid: int) -> bool:
    """Whether the pid recorded for this machine's pool is that pool, still running.

    Uses the boot identifier and start time in the pool's own record when it has them, so
    a pid that an unrelated process reused after a restart reads as not running. A record
    without them falls back to the plain pid check.
    """
    if not pid:
        return False
    info = read_worker_info(root, hostname) or {}
    record = info if info.get("pid") == pid else None
    return not process_gone(pid, record)


def read_worker_info(root: Path, hostname: str) -> dict | None:
    """The recorded pool sidecar for this machine, or None when absent/unreadable."""
    state, rec = read_json_dict(worker_info_path(root, hostname))
    return rec if state == "ok" else None


# How long, and how often, the lock is asked again when it says a pool is running on a
# machine whose pid file names none. Long enough for a child a pool forked just before
# it left to reach its exec and drop the copy of the lock it inherited.
POOL_LOCK_SETTLE_TRIES = 5
POOL_LOCK_SETTLE_S = 0.05


def pool_lock_held(root: Path, hostname: str) -> bool | None:
    """Whether this machine's pool lock is held right now; ``None`` when it cannot be told.

    The lock is what makes one pool per machine and queue folder, and the kernel drops it
    when that process ends however it ends, so trying to take it is the one answer that
    does not depend on any file a pool wrote. Nothing is created: a lock file that is not
    there means no pool has ever run here.
    """
    path = worker_lock_path(root, hostname)
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return None  # this filesystem cannot answer the question
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def pool_state(root: Path, hostname: str) -> dict:
    """Whether a pool is alive on this machine, with its pid and current log path.

    ``alive`` is only ever answered for this machine: a pid from another machine says
    nothing here. It comes from the pool lock (:func:`pool_lock_held`), which a live pool
    holds for its whole life; the pid file is read for the number to show and stands in
    only where the lock cannot answer, so a pool whose pid file is missing still reads as
    running.
    """
    pid = read_worker_pid(root, hostname)
    info = read_worker_info(root, hostname) or {}
    log = info.get("log") if info.get("pid") == pid else None
    if log is None and not pid:
        log = info.get("log")
    held = pool_lock_held(root, hostname)
    if held is None:
        alive = pool_pid_is_live(root, hostname, pid)
    elif held and not pool_pid_is_live(root, hostname, pid):
        # The lock and the pid file disagree. A pool that has just left can still be
        # held to by one of its own children between the fork and the exec, which owns
        # a copy of the lock for that moment, so the question is asked again for a
        # short while before a machine with no pid file is reported as running.
        alive = True
        for _ in range(POOL_LOCK_SETTLE_TRIES):
            time.sleep(POOL_LOCK_SETTLE_S)
            if pool_lock_held(root, hostname) is not True:
                alive = False
                break
    else:
        alive = held
    return {
        "alive": alive,
        "pid": pid or (info.get("pid") if alive else None) or None,
        "log": log,
    }


def read_worker_pid(root: Path, hostname: str) -> int:
    """The pid in this machine's pool pid file, or 0 when there is none to read."""
    try:
        host, pid_s = worker_pid_path(root, hostname).read_text().strip().split(":")
    except (OSError, ValueError):
        return 0
    if host != hostname:
        return 0
    try:
        return int(pid_s)
    except ValueError:
        return 0


# Temporary files an interrupted atomic write leaves behind, and how old one must be
# before a starting pool removes it. An hour is far longer than any write takes, so a file
# older than that belongs to a process that is gone.
TEMP_FILE_MARKER = ".tmp."
TEMP_FILE_MAX_AGE_S = 3600.0


def remove_stale_temp_files(root: Path, *, max_age_s: float = TEMP_FILE_MAX_AGE_S) -> list[str]:
    """Remove leftover temporary files from the queue folder and each queue directory.

    An atomic write creates ``<name>.tmp.<token>`` beside its destination and renames it
    into place; a process killed in between leaves the temporary file, which nothing will
    ever rename or read. Returns the paths removed.
    """
    removed: list[str] = []
    root = Path(root)
    dirs = [root]
    try:
        dirs += [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return removed
    now = time.time()
    for d in dirs:
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if TEMP_FILE_MARKER not in entry.name or not entry.is_file():
                continue
            try:
                if now - entry.stat().st_mtime <= max_age_s:
                    continue
                entry.unlink()
            except OSError:
                continue
            removed.append(str(entry))
    return removed


def stop_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"stop.{hostname}"


def stop_now_path(root: Path, hostname: str) -> Path:
    """The request to end this machine's running jobs at once, beside the stop file.

    ``jobq stop --now`` writes this and the stop file together. A pool that reads only the
    stop file drains; a pool that reads this one ends its running jobs and puts them back
    in the queue. The pool removes this file when it leaves, so the request belongs to the
    pool it was aimed at; the stop file stays until it is cleared.
    """
    return Path(root) / f"stop.now.{hostname}"


def this_host() -> str:
    return socket.gethostname()
