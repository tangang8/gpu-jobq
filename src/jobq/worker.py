"""The threaded worker pool: claim -> acquire GPU -> run -> record, log-and-continue.

One process, ``--workers N`` threads, all sharing one :class:`GpuManager` (so the per-GPU
slot cap is global). Each thread loops over the ready queues (machine filter + deps satisfied,
ordered by priority desc then created_utc), claims the first pending job, runs it, records
the result, and moves on. A job failure never aborts the pool. Between jobs it honors the
``stop.<hostname>`` file.

Master-log verbs (``[<iso-utc>] VERB ...``): WORKERS once at startup with the thread count
and the figures behind it, then CLAIM, START, END rc=<rc>, STEAL, WAIT, RESERVE / RELEASE
for a GPU held for a job that waited too long, QUEUE COMPLETE <name>, and one final exit
line: ALL QUEUES COMPLETE when the root really drained, POOL EXIT ... when the pool left
with work outstanding. The opt-in GPU-yielding
framework (:mod:`jobq.yielding`) adds YIELD gpu=N foreign=K, KILL <queue> <key> gpu=N,
DRAIN <queue> <key> gpu=N pct=P (finishing) (a near-done job spared under
drain_if_near_done), and RECLAIM gpu=N. PAUSE <queue> ... / RESUME <queue> report a queue
that stopped being claimed after a run of consecutive job failures, and the ``jobq resume``
or success that ended it; PAUSE LAPSED <queue>: ... reports a pause that ran out, after
which this worker alone claims one job to try the queue. SIDECAR <queue> <key> ... reports a job made
terminal because this pool could not update its attempts sidecar. WAIT queue <queue>:
working directory ... is not reachable ... reports a queue whose jobs are left for other
machines because the directory they run in is missing here. The
requeue paths add OOM REQUEUE / OOM CEILING / OOM BACKSTOP, TEMPFAIL / TEMPFAIL BACKSTOP
and YIELD REQUEUE, and the supervisor adds RESPAWN, SCALE and REAP. An interrupt or a
termination signal asks the pool to drain and logs DRAIN REQUEST ...; ``jobq stop --now``
asks it to end its jobs at once and logs END NOW ..., then INTERRUPT <queue> <key> ... for
each job it ends and INTERRUPT REQUEUE <queue> <key> ... when that job is back in the
queue.
ENDED SURVIVOR <queue> <jobkey> ... reports a job that outlived the pool which started it
and what became of it, and CLAIM KEPT <queue> <jobkey> ... a claim recovery looked at and
left where it is, with the reason: its surviving job could not be ended, its owner record
cannot be read, or its attempts record cannot be. RUN ID <queue> <key> ... reports a job
handed back unrun because its claim carries no run identifier.

Recovery of claims left behind by a pool that is gone is a pass of its own
(``_Pool.recover_claims``), run at pool start and on every supervisor tick over every
queue in the folder whatever its state, because a claim nobody hands back counts as work
in progress for ever.

Job environment, applied in this order: the pool process's environment, the policy ``env``,
the queue default ``env``, the job's own ``env``. On top of that the worker always sets
``CUDA_VISIBLE_DEVICES`` to the granted GPU (empty for a CPU-only job) and exports
``JOBQ_QUEUE``, ``JOBQ_JOB_KEY``, ``JOBQ_ATTEMPT``, ``JOBQ_NODE``, ``JOBQ_GPU`` and
``JOBQ_RUN_ID``, the claim's identifier for this run of the job.

A job killed by CUDA OOM is not a failure: its log tail is classified (:func:`is_oom_text`)
and it is requeued with an escalated ``mem_mib`` floor persisted in the attempts sidecar, so
the memory gate — not a timer or a retry counter — decides when it runs again. The only
terminal case is a job that OOMs while already asking for the whole GPU (``OOM ceiling``),
which is recorded as an ordinary failure. A machine's policy may replace the built-in OOM
markers with its own ``oom_patterns``.

Each attempt writes its own log, ``logs/<stamp>_<jobkey>.a<attempt>.<host>.log``, so the
out-of-memory classification reads the output of the attempt it is judging and no other.

Peak GPU memory is self-reported: the worker sets ``JOBQ_REPORT_PEAK_MEM=1`` and parses the
last ``JOBQ_PEAK_GPU_MEM_MIB=<int>`` line (optionally followed by
``JOBQ_PEAK_GPU_MEM_ALLOC_MIB=<int>``) from the job's log tail — see :mod:`jobq.report` for
a helper a PyTorch job can register. The reading is stored on the result as ``peak_mem_mib``
and appended to the END line; an attempt that is requeued (OOM, rc=75) records no result, so
its reading goes on the OOM/TEMPFAIL log line instead.
"""

from __future__ import annotations

import contextlib
import fcntl
import math
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from jobq import gpu as gpu_defaults
from jobq import io, monitor, store, yielding
from jobq.gpu import (
    MPS_AUTO,
    GpuInterface,
    GpuManager,
    ensure_mps_daemon,
    load_policy,
    mps_env,
)
from jobq.model import Job
from jobq.store import QueueMeta, now_iso, utc_stamp

# Env var the worker sets on every job, asking it to print its peak-memory line.
REPORT_PEAK_MEM_ENV = "JOBQ_REPORT_PEAK_MEM"
# OOM escalation: new_mem = max(ceil(old * oom_mem_factor), oom_mem_floor_mib), capped at
# what a GPU can grant.
# Priority-ordered parking. Without it every parked waiter polls a freed slot on equal
# terms, so a queue's priority orders claims but never admission and a crowd of parked
# low-priority waiters takes each freed slot before a high-priority one gets its single
# non-blocking try. A parked waiter therefore defers its acquire while a strictly
# higher-priority waiter is parked and eligible (see ``_Pool._higher_eligible_waiter``);
# the deferral is bounded so an unsatisfiable head-of-line request can delay, never wedge,
# the rest of the pool.
PARK_DEFER_MAX_S = gpu_defaults.DEFAULT_PARK_DEFER_MAX_S
# Returned by ``_Pool._acquire_gpu`` when a capped job's stage group saturated while it was
# parked: the caller hands the claim back (capped jobs never park on a saturated group).
_YIELD = object()

OOM_MEM_FACTOR = gpu_defaults.DEFAULT_OOM_MEM_FACTOR
OOM_MEM_FLOOR_MIB = gpu_defaults.DEFAULT_OOM_MEM_FLOOR_MIB
# An idle GPU never reports its total memory as free (driver/context reserve), so a floor
# persisted at the total can never pass the memory gate: the job parks forever and the
# backstop never fires because it never runs again. Cap the ladder this far below the total.
OOM_CEILING_HEADROOM_MIB = gpu_defaults.DEFAULT_OOM_CEILING_HEADROOM_MIB
# Backstop on OOM requeues per job. The memory-gate ladder is the designed terminator
# (escalate 1.5x until the request hits the GPU total, then record a real failure); it is
# finite whenever the ceiling is readable. This counter only bites when it is not (a broken
# nvidia-smi leaves the ceiling unknown and the escalation unbounded) or when a non-OOM
# failure whose log merely contains an OOM marker keeps being misclassified; without it both
# loop forever without ever counting as a failure. 8 covers any real ladder (4 GiB to an
# 80 GiB GPU is about 7 steps of 1.5x).
OOM_REQUEUE_BACKSTOP = gpu_defaults.DEFAULT_OOM_MAX_REQUEUES
OOM_LOG_TAIL_BYTES = 50_000
# EX_TEMPFAIL (sysexits.h). A job exiting 75 means "not my turn, try again later", never
# "failed": a job may use it when a resource it needs is busy elsewhere, or when it was
# killed by a signal from outside this pool's own yield path.
TEMPFAIL_RC = 75
# How long a tempfailed job stays unclaimable.
TEMPFAIL_RETRY_S = gpu_defaults.DEFAULT_TEMPFAIL_RETRY_S
# Backstop: a job that is "temporarily" unavailable this many times is really stuck, so
# record the 75 as the failure it looks like.
TEMPFAIL_REQUEUE_BACKSTOP = gpu_defaults.DEFAULT_TEMPFAIL_MAX_REQUEUES
# Co-tenant OOM guard: an OOM whose message shows this process holding less than this
# fraction of its own memory ask was starved by neighbours on a shared GPU, not
# under-reserved. Escalating the floor there is wrong twice over — the job never needed the
# memory, and a few such escalations park the claim behind a request no GPU can satisfy.
# Such an OOM is still requeued (and still counts toward the backstop) but at the same
# memory ask.
OOM_OWN_USAGE_ESCALATE_FRACTION = gpu_defaults.DEFAULT_OOM_OWN_USAGE_FRACTION

# Each tunable's name in :class:`~jobq.gpu.Tunables` mapped to the module-level name above
# holding the value the pool uses when nothing supplies a policy (see ``_Pool._tunable``).
_TUNABLE_FALLBACKS = {
    "oom_mem_factor": "OOM_MEM_FACTOR",
    "oom_mem_floor_mib": "OOM_MEM_FLOOR_MIB",
    "oom_ceiling_headroom_mib": "OOM_CEILING_HEADROOM_MIB",
    "oom_max_requeues": "OOM_REQUEUE_BACKSTOP",
    "oom_own_usage_fraction": "OOM_OWN_USAGE_ESCALATE_FRACTION",
    "tempfail_retry_s": "TEMPFAIL_RETRY_S",
    "tempfail_max_requeues": "TEMPFAIL_REQUEUE_BACKSTOP",
    "park_defer_max_s": "PARK_DEFER_MAX_S",
    "kill_grace_s": "KILL_GRACE_S",
    "orphan_claim_grace_s": "ORPHAN_CLAIM_GRACE_S",
    "failure_pause_s": "FAILURE_PAUSE_S",
}
# Seconds between the termination signal and the kill signal in a force-kill.
KILL_GRACE_S = gpu_defaults.DEFAULT_KILL_GRACE_S
# How old an owner-less claim must be before any machine may reclaim it.
ORPHAN_CLAIM_GRACE_S = store.ORPHAN_CLAIM_GRACE_S
# How long a failure pause lasts before one job is tried again (0: until jobq resume).
FAILURE_PAUSE_S = gpu_defaults.DEFAULT_FAILURE_PAUSE_S

# Default out-of-memory markers: generic CUDA / PyTorch messages. A machine's policy may
# replace them wholesale with its own ``oom_patterns``.
OOM_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUDA error: out of memory",
    "cudaErrorMemoryAllocation",
)


def is_oom_text(text: str, patterns: tuple[str, ...] | None = None) -> bool:
    """Whether a job's output carries any out-of-memory marker.

    ``patterns`` (a machine's policy ``oom_patterns``) replaces :data:`OOM_MARKERS` when given
    and non-empty.
    """
    return any(m in text for m in (patterns or OOM_MARKERS))


_MEM_UNIT_MIB = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}
# Newer torch: "Including non-PyTorch memory, this process has 5.13 GiB memory in use."
_OOM_THIS_PROCESS_RE = re.compile(r"this process has ([\d.]+) (KiB|MiB|GiB) memory in use")
# Every torch OOM message: "Of the allocated memory 1.46 GiB is allocated by PyTorch, and
# 1.93 MiB is reserved by PyTorch but unallocated." (allocated + reserved = this process's
# CUDA footprint, minus the CUDA context).
_OOM_ALLOCATED_RE = re.compile(
    r"Of the allocated memory ([\d.]+) (KiB|MiB|GiB) is allocated by PyTorch, and "
    r"([\d.]+) (KiB|MiB|GiB) is reserved by PyTorch but unallocated"
)


def oom_own_usage_mib(text: str) -> int | None:
    """How much CUDA memory the OOM-ing process itself held, parsed from torch's message.

    Uses the last match (a tail can carry several OOM messages; the last is the fatal one).
    Prefers the explicit "this process has X" figure, else allocated + reserved-unallocated.
    ``None`` when the text carries no parseable figure — the caller must then assume the OOM
    is genuine and escalate.
    """
    m = None
    for m in _OOM_THIS_PROCESS_RE.finditer(text):  # noqa: B007 — want the last match
        pass
    if m is not None:
        return int(round(float(m.group(1)) * _MEM_UNIT_MIB[m.group(2)]))
    for m in _OOM_ALLOCATED_RE.finditer(text):  # noqa: B007 — want the last match
        pass
    if m is not None:
        return int(
            round(
                float(m.group(1)) * _MEM_UNIT_MIB[m.group(2)]
                + float(m.group(3)) * _MEM_UNIT_MIB[m.group(4)]
            )
        )
    return None


# Every torch OOM message opens with the size of the allocation that failed:
# "CUDA out of memory. Tried to allocate 20.00 GiB (GPU 0; ...)".
_OOM_TRIED_ALLOC_RE = re.compile(r"Tried to allocate ([\d.]+) (KiB|MiB|GiB)")


def oom_requested_mib(text: str) -> int | None:
    """The size of the allocation that failed, parsed from torch's message (last match).

    ``None`` when unparseable — the caller then falls back to the own-usage fraction alone.
    """
    m = None
    for m in _OOM_TRIED_ALLOC_RE.finditer(text):  # noqa: B007 — want the last match
        pass
    return None if m is None else int(round(float(m.group(1)) * _MEM_UNIT_MIB[m.group(2)]))


def _read_log_tail(path: Path | str, tail_bytes: int) -> str | None:
    """The last ``tail_bytes`` of a log as text; ``None`` when missing or unreadable."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            tail = f.read()
    except OSError:
        return None
    return tail.decode("utf-8", "replace")


PEAK_MEM_LOG_TAIL_BYTES = 256 * 1024
_PEAK_MEM_RE = re.compile(
    r"JOBQ_PEAK_GPU_MEM_MIB=(\d+)(?:\s+JOBQ_PEAK_GPU_MEM_ALLOC_MIB=(\d+))?"
)


def log_tail_peak_mem_mib(
    path: Path | str, *, tail_bytes: int = PEAK_MEM_LOG_TAIL_BYTES
) -> tuple[int | None, int | None]:
    """``(reserved, allocated)`` peak MiB from a job log's tail; ``(None, None)`` if absent.

    A job may print the line more than once (one per stage of an ``&&`` chain), so the
    job's number is the maximum over the lines found.
    """
    tail = _read_log_tail(path, tail_bytes)
    if not tail:
        return None, None
    res: list[int] = []
    alloc: list[int] = []
    for m in _PEAK_MEM_RE.finditer(tail):
        res.append(int(m.group(1)))
        if m.group(2) is not None:
            alloc.append(int(m.group(2)))
    return (max(res) if res else None), (max(alloc) if alloc else None)


def log_tail_is_oom(
    path: Path | str,
    *,
    tail_bytes: int = OOM_LOG_TAIL_BYTES,
    patterns: tuple[str, ...] | None = None,
) -> bool:
    """Classify a finished job's log as OOM from its last ``tail_bytes``.

    A missing or unreadable log is not an OOM (the caller must fall through to normal
    failure handling); the read is byte-bounded so a multi-GB log costs one seek.
    """
    tail = _read_log_tail(path, tail_bytes)
    return bool(tail) and is_oom_text(tail, patterns)


def log_tail_oom_own_usage_mib(
    path: Path | str, *, tail_bytes: int = OOM_LOG_TAIL_BYTES
) -> int | None:
    """``oom_own_usage_mib`` over the log's tail; ``None`` when unreadable or unparseable."""
    tail = _read_log_tail(path, tail_bytes)
    return None if tail is None else oom_own_usage_mib(tail)


def log_tail_oom_requested_mib(
    path: Path | str, *, tail_bytes: int = OOM_LOG_TAIL_BYTES
) -> int | None:
    """``oom_requested_mib`` over the log's tail; ``None`` when unreadable or unparseable."""
    tail = _read_log_tail(path, tail_bytes)
    return None if tail is None else oom_requested_mib(tail)


class MasterLog:
    """Thread-safe append-only master log; also mirrors each line to loguru."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        store.ensure_dir(self.path.parent)

    def log(self, verb: str, msg: str = "") -> None:
        line = f"[{now_iso()}] {verb} {msg}".rstrip()
        try:
            with self._lock:
                with open(self.path, "a") as f:
                    f.write(line + "\n")
        except OSError:
            # A full disk must never crash a worker mid-iteration; the mirror below still
            # records the line.
            pass
        logger.info(line)


@dataclass
class _RunHandle:
    """A live job subprocess, tracked per-GPU so the yield watchdog can kill it.

    ``killed`` distinguishes "I killed this to yield the GPU" from "it exited on its own"
    so the runner requeues (not fails) a yield-killed job. ``log_path``/``start_ts`` feed the
    ``drain_if_near_done`` progress estimate (log-tail regex + elapsed/median); ``drained``
    records that a spared job's DRAIN line was already logged (so it is not re-logged every
    poll while the GPU stays yielded).
    """

    proc: subprocess.Popen
    queue: str
    job: Job
    gpu: int
    killed: bool = False
    # The verb and the sentence of the line that records the requeue. Yielding a GPU and
    # a pool asked to leave at once are separate features that both end a job and hand it
    # back, so each says so under its own verb.
    kill_verb: str = "YIELD REQUEUE"
    kill_note: str = "killed for a yield; back in the queue"
    log_path: str = ""
    start_ts: float = 0.0
    drained: bool = False
    # The memory ask the job was admitted with, carried here because an out-of-memory
    # requeue escalates from it: the queue's settings may have been edited, or its
    # learned request moved, since this job was placed.
    admitted_mem_mib: int = 0


def _kill_group(
    proc: subprocess.Popen, *, grace_s: float = KILL_GRACE_S, poll_s: float = 0.5
) -> None:
    """SIGTERM the process's group, wait ``grace_s``, then SIGKILL (best-effort, no raise).

    The job is launched with ``start_new_session=True`` so ``bash -c`` and every child share
    a process group we can signal in one shot.

    A termination signal that fails for any reason other than "the process is already
    gone" is logged and the kill signal is still attempted: the GPU has been promised
    away, so an unexpected signalling error must not silently leave the job running. A
    failing kill signal is logged too.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return  # exited between the poll and the signal
    except OSError as exc:  # includes PermissionError
        logger.warning(
            "could not signal job process group (pid {}) to terminate: {}; "
            "trying the kill signal anyway",
            proc.pid,
            exc,
        )
    else:
        deadline = time.time() + grace_s
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(poll_s)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        logger.warning("could not kill job process group (pid {}): {}", proc.pid, exc)


def _pgid_of(proc: subprocess.Popen) -> int:
    """The process group of a job's process; its own pid when the group cannot be read.

    Every job is started with ``start_new_session=True``, so it leads its own group and
    the two numbers are the same.
    """
    try:
        return os.getpgid(proc.pid)
    except OSError:
        return proc.pid


def _log_field(log_path) -> str:
    """The ``log`` a result records: a file that is there, or nothing at all.

    A result that points at a directory, or at a file that could never be opened, sends
    whoever reads it looking for output that does not exist. The path is only written
    when there is something at the end of it.
    """
    try:
        return str(log_path) if log_path and Path(log_path).is_file() else ""
    except OSError:
        return ""


def _pid_file_content(hostname: str) -> str:
    return f"{hostname}:{os.getpid()}"


def _acquire_pool_lock(root: Path, hostname: str) -> int:
    """Lock this machine's pool file for the pool's whole life; return the open descriptor.

    The lock, not the pid file, is what makes one pool per machine and queue folder: it is
    taken before anything else is read or written, the kernel drops it when this process
    ends however it ends, and the file behind it is never removed, so there is no window
    in which a second starter can decide the record is left over and take over. Whatever
    the pid file holds — a pid, nothing, a half-written line, no file at all — a second
    pool is refused while the first one lives.

    Raises:
        RuntimeError: another pool holds the lock, or this filesystem cannot lock it.
    """
    p = store.worker_lock_path(root, hostname)
    store.ensure_dir(Path(root))
    fd = os.open(p, os.O_CREAT | os.O_RDWR, io.lock_file_mode())
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        pid = store.read_worker_pid(root, hostname)
        held_by = f" (pid {pid})" if pid else ""
        raise RuntimeError(
            f"a worker pool is already running on {hostname} for this queue folder"
            f"{held_by}; it holds {p}"
        ) from None
    except OSError as exc:
        os.close(fd)
        raise RuntimeError(
            f"the filesystem holding {p} does not support the lock a pool needs to be the "
            f"only one on {hostname}: {exc}"
        ) from exc
    return fd


def _write_pid_file(root: Path, hostname: str) -> None:
    """Record this pool's pid for the reader; the pool lock is what reserves the machine."""
    io.atomic_write_text(store.worker_pid_path(root, hostname), _pid_file_content(hostname))


def _release_pid_file(root: Path, hostname: str) -> None:
    """Remove this machine's pid file, but only while it still names this process.

    A pool that was refused at startup never wrote the file, and a pool that has since
    started on this machine has rewritten it with its own line. Either way, removing a
    file that names another process would leave the running pool unreadable to ``jobq
    status``, so the content is compared first.
    """
    p = store.worker_pid_path(root, hostname)
    try:
        if p.read_text().strip() != _pid_file_content(hostname):
            return
    except OSError:
        return
    with contextlib.suppress(OSError):
        p.unlink()


class CwdUnreachable(RuntimeError):
    """The directory a job asked to run in does not exist on this machine."""


class MpsUnavailable(RuntimeError):
    """The policy asks for MPS with ``mps: true`` and this machine cannot use it."""


def resolve_cwd(cwd: str | None, fallback: str | None) -> str | None:
    """The directory to run a job in on this machine.

    ``None`` in and ``None`` out means "inherit the pool's directory". A named directory
    that is reachable here is used as-is; one that is not falls back to the policy's
    ``cwd_fallback`` when that is itself reachable, and otherwise raises
    :class:`CwdUnreachable`, and the pool leaves the job to other machines.
    """
    if cwd is None:
        return None
    if os.path.isdir(cwd) and os.access(cwd, os.X_OK):
        return cwd
    if fallback and os.path.isdir(fallback) and os.access(fallback, os.X_OK):
        return fallback
    raise CwdUnreachable(
        f"working directory {cwd!r} does not exist (or is not searchable) on this machine "
        "and this machine's GPU policy sets no usable 'cwd_fallback'"
    )


def _dependency_rings(metas: dict[str, QueueMeta]) -> dict[str, str]:
    """Each queue that sits in a ring of queues waiting on each other, and the ring.

    Queues that wait on each other can never complete: each is waiting for another to
    finish first. Without this a pool stays alive on them for ever, since nothing about
    any single queue of the ring says anything is wrong with it.

    Found by walking each queue's dependencies depth first and looking for a name
    already on the path; everything from that name onwards is in the ring, which is
    written as the names in the order they wait on each other.
    """
    rings: dict[str, str] = {}
    settled: set[str] = set()

    def _walk(name: str, path: list[str]) -> None:
        if name in path:
            ring = path[path.index(name) :]
            text = " -> ".join([*ring, name])
            for member in ring:
                rings.setdefault(member, text)
            return
        if name in settled or name not in metas:
            return
        path.append(name)
        for dep in metas[name].depends_on:
            _walk(dep, path)
        path.pop()
        settled.add(name)

    for name in metas:
        _walk(name, [])
    return rings


class _Pool:
    """Shared coordinator across worker threads (deps, once-only completion logs, stop)."""

    def __init__(
        self,
        root: Path,
        hostname: str,
        gpu: GpuInterface,
        master: MasterLog,
        *,
        queues_filter: set[str] | None,
        poll_s: float,
        gpu_wait_s: float,
        mps_env: dict[str, str] | None = None,
        shutdown: _Shutdown | None = None,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.gpu = gpu
        self.master = master
        self.queues_filter = queues_filter
        self.poll_s = poll_s
        self.gpu_wait_s = gpu_wait_s
        # Injected only when ensure_mps_daemon succeeded, empty otherwise. Merged into every
        # job's env so clients find the shared MPS server.
        self.mps_env = mps_env or {}
        self._lock = threading.Lock()
        self._complete_logged: set[str] = set()
        self._dep_warned: set[str] = set()
        # Live jobs by physical GPU so the yield watchdog can kill this GPU's jobs:
        # {gpu: {(queue, jobkey): _RunHandle}}. Guarded by its own lock (touched from worker
        # threads + the watchdog thread).
        self._running_lock = threading.Lock()
        # Queue is part of the identity: queues may reuse a job key on one GPU.
        self._running: dict[int, dict[tuple[str, str], _RunHandle]] = {}
        # Claims this process currently owns, counted per (queue, jobkey). Noted once per
        # iteration right after claim_next and dropped exactly once by that iteration's
        # bracket finally (_admit_and_run). The supervisor's reap_orphan_claims() removes
        # any on-disk claim stamped with our pid that has no count here: a claim stranded by
        # a crash mid-iteration would otherwise count as RUNNING forever, invisible to
        # steal_stale because the pool pid stays alive. A count (not a set)
        # because a handed-back claim can be re-claimed by a peer thread before the first
        # thread's drop runs: with a set, that drop would erase the peer's live entry and
        # the janitor would later reap a RUNNING job's claim (duplicate execution). Guarded
        # by its own lock.
        self._owned_lock = threading.Lock()
        self._owned_claims: dict[tuple[str, str], int] = {}
        # Threads parked in _acquire_gpu: {thread ident: (priority, cap_per_gpu, cap_group)}.
        # Read on every poll by every other parked thread (priority-ordered admission);
        # guarded by its own lock.
        self._park_lock = threading.Lock()
        self._parked: dict[int, tuple[int, int | None, str | None]] = {}
        # Queues this pool has already logged a PAUSE for, so the line appears once per
        # pause rather than once per claim-loop pass.
        self._paused_seen: set[str] = set()
        # The queue whose lapsed pause this thread took the probe of in its current pass.
        self._probe_tls = threading.local()
        # Queues whose unreachable working directory this pool has reported, and the jobs
        # it handed back because their own working directory is unreachable here: those
        # are left to other machines and never claimed again by this pool.
        self._cwd_warned: set[str] = set()
        self._cwd_skipped: set[tuple[str, str]] = set()
        # Queues whose zero memory ask this pool has already reported raising to the
        # machine's smallest grantable one (see _effective_mem), once per queue.
        self._zero_ask_queues: set[str] = set()
        # Median job durations already read for a yield of a GPU, keyed by the GPU, the
        # moment that yield started and the queue (see _drain_median).
        self._median_lock = threading.Lock()
        self._drain_medians: dict[tuple[int, float, str], float | None] = {}
        # Reasons for leaving already reported for the pool as a whole (see
        # _log_exit_reason), so the line appears once however many threads there are.
        self._exit_logged: set[str] = set()
        # The requests to leave, shared with the signal handlers the pool installs
        # before it builds this (see :class:`_Shutdown`). ``draining`` is set by an
        # interrupt or termination signal: the pool claims no further job and its
        # running jobs finish and record their results. The worker threads read it
        # wherever they read the stop file, so they leave their waits. ``end_now`` is
        # ``jobq stop --now``: the running jobs go back in the queue and the pool leaves.
        self._shutdown = shutdown if shutdown is not None else _Shutdown()
        self.draining = self._shutdown.draining
        self.end_now = self._shutdown.end_now
        # Queues this pool has already reported as unreadable by this account.
        self._unreadable_warned: set[str] = set()
        # How many worker threads the pool is aiming for. Read by every worker thread and
        # written by the supervisor, so both go through the property below, under the
        # pool's lock.
        self._worker_target = 1

    @property
    def interrupt_signal(self) -> str | None:
        """The name of the signal that asked this pool to drain, for the exit line."""
        return self._shutdown.signal_name

    @interrupt_signal.setter
    def interrupt_signal(self, value: str | None) -> None:
        self._shutdown.signal_name = value

    def waiting_count(self) -> int:
        """How many of this pool's claimed jobs are parked waiting for capacity."""
        with self._park_lock:
            return len(self._parked)

    def parked_tids(self) -> set[int]:
        """The worker threads parked waiting for capacity right now.

        A GPU is only held for a job while the thread holding the claim is still
        waiting for it, so the supervisor gives back a GPU held for a thread that is
        not in this set (see :meth:`jobq.gpu.GpuInterface.release_dead_reservations`).
        """
        with self._park_lock:
            return set(self._parked)

    @property
    def worker_target(self) -> int:
        """The number of worker threads the pool is aiming for."""
        with self._lock:
            return self._worker_target

    @worker_target.setter
    def worker_target(self, value: int) -> None:
        with self._lock:
            self._worker_target = int(value)

    def _tunable(self, name: str):
        """One tunable: this machine's policy value, or the module-level default.

        A :class:`~jobq.gpu.GpuInterface` with no policy behind it answers ``None``, and
        the module-level name is then read fresh on every call.
        """
        t = self.gpu.tunables()
        if t is not None:
            return getattr(t, name)
        return globals()[_TUNABLE_FALLBACKS[name]]

    # ------------------- queue selection -------------------

    def _node_queues(self) -> list[str]:
        """Queue names this machine may claim from (machine tie + any --queues restriction).

        A queue whose settings cannot be read is skipped, once with a line in the log: it
        cannot be claimed from, and raising here would stop the pool draining the queues
        that are fine.
        """
        out = []
        for name in store.list_queues(self.root):
            if self.queues_filter is not None and name not in self.queues_filter:
                continue
            try:
                meta = store.read_meta(self.root, name)
            except (store.QueueMetaUnreadable, store.QueuePathUnsafe) as exc:
                self._log_unreadable(name, exc)
                continue
            if self._tie_allows_here(meta):
                out.append(name)
        return out

    def _tie_gpus(self, meta: QueueMeta) -> tuple[int, ...] | None:
        """The GPUs of this machine a queue is allowed on; ``None`` when it may use any.

        The GPUs both its tie and this machine's policy name. An empty tuple means the
        queue names GPUs here and this machine's policy selects none of them, so no job
        of it that wants a GPU can ever run here.
        """
        named = meta.gpus_on(self.hostname)
        if named is None:
            return None
        policy_gpus = self.gpu.policy_gpus()
        if policy_gpus is None:
            return named  # no policy to read: the tie is all this machine knows
        return tuple(g for g in named if g in policy_gpus)

    def _usable_gpus(self, meta: QueueMeta) -> tuple[int, ...] | None:
        """The GPUs of this machine a queue may be placed on right now; ``None`` = any.

        :meth:`_tie_gpus` less the GPUs yielded to another user, which is the set a job
        of this queue may actually be dispatched to. A GPU that is merely yielded comes
        back, so an empty answer here is a wait rather than a refusal; whether the queue
        can ever run here is :meth:`_tie_gpus`.
        """
        named = self._tie_gpus(meta)
        if named is None:
            return None
        yielded = yielding.yielded_gpus(self.root, self.hostname)
        return tuple(g for g in named if g not in yielded)

    def _tie_allows_here(self, meta: QueueMeta) -> bool:
        """Whether this machine may claim from a queue, by machine and by GPU.

        A queue whose tie names GPUs this machine's policy does not select can never run
        here, so it is left alone rather than claimed and parked on. A GPU that is
        merely yielded is a wait, not a refusal, so it still counts as one this machine
        could work on.

        The GPUs of a tie say nothing about a job that takes no GPU, so a queue holding
        such a job is claimable here for it whatever its GPU list names. Each job's own
        ``slots`` decides that, since the queue default only applies to the jobs that
        state none.
        """
        if not meta.allows_machine(self.hostname):
            return False
        allowed = self._tie_gpus(meta)
        if allowed is None or allowed:
            return True
        return self._has_cpu_only_job(meta)

    def _claimable_job(self, meta: QueueMeta):
        """Which jobs of a queue this machine may take, or ``None`` when every one of them.

        A queue whose GPU tie names no GPU this machine's policy selects is still
        claimable here for the jobs that take no GPU, and only for those: a job of it
        that wants a GPU has none to wait for here and is left to a machine the tie
        allows. A job this pool handed back because its working directory is unreachable
        here is not taken again.
        """
        name = meta.name
        with self._lock:
            skipped = {k for q, k in self._cwd_skipped if q == name}
        allowed = self._tie_gpus(meta)
        if allowed is None or allowed:
            if not skipped:
                return None
            return lambda job: job.jobkey not in skipped
        default = int(meta.defaults.get("slots", 1))
        return lambda job: (
            (default if job.slots is None else job.slots) == 0 and job.jobkey not in skipped
        )

    def _has_cpu_only_job(self, meta: QueueMeta) -> bool:
        """Whether any job of a queue takes no GPU, by its own field or the queue default.

        Read from the jobs file, and only for a queue whose GPU tie rules this machine
        out, which is the one case where the answer changes what the pool does.
        """
        default = int(meta.defaults.get("slots", 1))
        try:
            jobs = store.load_jobs(self.root, meta.name)
        except (OSError, ValueError):
            return default == 0
        return any((default if job.slots is None else job.slots) == 0 for job in jobs)

    def _maybe_log_complete(self, name: str) -> None:
        with self._lock:
            if name in self._complete_logged:
                return
            self._complete_logged.add(name)
        self.master.log("QUEUE COMPLETE", name)

    def _deps_ok(self, meta: QueueMeta) -> bool:
        for dep in meta.depends_on:
            if not (store.queue_exists(self.root, dep) and store.queue_complete(self.root, dep)):
                # A dep that exists but holds no jobs is never "complete", so this queue
                # would park forever with nothing to show for it; say so once.
                if store.queue_exists(self.root, dep) and not store.load_jobs(self.root, dep):
                    with self._lock:
                        warn_key = f"{meta.name}<-{dep}:empty"
                        warn = warn_key not in self._dep_warned
                        self._dep_warned.add(warn_key)
                    if warn:
                        self.master.log(
                            "WAIT",
                            f"dep {dep} exists but has no jobs; it can never complete and "
                            f"{meta.name} will never start (submit its jobs, or drop the dep)",
                        )
                return False
            failed = store.queue_status(self.root, dep)["failed"]
            if failed:
                strict = meta.strict_deps
                with self._lock:
                    warn_key = f"{meta.name}<-{dep}"
                    warn = warn_key not in self._dep_warned
                    self._dep_warned.add(warn_key)
                if warn:
                    self.master.log(
                        "WAIT",
                        f"dep {dep} completed with {failed} failed "
                        + ("job " if failed == 1 else "jobs ")
                        + (
                            f"— strict_deps blocks {meta.name} until "
                            + ("it is" if failed == 1 else "they are")
                            + " requeued or fixed"
                            if strict
                            else "(treated as terminal)"
                        ),
                    )
                if strict:
                    return False
        return True

    def _observe_pause(self, name: str) -> bool:
        """Whether ``name`` is paused right now; logs the change once per pause.

        One stat per queue per pass while no pause file exists — the claim loop must not
        read result files to learn this. When a pause's ``until_utc`` has passed, the one
        worker that takes the probe (:func:`store.take_pause_probe`) logs ``PAUSE LAPSED``
        and claims one job; one cleared by a success or ``jobq resume`` is logged as
        ``RESUME``.
        """
        paused = store.queue_paused(self.root, name)
        probe = False
        if not paused and store.paused_path(self.root, name).exists():
            # A lapsed pause: one worker on one machine takes the probe under the lock;
            # every other one, and a draining pool, reads the queue as still paused.
            probe = not self.draining.is_set() and store.take_pause_probe(
                self.root, name, self._tunable("failure_pause_s")
            )
            paused = not probe and store.paused_path(self.root, name).exists()
        if probe:
            self._probe_tls.queue = name
            with self._lock:
                self._paused_seen.add(name)
            self.master.log(
                "PAUSE", f"LAPSED {name}: the failure pause ended; trying one job again"
            )
            return False
        with self._lock:
            seen = name in self._paused_seen
            if paused and not seen:
                self._paused_seen.add(name)
            elif not paused and seen:
                self._paused_seen.discard(name)
            else:
                return paused
        if paused:
            rec = store.read_pause(self.root, name) or {}
            keys = ", ".join(str(k) for k in (rec.get("keys") or [])) or "-"
            self.master.log(
                "PAUSE",
                f"{name} paused since {rec.get('since_utc', '?')} after "
                f"{rec.get('limit', '?')} consecutive job failures: {keys}; "
                f"resume with: jobq resume {name}",
            )
        else:
            self.master.log("RESUME", name)
        return paused

    def ready_queues(self) -> list[tuple[str, QueueMeta]]:
        """Ready = visible here, not parked, not paused, not complete, deps met — ordered.

        Ordered by priority descending, then creation time. A parked queue is skipped on
        every machine until ``jobq unpark``; the jobs it already has running finish
        normally, since nothing here touches a claim that exists.
        """
        ready: list[tuple[str, QueueMeta]] = []
        self._probe_tls.queue = None
        for name in self._node_queues():
            # A queue whose files this uid cannot read (written from another machine under a
            # restrictive umask) would otherwise raise PermissionError here and crash every
            # loop iteration before a single claim; skip it (logged once) so the readable
            # queues still drain. It is also excluded from all_complete().
            try:
                meta = store.read_meta(self.root, name)
                if meta.parked:
                    continue
                if self._observe_pause(name):
                    continue
                if store.queue_complete(self.root, name):
                    self._maybe_log_complete(name)
                    continue
                if not self._deps_ok(meta):
                    continue
                if self._queue_cwd_unreachable(meta) is not None:
                    self._log_cwd_unreachable(name, str(meta.defaults.get("cwd")))
                    continue
            except (PermissionError, store.QueueMetaUnreadable) as e:
                # The queue itself, or one of the queues it depends on.
                self._log_unreadable(name, e)
                continue
            ready.append((name, meta))
        ready.sort(key=lambda nm: store.queue_order_key(nm[1]))
        return ready

    def _queue_cwd_unreachable(self, meta: QueueMeta) -> str | None:
        """Why the queue's default working directory cannot be used here, or ``None``."""
        d = meta.defaults.get("cwd")
        if d is None:
            return None
        try:
            resolve_cwd(str(d), self._cwd_fallback())
        except CwdUnreachable:
            return (
                f"working directory {d} is not reachable on this machine and the policy "
                "sets no usable cwd_fallback"
            )
        return None

    def _log_cwd_unreachable(self, name: str, cwd: str) -> None:
        """Say once per queue that its jobs are left for other machines."""
        with self._lock:
            if name in self._cwd_warned:
                return
            self._cwd_warned.add(name)
        self.master.log(
            "WAIT",
            f"queue {name}: working directory {cwd} is not reachable on this machine and "
            "the policy sets no usable cwd_fallback; its jobs are left for other machines",
        )

    def _hand_back_cwd(self, name: str, job: Job, cwd: str) -> None:
        """Hand back a claimed job whose own working directory is unreachable here."""
        with self._lock:
            self._cwd_skipped.add((name, job.jobkey))
        store.remove_claim(self.root, name, job.jobkey)
        self._log_cwd_unreachable(name, cwd)

    def _log_unreadable(self, name: str, err: Exception) -> None:
        with self._lock:
            if name in self._unreadable_warned:
                return
            self._unreadable_warned.add(name)
        self.master.log("WAIT", f"queue {name} unreadable by this account; skipping ({err})")

    def _incomplete_metas(self) -> list[tuple[str, QueueMeta]]:
        """Every queue in the folder that is not complete and whose settings can be read.

        The whole folder, not only the queues this pool may claim from: whether a
        dependency can still be met is decided by the queues it waits on wherever they
        run.
        """
        out = []
        for name in store.list_queues(self.root):
            try:
                if store.queue_complete(self.root, name):
                    continue
                out.append((name, store.read_meta(self.root, name)))
            except (store.QueueMetaUnreadable, store.QueuePathUnsafe, PermissionError):
                continue
        return out

    def _never_completes(self) -> dict[str, str]:
        """Queues that can never complete anywhere, with the reason for each.

        A queue is stuck in itself when it is parked, holds no jobs at all, waits on a
        queue that is not in this folder, or sits in a ring of queues waiting on each
        other. It is stuck by its dependencies when one of the queues it waits on is
        stuck, however far down the chain. A queue tied to another machine is not stuck:
        a pool there can still finish it. Neither is one a run of failures has paused,
        nor anything waiting on that one: a person clears a pause, often while the pool
        runs, so it is still work this pool is here to do.
        """
        metas = dict(self._incomplete_metas())
        reason: dict[str, str] = {}
        rings = _dependency_rings(metas)
        for name, meta in metas.items():
            missing = [
                dep for dep in meta.depends_on if not store.queue_exists(self.root, dep)
            ]
            if meta.parked:
                reason[name] = "is parked"
            elif not store.load_jobs(self.root, name):
                reason[name] = "holds no jobs"
            elif missing:
                reason[name] = (
                    f"waits on {missing[0]}, which is not a queue in this folder, so "
                    "nothing can ever complete it"
                )
            elif name in rings:
                reason[name] = (
                    "sits in a ring of queues waiting on each other: " + rings[name]
                )
            else:
                continue
        while True:
            fresh = {
                name: f"waits on {dep}, which {reason[dep]}"
                for name, meta in metas.items()
                if name not in reason
                for dep in meta.depends_on
                if dep in reason
            }
            if not fresh:
                return reason
            reason.update(fresh)

    def set_aside_reasons(self) -> dict[str, str]:
        """Queues with work left that no pool on this machine can bring closer to done.

        A queue set aside with ``jobq park``, one tied to machines this is not among (or
        to GPUs this machine's policy does not select), one that holds no jobs of its
        own, and one whose dependencies can never be met anywhere. They are counted rather than waited for: a queue folder
        can carry parked queues for weeks, and a pool that stayed alive for them would
        never finish its own work.

        A queue under the failure pause is not among them, and neither is one waiting on
        such a queue: a person clears that pause, often while the pool runs, so both are
        still work this pool is here to do. Nor is a queue that waits on one running on
        another machine, which that machine can still finish.
        """
        reason = self._never_completes()
        aside: dict[str, str] = {}
        for name, meta in self._incomplete_metas():
            if self.queues_filter is not None and name not in self.queues_filter:
                continue
            if meta.parked:
                aside[name] = "parked"
            elif not meta.allows_machine(self.hostname):
                aside[name] = "tied to machines this one is not among"
            elif not self._tie_allows_here(meta):
                aside[name] = "tied to GPUs this machine's policy does not select"
            elif self._queue_cwd_unreachable(meta) is not None:
                aside[name] = self._queue_cwd_unreachable(meta)
            elif name in reason:
                # Including the queues stuck in themselves. A queue that holds no jobs
                # is never complete and never will be, so a pool that waited for it
                # would stay alive for ever with nothing to do.
                aside[name] = reason[name]
        return aside

    def set_aside(self) -> list[str]:
        """The names :meth:`set_aside_reasons` gives, in order."""
        return sorted(self.set_aside_reasons())

    def all_complete(self) -> bool:
        """True once every queue this pool could work on is complete (or there are none).

        Unreadable queues (see ``ready_queues``) are ignored here: they can neither be
        drained nor verified from this uid, so they must not keep the pool alive forever.
        The queues :meth:`set_aside` names are ignored for the same reason — nothing this
        pool does can bring them any closer to done.
        """
        aside = set(self.set_aside())
        for n in self._node_queues():
            if n in aside:
                continue
            try:
                if not store.queue_complete(self.root, n):
                    return False
            except PermissionError:
                continue
        return True

    def root_complete(self) -> bool:
        """True once every queue in the queue folder is complete, whoever they belong to.

        Wider than :meth:`all_complete`, which only looks at the queues this pool may
        claim from: a pool restricted with ``--queues``, or one on a machine other queues
        are pinned to, can drain its own share while the folder still holds work.
        """
        for n in store.list_queues(self.root):
            try:
                if not store.queue_complete(self.root, n):
                    return False
            except PermissionError:
                continue
        return True

    # ------------------- run one job -------------------

    def _learned_mem(self, name: str) -> int | None:
        """What this queue's own jobs have shown they need, bounded by an idle GPU here.

        A queue that states no memory starts at the machine's modest default and then
        asks for what its jobs turned out to use (see
        :func:`jobq.store.note_peak_mem`). The figure can never rise above what a GPU of
        this machine grants while it is idle (see
        :meth:`jobq.gpu.GpuInterface.idle_mem_ceiling_mib`), since a request no GPU can
        meet would leave the queue waiting for ever. A request the ceiling cut down is
        noted on the queue's record, so ``jobq status QUEUE`` says the queue asks for
        less than its jobs were measured to need.
        """
        learned = store.learned_mem_mib(self.root, name)
        if learned is None:
            return None
        ceiling = self._idle_mem_ceiling()
        if not ceiling or learned <= ceiling:
            return learned
        try:
            store.note_learned_mem_capped(self.root, name, ceiling, self.hostname)
        except OSError as exc:
            logger.warning("could not record the capped memory request for {}: {}", name, exc)
        return ceiling

    def _idle_mem_ceiling(self) -> int:
        """Largest request an idle GPU here could grant; ``0`` when it is not known."""
        try:
            return int(self.gpu.idle_mem_ceiling_mib() or 0)
        except Exception:  # noqa: BLE001 — a broken smi must not fail a job
            logger.exception("could not read the idle GPU memory ceiling; treating as unbounded")
            return 0

    def _effective_mem(self, name: str, meta: QueueMeta, job: Job) -> int:
        """The job's memory request: max(static resolution, persisted OOM floor).

        Statically it is the job's own ``mem_mib``, then the queue's, then what the
        queue's finished jobs have shown they need, then this machine's ``free_mem_mib``.
        A job or a queue that states an amount is therefore never affected by what other
        jobs used.

        The floor lives on disk (attempts sidecar), so an escalation survives a worker
        restart and is honored by whichever machine claims the job next. Operator reset:
        ``jobq reset-mem-floor``.

        A machine whose policy caps nothing but memory has no room for an ask of zero, which
        would pass every memory gate: such an ask is raised to the machine's smallest
        grantable one (see :meth:`GpuInterface.min_mem_ask_mib`), logged once per queue.
        """
        if job.mem_mib is not None:
            static = job.mem_mib
        else:
            d = meta.defaults.get("mem_mib")
            if d is not None:
                static = int(d)
            else:
                learned = self._learned_mem(name)
                static = (
                    learned if learned is not None else self.gpu.default_mem_mib()
                )
        mem_mib = max(static, store.read_mem_floor(self.root, name, job.jobkey))
        if mem_mib > 0:
            return mem_mib
        minimum = int(self.gpu.min_mem_ask_mib() or 0)
        if minimum <= 0:
            return mem_mib
        with self._lock:
            first = name not in self._zero_ask_queues
            self._zero_ask_queues.add(name)
        if first:
            logger.info(
                "queue {}: a memory ask of 0 cannot be placed on a machine without "
                "cap_per_gpu; asking for the policy's free_mem_mib ({} MiB) instead",
                name,
                minimum,
            )
        return minimum

    def _cwd_fallback(self) -> str | None:
        """This machine's policy ``cwd_fallback``; ``None`` when there is no policy to read."""
        return self.gpu.cwd_fallback()

    def _oom_patterns(self) -> tuple[str, ...]:
        """This machine's policy ``oom_patterns``; empty means the built-in markers."""
        return tuple(self.gpu.oom_patterns() or ())

    def _mem_ceiling(self) -> int:
        """Highest grantable ``mem_mib`` on this machine; ``0`` when unknown (= no ceiling)."""
        try:
            return int(self.gpu.mem_ceiling_mib() or 0)
        except Exception:  # noqa: BLE001 — a broken smi must not fail a job
            logger.exception("could not read the GPU memory ceiling; treating as unbounded")
            return 0

    def _oom_requeue(
        self, name: str, meta: QueueMeta, job: Job, log_path: Path, settled: list,
        *, admitted_mib: int | None = None, peak_mib: int | None = None,
    ) -> bool:
        """Requeue an OOM-killed job with an escalated memory floor; False = handle normally.

        False means "not an OOM", "already asking for the whole GPU", or "backstop
        exceeded". The last two end the job for good, and it is recorded as the job's own
        failure: a job that ran out of memory at the top of the ladder, or that kept
        running out of memory until the retries ran out, has failed for a reason of its
        own, and a queue whose jobs all do that is exactly what the failure pause is for.
        The memory gate remains the designed terminator — the ladder is finite
        whenever the ceiling is readable; ``OOM_REQUEUE_BACKSTOP`` exists for when it is
        not (broken smi ⇒ unbounded escalation) and for misclassified OOMs (a non-OOM
        failure whose log tail merely contains an OOM marker), both of which would
        otherwise requeue forever without ever counting as a failure.

        ``admitted_mib`` is the ask the job was actually admitted with, which is what
        the escalation multiplies. Working it out again here would read whatever the
        queue's settings say now, and a queue whose ``mem_mib`` was edited, or whose
        learned request moved, while the job ran would have its ladder restart from the
        new number.
        """
        if not log_tail_is_oom(log_path, patterns=self._oom_patterns()):
            return False
        requeues = store.read_oom_requeues(self.root, name, job.jobkey)
        backstop = self._tunable("oom_max_requeues")
        if requeues >= backstop:
            self.master.log(
                "OOM BACKSTOP",
                f"{name} {job.jobkey} OOM-requeued {requeues}x without resolving "
                "(misclassified OOM, or memory ceiling unknown); recording failure",
            )
            return False
        old = (
            admitted_mib
            if admitted_mib
            else self._effective_mem(name, meta, job)
        )
        ceiling = self._mem_ceiling()
        # Attainable, not total: see OOM_CEILING_HEADROOM_MIB.
        headroom = self._tunable("oom_ceiling_headroom_mib")
        floor_mib = self._tunable("oom_mem_floor_mib")
        attainable = max(ceiling - headroom, floor_mib) if ceiling else 0
        if attainable and old >= attainable:
            self.master.log(
                "OOM CEILING",
                f"{name} {job.jobkey} mem_mib {old} >= attainable GPU memory {attainable} "
                f"(total {ceiling}); recording failure",
            )
            return False
        own = log_tail_oom_own_usage_mib(log_path)
        want = log_tail_oom_requested_mib(log_path)
        # A small footprint alone does not mean "starved by neighbours": a job holding 5 GiB
        # of a 16 GiB ask that asks for 20 GiB is under-reserved, and retrying at the
        # same floor just burns the backstop. Co-tenant only when what it held plus what it
        # asked for still fits inside the ask.
        cotenant = (
            own is not None
            and own < old * self._tunable("oom_own_usage_fraction")
            and (want is None or own + want <= old)
        )
        if cotenant:
            # Starved by neighbours, not under-reserved: retry at the same ask.
            new = old
        else:
            new = max(math.ceil(old * self._tunable("oom_mem_factor")), floor_mib)
            if attainable:
                new = min(new, attainable)
        # All fallible writes happen before remove_claim (settled contract, see _run_job):
        # a throw up to here leaves the claim ours; after the removal a peer may re-claim,
        # so only the swallowing master.log may follow.
        store.note_oom_requeue(
            self.root, name, job.jobkey, mem_mib_floor=None if new == old else new
        )
        store.remove_claim(self.root, name, job.jobkey)
        settled.append(True)
        wanted = f" (wanted {want} MiB)" if want is not None else ""
        # No result record for a requeued attempt, so the measurement lives in the log line.
        peak_s = f" peak_mem={peak_mib}MiB" if peak_mib is not None else ""
        detail = (
            f"co-tenant pressure: process held {own} MiB of its {old} MiB ask"
            f"{wanted}; floor kept"
            if cotenant
            else f"mem_mib {old} -> {new}{wanted}"
        )
        self.master.log(
            "OOM REQUEUE",
            f"{name} {job.jobkey} {detail} ({requeues + 1}/{backstop}){peak_s}",
        )
        return True

    def _tempfail_requeue(self, name: str, job: Job, settled: list, *,
                          peak_mib: int | None = None) -> bool:
        """Requeue an rc=75 job after a delay; False = record it as a failure (backstop).

        A job that has used up its retries is recorded as having failed, and counts
        towards its queue's run of failures: something it needs has been busy every time
        it ran, and a whole queue behaving that way is worth pausing.

        Ordering follows the settled contract exactly like :meth:`_oom_requeue`: every
        fallible store write happens before ``remove_claim``, and only the swallowing
        ``master.log`` may follow it.
        """
        tempfails = store.read_tempfails(self.root, name, job.jobkey)
        backstop = self._tunable("tempfail_max_requeues")
        if tempfails >= backstop:
            self.master.log(
                "TEMPFAIL BACKSTOP",
                f"{name} {job.jobkey} tempfail-requeued {tempfails}x without ever running "
                "to completion; recording failure",
            )
            return False
        retry_at = time.time() + self._tunable("tempfail_retry_s")
        store.bump_tempfails(self.root, name, job.jobkey, not_before=retry_at)
        store.bump_attempt(self.root, name, job.jobkey)
        store.remove_claim(self.root, name, job.jobkey)
        settled.append(True)
        when = time.strftime("%H:%M:%SZ", time.gmtime(retry_at))
        peak_s = f" peak_mem={peak_mib}MiB" if peak_mib is not None else ""
        self.master.log(
            "TEMPFAIL",
            f"{name} {job.jobkey} rc={TEMPFAIL_RC} requeued "
            f"({tempfails + 1}/{backstop}), retry after {when}{peak_s}",
        )
        return True

    def _effective_slots(self, meta: QueueMeta, job: Job) -> int:
        """The job's weight in global slot-units (job field > queue default > 1)."""
        if job.slots is not None:
            return job.slots
        return int(meta.defaults.get("slots", 1))

    def _effective_cwd(self, meta: QueueMeta, job: Job) -> str | None:
        """The directory the job asks to run in: job field, else the queue default.

        ``None`` means the queue carries no default either, and the job then inherits the
        pool process's directory. ``jobq submit`` records a default, so that is only the
        case for a queue whose meta was written by hand.
        """
        if job.cwd is not None:
            return job.cwd
        d = meta.defaults.get("cwd")
        return None if d is None else str(d)

    def _effective_cap(self, meta: QueueMeta) -> tuple[int | None, str | None]:
        """Queue-level stage ceiling: (cap_per_gpu, cap_group) from meta.defaults.

        ``cap_group`` defaults to the queue name so each capped queue gets its own ceiling;
        set the same ``cap_group`` on several queues to co-cap them.
        """
        cap = meta.defaults.get("cap_per_gpu")
        if cap is None:
            return None, None
        return int(cap), str(meta.defaults.get("cap_group") or meta.name)

    def _higher_eligible_waiter(self, me: int, priority: int) -> bool:
        """Is a strictly higher-priority thread parked that could take a slot right now?

        Eligible = uncapped, or capped with stage-group room (a capped waiter whose group is
        saturated is about to yield and must not hold lower waiters back). The room probe is
        the same racy probe-and-release as the pre-claim check.
        """
        with self._park_lock:
            ahead = [
                (cap, group) for tid, (p, cap, group) in self._parked.items()
                if tid != me and p > priority
            ]
        return any(
            cap is None or self.gpu.group_has_room(group, cap) for cap, group in ahead
        )

    def _wait_no_longer_allowed(
        self, name: str, slots: int, tie: tuple[int, ...] | None
    ) -> str | None:
        """Why a thread waiting for capacity should give its claim back, or ``None``.

        ``tie`` is the GPUs of this machine the queue allowed when the job was claimed
        (``None`` = any of them). A job that takes no GPU is unaffected by a change to
        that list, since it never had one.
        """
        try:
            meta = store.read_meta(self.root, name)
        except (store.QueueMetaUnreadable, store.QueuePathUnsafe) as exc:
            return f"its queue's settings cannot be read ({exc})"
        if meta.parked:
            return "its queue has been parked"
        if store.queue_paused(self.root, name) and getattr(
            self._probe_tls, "queue", None
        ) != name:
            return "its queue has been paused"
        if not meta.allows_machine(self.hostname):
            return "its queue is not tied to this machine any more"
        if slots > 0 and self._tie_gpus(meta) != tie:
            return "its queue does not allow the GPUs this job was going to use"
        return None

    def _acquire_gpu(
        self,
        mem_mib: int,
        slots: int,
        cap_per_gpu: int | None,
        cap_group: str | None,
        job_key: str,
        priority: int = 0,
        gpus: tuple[int, ...] | None = None,
        name: str | None = None,
        tie: tuple[int, ...] | None = None,
    ):
        """Block until capacity is free or a stop is requested (returns None on stop).

        Admission among parked threads is priority-ordered: while a strictly higher-priority
        eligible waiter is parked, this thread does not even attempt the acquire (bounded by
        ``PARK_DEFER_MAX_S``, after which it tries on every poll). A capped job
        parks only while its stage group has room; once the group saturates it returns
        ``_YIELD`` and the caller hands the claim back.

        WAIT lines are rate-limited (first miss, then ~once a minute) and name the waiting
        job, so an unsatisfiable request (mem_mib/slots no GPU can ever meet) is visible in
        the master log instead of silently wedging a thread.

        The queue's settings are re-read on every poll. A wait can last hours, and in
        that time the queue may be parked, paused, or tied away from this machine or
        from the GPUs this job was going to use — all of which mean no pool here should
        be holding one of its jobs. The claim is then handed back (``_YIELD``), which is
        not an attempt and not a failure: the job is pending again for whichever machine
        the queue now belongs to.
        """
        misses = 0
        me = threading.get_ident()
        defer_max_s = self._tunable("park_defer_max_s")
        defer_since: float | None = None
        valve_logged = False
        waiting_since = time.monotonic()
        reserved: int | None = None
        unplaceable_logged = False
        with self._park_lock:
            self._parked[me] = (priority, cap_per_gpu, cap_group)
        try:
            while True:
                if self.draining.is_set():
                    return None
                if store.stop_path(self.root, self.hostname).exists():
                    return None
                if name is not None:
                    changed = self._wait_no_longer_allowed(name, slots, tie)
                    if changed is not None:
                        self.master.log(
                            "WAIT",
                            f"{name} {job_key} was waiting for capacity and is back in "
                            f"the queue: {changed}",
                        )
                        return _YIELD
                if cap_per_gpu is not None and not self.gpu.group_has_room(
                    cap_group, cap_per_gpu
                ):
                    return _YIELD
                if self._higher_eligible_waiter(me, priority):
                    now = time.monotonic()
                    defer_since = now if defer_since is None else defer_since
                    if now - defer_since < defer_max_s:
                        time.sleep(self.gpu_wait_s)
                        continue
                    if not valve_logged:
                        valve_logged = True
                        self.master.log(
                            "WAIT",
                            f"{job_key} deferred to a higher-priority waiter for "
                            f"{int(defer_max_s / 60)} min; trying anyway",
                        )
                else:
                    defer_since = None
                slot = self.gpu.acquire(
                    mem_mib,
                    slots=slots,
                    cap_per_gpu=cap_per_gpu,
                    cap_group=cap_group,
                    **({} if gpus is None else {"gpus": gpus}),
                )
                if slot is not None:
                    return slot
                misses += 1
                self._log_wait_miss(misses, job_key, mem_mib, slots)
                if slots > 0:
                    reserved, unplaceable_logged = self._maybe_reserve(
                        mem_mib,
                        slots,
                        job_key,
                        priority,
                        time.monotonic() - waiting_since,
                        reserved,
                        unplaceable_logged,
                        gpus,
                    )
                time.sleep(self.gpu_wait_s)
        finally:
            with self._park_lock:
                self._parked.pop(me, None)
            self._release_reservation(job_key, reserved)

    def _maybe_reserve(
        self,
        mem_mib: int,
        slots: int,
        job_key: str,
        priority: int,
        waited_s: float,
        reserved: int | None,
        unplaceable_logged: bool,
        gpus: tuple[int, ...] | None = None,
    ) -> tuple[int | None, bool]:
        """Ask this machine to hold a GPU for a job that has waited too long.

        Returns the GPU held for the job, if any, and whether the log has already said
        that the request is one no GPU here can ever grant. Both lines are written once
        per wait: a thread asks again on every poll, and the answer rarely changes.

        A GPU the job already holds is kept for it (see
        :meth:`jobq.gpu.GpuManager.reserve_for_wait`), so the answer only changes when a
        GPU is first held, or when a waiting job that outranks this one takes it over.
        That is one RESERVE line per GPU held, and one RELEASE when it goes: the job
        that takes a GPU over writes the RELEASE for the job it took it from, and that
        job sees the GPU held for another and says nothing more about it.
        """
        if not unplaceable_logged and not self.gpu.request_fits_somewhere(
            mem_mib, slots, gpus
        ):
            self.master.log(
                "WAIT",
                f"{job_key} asks for {mem_mib} MiB, which is more than any GPU of this "
                "machine can grant even when it is empty, so no GPU is held for it",
            )
            return reserved, True
        displaced: list = []
        gpu = self.gpu.reserve_for_wait(
            mem_mib,
            slots=slots,
            job_key=job_key,
            priority=priority,
            waited_s=waited_s,
            gpus=gpus,
            displaced=displaced,
        )
        if gpu is not None and gpu != reserved:
            for other in displaced:
                self.master.log(
                    "RELEASE",
                    f"gpu {gpu} was held for {other} and is held for {job_key} now, "
                    f"which has the higher priority; {other} waits for a GPU like any "
                    "other job",
                )
            self.master.log(
                "RESERVE",
                f"gpu {gpu} is held for {job_key} ({mem_mib} MiB, priority {priority}, "
                f"waiting {int(waited_s)}s); no other job of this queue folder starts "
                "there until it does",
            )
        elif gpu is None and reserved is not None:
            # A GPU that is held for another job now was taken over, and the job that
            # took it has already said so.
            if self.gpu.reservation_holder(reserved) is None:
                self.master.log(
                    "RELEASE",
                    f"gpu {reserved} is free for other jobs again; it was held for "
                    f"{job_key}",
                )
        return gpu, unplaceable_logged

    def _release_reservation(self, job_key: str, reserved: int | None) -> None:
        """Give back the GPU held for a job that has stopped waiting for one.

        Only a GPU this job still holds is given back, and only that is reported: a
        GPU taken over by a job that outranked this one belongs to that job, and it
        wrote the line about the change when it took it.
        """
        try:
            gpu = self.gpu.release_reservation(job_key)
        except OSError as exc:
            logger.warning("could not release the GPU held for {}: {}", job_key, exc)
            return
        if gpu is not None:
            self.master.log(
                "RELEASE",
                f"gpu {gpu} is free for other jobs again; it was held for {job_key}",
            )

    def _log_wait_miss(self, misses: int, job_key: str, mem_mib: int, slots: int) -> None:
        """Rate-limited WAIT line for a parked thread (first miss, then ~once a minute)."""
        per_min = max(1, int(60 / max(self.gpu_wait_s, 0.001)))
        if misses != 1 and misses % per_min != 0:
            return
        waited_min = int(misses * self.gpu_wait_s / 60)
        if slots == 0:
            # CPU-only jobs wait on the cpu_cap budget, never on GPU capacity or
            # the memory gate — pointing at mem_mib/gpu_policy here misdirects.
            need = "a cpu_cap slot (CPU-only job, slots=0)"
            hint = " — all cpu_cap slots held; check cpu_cap in this machine's gpu_policy"
        else:
            need = f"{mem_mib} MiB free + {slots} slot-unit(s)"
            hint = " — check mem_mib/slots against this machine's gpu_policy"
        self.master.log(
            "WAIT",
            f"{job_key} needs {need}; ~{waited_min}min so far"
            + (hint if waited_min >= 10 else ""),
        )

    def _startup_hold_s(self) -> float:
        """This machine's start-up window, the span a footprint is measured over."""
        pol = self.gpu._policy_or_none()
        return gpu_defaults.DEFAULT_STARTUP_HOLD_S if pol is None else pol.startup_hold_s

    def _wait_and_measure(self, proc, slot) -> int:
        """Wait for a job, taking one memory reading at the end of its start-up window.

        The footprint is the fall in the GPU's free memory over that window (see
        :meth:`jobq.gpu.GpuManager.measure_footprint`), so it has to be read while the
        job is running: once the process ends, its memory is gone from the GPU. A job
        shorter than the window is never measured, and neither is a job on a machine
        whose window is zero.
        """
        window = self._startup_hold_s()
        granted = slot.granted_ts
        if window > 0 and granted and not slot.is_cpu_only:
            remaining = max(0.0, granted + window - time.time())
            try:
                return proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    self.gpu.measure_footprint(slot)
                except Exception:  # noqa: BLE001 — a reading is never worth a failed job
                    logger.exception("could not measure the job's memory footprint")
        return proc.wait()

    def _note_peak(self, name: str, peak_mib, measured_mib, shared: bool = False) -> None:
        """Teach the queue what its jobs need, from a reported peak or a measured footprint.

        A peak the job reported is the better figure and is preferred; the measurement
        from the GPU stands in when the job reported nothing. ``shared`` marks a
        measurement that was divided among jobs granted together, which the queue treats
        with more care (see :func:`jobq.store.note_peak_mem`).
        """
        value, measured = (
            (peak_mib, False) if peak_mib is not None else (measured_mib, True)
        )
        if value is None:
            return
        try:
            store.note_peak_mem(
                self.root, name, int(value), measured=measured,
                shared=shared and measured,
            )
        except OSError as exc:
            logger.warning("could not record the memory {} used: {}", name, exc)

    def _run_job(
        self,
        name: str,
        meta: QueueMeta,
        job: Job,
        slot,
        attempt: int,
        settled: list,
        peak_out: list | None = None,
    ) -> int | None:
        """Run one job to completion. Returns its rc, or ``None`` if it was requeued.

        A yield-killed job — and, on the same path, one classified as CUDA OOM, or one that
        exited ``TEMPFAIL_RC`` — is requeued to PENDING (claim removed, no result recorded,
        attempt bumped) so it retries later; none is ever recorded as failed.

        ``settled`` is a mutable flag (append-once list): it is marked immediately after
        every ``remove_claim`` here, and each removal is the last fallible operation on its
        path (attempt bumps and result writes come before it). The caller's crash cleanup
        removes the claim only when ``settled`` is empty — once ownership is released a
        peer thread may have re-claimed this jobkey, and an unconditional second removal
        would delete the peer's live claim (a third thread then duplicates the run).
        """
        # The claim's identifier for this run, inherited by the job and every process it
        # starts: it is how the next pool on this machine finds what the run left behind
        # when this pool is killed. A job started without it is a run nobody can end, so
        # the claim goes back instead.
        run_id = store.read_run_id(self.root, name, job.jobkey)
        if not run_id:
            store.remove_claim(self.root, name, job.jobkey)
            settled.append(True)
            self.master.log(
                "RUN ID",
                f"{name} {job.key} not started: its claim carries no run identifier, so "
                "nothing could find the job's processes if this pool were killed; back "
                "in the queue",
            )
            return None
        store.update_owner_gpu(self.root, name, job.jobkey, slot.gpu)
        pid = os.getpid()
        stamp = utc_stamp()
        # The attempt is part of the name, so each attempt of a job has a log of its
        # own: the out-of-memory classification reads the tail of the log, and two
        # attempts sharing a file would let one attempt's message decide the next one's
        # fate.
        log_path = (
            store.queue_logs_dir(self.root, name)
            / f"{stamp}_{job.jobkey}.a{attempt}.{self.hostname}.log"
        )
        store.ensure_dir(log_path.parent)
        start = now_iso()
        env = os.environ.copy()
        env.update(self.gpu.env())
        env.update(self.mps_env)  # empty unless MPS was ensured up at pool startup
        env.update({str(k): str(v) for k, v in (meta.defaults.get("env") or {}).items()})
        env.update(job.env or {})
        # Ask the job to self-report its peak GPU memory; an explicit policy/job value wins.
        env.setdefault(REPORT_PEAK_MEM_ENV, "1")
        # slots=0 (declared CPU-only): no GPU was reserved, so hide every device rather than
        # leaving the inherited value visible -- a job that claimed no slot must not be able
        # to quietly grab a GPU and oversubscribe it.
        env["CUDA_VISIBLE_DEVICES"] = "" if slot.is_cpu_only else str(slot.gpu)
        env.update(
            {
                "JOBQ_QUEUE": name,
                "JOBQ_JOB_KEY": job.key,
                "JOBQ_ATTEMPT": str(attempt),
                "JOBQ_NODE": self.hostname,
                "JOBQ_GPU": "" if slot.is_cpu_only else str(slot.gpu),
            }
        )
        env[store.RUN_ID_ENV] = run_id
        try:
            cwd = resolve_cwd(self._effective_cwd(meta, job), self._cwd_fallback())
        except CwdUnreachable:
            # The directory went away after the claim: left for other machines as well.
            self._hand_back_cwd(name, job, self._effective_cwd(meta, job) or "")
            settled.append(True)
            return None
        try:
            log_file = open(log_path, "w")
        except OSError as exc:
            self._pool_fault(
                name, job, f"the job's log file could not be opened: {exc}", settled,
                slot=slot, start=start, log_path=log_path, attempt=attempt,
            )
            return 1
        with log_file as f:
            f.write(
                f"=== node={self.hostname} pid={pid} gpu={slot.gpu} "
                f"queue={name} job={job.key} start={start} ===\n"
            )
            f.flush()
            if self.end_now.is_set():
                # The pool has been told to end its running jobs. Starting one now would
                # leave a process behind the shutdown, so the job is handed back instead.
                return self._give_back_before_start(name, job, settled)
            # start_new_session=True: bash -c and its children get their own process group
            # so the yield watchdog can kill the whole tree in one killpg.
            try:
                proc = subprocess.Popen(
                    ["bash", "-c", job.cmd],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=cwd,
                    start_new_session=True,
                )
            except (OSError, ValueError) as exc:
                self._pool_fault(
                    name, job, f"the job's process could not be started: {exc}", settled,
                    slot=slot, start=start, log_path=log_path, attempt=attempt,
                )
                return 1
            handle = _RunHandle(
                proc=proc,
                queue=name,
                job=job,
                gpu=slot.gpu,
                log_path=str(log_path),
                start_ts=time.time(),
                admitted_mem_mib=int(slot.mem_mib),
            )
            # Registered before anything else is done with the process, and the shutdown
            # is read immediately afterwards: a pool told to end its jobs while this one
            # was starting ends it here, so no process of this pool outlives it whatever
            # the timing.
            with self._running_lock:
                self._running.setdefault(slot.gpu, {})[(name, job.jobkey)] = handle
            if self.end_now.is_set():
                self._end_one_running(handle, self._end_now_reason())
            # The claim now names the job itself, not only this pool: a pool that is killed
            # leaves the job running in its own session, and the next pool on this machine
            # needs the process group to end it before the job goes back in the queue.
            store.record_job_process(
                self.root, name, job.jobkey, pid=proc.pid, pgid=_pgid_of(proc)
            )
            try:
                rc = self._wait_and_measure(proc, slot)
            finally:
                with self._running_lock:
                    key = (name, job.jobkey)
                    if self._running.get(slot.gpu, {}).get(key) is handle:
                        self._running.get(slot.gpu, {}).pop(key, None)
        # Read the peak before any branch that returns without recording a result (a log
        # read, no store write, so the settled contract is untouched): a yield-killed, OOM or
        # tempfail attempt records nothing, and those are the attempts whose memory matters
        # most.
        peak_mib, peak_alloc_mib = log_tail_peak_mem_mib(log_path)
        measured_mib = slot.measured_mib
        if peak_out is not None and peak_mib is not None:
            peak_out.append(peak_mib)
        peak_s = f" peak_mem={peak_mib}MiB" if peak_mib is not None else ""
        # Every requeue path below rewrites the attempts sidecar, which refuses when the
        # record is present but unreadable (store.AttemptRecordUnreadable). Requeuing
        # without the counters would re-run the job in a tight loop, so the job is made
        # terminal instead — see _sidecar_failure. The raise always happens before the
        # path's remove_claim, so the claim is still ours to settle.
        try:
            if handle.killed:  # yielded the GPU: requeue, don't record a result
                # Bump before removing: the claim dir still shields the job from re-claim,
                # so a throw here leaves ownership with us and the caller's
                # cleanup is safe; after remove_claim nothing fallible may run (see the
                # settled contract).
                store.bump_attempt(self.root, name, job.jobkey)
                store.remove_claim(self.root, name, job.jobkey)
                settled.append(True)
                self.master.log(
                    handle.kill_verb, f"{name} {job.key} {handle.kill_note}{peak_s}"
                )
                return None
            # Exit 75 is the job saying "not my turn", and it is answered before the
            # log is classified: a job that asks to be retried later while its log
            # happens to carry an out-of-memory message from an earlier stage is not an
            # out-of-memory exit, and reading it as one escalates a memory floor the job
            # never needed.
            if rc == TEMPFAIL_RC and self._tempfail_requeue(name, job, settled,
                                                            peak_mib=peak_mib):
                return None  # EX_TEMPFAIL: deferred back to pending, not a failure
            if rc not in (0, TEMPFAIL_RC) and self._oom_requeue(
                name, meta, job, log_path, settled,
                admitted_mib=handle.admitted_mem_mib, peak_mib=peak_mib,
            ):
                return None  # OOM: requeued behind an escalated memory gate, not a failure
        except store.AttemptRecordUnreadable as exc:
            self._sidecar_failure(
                name, job, exc, settled, rc=rc, slot=slot, start=start, log_path=log_path,
                attempt=attempt, peak_mib=peak_mib, peak_alloc_mib=peak_alloc_mib,
            )
            return None  # terminal and visible, but not the job's failure: no streak
        self._note_peak(name, peak_mib, measured_mib, slot.measured_shared)
        try:
            store.record_result(
                self.root,
                name,
                job,
                rc=rc,
                node=self.hostname,
                gpu=slot.gpu,
                start_utc=start,
                end_utc=now_iso(),
                log=str(log_path),
                attempt=attempt,
                peak_mem_mib=peak_mib,
                peak_mem_alloc_mib=peak_alloc_mib,
                measured_mem_mib=measured_mib,
            )
        except store.AttemptRecordUnreadable as exc:
            # The result is written first and the attempt counter last, so the job is
            # already terminal with its true rc — only the counter is stale. Settle the
            # claim and say so; nothing re-runs, so this is not the tight loop above.
            self._note_outcome(name, meta, job, rc)
            store.remove_claim(self.root, name, job.jobkey)
            settled.append(True)
            self.master.log(
                "SIDECAR",
                f"{name} {job.key} rc={rc} recorded, but its attempts sidecar {exc} is "
                "unreadable so the attempt counter was not updated",
            )
            return None
        self._note_outcome(name, meta, job, rc)
        store.remove_claim(self.root, name, job.jobkey)
        settled.append(True)
        return rc

    def _note_outcome(self, name: str, meta: QueueMeta, job: Job, rc: int) -> None:
        """Feed a job's own exit into its queue's run of consecutive failures.

        Reached only for a job process that ran and exited by itself: everything the pool
        requeues (a yielded GPU, exit 75, an out-of-memory exit that goes back at a
        higher ask) returns before this, and every outcome that is this machine's fault
        rather than the job's records its reason on the result instead. A pause is logged
        by whichever pool caused it; the others see it on their next pass.
        """
        try:
            if rc == 0:
                store.note_job_success(self.root, name)
                return
            limit = store.max_consecutive_failures(meta)
            pause_s = self._tunable("failure_pause_s")
            if store.note_job_failure(self.root, name, job.jobkey, limit, pause_s):
                with self._lock:
                    self._paused_seen.add(name)
                rec = store.read_pause(self.root, name) or {}
                keys = ", ".join(str(k) for k in (rec.get("keys") or [])) or "-"
                until = rec.get("until_utc")
                self.master.log(
                    "PAUSE",
                    f"{name} paused: {limit} consecutive job failures ({keys}); "
                    + (f"no pool will claim from it until {until} or: jobq resume {name}"
                       if until else f"no pool will claim from it until: jobq resume {name}"),
                )
        except OSError as exc:
            # The job is already terminal; losing the count must not turn into a crash on
            # the settle path.
            logger.warning("could not update the failure run of queue {}: {}", name, exc)

    def _sidecar_failure(
        self, name: str, job: Job, exc: Exception, settled: list, *, rc: int, slot,
        start: str, log_path: Path, attempt: int,
        peak_mib: int | None, peak_alloc_mib: int | None,
    ) -> None:
        """Make a job terminal because this pool cannot update its attempts sidecar.

        A requeue whose counters cannot be written would hand the job straight back to the
        claim loop with the same unreadable record — a tight re-run loop with no visible
        cause. Instead the job is recorded as failed (rc from the run, or 1) with the
        sidecar named in the ``SIDECAR`` master-log line: the fault is this machine's, not the
        job's. Operator fix: repair or delete ``attempts/<jobkey>.json``, then
        ``jobq requeue``.
        """
        try:
            store.record_result(
                self.root, name, job,
                rc=rc or 1,
                node=self.hostname,
                gpu=slot.gpu,
                start_utc=start,
                end_utc=now_iso(),
                log=_log_field(log_path),
                attempt=attempt,
                peak_mem_mib=peak_mib,
                peak_mem_alloc_mib=peak_alloc_mib,
                not_failure_reason="attempts record unreadable on this machine",
            )
        except store.AttemptRecordUnreadable:
            pass  # expected: the result file itself is written before the counter
        store.remove_claim(self.root, name, job.jobkey)
        settled.append(True)
        self.master.log(
            "SIDECAR",
            f"{name} {job.key} recorded FAILED: attempts sidecar {exc} is present but "
            "unreadable, so this pool cannot update its retry counters (fix or delete "
            "that file, then requeue)",
        )

    def _pool_fault(
        self, name: str, job: Job, detail: str, settled: list, *, slot,
        start: str, log_path: Path | None, attempt: int, verb: str = "START FAILED",
    ) -> None:
        """Record a job as failed because this pool could not run it, and say why.

        The job never ran, so nothing about it is known to be wrong; handing it straight
        back to the claim loop would re-run the same failure as fast as the pool can
        claim. It is recorded with its reason instead, which keeps it out of the queue's
        run of consecutive job failures and leaves it to ``jobq requeue``.
        """
        try:
            store.record_result(
                self.root, name, job,
                rc=1,
                node=self.hostname,
                gpu=slot.gpu,
                start_utc=start,
                end_utc=now_iso(),
                log=_log_field(log_path),
                attempt=attempt,
                not_failure_reason=detail,
            )
        except store.AttemptRecordUnreadable:
            pass  # the result file itself is written before the counter
        except OSError as exc:
            logger.warning("could not record the failed start of {} {}: {}", name, job.key, exc)
        store.remove_claim(self.root, name, job.jobkey)
        settled.append(True)
        self.master.log(verb, f"{name} {job.key} {detail}")

    def yield_kill_gpu(self, gpu: int, spec: yielding.DrainSpec | None = None) -> None:
        """Yield ``gpu`` (called by the watchdog); kill our jobs there, or spare near-done ones.

        Only handles whose process is still alive (``poll() is None``) are considered: a job
        that already exited by itself (but whose handle is not yet popped — ``_run_job`` pops
        only after ``wait()`` returns) is left alone so its real rc is recorded normally
        instead of being discarded and requeued as if we killed it.

        ``spec is None`` (``yield_action == "kill"``) kills every live job on the GPU. With
        a :class:`~yielding.DrainSpec`
        (``drain_if_near_done``) each live job is evaluated: a job estimated ``>= threshold``
        done is spared (runs to completion, records its real rc), one below threshold is
        killed and requeued, and any spared job that has held the GPU past
        ``max_s`` since the yield began is killed with a reason suffix. The GPU stays marked
        yielded (that is the watchdog's marker), so no new work lands here either way.
        """
        to_kill: list[tuple[_RunHandle, str | None]] = []
        to_drain: list[tuple[_RunHandle, int]] = []
        with self._running_lock:
            handles = [
                h for h in self._running.get(gpu, {}).values() if h.proc.poll() is None
            ]
        # Reading every result file of a queue is what the median costs, and it happens
        # outside the lock that the worker threads take to register and drop running jobs:
        # holding that lock through a directory walk would stall every job start and end
        # on this machine. It is computed at most once per queue per yield of this GPU.
        medians = (
            {}
            if spec is None
            else {h.queue: self._drain_median(gpu, spec, h.queue) for h in handles}
        )
        with self._running_lock:
            for h in handles:
                if h.proc.poll() is not None:
                    continue  # finished while the medians were read
                if spec is None:
                    h.killed = True
                    to_kill.append((h, None))
                    continue
                verb, pct, reason = self._drain_decision(h, spec, medians.get(h.queue))
                if verb == "kill":
                    h.killed = True
                    to_kill.append((h, reason))
                elif not h.drained:  # spare, and log DRAIN exactly once
                    h.drained = True
                    to_drain.append((h, pct))
        for h, reason in to_kill:
            suffix = f" ({reason})" if reason else ""
            self.master.log("KILL", f"{h.queue} {h.job.key} gpu={gpu}{suffix}")
            _kill_group(h.proc, grace_s=self._tunable("kill_grace_s"))
        for h, pct in to_drain:
            self.master.log("DRAIN", f"{h.queue} {h.job.key} gpu={gpu} pct={pct} (finishing)")

    def _end_now_reason(self) -> str:
        """The sentence the log gives for ending the running jobs."""
        return "jobq stop --now was requested on this machine"

    def _end_one_running(self, handle: _RunHandle, why: str) -> None:
        """End one running job and mark it to go back in the queue instead of finishing.

        The thread that started the job reads the mark when the process ends: it removes
        the claim, raises the attempt counter and records no result, so the job is pending
        again for the next pool and nothing is counted as a failure. Ending a job whose
        process has already ended does nothing, so calling this twice for one job is the
        same as calling it once.
        """
        with self._running_lock:
            if handle.killed or handle.proc.poll() is not None:
                return
            handle.killed = True
            handle.kill_verb = "INTERRUPT REQUEUE"
            handle.kill_note = f"ended because {why}; back in the queue"
        self.master.log("INTERRUPT", f"{handle.queue} {handle.job.key} ended because {why}")
        _kill_group(handle.proc, grace_s=self._tunable("kill_grace_s"))

    def end_running_jobs(self, why: str) -> list[str]:
        """End every job this pool is running and have it put back in the queue.

        Used when ``jobq stop --now`` asks the pool to leave at once: each job's process
        group gets the termination signal, then the kill signal after ``kill_grace_s``,
        exactly as a yielded GPU does. Returns the job keys ended by this call. A job
        started while this runs is ended by the thread that started it, which reads the
        same request the moment it has registered its process.
        """
        with self._running_lock:
            handles = [
                h
                for jobs in self._running.values()
                for h in jobs.values()
                if not h.killed and h.proc.poll() is None
            ]
        ended: list[str] = []
        for h in handles:
            self._end_one_running(h, why)
            ended.append(h.job.key)
        return ended

    def _give_back_before_start(self, name: str, job: Job, settled: list) -> None:
        """Hand a claimed job back unrun, because the pool is ending its jobs.

        Same outcome as a job that was ended: the attempt counter is raised, the claim is
        removed, no result is recorded and nothing is counted as a failure.
        """
        why = self._end_now_reason()
        try:
            store.bump_attempt(self.root, name, job.jobkey)
        except store.AttemptRecordUnreadable:
            return None  # already reported; leave the claim rather than lose the counters
        store.remove_claim(self.root, name, job.jobkey)
        settled.append(True)
        self.master.log(
            "INTERRUPT REQUEUE",
            f"{name} {job.key} not started because {why}; back in the queue",
        )
        return None

    def _drain_median(self, gpu: int, spec: yielding.DrainSpec, queue: str) -> float | None:
        """The median duration of a queue's finished jobs, read once per yield of a GPU.

        One yield of one GPU is identified by the moment it started, so the numbers are
        read again the next time the GPU is yielded (by then the queue has more results)
        but not on every poll while it stays yielded.
        """
        key = (gpu, spec.yield_started_ts, queue)
        with self._median_lock:
            if key in self._drain_medians:
                return self._drain_medians[key]
        value = yielding.median_duration_s(store.load_results(self.root, queue))
        with self._median_lock:
            # Entries for earlier yields of this GPU can never be asked for again.
            for old in [k for k in self._drain_medians
                        if k[0] == gpu and k[1] != spec.yield_started_ts]:
                del self._drain_medians[old]
            self._drain_medians[key] = value
        return value

    def _drain_decision(
        self, handle: _RunHandle, spec: yielding.DrainSpec, median_s: float | None
    ) -> tuple[str, int | None, str | None]:
        """Decide a live job's fate under ``drain_if_near_done`` -> (verb, pct, reason).

        The safety cap wins first (a spared job may not keep a promised-away GPU forever);
        otherwise progress drives it (>= threshold spares, below kills). See
        :func:`yielding.estimate_progress` for the log-regex -> median -> 0.0 priority chain.
        """
        if spec.now - spec.yield_started_ts >= spec.max_s:
            return "kill", None, "drain cap expired"
        progress = yielding.estimate_progress(
            log_path=handle.log_path or None,
            progress_regex=spec.progress_regex,
            elapsed_s=spec.now - handle.start_ts,
            median_s=median_s,
        )
        pct = int(round(progress * 100))
        if progress >= spec.threshold:
            return "spare", pct, None
        return "kill", None, None

    # ------------------- thread loop -------------------

    def loop(self, tid: int) -> str:
        """Run until stop / scale-down / all complete. Returns why: "stop" | "scale" | "complete".

        The reason lets the supervisor respawn a thread that left for a stop file that has
        since gone: a short-lived stop file would otherwise silently retire every idle
        thread, since they exit through the acquire wait rather than the loop head.

        The acquire-wait exit is always "stop": ``_acquire_gpu`` returns None only because
        a stop file was seen or the pool was asked to drain. Re-checking the file here and
        answering "complete" when it had already been cleared would retire the thread for
        the pool's lifetime with no log line, because the supervisor only respawns
        "stop"/"scale". The all-complete exit is
        therefore reported by ``_loop_once`` under its own code ("complete", never "exit").
        """
        while True:
            if self.draining.is_set():
                self._log_exit_reason(
                    "drain",
                    "the pool was asked to drain; the workers are exiting once their "
                    "jobs are done",
                )
                return "stop"
            if store.stop_path(self.root, self.hostname).exists():
                self._log_exit_reason("stop", "stop file present; the workers are exiting")
                return "stop"
            if tid >= self.worker_target:
                self.master.log("WAIT", f"worker {tid} above scaled-down target; exiting")
                return "scale"
            try:
                rc = self._loop_once()
                if rc == "exit":
                    self._log_exit_reason(
                        "stop", "stop file seen while waiting for a GPU; the workers are "
                        "exiting"
                    )
                    return "stop"
                if rc == "complete":
                    self._log_exit_reason(
                        "complete", "every queue this pool works on is complete; the "
                        "workers are exiting"
                    )
                    return "complete"
            except Exception:  # noqa: BLE001 — a runner thread must never die.
                # An error raised inside claim/acquire (a malformed policy file, say) would
                # otherwise kill threads one by one until the pool quietly ran on a handful.
                # Log, back off, keep looping.
                logger.exception("worker {} loop iteration crashed; continuing", tid)
                time.sleep(self.poll_s)

    def _log_exit_reason(self, reason: str, message: str) -> None:
        """Log why the workers are leaving, once for the pool rather than once each.

        Every worker thread reaches the same stop file or the same finished queues at
        about the same moment, so a line each says the same thing as many times as the
        pool has threads. A thread that leaves for a reason of its own — scaled down, or
        restarted after ending abnormally — still gets its own line elsewhere.
        """
        with self._lock:
            if reason in self._exit_logged:
                return
            self._exit_logged.add(reason)
        self.master.log("WAIT", message)

    def _claim_note(self, name: str, jobkey: str) -> None:
        with self._owned_lock:
            k = (name, jobkey)
            self._owned_claims[k] = self._owned_claims.get(k, 0) + 1

    def _claim_drop(self, name: str, jobkey: str) -> None:
        with self._owned_lock:
            k = (name, jobkey)
            n = self._owned_claims.get(k, 0) - 1
            if n > 0:
                self._owned_claims[k] = n
            else:
                self._owned_claims.pop(k, None)

    def reap_orphan_claims(self, grace_s: float = 300.0) -> list[str]:
        """Drop on-disk claims stamped (this machine, this pid) that no live thread owns.

        Self-healing for claims stranded by a crash between ``claim_next`` and cleanup:
        jobq counts them RUNNING forever and ``steal_stale`` cannot help because the pool's
        pid is alive. Anything under our pid absent from the
        ``_owned_claims`` registry for longer than ``grace_s`` (covers the sub-second
        claim->note window with a huge margin) is debris. Returns "queue/jobkey" reaped.
        """
        reaped: list[str] = []
        now = time.time()
        pid = os.getpid()
        try:
            qdirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        except OSError:
            return reaped
        for qdir in qdirs:
            cdir = qdir / "claims"
            if not cdir.is_dir():
                continue
            name = qdir.name
            try:
                entries = list(cdir.iterdir())
            except OSError:
                continue
            for entry in entries:
                if not entry.is_dir():
                    continue
                jk = entry.name
                with self._owned_lock:
                    if (name, jk) in self._owned_claims:
                        continue
                owner = store.read_owner(self.root, name, jk)
                if (
                    not owner
                    or owner.get("node") != self.hostname
                    or owner.get("pid") != pid
                ):
                    continue
                try:
                    age = now - entry.stat().st_mtime
                except OSError:
                    continue
                if age < grace_s:
                    continue
                store.remove_claim(self.root, name, jk)
                self.master.log(
                    "REAP", f"{name} {jk} orphan claim (our pid, no owning thread; {int(age)}s old)"
                )
                reaped.append(f"{name}/{jk}")
        return reaped

    def recover_claims(self) -> list[str]:
        """Hand back every claim of this machine whose pool is gone, wherever it is.

        A pass of its own, asked at pool start and on every supervisor tick, and
        deliberately not tied to whether the queue can be claimed from: a claim left
        behind by a pool that was killed counts as work in progress until somebody
        recovers it, and the queue it belongs to may be paused, parked, waiting on a
        dependency, tied elsewhere, outside this pool's ``--queues``, or have settings
        this account cannot read. None of that is a reason to leave a job that is not
        running counted as running, and the claim directory is readable in every one of
        those cases.

        The pass stops between claims once the pool has been asked to drain, so a drain
        is honoured without waiting for it to walk the whole folder.
        """
        recovered: list[str] = []
        for name in store.queues_with_claims(self.root):
            if self.draining.is_set():
                break
            recovered += self._recover_queue_claims(name)
        return recovered

    def _recover_queue_claims(self, name: str) -> list[str]:
        """One queue's recovery pass, reporting whatever it did and did not do."""

        def _event(kind: str, jobkey: str, detail) -> None:
            if kind == "ended":
                self.master.log(
                    "ENDED SURVIVOR",
                    f"{name} {jobkey} process group {detail} outlived the pool that "
                    "started it and was ended before the job went back in the queue",
                )
            elif kind == store.KEPT_SURVIVOR:
                self.master.log(
                    "CLAIM KEPT",
                    f"{name} {jobkey} the pool that claimed it is gone, but its process "
                    f"group {detail} is still running and could not be ended; the claim "
                    "stays and the job is not started again here",
                )
            elif kind == store.KEPT_OWNER_UNREADABLE:
                self.master.log(
                    "CLAIM KEPT",
                    f"{name} {jobkey} its owner record {detail} is there but cannot be "
                    "read, so there is no telling whose claim it is; fix or remove that "
                    f"file, or run: jobq release {name} {jobkey} --force",
                )
            elif kind == store.KEPT_ATTEMPTS_UNREADABLE:
                self.master.log(
                    "CLAIM KEPT",
                    f"{name} {jobkey} its run was ended, but its attempts record "
                    f"{detail} is there and cannot be read, so the job is not handed "
                    "back with counters nobody can see; fix or delete that file",
                )

        try:
            stolen = store.steal_stale(
                self.root, name, self.hostname,
                grace_s=self._tunable("orphan_claim_grace_s"),
                kill_grace_s=self._tunable("kill_grace_s"),
                on_event=_event,
                should_stop=self.draining.is_set,
            )
        except (OSError, store.InvalidQueueName, store.QueuePathUnsafe) as exc:
            self.master.log(
                "WAIT", f"queue {name}: its claims could not be looked at ({exc})"
            )
            return []
        for jk in stolen:
            self.master.log("STEAL", f"{name} {jk}")
        return [f"{name}/{jk}" for jk in stolen]

    def _loop_once(self) -> str | None:
        """One claim -> acquire -> run iteration; None when it ran (or idled) normally.

        The two stopping outcomes are distinct codes, because the caller must respawn one
        and not the other: "exit" = a stop file was seen inside the acquire wait (transient,
        respawnable), "complete" = every machine-visible queue is done (terminal).

        Everything after a successful ``claim_next`` is owned by ``_admit_and_run``, whose
        note/drop bracket guarantees the registry entry (and any acquired slot) is released
        no matter where an exception escapes; the stranded on-disk claim is then unregistered,
        which is exactly the state the supervisor's janitor collects after its grace.
        """
        ready = self.ready_queues()
        # Recovery of what a dead pool left behind is its own pass (see
        # :meth:`recover_claims`), run at pool start and on every supervisor tick. This
        # one is only an optimisation: a queue this thread is about to claim from is
        # worth looking at now rather than at the next tick.
        for name, _ in ready:
            self._recover_queue_claims(name)
        for name, meta in ready:
            # Fall-through: a claimed job whose stage cap (cap_per_gpu / cap_group) is
            # saturated on every GPU would otherwise park this thread in _acquire_gpu while
            # holding the claim, starving lower-priority queues behind a capped one. Probe
            # the group ceiling first and skip the queue this round.
            cap, group = self._effective_cap(meta)
            if cap is not None and not self.gpu.group_has_room(group, cap):
                continue
            # Same fall-through for the CPU lane: a queue-level ``slots: 0`` job would
            # otherwise park a thread on the cpu_cap budget (probe, then one non-blocking
            # acquire below).
            cpu_lane = int(meta.defaults.get("slots", 1)) == 0
            if cpu_lane and not self.gpu.cpu_has_room():
                continue
            job = store.claim_next(
                self.root, name, self.hostname, os.getpid(),
                accept=self._claimable_job(meta),
            )
            if job is None:
                continue
            outcome = self._admit_and_run(name, meta, job, cap, group)
            if outcome == "next":  # gave the job back: keep walking the ready list
                continue
            return outcome  # None (ran) or "exit"
        if self.all_complete():
            return "complete"
        time.sleep(self.poll_s)
        return None

    def _admit_and_run(
        self, name: str, meta: QueueMeta, job: Job, cap, group
    ) -> str | None:
        """Own a just-made claim from registry note to terminal state.

        Returns "next" (job handed back — caller keeps walking the ready list), "exit"
        (stop requested), or None (job ran). This is the one note/drop bracket per claim:
        the finally always drops this iteration's registry note and releases any slot
        (idempotent), so no escape path can leave the registry shielding a dead claim from
        reap_orphan_claims or a dead slot's fds from reap_leaked_locks. It deliberately
        does not remove the disk claim in the finally: on the explicit hand-back paths the
        claim is still ours to remove, but on an unforeseen throw a peer may have already
        re-claimed it, so removal is left to the inline paths and (for true debris) the
        janitor's post-grace pass.
        """
        self._claim_note(name, job.jobkey)
        slot = None
        try:
            if job.cwd is not None:
                try:
                    resolve_cwd(job.cwd, self._cwd_fallback())
                except CwdUnreachable:
                    self._hand_back_cwd(name, job, job.cwd)
                    return "next"
            mem_mib = self._effective_mem(name, meta, job)
            slots = self._effective_slots(meta, job)
            # A job that declares ``slots: 0`` takes the CPU lane whatever its queue's
            # default is: what it needs is a cpu_cap slot, and parking it on GPU
            # capacity would hold a worker thread on a budget it never draws from.
            cpu_only = slots == 0
            # A job that uses no GPU follows the machine part of its queue's tie and
            # ignores the GPU part, since it takes no GPU.
            gpus = None if cpu_only else self._usable_gpus(meta)
            if cap is not None or cpu_only:
                # The probe above races (many threads pass it before any acquires), so a
                # capped / CPU-lane claim first gets one non-blocking acquire.
                try:
                    slot = self.gpu.acquire(
                        mem_mib,
                        slots=slots,
                        cap_per_gpu=cap,
                        cap_group=group,
                        **({} if gpus is None else {"gpus": gpus}),
                    )
                except Exception:
                    store.remove_claim(self.root, name, job.jobkey)
                    raise
                if slot is None and cpu_only:
                    # CPU lane: no slot -> give the job back and keep walking the ready
                    # list; a cpu_cap waiter never parks a thread.
                    store.remove_claim(self.root, name, job.jobkey)
                    return "next"
            self.master.log("CLAIM", f"{name} {job.key}")
            if slot is None:
                # Park on the global slots with priority-ordered admission. A capped job
                # parks too: its stage group has room (the probe said so and the wait
                # re-checks every poll), so what it waits on is a global slot, exactly like
                # an uncapped job. The fall-through above is about a saturated group, and
                # that case still hands the claim back (_YIELD).
                try:
                    slot = self._acquire_gpu(
                        mem_mib, slots, cap, group, job.key,
                        priority=meta.priority, gpus=gpus,
                        name=name, tie=self._tie_gpus(meta),
                    )
                except Exception:
                    store.remove_claim(self.root, name, job.jobkey)  # never strand a claim
                    raise
                # Group saturated while parked, or the queue was parked, paused or tied
                # away while this thread waited: either way the job goes back.
                if slot is _YIELD:
                    slot = None
                    store.remove_claim(self.root, name, job.jobkey)
                    return "next"
                if slot is None:  # stop requested while waiting -> let a peer take it
                    store.remove_claim(self.root, name, job.jobkey)
                    return "exit"
            self._run_claimed(name, meta, job, slot)
            return None
        finally:
            if slot is not None:
                slot.release()  # idempotent; backstop for paths that already released
            self._claim_drop(name, job.jobkey)

    def _run_claimed(self, name, meta, job, slot) -> None:
        # everything sits inside the try (even the START log / attempt read): an exception
        # escaping before the finally would leak the slot until the caller's bracket, and
        # any throw here must land in the except below, not propagate past the cleanup.
        settled: list = []  # marked by _run_job the moment it removes the claim
        # Read before the try, so the result written for a fault of this pool's own
        # carries the attempt the job is really on rather than a stand-in.
        attempt = store.read_attempt(self.root, name, job.jobkey)
        try:
            self.master.log("START", f"{name} {job.key} gpu={slot.gpu}")
            peak_out: list = []  # filled by _run_job when the job reported a peak
            rc = self._run_job(name, meta, job, slot, attempt, settled, peak_out=peak_out)
            if rc is not None:  # None: yield-killed or requeued, already logged
                # Field appended to the END line, so the rc= prefix form stays greppable.
                peak_s = f" peak_mem={peak_out[0]}MiB" if peak_out else ""
                self.master.log("END", f"{name} {job.key} rc={rc}{peak_s}")
        except Exception as exc:  # noqa: BLE001 — log-and-continue; never abort the pool
            logger.exception("job {} in queue {} crashed the runner", job.key, name)
            if not settled:
                # Only while the claim is still ours. Once _run_job removed it, a peer
                # thread may have re-claimed this jobkey (same machine, same pid —
                # owner.json cannot tell the generations apart), and removing again would
                # delete the peer's live claim and let a third thread run the job
                # concurrently. The job is recorded with its reason rather than handed
                # back, so a fault in the pool cannot re-run it in a loop, and it does not
                # count as one of the queue's consecutive job failures.
                self._pool_fault(
                    name, job,
                    f"the pool failed while running this job: {type(exc).__name__}: {exc}",
                    settled, slot=slot, start=now_iso(),
                    log_path=None, attempt=attempt,
                    verb="INTERNAL ERROR",
                )
        finally:
            # The registry note is dropped by the caller's bracket (_admit_and_run), not
            # here — dropping in both places would double-decrement the counted registry.
            slot.release()
        if store.queue_complete(self.root, name):
            self._maybe_log_complete(name)


def run_pool(
    root: Path,
    *,
    workers: int | None = None,
    queues: list[str] | None = None,
    gpu: GpuInterface | None = None,
    hostname: str | None = None,
    poll_s: float = gpu_defaults.DEFAULT_POLL_S,
    gpu_wait_s: float = gpu_defaults.DEFAULT_GPU_WAIT_S,
    lock_prefix: str | None = None,  # resolved via gpu.resolve_lock_prefix when absent
    supervise_s: float = gpu_defaults.DEFAULT_SUPERVISE_S,
    monitor_samples: bool = True,
) -> None:
    """Drain this machine's share of the queue folder until it is complete or stopped.

    ``jobq work`` checks that this machine has a policy file before it calls this, and
    refuses to start without one: a pool with no policy holds claims it can never admit.

    Args:
        root: The queue folder.
        workers: Number of worker threads (the per-GPU slot cap is global). ``None``
            chooses one from the cores this process may use and the cores a job is
            assumed to need (see :func:`~jobq.gpu.default_worker_count`).
        queues: Restrict the pool to these queue names.
        gpu: A :class:`~jobq.gpu.GpuInterface` to use instead of building a manager.
        hostname: This machine's name (default: the system hostname).
        poll_s: Seconds an idle thread waits before re-deriving the ready list; a policy
            naming ``poll_s`` overrides it (same for ``gpu_wait_s`` and ``supervise_s``).
        gpu_wait_s: Seconds a parked thread waits between acquire attempts.
        lock_prefix: Optional override for GPU slot flock paths (default: derived from the
            queue root — see ``gpu.resolve_lock_prefix``).
        supervise_s: Supervisor tick: respawn dead threads, re-read the scale file, run the
            janitors.
        monitor_samples: Append utilisation samples to the queue folder while the pool
            runs (see :mod:`jobq.monitor`). False turns sampling off for this pool.
    """
    root = Path(root)
    hostname = hostname or store.this_host()
    # The pool lock comes before anything else this function reads or writes: it is what
    # makes this the only pool on this machine for this queue folder, and it is held until
    # the process ends.
    lock_fd = _acquire_pool_lock(root, hostname)
    try:
        _run_pool_locked(
            root,
            hostname,
            workers=workers,
            queues=queues,
            gpu=gpu,
            poll_s=poll_s,
            gpu_wait_s=gpu_wait_s,
            lock_prefix=lock_prefix,
            supervise_s=supervise_s,
            monitor_samples=monitor_samples,
        )
    finally:
        os.close(lock_fd)  # the lock file itself stays: a pool may be waiting on it


def _run_pool_locked(
    root: Path,
    hostname: str,
    *,
    workers: int | None,
    queues: list[str] | None,
    gpu: GpuInterface | None,
    poll_s: float,
    gpu_wait_s: float,
    lock_prefix: str | None,
    supervise_s: float,
    monitor_samples: bool = True,
) -> None:
    """Run the pool with this machine's pool lock already held (see :func:`run_pool`)."""
    # Opt-in world-writable queue state, for an account whose uid differs between the machines
    # sharing this queue folder. Read before anything is written, and applied both to this
    # process's own writes and to the jobs it launches.
    try:
        _policy = load_policy(root, hostname)
    except FileNotFoundError:
        _policy = None
    if _policy is not None and _policy.shared_perms:
        io.set_shared_perms(True)
        os.umask(0)
    if _policy is not None:  # a policy that names a timing key wins over the call's value
        if "poll_s" in _policy.declared:
            poll_s = _policy.poll_s
        if "gpu_wait_s" in _policy.declared:
            gpu_wait_s = _policy.gpu_wait_s
        if "supervise_s" in _policy.declared:
            supervise_s = _policy.supervise_s
    # The handlers come before anything is written or started, so a termination signal
    # arriving during start-up ends this pool as cleanly as one arriving later: the pid
    # file, the temp sweep, the recovery pass and the MPS daemon all happen under them.
    shutdown = _Shutdown()
    with _signal_handlers(shutdown):
        _write_pid_file(root, hostname)
        try:
            _run_pool_started(
                root,
                hostname,
                shutdown=shutdown,
                policy=_policy,
                workers=workers,
                queues=queues,
                gpu=gpu,
                poll_s=poll_s,
                gpu_wait_s=gpu_wait_s,
                lock_prefix=lock_prefix,
                supervise_s=supervise_s,
                monitor_samples=monitor_samples,
            )
        finally:
            # Whatever happened — an unusable MPS daemon, a queue folder that cannot be
            # read while the exit reason is worked out — this machine must not be left
            # looking as though a pool were still running on it.
            _release_pid_file(root, hostname)
            store.worker_info_path(root, hostname).unlink(missing_ok=True)
            # The request to end the jobs was aimed at this pool and this pool has
            # answered it; the stop file stays until it is cleared, as the lasting
            # request it is.
            store.stop_now_path(root, hostname).unlink(missing_ok=True)


def _run_pool_started(
    root: Path,
    hostname: str,
    *,
    shutdown: _Shutdown,
    policy,
    workers: int | None,
    queues: list[str] | None,
    gpu: GpuInterface | None,
    poll_s: float,
    gpu_wait_s: float,
    lock_prefix: str | None,
    supervise_s: float,
    monitor_samples: bool,
) -> None:
    """The pool's own life, with the lock held, the handlers installed and a pid file."""
    _policy = policy
    master = MasterLog(root / "logs" / f"{utc_stamp()}_worker.{hostname}.log")
    for path in store.remove_stale_temp_files(root):
        master.log("REAP", f"leftover temporary file removed: {path}")
    gpu = gpu or GpuManager(root, hostname, lock_prefix=lock_prefix)
    pool = _Pool(
        root,
        hostname,
        gpu,
        master,
        queues_filter=set(queues) if queues else None,
        poll_s=poll_s,
        gpu_wait_s=gpu_wait_s,
        shutdown=shutdown,
    )
    # Before the stop-file check and before MPS: a pool that starts only to find it has
    # been told to stop is still the one thing on this machine that can hand back the
    # claims of the pool before it, and those jobs count as running until it does.
    recovered = pool.recover_claims()
    if store.stop_path(root, hostname).exists():
        master.log(
            "POOL EXIT",
            "a stop request was already in place when this pool started, so no job was "
            f"claimed; {len(recovered)} claim(s) left by a pool that has ended were "
            "handed back first",
        )
        return
    if workers is None:
        plan = gpu_defaults.default_worker_count(_policy)
        machine = (
            f"{plan.machine_gpus} gpu(s) on the machine"
            if plan.machine_gpus is not None
            else f"the machine's gpu list could not be read, so its {plan.policy_gpus} "
            "policy gpu(s) stand in"
        )
        master.log(
            "WORKERS",
            f"{plan.workers} worker thread(s): {plan.cores} usable core(s), "
            f"{plan.reserve} reserved, {machine}, {plan.policy_gpus} gpu(s) in the "
            f"policy, {plan.cores_per_job} core(s) per job, "
            f"{plan.workers_per_gpu} worker(s) per gpu",
        )
        workers = plan.workers
    if workers <= 0:
        master.log(
            "POOL EXIT",
            "this machine's policy allows no work: it selects no GPU and its cpu_cap is "
            "0, so there is nothing a worker thread here could ever be admitted",
        )
        return
    # Sidecar beside the pid file, so `jobq status` can name the log this pool writes.
    store.write_worker_info(root, hostname, pid=os.getpid(), log=str(master.path))
    # CUDA MPS: bring the per-user daemon up once before workers spawn (a shared,
    # user-level resource) and only inject its env into jobs if it actually came up. With
    # the default ``"auto"`` a machine that cannot use MPS runs without it and says why;
    # with ``true`` the pool refuses to start instead, since the person asked for it. The
    # daemon is deliberately not torn down at pool exit, since other pools may share it.
    active_mps_env: dict[str, str] = {}
    if _policy is not None and _policy.mps is not False:
        usable, reason = ensure_mps_daemon(_policy)
        if usable:
            active_mps_env = mps_env(_policy)
            logger.info("MPS enabled")
        elif _policy.mps is True:
            raise MpsUnavailable(
                f"the policy asks for MPS with mps: true, and {reason}; set mps to "
                f'"{MPS_AUTO}" to run without it when the machine cannot'
            )
        else:
            logger.info("running without MPS: {}", reason)
    pool.mps_env = active_mps_env
    # Opt-in GPU yielding: always start the watchdog when this machine has a policy file, and
    # let poll_once no-op while yield_to_foreign is false. This makes the switch hot-enablable:
    # flip the flag in the policy JSON on a running pool and yielding begins within one
    # yield_poll_s — no restart (a restart here would kill and requeue every running job,
    # exactly the pain this feature manages). A missing policy means no thread; if one
    # exists, poll_once returns harmlessly on FileNotFoundError.
    stop_event = threading.Event()
    watch_thread: threading.Thread | None = None
    try:
        load_policy(root, hostname)
        have_policy = True
    except FileNotFoundError:
        have_policy = False
    if have_policy:
        watchdog = yielding.YieldWatchdog(
            root,
            hostname,
            load_policy=lambda: load_policy(root, hostname),
            master=master,
            on_yield=pool.yield_kill_gpu,
        )
        watch_thread = threading.Thread(
            target=watchdog.run, args=(stop_event,), name="jobq-yield-watchdog", daemon=True
        )
        watch_thread.start()
    monitor_thread = _start_sampler(
        root, hostname, gpu, pool, stop_event, enabled=monitor_samples
    )
    with _shutdown_watch(pool, master, root, hostname):
        supervisor_error = _supervise_workers(
            pool, master, root, hostname, workers, supervise_s
        )
    stop_event.set()  # let the daemon watchdog exit promptly (it also dies at process end)
    if watch_thread is not None:
        watch_thread.join(timeout=5.0)
    if monitor_thread is not None:
        monitor_thread.join(timeout=5.0)
    # "ALL QUEUES COMPLETE" says the whole queue folder drained, so it is logged only when
    # every queue in it is complete — not merely the ones this pool was allowed to claim
    # from. A pool that left on a stop file, on a scale-down, or with a supervisor that
    # ended abnormally still has claimable work and says so under its own verb. A
    # supervisor that did not end normally cannot vouch for its workers at all, so its
    # exit never reads as a drain even if the queues happen to look complete.
    if pool.end_now.is_set():
        master.log(
            "POOL EXIT",
            "jobq stop --now was requested: the running jobs were ended and put back in "
            "the queue",
        )
    elif pool.draining.is_set():
        master.log(
            "POOL EXIT",
            f"asked to drain by {pool.interrupt_signal or 'a signal'}: the running jobs "
            "finished and no further job was claimed",
        )
    elif supervisor_error is None and _queue_state_problem(pool) is not None:
        # Reading the queue folder is how the exit reason is decided, and a folder that
        # cannot be read must still produce an exit line naming the problem rather than
        # an exception out of the pool's last few statements.
        master.log(
            "POOL EXIT",
            "the workers exited and the queue folder could not be read to say whether "
            f"anything is left: {_queue_state_problem(pool)}",
        )
    elif supervisor_error is None and pool.all_complete() and pool.root_complete():
        master.log("ALL QUEUES COMPLETE")
    elif supervisor_error is None and pool.all_complete():
        aside = pool.set_aside_reasons()
        rest = (
            f"{len(aside)} queue is left where it is: "
            if len(aside) == 1
            else f"{len(aside)} queues are left where they are: "
        )
        master.log(
            "POOL EXIT",
            "the queues this pool works on are complete; "
            + (
                rest + ", ".join(f"{n} ({aside[n]})" for n in sorted(aside))
                if aside
                else "other queues in the queue folder are not"
            ),
        )
    elif supervisor_error is not None:
        master.log("POOL EXIT", f"supervisor ended abnormally: {supervisor_error}")
    elif store.stop_path(root, hostname).exists():
        master.log(
            "POOL EXIT",
            "a stop request was made: the running jobs finished and the queues are still "
            "incomplete",
        )
    else:
        master.log("POOL EXIT", "workers exited with queues still incomplete")


def _queue_state_problem(pool: _Pool) -> str | None:
    """Why the queue folder cannot say whether this pool has anything left, or ``None``.

    The three questions the exit line is decided by all read the queue folder, and each
    of them can refuse: a queue whose settings became unreadable, or one of its own
    directories turned into a link, while the pool ran.
    """
    try:
        pool.all_complete()
        pool.root_complete()
        pool.set_aside_reasons()
    except (store.QueueMetaUnreadable, store.QueuePathUnsafe, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _occupancy_of(gpu: GpuInterface):
    """How to ask this GPU manager what it holds; an empty answer when it cannot say."""
    return getattr(gpu, "occupancy", None) or (lambda: {})


def pool_slot_state(
    root: Path, hostname: str, gpu: GpuInterface, pool: _Pool
) -> monitor.SlotState:
    """The slots row of a running pool: what it holds, what it may hold, who is waiting.

    The capacity is the cap in force times the policy GPUs that are not yielded, which is
    what the pool could be running on right now; the GPUs yielded to another user are
    counted separately rather than silently dropped.
    """
    try:
        policy = load_policy(root, hostname)
    except (FileNotFoundError, gpu_defaults.PolicyError):
        policy = None
    occupancy = _occupancy_of(gpu)()
    yielded = yielding.read_yielded(root, hostname) or {}
    gpus = list(policy.gpus) if policy is not None else sorted(occupancy)
    cap = policy.slot_units if policy is not None else 0
    yielded_here = sum(1 for g in gpus if g in yielded)
    return monitor.SlotState(
        used=sum(int(n) for g, n in occupancy.items() if g in gpus),
        slots=max(0, len(gpus) - yielded_here) * cap,
        wait=pool.waiting_count(),
        gpus=len(gpus),
        yielded=yielded_here,
        cap_per_gpu=cap,
        live=True,
    )


def _start_sampler(
    root: Path,
    hostname: str,
    gpu: GpuInterface,
    pool: _Pool,
    stop_event: threading.Event,
    *,
    enabled: bool,
) -> threading.Thread | None:
    """Start the utilisation sampler on its own thread, or return ``None`` when it is off.

    Off means either this pool was asked not to sample or the policy sets the interval to
    zero. The thread is a daemon and takes nothing the admission path holds, so a reading
    that hangs delays no job.
    """
    if not enabled:
        return None
    try:
        policy = load_policy(root, hostname)
    except (FileNotFoundError, gpu_defaults.PolicyError):
        policy = None
    config = monitor.config_from_policy(policy)
    if config.interval_s <= 0:
        return None
    sampler = monitor.Sampler(
        root,
        hostname,
        config=config,
        gpus=tuple(policy.gpus) if policy is not None else (),
        occupancy=_occupancy_of(gpu),
        slot_state=lambda: pool_slot_state(root, hostname, gpu, pool),
    )
    thread = threading.Thread(
        target=sampler.run, args=(stop_event,), name="jobq-monitor", daemon=True
    )
    thread.start()
    return thread


def drain_sentence(running: int, signal_name: str | None, *, first: bool) -> str:
    """What a pool says when a signal asks it to drain, for one job and for several.

    ``first`` is the line the signal itself produces; the other is the reminder a repeat
    of the signal prints.
    """
    one = running == 1
    jobs = f"{running} running job" if one else f"{running} running jobs"
    if first:
        return (
            f"{signal_name or 'a signal'} received: no further job is claimed and the "
            f"{jobs} {'is' if one else 'are'} allowed to finish. To end "
            f"{'it' if one else 'them'} at once, run: jobq stop --now"
        )
    return (
        f"still letting {jobs} finish; jobq stop --now ends "
        f"{'it' if one else 'them'} at once"
    )


def _running_job_count(pool: _Pool) -> int:
    """How many job processes of this pool are running right now."""
    with pool._running_lock:
        return sum(
            1 for jobs in pool._running.values() for h in jobs.values() if h.proc.poll() is None
        )


class _Shutdown:
    """The two requests to leave, shared by the signal handlers and the pool.

    It exists on its own because the handlers are installed before the pool is built:
    a termination signal during start-up — while the pid file is written, while claims
    left by the pool before this one are recovered, while the MPS daemon comes up — has
    to reach the same state the pool later reads, or it would be lost.

    The counters the handler touches are guarded by a lock, because a signal can arrive
    while the watching thread is reading them.
    """

    def __init__(self) -> None:
        self.draining = threading.Event()
        self.end_now = threading.Event()
        self._lock = threading.Lock()
        self._seen = 0
        self._announced = 0
        self.signal_name: str | None = None

    def note_signal(self, signum: int) -> None:
        """Record one interrupt or termination signal and ask the pool to drain."""
        with self._lock:
            self._seen += 1
            self.signal_name = signal.Signals(signum).name
        self.draining.set()

    def take_announcement(self) -> bool | None:
        """``True`` for the first signal to announce, ``False`` for a repeat, else ``None``."""
        with self._lock:
            if self._seen == self._announced:
                return None
            first = self._announced == 0
            self._announced = self._seen
            return first


@contextlib.contextmanager
def _signal_handlers(shutdown: _Shutdown):
    """Handle the interrupt and termination signals for the length of the block.

    Installed before the pool writes or starts anything, and the previous handlers are
    put back on the way out. Signals are only handled in the main thread, which is where
    a pool runs; anywhere else this does nothing.
    """
    previous: dict[int, object] = {}

    def _handler(signum, _frame) -> None:
        shutdown.note_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _handler)
        except ValueError:
            break  # not the main thread: nothing here can handle a signal
    try:
        yield
    finally:
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(sig, handler)


@contextlib.contextmanager
def _shutdown_watch(pool: _Pool, master: MasterLog, root: Path, hostname: str):
    """Watch for the two ways this pool is asked to leave, and act on them.

    An interrupt or a termination signal (Ctrl-C, ``kill``) is a request to drain: the
    pool claims no further job, the jobs it is running finish and record their results,
    and the pool exits. The handler itself only counts the signal; the thread below says
    on the first one how many jobs are finishing and that ``jobq stop --now`` ends them at
    once, and answers a further signal with a shorter reminder of the same.

    ``jobq stop --now`` puts its request in the queue folder, so it works from any
    terminal on this machine. The same thread reads it and ends the running jobs, and it
    keeps ending them until every worker thread has returned: a job whose process starts
    while this is happening is ended by the thread that started it, which reads the same
    request the moment it has registered the process.

    The signal handlers themselves are installed before this, at the very start of the
    pool (see :func:`_signal_handlers`); this thread is what acts on what they recorded.
    """
    finished = threading.Event()

    def _announce() -> None:
        first = pool._shutdown.take_announcement()
        if first is None:
            return
        line = drain_sentence(
            _running_job_count(pool), pool.interrupt_signal, first=first
        )
        print(line, flush=True)
        master.log("DRAIN REQUEST", line)
        store.mark_pool_draining(root, hostname)

    def _watch() -> None:
        # Once the jobs are being ended there is nothing left to react to quickly: the
        # only work is catching a job that started in the meantime, which its own thread
        # ends as well. Polling ten times a second for the rest of the shutdown would
        # spin a core for no gain.
        poll_s = 0.1
        while not finished.wait(poll_s):
            _announce()
            if not pool.end_now.is_set() and store.stop_now_path(root, hostname).exists():
                pool.end_now.set()
                pool.draining.set()
                master.log(
                    "END NOW",
                    "jobq stop --now was requested: ending the running jobs and putting "
                    "them back in the queue",
                )
            if pool.end_now.is_set():
                pool.end_running_jobs(pool._end_now_reason())
                poll_s = 1.0

    watcher = threading.Thread(target=_watch, name="jobq-shutdown", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()  # release the watching thread when the pool ends
        # Waited for rather than given a deadline: it is the thread that ends the
        # running jobs, and the exit line below says what became of them.
        watcher.join()


def workers_path(root: Path, hostname: str) -> Path:
    """Optional hot-scale file: an integer thread target for this machine's running pool."""
    return Path(root) / f"workers.{hostname}"


def _read_workers_target(root: Path, hostname: str, default: int) -> int:
    try:
        return max(1, int(workers_path(root, hostname).read_text().strip()))
    except (FileNotFoundError, ValueError, OSError):
        return default


def _supervise_workers(
    pool: _Pool, master: MasterLog, root: Path, hostname: str, workers: int, supervise_s: float
) -> str | None:
    """Run worker threads under a supervisor: respawn dead ones, hot-scale from a file.

    An exception escaping ``loop()`` kills that thread, and without a supervisor the pool
    would run short-handed with no restart path short of a full drain. Every ``supervise_s``
    the supervisor (a) respawns any thread that ended without returning normally (crash) and
    logs ``RESPAWN``, and (b) re-reads
    ``workers.<host>`` — write an integer there to grow or shrink the live pool without a
    restart (shrink = threads above the target exit after their current job). Threads that
    return normally (stop file / all queues complete) are not respawned; the supervisor
    returns once every live thread has done so. Cost: ``is_alive`` on ≤N Thread objects +
    one stat() per tick.

    Returns ``None`` after a normal end, or a description of the exception that ended it.
    An error inside the tick itself (an unreadable scale file, a janitor raising something
    unforeseen) is logged and the loop continues: the supervisor is the one thing holding
    the workers up, so it must outlive its own mistakes. It only gives up when starting or
    tracking the threads fails, which is what the caller reports.
    """
    finished: set[int] = set()  # tids that returned normally
    finished_reason: dict[int, str] = {}  # tid -> "stop" | "scale" | "complete"
    threads: dict[int, threading.Thread] = {}
    # Wakes the supervisor the moment a worker returns, instead of waiting out the rest of
    # a supervise_s tick, so the pool exits promptly after a drain or a stop.
    wake = threading.Event()

    def _target(tid: int) -> None:
        reason = pool.loop(tid) or "complete"
        finished_reason[tid] = reason
        finished.add(tid)
        wake.set()

    def _spawn(tid: int) -> None:
        t = threading.Thread(target=_target, args=(tid,), name=f"jobq-worker-{tid}")
        threads[tid] = t
        t.start()

    target = max(1, workers)
    pool.worker_target = target
    for i in range(target):
        _spawn(i)
    while True:
        try:
            if _supervise_tick(pool, master, root, hostname, threads, finished,
                               finished_reason, target, wake, supervise_s, _spawn):
                return None
        except Exception as exc:  # noqa: BLE001 — the supervisor must not end on a tick
            logger.exception("supervisor tick failed; continuing")
            if not [t for t in threads.values() if t.is_alive()]:
                return f"{type(exc).__name__}: {exc}"
        target = pool.worker_target


def _supervise_tick(
    pool: _Pool,
    master: MasterLog,
    root: Path,
    hostname: str,
    threads: dict[int, threading.Thread],
    finished: set[int],
    finished_reason: dict[int, str],
    target: int,
    wake: threading.Event,
    supervise_s: float,
    spawn,
) -> bool:
    """One supervisor tick: scale, respawn, run the janitors. True = the pool is done.

    Split out so the supervisor loop can run it inside one try and carry on after an
    error. ``pool.worker_target`` carries the (possibly new) target back to the caller.
    """
    wake.wait(timeout=supervise_s)
    wake.clear()
    # The heartbeat says on every tick that this pool is still going, so a reader on
    # another machine can tell a live pool from one that fell silent.
    store.write_heartbeat(root, hostname)
    new_target = _read_workers_target(root, hostname, target)
    if new_target != target:
        master.log("SCALE", f"worker target {target} -> {new_target} (workers.{hostname})")
        target = new_target
    pool.worker_target = target
    leaving = pool.draining.is_set()
    stop_present = store.stop_path(root, hostname).exists() or leaving
    for tid in range(target):
        t = threads.get(tid)
        if t is None:
            if leaving:
                continue  # the pool is on its way out; do not start more workers
            spawn(tid)
        elif not t.is_alive() and tid not in finished and not leaving:
            master.log("RESPAWN", f"worker {tid} died without returning; respawning")
            spawn(tid)
        elif (
            not t.is_alive()
            and finished_reason.get(tid) in ("stop", "scale")
            and not stop_present
        ):
            # A stop file that has since been cleared (or a scale-up past a prior
            # scale-down) is not a reason to run short-handed for the pool's lifetime.
            master.log(
                "RESPAWN",
                f"worker {tid} left on {finished_reason[tid]}; stop cleared, respawning",
            )
            finished.discard(tid)
            finished_reason.pop(tid, None)
            spawn(tid)
    # Janitor tick: return leaked slot locks and stranded own-pid claims to the pool.
    # Two independent try blocks: a crash in one reaper must not skip the other.
    try:
        for path in pool.gpu.reap_leaked_locks():
            master.log("REAP", f"leaked slot lock closed: {path}")
    except Exception:  # noqa: BLE001 — the janitor must never kill the supervisor
        logger.exception("lock janitor tick failed; continuing")
    try:
        pool.reap_orphan_claims()
    except Exception:  # noqa: BLE001 — the janitor must never kill the supervisor
        logger.exception("claim janitor tick failed; continuing")
    # Claims left by a pool that is gone are recovered here rather than only where a job
    # is claimed, so a queue nothing can claim from right now still gets its jobs back.
    try:
        pool.recover_claims()
    except Exception:  # noqa: BLE001 — the janitor must never kill the supervisor
        logger.exception("claim recovery tick failed; continuing")
    try:
        for g in pool.gpu.release_dead_reservations(pool.parked_tids()):
            master.log(
                "RELEASE",
                f"gpu {g} is free for other jobs again; the job it was held for is not "
                "waiting for it any more",
            )
    except Exception:  # noqa: BLE001 — the janitor must never kill the supervisor
        logger.exception("reservation janitor tick failed; continuing")
    alive = [t for t in threads.values() if t.is_alive()]
    return not alive and bool(finished)
