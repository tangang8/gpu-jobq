"""One compact summary per queue, and the plain-text table ``jobq status`` prints.

:func:`queue_summaries` reads a queue folder and returns one :class:`QueueSummary` per
queue, in the order a worker pool would take them: the first entry is where the next job
comes from. It only reads — no cache file, no lock file, no directory is created — so it
is safe to call from any machine while pools are running.

The time estimate is deliberately small: each queue is estimated from the durations of its
own finished jobs and nothing else. There is no model of what the commands do and no
simulation of the machines, so the figure is a rough one and is always stated rounded.
"""

from __future__ import annotations

import os
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jobq import store
from jobq.gpu import PolicyError, load_policy

# Below this many successful jobs a queue has not shown how long its work takes, so it
# gets no estimate at all rather than one drawn from one or two runs.
MEDIAN_MIN_SAMPLES = 3

# How many result files one queue's median may read. A queue can hold tens of thousands of
# them, and the newest ones describe the work that is still running.
MAX_RESULTS_READ = 200

RUNNING = "running"
READY = "ready"
OTHER_MACHINE = "other machine"
WAITING = "waiting"
PAUSED = "paused"
PARKED = "parked"
EMPTY = "empty"
UNREADABLE = "unreadable"
COMPLETE = "complete"

# Rows are grouped by what a pool on this machine can do with the queue: work on it now,
# leave it to another machine, wait for it, nothing until a person acts, then the queues
# with nothing to do and the ones that cannot be read.
_STATE_GROUP = {
    RUNNING: 0,
    READY: 0,
    OTHER_MACHINE: 1,
    WAITING: 2,
    PAUSED: 3,
    PARKED: 4,
    EMPTY: 5,
    UNREADABLE: 6,
    COMPLETE: 7,
}


@dataclass(frozen=True)
class QueueSummary:
    """What one queue is doing, what it costs per job, and how long it still needs."""

    name: str
    state: str
    pending: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    total: int = 0
    deferred: int = 0
    priority: int | None = None
    node: str | None = None
    # The tie as it is written (``gpu-host-1:0,1 gpu-host-2:4-7``), empty when any machine may
    # take the queue, and whether a pool on this machine could work on it.
    machines: str = ""
    runs_here: bool = True
    waiting_for: list[str] = field(default_factory=list)
    mem_mib: int | None = None
    mem_from_machine_default: bool = False
    mem_learned: bool = False
    mem_learned_jobs: int = 0
    mem_learned_reported: int = 0
    mem_learned_measured: int = 0
    mem_learned_capped: bool = False
    mem_job_min_mib: int | None = None
    mem_job_max_mib: int | None = None
    parked_reason: str | None = None
    slots: int = 1
    no_gpu: bool = False
    median_run_s: float | None = None
    median_samples: int = 0
    eta_s: float | None = None
    last_activity_epoch: float | None = None

    @property
    def last_activity_utc(self) -> str | None:
        """When this queue was last touched, as a time stamp, or ``None`` if unknown."""
        if self.last_activity_epoch is None:
            return None
        return datetime.fromtimestamp(self.last_activity_epoch, UTC).isoformat()

    @property
    def last_activity_text(self) -> str:
        """How long ago this queue was last touched, in plain words."""
        if self.last_activity_epoch is None:
            return "unknown"
        return age_text(max(0.0, time.time() - self.last_activity_epoch))

    @property
    def state_text(self) -> str:
        """The state: the dependencies waited for, or the machine a queue is tied to."""
        if self.state == WAITING and self.waiting_for:
            return f"waiting for {', '.join(self.waiting_for)}"
        if self.state == OTHER_MACHINE:
            return f"only on {self.runs_on}"
        return self.state

    @property
    def pending_text(self) -> str:
        """Pending jobs, saying how many of them are parked for a later retry."""
        if self.deferred:
            return f"{self.pending} ({self.deferred} deferred)"
        return str(self.pending)

    @property
    def runs_on(self) -> str:
        """The machines this queue is tied to, written as they are typed."""
        return self.machines or self.node or "any machine"

    @property
    def memory_text(self) -> str:
        """Memory per job: one figure, and the jobs' own values only when they differ.

        The figure is the queue's own ask, or what its jobs have been learned to need, or
        this machine's default, each said as what it is. The jobs' own values are added
        only when they are not all the same as it, since repeating one number twice says
        nothing.
        """
        if self.no_gpu:
            return "no GPU"
        if self.mem_mib is None:
            base = "unknown"
        elif self.mem_learned:
            base = f"{self.mem_mib} learned"
            if self.mem_learned_capped:
                base += ", capped to what a GPU grants"
        elif self.mem_from_machine_default:
            base = f"{self.mem_mib} machine default"
        else:
            base = str(self.mem_mib)
        same_as_queue = (
            self.mem_job_min_mib == self.mem_job_max_mib == self.mem_mib
        )
        if self.mem_job_min_mib is not None and not same_as_queue:
            if self.mem_job_min_mib == self.mem_job_max_mib:
                base += f", jobs {self.mem_job_min_mib}"
            else:
                base += f", jobs {self.mem_job_min_mib}-{self.mem_job_max_mib}"
        if self.slots != 1:
            base += f", slots {self.slots}"
        return base

    @property
    def time_per_job_text(self) -> str:
        """The median run time and how many finished jobs it comes from."""
        if self.median_run_s is None:
            return "unknown"
        return f"{run_time_text(self.median_run_s)} (n={self.median_samples})"

    @property
    def time_left_text(self) -> str:
        """The rounded estimate, or why there is none."""
        if self.median_run_s is None:
            return "unknown"
        if self.eta_s is None:
            return "not running"
        return duration_text(self.eta_s)

    def to_dict(self) -> dict:
        """Every field plus the text the table shows, for the JSON output."""
        d = asdict(self)
        d.update(
            state_text=self.state_text,
            pending_text=self.pending_text,
            runs_on=self.runs_on,
            memory_text=self.memory_text,
            time_per_job_text=self.time_per_job_text,
            time_left_text=self.time_left_text,
            last_activity_utc=self.last_activity_utc,
            last_activity_text=self.last_activity_text,
        )
        return d


def duration_text(seconds: float) -> str:
    """A rounded, approximate wall-clock duration, never stated to the second."""
    if seconds < 60:
        return "under 1m"
    if seconds < 3600:
        return f"about {round(seconds / 60)}m"
    hours = seconds / 3600
    return f"about {hours:.0f}h" if hours >= 10 else f"about {hours:.1f}h".replace(".0h", "h")


def run_time_text(seconds: float) -> str:
    """How long a job takes, said to the second below a minute and to the minute above.

    A queue of short jobs is exactly the one whose per-job time matters most, so the
    seconds are said rather than rounded away; from an hour up the rounded form of
    :func:`duration_text` reads better than a count of minutes.
    """
    if seconds < 60:
        return f"{round(seconds)}s"
    if seconds < 3600:
        minutes, rest = divmod(int(round(seconds)), 60)
        return f"{minutes}m {rest}s" if rest else f"{minutes}m"
    return duration_text(seconds)


def age_text(seconds: float) -> str:
    """How long ago something happened, in plain words (``just now``, ``3h ago``)."""
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


class SinceError(ValueError):
    """A time window that cannot be read as a span of time."""


_SINCE_PART = re.compile(r"(\d+)([dhm])")
_SINCE_UNIT_S = {"d": 86400.0, "h": 3600.0, "m": 60.0}
SINCE_FORMS = "a number and a unit, such as 30m, 6h, 24h, 1d or 2d12h"


def parse_since(text: str) -> float:
    """Seconds in a window written as days, hours and minutes, such as ``2d12h``.

    Raises:
        SinceError: the text is not one or more number-and-unit pairs, or it adds up to
            no time at all.
    """
    cleaned = str(text).strip().lower()
    if not cleaned or "".join(m.group(0) for m in _SINCE_PART.finditer(cleaned)) != cleaned:
        raise SinceError(f"a time window is written as {SINCE_FORMS}, not {text!r}")
    seconds = sum(
        int(m.group(1)) * _SINCE_UNIT_S[m.group(2)] for m in _SINCE_PART.finditer(cleaned)
    )
    if seconds <= 0:
        raise SinceError(f"a time window has to cover some time; {text!r} covers none")
    return seconds


def _since_seconds(since: float | timedelta | None) -> float | None:
    """A window given as seconds or as a :class:`datetime.timedelta`, in seconds."""
    if since is None:
        return None
    if isinstance(since, timedelta):
        return since.total_seconds()
    return float(since)


def _last_activity_epoch(root: Path, name: str) -> float | None:
    """When a queue was last touched, from what the filesystem records about four paths.

    The most recent modification time of the results directory (a job finished), the
    claims directory (a job started), the jobs file (work was submitted) and the
    completion state file. This is a constant number of file status calls per queue and
    never reads a result, so it is as accurate as the filesystem's own timestamps.
    """
    qdir = store.queue_dir(root, name)
    stamps = []
    for path in (qdir / "results", qdir / "claims", qdir / "jobs.jsonl",
                 store.complete_state_path(root, name)):
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(stamps) if stamps else None


def _duration_s(start_iso: object, end_iso: object) -> float | None:
    """Seconds between two recorded stamps, or ``None`` when either cannot be read."""
    if not isinstance(start_iso, str) or not isinstance(end_iso, str):
        return None
    try:
        return (
            datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)
        ).total_seconds()
    except (TypeError, ValueError):
        return None


def _elapsed_s(start_iso: object) -> float:
    """How long a claim has been held, or ``0`` when its stamp cannot be read."""
    if not isinstance(start_iso, str):
        return 0.0
    try:
        start = datetime.fromisoformat(start_iso)
    except ValueError:
        return 0.0
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - start).total_seconds())


def _recent_results(root: Path, name: str, limit: int = MAX_RESULTS_READ) -> list[dict]:
    """The most recently written result records of a queue, at most ``limit`` of them.

    Which ones are recent comes from the file modification times the directory scan
    already carries, so a queue of thousands of results costs one scan plus ``limit``
    reads.
    """
    rdir = store.queue_dir(root, name) / "results"
    try:
        with os.scandir(rdir) as it:
            entries = []
            for entry in it:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    entries.append((entry.stat().st_mtime, entry.path))
                except OSError:
                    continue
    except OSError:
        return []
    entries.sort(reverse=True)
    out: list[dict] = []
    for _mtime, path in entries[:limit]:
        state, rec = store.read_json_dict(Path(path))
        if state == "ok":
            out.append(rec)
    return out


def _median_run_s(results: list[dict]) -> tuple[float | None, int]:
    """Median duration of the successful results and how many they are.

    Failed jobs are left out: a job that died early says nothing about how long the work
    takes.
    """
    durations = []
    for rec in results:
        if rec.get("rc", 1) != 0:
            continue
        d = _duration_s(rec.get("start_utc"), rec.get("end_utc"))
        if d is not None and d >= 0:
            durations.append(d)
    if len(durations) < MEDIAN_MIN_SAMPLES:
        return None, len(durations)
    return statistics.median(durations), len(durations)


def _eta_s(median_s: float, pending: int, running_jobs: list[dict]) -> float | None:
    """Seconds of work left divided by the jobs of this queue running right now.

    A running job counts for the part of a median run it has not reached yet, so one that
    has already run longer than the median adds nothing. With no job running there is no
    rate to divide by and the answer is ``None``.
    """
    if not running_jobs:
        return None
    work = pending * median_s
    for job in running_jobs:
        work += max(0.0, median_s - _elapsed_s(job.get("start_utc")))
    return work / len(running_jobs)


def _machine_default_mem_mib(root: Path, host: str, cache: dict) -> int | None:
    """The ``free_mem_mib`` of a machine's policy, or ``None`` when it cannot be read."""
    if host not in cache:
        try:
            cache[host] = load_policy(root, host).free_mem_mib
        except (FileNotFoundError, PolicyError, OSError):
            cache[host] = None
    return cache[host]


def _is_complete(root: Path, name: str) -> bool:
    """Whether every job of a queue has a result, without writing a cache."""
    if store.complete_cache_valid(root, name):
        return True
    try:
        jobs = store.load_jobs(root, name)
    except OSError:
        return False
    return bool(jobs) and all(store.has_result(root, name, j.jobkey) for j in jobs)


def _complete_counts(root: Path, name: str) -> dict:
    """The counts of a queue whose completion cache still matches, from its results.

    Every job of such a queue is terminal, so there is nothing pending, nothing running
    and no claim to read: one pass over the results directory says how many finished and
    how many of those failed.
    """
    done = 0
    failed = 0
    rdir = store.queue_dir(root, name) / "results"
    try:
        with os.scandir(rdir) as it:
            paths = [e.path for e in it if e.name.endswith(".json")]
    except OSError:
        paths = []
    for path in paths:
        state, rec = store.read_json_dict(Path(path))
        rc = 1 if state != "ok" else rec.get("rc", 1)
        if rc == 0:
            done += 1
        else:
            failed += 1
    return {
        store.PENDING: 0,
        store.RUNNING: 0,
        store.DONE: done,
        store.FAILED: failed,
        "total": done + failed,
        "running_jobs": [],
        "deferred": 0,
        "deferred_until": None,
    }


def _unmet_deps(root: Path, meta: store.QueueMeta) -> list[str]:
    """The dependencies that keep this queue from being claimed, in the worker's terms."""
    unmet = []
    for dep in meta.depends_on:
        if not store.queue_exists(root, dep) or not _is_complete(root, dep):
            unmet.append(dep)
            continue
        if meta.strict_deps and store.queue_status(root, dep)["failed"]:
            unmet.append(dep)
    return unmet


def _this_machine_gpus(root: Path, cache: dict) -> tuple[int, ...] | None:
    """The GPUs this machine's policy selects, or ``None`` when it cannot be read."""
    key = ("gpus", store.this_host())
    if key not in cache:
        try:
            cache[key] = tuple(load_policy(root, store.this_host()).gpus)
        except (FileNotFoundError, PolicyError, OSError):
            cache[key] = None
    return cache[key]


def _runs_here(root: Path, meta: store.QueueMeta, cache: dict) -> bool:
    """Whether a pool on this machine could work on a queue, by machine and by GPU.

    A queue counts as one this machine can run when this machine is among the machines
    it is tied to, or it is tied to none, and when it names GPUs here at least one of
    them is a GPU this machine's policy selects.
    """
    host = store.this_host()
    if not meta.allows_machine(host):
        return False
    if int(meta.defaults.get("slots", 1)) == 0:
        return True  # its jobs take no GPU, so the GPUs of the tie say nothing
    named = meta.gpus_on(host)
    policy_gpus = _this_machine_gpus(root, cache)
    if named is None or policy_gpus is None:
        return True
    return any(g in policy_gpus for g in named)


def _count(rec: dict | None, key: str) -> int:
    """One count of a learned-memory record, or zero when it says nothing usable."""
    value = None if rec is None else rec.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _memory_fields(root: Path, meta: store.QueueMeta, jobs: list, cache: dict) -> dict:
    """The memory and slot fields of a summary, from the queue, the jobs and the policy."""
    queue_slots = int(meta.defaults.get("slots", 1) or 0)
    slots = [queue_slots if j.slots is None else j.slots for j in jobs]
    no_gpu = queue_slots == 0 if not slots else all(s == 0 for s in slots)
    queue_mem = meta.defaults.get("mem_mib")
    learned = None if queue_mem is not None else store.read_learned_mem(root, meta.name)
    from_machine = queue_mem is None and learned is None
    capped_to = _count(learned, "capped_to_mib")
    if queue_mem is not None:
        mem = int(queue_mem)
    elif learned is not None:
        mem = min(int(learned["request_mib"]), capped_to) if capped_to else int(
            learned["request_mib"]
        )
    else:
        mem = _machine_default_mem_mib(root, meta.machine or store.this_host(), cache)
    per_job = sorted(j.mem_mib for j in jobs if j.mem_mib is not None)
    return {
        "mem_mib": mem,
        "mem_from_machine_default": from_machine and mem is not None,
        "mem_learned": learned is not None,
        "mem_learned_jobs": _count(learned, "jobs"),
        "mem_learned_reported": _count(learned, "reported"),
        "mem_learned_measured": _count(learned, "measured"),
        "mem_learned_capped": bool(
            learned is not None and capped_to and capped_to < int(learned["request_mib"])
        ),
        "mem_job_min_mib": per_job[0] if per_job else None,
        "mem_job_max_mib": per_job[-1] if per_job else None,
        "slots": queue_slots,
        "no_gpu": no_gpu,
    }


def _summarize(
    root: Path,
    name: str,
    cache: dict,
    include_complete: bool,
    since_s: float | None = None,
) -> tuple[QueueSummary, tuple[int, str]] | None:
    """One queue's summary and its sort key, or ``None`` when it is a hidden complete one."""
    last_activity = _last_activity_epoch(root, name)
    show_complete = include_complete or (
        since_s is not None
        and last_activity is not None
        and time.time() - last_activity <= since_s
    )
    try:
        meta = store.read_meta(root, name)
    except (store.QueueMetaUnreadable, OSError, PermissionError):
        return QueueSummary(
            name=name, state=UNREADABLE, last_activity_epoch=last_activity
        ), (0, name)
    order = store.queue_order_key(meta)
    try:
        cached_complete = store.complete_cache_valid(root, name)
        if cached_complete and not show_complete:
            return None
        # A queue the completion cache vouches for has no claim to look at and no job
        # whose state has to be derived, so its counts come from its result files alone.
        # Without that, listing a folder of thousands of settled queues costs a claim
        # stat and an owner read per job of every one of them.
        counts = (
            _complete_counts(root, name)
            if cached_complete
            else store.queue_status(root, name, with_deferred=True)
        )
        jobs = store.load_jobs(root, name)
    except (OSError, ValueError):
        # The settings read, so the queue keeps its priority and machine; what its jobs
        # are doing could not be determined.
        return QueueSummary(
            name=name,
            state=UNREADABLE,
            priority=meta.priority,
            node=meta.machine,
            last_activity_epoch=last_activity,
        ), order
    running_jobs = counts["running_jobs"]
    complete = cached_complete or (
        counts["total"] > 0 and not counts["pending"] and not counts["running"]
    )
    waiting_for: list[str] = []
    # Parked and paused come before running: those two say that no pool will claim from
    # the queue again until a person acts, which is the thing to know about it. A job
    # that was already running when it was set aside finishes, and its count is on the
    # row, but the queue is not one that is being worked through.
    if complete:
        state = COMPLETE
    elif counts["total"] == 0:
        state = EMPTY
    elif meta.parked:
        state = PARKED
    elif store.queue_paused(root, name):
        state = PAUSED
    elif running_jobs:
        state = RUNNING
    else:
        waiting_for = _unmet_deps(root, meta)
        state = WAITING if waiting_for else READY
    runs_here = _runs_here(root, meta, cache)
    if state == READY and not runs_here:
        # A queue this machine may not claim from is work for another machine, not work
        # waiting here, so it sits below what this machine can do now.
        state = OTHER_MACHINE
    if state == COMPLETE and not show_complete:
        return None
    median_s, samples = _median_run_s(_recent_results(root, name))
    summary = QueueSummary(
        name=name,
        state=state,
        pending=counts["pending"],
        running=counts["running"],
        done=counts["done"],
        failed=counts["failed"],
        total=counts["total"],
        deferred=counts.get("deferred", 0),
        priority=meta.priority,
        node=meta.machine,
        machines=meta.runs_on_text if meta.ties else "",
        runs_here=runs_here,
        parked_reason=meta.parked_reason,
        waiting_for=waiting_for,
        median_run_s=median_s,
        median_samples=samples,
        eta_s=None if median_s is None else _eta_s(median_s, counts["pending"], running_jobs),
        last_activity_epoch=last_activity,
        **_memory_fields(root, meta, jobs, cache),
    )
    return summary, order


def queue_summaries(
    root: Path | str,
    *,
    include_complete: bool = False,
    since: float | timedelta | None = None,
) -> list[QueueSummary]:
    """One summary per queue under ``root``, ordered as a worker pool would take them.

    Queues a pool on this machine can work on now come first, ordered by
    :func:`jobq.store.queue_order_key` exactly as the worker orders its ready queues,
    then the queues tied to another machine, the queues waiting for a dependency, the
    queues a failure pause stopped, the queues set aside with ``jobq park``, the queues
    holding no jobs, and the queues whose settings or jobs could not be read. Complete
    queues are left out unless ``include_complete``, and then come last.

    Each summary carries the median run time of that queue's recent successful jobs — at
    most :data:`MAX_RESULTS_READ` results are read, the most recently written ones — and a
    rough estimate of the time the queue still needs, which exists only once
    :data:`MEDIAN_MIN_SAMPLES` of its jobs have finished successfully and only while one
    of its jobs is running.

    Every summary also carries when its queue was last touched, measured from the
    modification times of its results directory, its claims directory, its jobs file and
    its completion state file, so it is accurate to what the filesystem records rather
    than to the stamps inside the results. A ``since`` window adds the complete queues
    touched within it; queues with work left are summarized whatever the window says.

    Nothing is written: no cache, no lock, no directory. A queue whose settings, jobs or
    results cannot be read still gets a summary, with the state ``unreadable``.

    Args:
        root: The queue folder.
        include_complete: Also summarize queues with no work left.
        since: A window ending now, in seconds or as a :class:`datetime.timedelta`;
            complete queues touched inside it are summarized too.

    Returns:
        The summaries, in the order described above.
    """
    root = Path(root)
    since_s = _since_seconds(since)
    cache: dict = {}
    rows = []
    for name in store.list_queues(root):
        got = _summarize(root, name, cache, include_complete, since_s)
        if got is not None:
            rows.append(got)
    rows.sort(key=lambda r: (_row_group(r[0]), r[1]))
    return [summary for summary, _order in rows]


def _row_group(summary: QueueSummary) -> int:
    """Which block of the table a queue belongs to (see :data:`_STATE_GROUP`).

    A queue tied to another machine sits in its own block even while its jobs run there,
    since nothing on this machine will take it.
    """
    if summary.state == RUNNING and not summary.runs_here:
        return _STATE_GROUP[OTHER_MACHINE]
    return _STATE_GROUP[summary.state]


def queue_summary(root: Path | str, name: str) -> QueueSummary:
    """The summary of one queue, complete or not, read the same way and writing nothing."""
    root = Path(root)
    got = _summarize(root, name, {}, include_complete=True)
    if got is None:  # only a hidden complete queue, which include_complete rules out
        return QueueSummary(name=name, state=UNREADABLE)
    return got[0]


_COLUMNS: list[tuple[str, str]] = [
    ("queue", "name"),
    ("state", "state_text"),
    ("pending", "pending_text"),
    ("running", "running"),
    ("done", "done"),
    ("failed", "failed"),
    ("priority", "priority"),
    ("runs on", "runs_on"),
    ("memory per job", "memory_text"),
    ("time per job", "time_per_job_text"),
    ("time left", "time_left_text"),
    ("last activity", "last_activity_text"),
]


def render_table(summaries: list[QueueSummary]) -> list[str]:
    """The table as plain lines: a header and one row per summary, aligned with spaces.

    No box characters and no colours, so the table stays readable in a log file and when
    the output is piped. A column is as wide as its widest entry, so a long queue name is
    never cut off.
    """
    header = [h for h, _attr in _COLUMNS]
    rows = [header]
    for s in summaries:
        cells = []
        for _h, attr in _COLUMNS:
            value = getattr(s, attr)
            cells.append("unknown" if value is None else str(value))
        rows.append(cells)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in rows
    ]


def table_text(summaries: list[QueueSummary]) -> str:
    """:func:`render_table` as one string."""
    return "\n".join(render_table(summaries))
