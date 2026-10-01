"""Opt-in GPU yielding: hand a GPU to a foreign user instead of fighting for it.

When a foreign process (one this machine cannot see in its own namespace) appears on a GPU
we are using, an opt-in policy makes our jobs on that GPU die (or drain) and the GPU is
left off-limits until the foreign process leaves, so we yield the GPU rather than
co-loading it into an OOM.

Namespace-safe detection (why the /proc uid of an nvidia-smi pid is not read).
``nvidia-smi --query-compute-apps`` reports host-namespace PIDs, which are different
numbers from the PIDs inside a container — an smi pid resolved in our ``/proc`` lands on an
unrelated process, so looking up its uid is actively misleading. Instead we count from two
independent sides:

- the compute-proc count nvidia-smi sees on the GPU (mapped ``gpu_uuid`` -> index, since
  the compute-apps query has no index column);
- a walk of this machine's own ``/proc/<pid>/fd/*`` (:func:`scan_local_gpu_procs`) that
  attributes each local CUDA process to the specific GPU(s) it actually uses. Everything it
  finds is ours-or-this-machine's — including local root-owned daemons, which are therefore
  counted as non-foreign and never trigger a yield, unless the policy's ``yield_to_uids``
  names their owner.

The foreign count on a GPU is the smi total minus the local processes attributed to it,
floored at 0 (:func:`foreign_counts`). Anything on the GPU we cannot see locally is
another container or user.

CUDA opens every device node, so attribution is required.
A CUDA process does not hold fds on only the GPU it uses: at init it enumerates and opens
every visible ``/dev/nvidiaN`` node, holding a handful of fds on each and many more on the
GPU actually in use. So a naive "has an fd on ``/dev/nvidia<g>``" test counts one of our
jobs as local on every GPU, masking a real foreign process to ``foreign=0`` on the very
GPUs where it is present. We therefore attribute each local CUDA proc to specific GPU(s),
in priority order:

1. ``CUDA_VISIBLE_DEVICES`` from ``/proc/<pid>/environ``, but only for a job jobq
   launched, recognised by ``JOBQ_RUN_ID`` in the same environment. jobq sets both on
   every job it launches, and the value it sets is the physical index, so for those
   processes the variable is exact (a comma list means several GPUs; empty entries and
   entries that are not numbers are ignored). For any other local process the variable is
   not a fact about the machine's GPUs: a container renumbers the devices it shows, so the
   index inside it names a different GPU here, and a process may set the variable to a
   list it never opens.
2. Max-fd fallback, which is what every process without such a variable uses: among the
   ``/dev/nvidiaN`` nodes the proc has open, attribute it to the node(s) with strictly more
   fds than the minimum baseline. If every device node ties, fall through to 3.
3. Attribute to every policy GPU (conservative: an unattributable local CUDA process keeps
   every GPU from being yielded — erring toward keeping our jobs alive, not yielding wrong).

Limitation: a root-owned daemon in another container is indistinguishable from another
user's job — if that ever misfires, raise ``yield_min_foreign_procs``.

All ``nvidia-smi`` access is behind injectable callables (module-level defaults, overridable
per :class:`YieldWatchdog`) exactly like ``gpu.query_free_mem``, so a caller can supply its
own source of GPU facts.

The set of currently-yielded GPUs is persisted to ``yielded.<hostname>.json`` under the
queue root so :meth:`GpuManager.acquire` skips them and the marker survives a pool restart.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from jobq.io import atomic_write_text
from jobq.store import now_iso

_NVIDIA_DEV = re.compile(r"^/dev/nvidia(\d+)$")  # the device node only, not nvidiactl/-uvm

# Bound on every nvidia-smi call here, matching ``gpu.query_free_mem``: a wedged smi must
# expire into "no information" (which never yields) instead of hanging the watchdog thread
# for the pool's lifetime.
SMI_TIMEOUT_S = 30


# --------------------------- nvidia-smi (injectable) ---------------------------


def query_compute_apps() -> list[tuple[int, str]]:
    """(pid, gpu_uuid) for every compute proc via ``nvidia-smi``, in the host namespace.

    The pids are host-namespace and must not be resolved in our ``/proc`` — only the count
    per ``gpu_uuid`` is used. Raises ``CalledProcessError`` on smi failure (caller treats a
    raise as "no information" and does not yield).
    """
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=SMI_TIMEOUT_S,
    ).stdout
    apps: list[tuple[int, str]] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        apps.append((int(parts[0]), parts[2]))
    return apps


def query_gpu_uuids() -> dict[int, str]:
    """GPU index -> uuid via ``nvidia-smi`` (the compute-apps query carries no index)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=SMI_TIMEOUT_S,
    ).stdout
    uuids: dict[int, str] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx, uuid = (x.strip() for x in line.split(",", 1))
        uuids[int(idx)] = uuid
    return uuids


# --------------------------- detection ---------------------------


JOBQ_RUN_ID_KEY = b"JOBQ_RUN_ID="
CUDA_VISIBLE_DEVICES_KEY = b"CUDA_VISIBLE_DEVICES="


def read_proc_environ(pid: int, *, proc_root: str = "/proc") -> str | None:
    """``CUDA_VISIBLE_DEVICES`` of a job jobq launched, else ``None``.

    ``environ`` is a NUL-separated ``KEY=value`` blob; readable for our own uid (a foreign
    uid raises ``PermissionError`` -> ``None``, so we fall back to the fd heuristic).

    The variable is only answered for a process carrying ``JOBQ_RUN_ID``, which is a job
    jobq launched and therefore one whose variable jobq itself set to a physical GPU
    index. For anything else the value is container-local and says nothing
    about the GPUs on this machine, so ``None`` is answered and the caller attributes
    the process by the device nodes it actually has open.
    """
    try:
        blob = (Path(proc_root) / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    cvd = None
    ours = False
    for entry in blob.split(b"\x00"):
        if entry.startswith(CUDA_VISIBLE_DEVICES_KEY):
            cvd = entry[len(CUDA_VISIBLE_DEVICES_KEY) :].decode("utf-8", "replace")
        elif entry.startswith(JOBQ_RUN_ID_KEY) and entry[len(JOBQ_RUN_ID_KEY) :]:
            ours = True
    return cvd if ours else None


def _nvidia_fd_counts(pdir: Path) -> dict[int, int] | None:
    """``{gpu_index: fd_count}`` for the ``/dev/nvidiaN`` nodes ``pdir`` has open.

    ``None`` when the pid's ``/proc/<pid>/fd`` vanished mid-scan or is unreadable
    (``PermissionError``); ``{}`` when it holds no ``/dev/nvidiaN`` fd (not a CUDA proc).
    """
    try:
        fds = list((pdir / "fd").iterdir())
    except (FileNotFoundError, NotADirectoryError, ProcessLookupError, PermissionError):
        return None
    except OSError:
        return None
    counts: dict[int, int] = {}
    for fd in fds:
        try:
            dest = os.readlink(fd)
        except OSError:
            continue
        m = _NVIDIA_DEV.match(dest)
        if m:
            g = int(m.group(1))
            counts[g] = counts.get(g, 0) + 1
    return counts


def _attribute_gpus(pid: int, counts: dict[int, int], environ_reader) -> set[int] | None:
    """The GPU(s) a local CUDA proc uses, or ``None`` meaning "attribute to every policy GPU".

    Priority: ``CUDA_VISIBLE_DEVICES`` of one of our own jobs (physical indices) ->
    strictly-above-baseline fd count -> ``None`` (conservative catch-all). See the module
    docstring for why fds alone lie, and why the variable is read only for our own jobs.
    """
    cvd = environ_reader(pid)
    if cvd:
        idxs: set[int] = set()
        for tok in cvd.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idxs.add(int(tok))
            except ValueError:
                continue  # e.g. a GPU-UUID form -> ignore, fall through to fd heuristic
        if idxs:
            return idxs
    baseline = min(counts.values())
    hot = {g for g, c in counts.items() if c > baseline}
    return hot or None


def scan_local_gpu_procs(
    *, proc_root: str = "/proc", environ_reader=None
) -> tuple[dict[int, int], dict[int, set[int] | None]] | None:
    """One ``/proc`` walk -> ``({pid: uid}, {pid: attributed GPUs or None for "all"})``.

    ``None`` when the process listing itself cannot be read: that is "no information about
    what is running here", and the caller must not turn it into a foreign count. An empty
    listing would make every one of our own processes look foreign and yield every GPU.

    :func:`foreign_counts` filters this per GPU, so the watchdog scans once per poll rather
    than once per GPU.
    """
    reader = environ_reader
    if reader is None:
        reader = lambda pid: read_proc_environ(pid, proc_root=proc_root)  # noqa: E731
    uids: dict[int, int] = {}
    attributed: dict[int, set[int] | None] = {}
    proc = Path(proc_root)
    try:
        entries = list(proc.iterdir())
    except OSError:
        return None
    for pdir in entries:
        if not pdir.name.isdigit():
            continue
        counts = _nvidia_fd_counts(pdir)
        if not counts:  # None (vanished/unreadable) or {} (no nvidia fd) -> not our GPU proc
            continue
        pid = int(pdir.name)
        gpus = _attribute_gpus(pid, counts, reader)
        try:
            uids[pid] = os.stat(pdir).st_uid
        except OSError:
            continue  # /proc/<pid> vanished mid-scan
        attributed[pid] = gpus
    return uids, attributed


def _procs_on_gpu(
    gpu_index: int, uids: dict[int, int], attributed: dict[int, set[int] | None]
) -> dict[int, int]:
    """Filter one :func:`scan_local_gpu_procs` result down to ``gpu_index`` -> {pid: uid}."""
    return {
        pid: uid
        for pid, uid in uids.items()
        if attributed.get(pid) is None or gpu_index in (attributed.get(pid) or ())
    }


def foreign_counts(
    gpus,
    *,
    compute_apps_query=query_compute_apps,
    uuid_query=query_gpu_uuids,
    proc_root: str = "/proc",
    environ_reader=None,
    foreign_uids: frozenset[int] | set[int] = frozenset(),
) -> dict[int, int | None]:
    """Per-GPU foreign counts from one smi pair and one ``/proc`` scan.

    A GPU's count is the compute processes nvidia-smi sees on it minus the local processes
    attributed to it, floored at 0. One smi pair and one ``/proc`` walk serve every GPU, so
    the cost is per poll rather than per GPU.

    ``foreign_uids`` (policy ``yield_to_uids``): local procs owned by these uids are
    treated as foreign even though they are visible in our namespace, for a machine where
    the person we make room for has an account here rather than a container of their own.

    A GPU with no uuid in the index map stays ``None`` ("no information", never yield), as
    does every GPU when smi itself fails or when the local process listing cannot be read.
    """
    gpus = list(gpus)
    try:
        uuids = uuid_query()
        apps = compute_apps_query()
    except Exception:  # noqa: BLE001 — any smi/parse failure means "no information"
        return {g: None for g in gpus}
    scan = scan_local_gpu_procs(proc_root=proc_root, environ_reader=environ_reader)
    if scan is None:
        # No local information: every smi process would read as foreign, which would yield
        # every GPU we are using.
        return {g: None for g in gpus}
    uids, attributed = scan
    out: dict[int, int | None] = {}
    for g in gpus:
        uuid = uuids.get(g)
        if uuid is None:
            out[g] = None
            continue
        total = sum(1 for _pid, u in apps if u == uuid)
        local = _procs_on_gpu(g, uids, attributed)
        ours = sum(1 for uid in local.values() if uid not in foreign_uids)
        out[g] = max(0, total - ours)
    return out


# --------------------------- yield marker file ---------------------------


def marker_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"yielded.{hostname}.json"


# Last successfully parsed marker per (root, hostname), and whether the current outage has
# been warned about. A marker file that exists but cannot be read must not read as "nothing
# is yielded": that would put new work straight back onto a GPU promised to someone else.
# Keyed per machine file and bounded by the roots this process touches.
_MARKER_LAST_GOOD: dict[tuple[str, str], dict[int, dict]] = {}
_MARKER_WARNED: set[tuple[str, str]] = set()
_MARKER_LOCK = threading.Lock()


def _clear_marker_cache() -> None:
    """Drop the last-known-good marker cache, so the next read goes to disk."""
    with _MARKER_LOCK:
        _MARKER_LAST_GOOD.clear()
        _MARKER_WARNED.clear()


def _marker_unreadable(root: Path, hostname: str, key, reason) -> dict[int, dict]:
    """Serve the last known-good marker set, warning once per outage."""
    with _MARKER_LOCK:
        warn = key not in _MARKER_WARNED
        _MARKER_WARNED.add(key)
        last = {g: dict(v) for g, v in _MARKER_LAST_GOOD.get(key, {}).items()}
    if warn:
        logger.warning(
            "yield marker {} is present but unreadable ({}) — keeping the last set of "
            "yielded GPUs ({})",
            marker_path(root, hostname),
            reason,
            sorted(last) or "none",
        )
    return last


def read_yielded(root: Path, hostname: str) -> dict[int, dict]:
    """Read the yield marker as ``{gpu: {first_seen, last_seen, foreign}}`` ({} if absent).

    An absent marker means nothing is yielded. A marker that is present but unreadable
    serves the last set this process read successfully, and warns once per outage: the
    file's existence says a GPU may be promised away, so the fail-safe direction is to
    keep it off-limits. Unreadable covers an I/O error, invalid JSON, JSON of the wrong
    shape, a key that is not a GPU index and an entry that is not a mapping — a malformed
    entry is not dropped, because dropping it would put work back on that GPU.
    """
    key = (str(root), hostname)
    try:
        raw = json.loads(marker_path(root, hostname).read_text())
    except FileNotFoundError:
        with _MARKER_LOCK:
            _MARKER_LAST_GOOD.pop(key, None)
            _MARKER_WARNED.discard(key)
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        return _marker_unreadable(root, hostname, key, exc)
    out: dict[int, dict] = {}
    if not isinstance(raw, dict):
        return _marker_unreadable(root, hostname, key, f"top level is {type(raw).__name__}")
    gpus = raw.get("gpus") or {}
    if not isinstance(gpus, dict):
        return _marker_unreadable(root, hostname, key, f"'gpus' is {type(gpus).__name__}")
    for k, v in gpus.items():
        if not isinstance(v, dict):
            return _marker_unreadable(root, hostname, key, f"entry {k!r} is not a mapping")
        try:
            out[int(k)] = dict(v)
        except (ValueError, TypeError):
            return _marker_unreadable(root, hostname, key, f"key {k!r} is not a GPU index")
    with _MARKER_LOCK:
        _MARKER_LAST_GOOD[key] = {g: dict(v) for g, v in out.items()}
        _MARKER_WARNED.discard(key)
    return out


def write_yielded(root: Path, hostname: str, data: dict[int, dict]) -> None:
    """Persist the yield marker atomically (safe against concurrent read/write)."""
    body = {"hostname": hostname, "gpus": {str(g): v for g, v in data.items()}}
    atomic_write_text(marker_path(root, hostname), json.dumps(body, indent=2) + "\n")


def yielded_gpus(root: Path, hostname: str) -> set[int]:
    """The GPU indices currently off-limits on this machine (empty if the marker is missing)."""
    return set(read_yielded(root, hostname).keys())


# --------------------------- progress estimation (drain_if_near_done) ---------------------------
#
# Generic, best-effort, and deliberately fail-safe toward yielding promptly: the framework
# runs arbitrary shell commands, so there is no universal progress signal. We try, in order:
#   1. a per-policy log regex (opt-in, most accurate) — the last (current, total) match in the
#      tail of the job's log, such as the step counter a progress bar prints;
#   2. elapsed / median-duration-of-completed-jobs in the same queue (needs a min sample);
#   3. nothing known -> 0.0 (the job is killed — never hoard a GPU we promised to vacate).


@dataclass(frozen=True)
class DrainSpec:
    """Per-yield context the watchdog hands the pool for a ``drain_if_near_done`` evaluation.

    ``yield_started_ts`` and ``now`` are epoch seconds from the watchdog's clock (the safety
    cap kills a spared job once ``now - yield_started_ts >= max_s``); ``progress_regex`` opts
    into log-based progress (two numeric groups = current/total).
    """

    threshold: float
    max_s: float
    progress_regex: str | None
    yield_started_ts: float
    now: float


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _read_log_tail(path: str, *, max_bytes: int = 16384) -> str | None:
    """The last ``max_bytes`` of ``path`` decoded as text, or ``None`` if unreadable/missing.

    Only the tail is read: a job log can hold thousands of progress lines and the useful
    marker is always the most recent one.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))  # back to the tail's start (0 for a small file)
            data = f.read()
    except OSError:
        return None
    return data.decode("utf-8", "replace")


def progress_from_log(
    log_path: str | None, progress_regex: str | None, *, tail_reader=_read_log_tail
) -> float | None:
    """current/total from the last regex match in the log tail, or ``None`` to fall through.

    ``None`` on: no regex/path, missing/unreadable log, no match, a bad regex, a non-numeric
    group, or ``total <= 0``. ``tail_reader`` is injectable.
    """
    if not log_path or not progress_regex:
        return None
    text = tail_reader(log_path)
    if not text:
        return None
    try:
        pat = re.compile(progress_regex)
    except re.error:
        return None
    last = None
    for last in pat.finditer(text):  # noqa: B007 — want the last match
        pass
    if last is None:
        return None
    try:
        cur = float(last.group(1))
        total = float(last.group(2))
    except (IndexError, ValueError):
        return None
    if total <= 0:
        return None
    return _clamp01(cur / total)


def _duration_s(start_iso: str | None, end_iso: str | None) -> float | None:
    if not start_iso or not end_iso:
        return None
    try:
        return (datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)).total_seconds()
    except (TypeError, ValueError):
        return None


def median_duration_s(results: list[dict], *, min_samples: int = 5) -> float | None:
    """Median wall-clock duration of the completed (rc==0) results, or ``None``.

    ``None`` until at least ``min_samples`` usable durations exist — an unproven queue must
    not drive a keep-the-GPU decision.
    """
    durs: list[float] = []
    for res in results:
        if res.get("rc", 1) != 0:
            continue
        d = _duration_s(res.get("start_utc"), res.get("end_utc"))
        if d is not None and d >= 0:
            durs.append(d)
    if len(durs) < min_samples:
        return None
    return statistics.median(durs)


def estimate_progress(
    *,
    log_path: str | None,
    progress_regex: str | None,
    elapsed_s: float | None,
    median_s: float | None,
    tail_reader=_read_log_tail,
) -> float:
    """Best-effort [0.0, 1.0] progress via the priority chain (log regex -> median -> 0.0)."""
    p = progress_from_log(log_path, progress_regex, tail_reader=tail_reader)
    if p is not None:
        return p
    if median_s is not None and median_s > 0 and elapsed_s is not None:
        return _clamp01(elapsed_s / median_s)
    return 0.0


# --------------------------- watchdog ---------------------------


class YieldWatchdog:
    """Background monitor that flips policy GPUs to yielded when foreigners appear.

    Independent of pool internals: it maintains the marker file and, for
    ``yield_action == "kill"``, calls ``on_yield(gpu)`` (the pool kills our jobs there).
    Placement (acquire skipping the marker) is what enforces "stop claiming".

    The thread is started whenever a policy file exists; while ``yield_to_foreign`` is false
    :meth:`poll_once` does no detection work (no nvidia-smi call, no process scan) and gives
    back any GPU the marker still names. So the switch is hot in both directions: flip the
    flag in the policy file on a running pool and yielding begins, or ends and the GPUs come
    back, within one ``yield_poll_s``.
    """

    def __init__(
        self,
        root: Path,
        hostname: str,
        *,
        load_policy,
        master,
        on_yield,
        compute_apps_query=query_compute_apps,
        uuid_query=query_gpu_uuids,
        proc_root: str = "/proc",
        environ_reader=None,
        clock=time.time,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.load_policy = load_policy  # () -> GpuPolicy (raises FileNotFoundError if none)
        self.master = master  # has .log(verb, msg)
        self.on_yield = on_yield  # (gpu:int) -> None, only for the "kill" action
        self.compute_apps_query = compute_apps_query
        self.uuid_query = uuid_query
        self.proc_root = proc_root
        self.environ_reader = environ_reader
        self.clock = clock
        self._confirm: dict[int, int] = {}
        self._smi_fail_logged = False
        self._policy_fail_logged = False

    def _apply_yield(self, g: int, policy, entry: dict, now: float) -> None:
        """Fire the configured yield action for gpu ``g`` (kill / drain / drain_if_near_done).

        ``kill`` kills every live job on the GPU (``on_yield(g)``).
        ``drain_if_near_done`` hands the pool a :class:`DrainSpec` so it spares near-done jobs
        and kills the rest. ``drain`` never touches running jobs (placement alone enforces it).
        """
        action = policy.yield_action
        if action == "kill":
            self.on_yield(g)
        elif action == "drain_if_near_done":
            started = entry.get("first_seen_ts")
            started = float(started) if started is not None else now
            self.on_yield(
                g,
                DrainSpec(
                    threshold=policy.yield_drain_threshold,
                    max_s=policy.yield_drain_max_s,
                    progress_regex=policy.yield_progress_regex,
                    yield_started_ts=started,
                    now=now,
                ),
            )

    def foreign_counts(self, gpus, foreign_uids=frozenset()) -> dict[int, int | None]:
        """Per-GPU foreign counts for one poll (one smi pair + one /proc scan for all)."""
        return foreign_counts(
            gpus,
            compute_apps_query=self.compute_apps_query,
            uuid_query=self.uuid_query,
            proc_root=self.proc_root,
            environ_reader=self.environ_reader,
            foreign_uids=foreign_uids,
        )

    def _policy_or_none(self):
        """The current policy, or ``None`` when there is none / it does not load.

        A policy edited to something invalid while the pool runs (the hot-reload path)
        must not take the watchdog down or change what it already decided: it is logged
        once per outage and this poll is simply skipped. Every error the loader can raise
        is caught, whatever its type, because the thread that calls this is the only thing
        that can ever give a yielded GPU back.
        """
        try:
            policy = self.load_policy()
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 — see the docstring
            if not self._policy_fail_logged:
                self._policy_fail_logged = True
                self.master.log(
                    "WAIT", f"yield check: GPU policy does not load ({exc}); keeping current state"
                )
            return None
        self._policy_fail_logged = False
        return policy

    def _release_all(self) -> None:
        """Give every yielded GPU back, because yielding is switched off.

        Turning ``yield_to_foreign`` off has to be as hot as turning it on: while the
        marker names a GPU, admission skips it, so leaving the marker in place would keep
        those GPUs idle for the pool's lifetime.
        """
        self._confirm.clear()
        yielded = read_yielded(self.root, self.hostname)
        if not yielded:
            return
        write_yielded(self.root, self.hostname, {})
        self.master.log(
            "RECLAIM",
            "yield_to_foreign is off; taking back gpu(s) "
            + ", ".join(str(g) for g in sorted(yielded)),
        )

    def poll_once(self) -> None:
        """One detection pass: update confirm counters, the marker, and (kill) trigger."""
        policy = self._policy_or_none()
        if policy is None:
            return
        if not policy.yield_to_foreign:
            self._release_all()
            return
        now = self.clock()
        yielded = read_yielded(self.root, self.hostname)
        changed = False
        # A marker entry for a GPU outside the policy's gpus can never be reclaimed by the
        # loop below, which only walks the policy's GPUs.
        for g in [g for g in yielded if g not in policy.gpus]:
            del yielded[g]
            changed = True
            self.master.log("RECLAIM", f"gpu={g} is outside the policy's gpus")
        foreign_uids = frozenset(getattr(policy, "yield_to_uids", ()) or ())
        counts = self.foreign_counts(policy.gpus, foreign_uids)
        for g in policy.gpus:
            fc = counts.get(g)
            if fc is None:
                # No information about this GPU. Keep our jobs alive, keep any marker, and
                # reset both clocks: the confirm count must be consecutive sightings and the
                # cooldown must be observed quiet, and a poll that saw nothing is neither.
                self._confirm.pop(g, None)
                if g in yielded and yielded[g].pop("quiet_since", None) is not None:
                    changed = True
                if not self._smi_fail_logged:
                    self.master.log(
                        "WAIT",
                        f"yield check: no usable process information (gpu={g}); not yielding",
                    )
                    self._smi_fail_logged = True
                continue
            self._smi_fail_logged = False
            if g in yielded and yielded[g].get("first_seen_ts") is None:
                # An entry written without the epoch anchor (by hand, or before the GPU
                # was last yielded) still needs a start time, or the drain safety cap has
                # nothing to measure from.
                yielded[g]["first_seen_ts"] = now
                changed = True
            if fc >= policy.yield_min_foreign_procs:
                self._confirm[g] = self._confirm.get(g, 0) + 1
                if g in yielded and yielded[g].pop("quiet_since", None) is not None:
                    changed = True  # the GPU is busy again; the cooldown starts over
                if g in yielded:
                    yielded[g]["last_seen"] = now_iso()
                    yielded[g]["foreign"] = fc
                    changed = True
                    # Re-fire on every poll while the GPU stays yielded, not only on the
                    # transition. Anything that slipped onto the GPU despite
                    # the marker (a worker already inside acquire when
                    # we yielded, a manual placement) would otherwise run there forever.
                    # The marker is already durable on disk (written on a prior poll), so
                    # the "unacquirable before kill" invariant still holds. yield_kill_gpu
                    # is idempotent — it only marks/kills handles whose process is still
                    # alive and returns at once when there is nothing to kill — so calling
                    # it each poll is cheap and emits a KILL line only for real kills. For
                    # drain_if_near_done the per-poll re-evaluation is the mechanism that
                    # kills a spared job once it overruns yield_drain_max_s.
                    if policy.yield_action in ("kill", "drain_if_near_done"):
                        self._apply_yield(g, policy, yielded[g], now)
                elif self._confirm[g] >= policy.yield_confirm_polls:
                    stamp = now_iso()
                    yielded[g] = {
                        "first_seen": stamp,
                        "last_seen": stamp,
                        "foreign": fc,
                        "first_seen_ts": now,  # epoch anchor for the drain safety cap
                    }
                    changed = True
                    self.master.log("YIELD", f"gpu={g} foreign={fc}")
                    if policy.yield_action in ("kill", "drain_if_near_done"):
                        # Persist the marker before any kill so the invariant
                        # holds — a GPU is unacquirable before any of its jobs are killed.
                        # Each kill frees a slot a worker re-claims immediately; if the
                        # marker were written only at end-of-poll, acquire would read a
                        # GPU that still looks available and dispatch straight back onto
                        # the one we are vacating. The end-of-poll write below still runs
                        # (changed is True) and harmlessly re-persists the same state.
                        write_yielded(self.root, self.hostname, yielded)
                        self._apply_yield(g, policy, yielded[g], now)
            else:
                self._confirm[g] = 0
                if g in yielded:
                    # The moment the GPU first went quiet lives in the marker, not in
                    # this thread: a pool that is restarted mid-cooldown would otherwise
                    # start the count again and keep the GPU out of use for another
                    # whole cooldown, however long it had already been free.
                    first = yielded[g].get("quiet_since")
                    if not isinstance(first, (int, float)) or isinstance(first, bool):
                        first = now
                        yielded[g]["quiet_since"] = now
                        changed = True
                    if now - first >= policy.yield_cooldown_s:
                        del yielded[g]
                        changed = True
                        self.master.log("RECLAIM", f"gpu={g}")
        if changed:
            write_yielded(self.root, self.hostname, yielded)

    def _poll_interval(self) -> float:
        """Seconds until the next poll; the default whenever the policy does not load.

        Outside ``run``'s try block, so it relies on ``_policy_or_none`` swallowing every
        error the loader can raise — one escaping here would end the thread.
        """
        policy = self._policy_or_none()
        return 30.0 if policy is None else policy.yield_poll_s

    def run(self, stop_event, *, sleep_slice: float = 1.0) -> None:
        """Poll until ``stop_event`` is set; sleeps in slices so a stop is honored promptly."""
        while not stop_event.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 — a watchdog must never crash the pool
                logger.exception("yield watchdog poll failed")
            try:
                interval = self._poll_interval()
            except Exception:  # noqa: BLE001 — a watchdog must never crash the pool
                logger.exception("yield watchdog could not read its poll interval")
                interval = 30.0
            waited = 0.0
            while waited < interval and not stop_event.is_set():
                nap = min(sleep_slice, interval - waited)
                time.sleep(nap)
                waited += nap
