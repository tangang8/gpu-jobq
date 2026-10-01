"""Per-machine GPU slot locks and memory gating for the worker pool.

Four guards, all machine-local and shared across all queues, so the combined per-GPU load
of this user's queues is capped globally:

- Slot locks — ``fcntl.flock(LOCK_EX|LOCK_NB)`` on ``<prefix>_gpu{g}_slot{s}.lock`` for
  ``s`` in ``range(cap_per_gpu)``. Holding one reserves a run slot on GPU ``g``. A policy
  that names no ``cap_per_gpu`` leaves the count of jobs to the memory gate and probes
  ``MAX_SLOT_UNITS_PER_GPU`` slots instead.
- CPU slot locks — the same mechanism on ``<prefix>_cpu_slot{s}.lock`` for ``s`` in
  ``range(cpu_cap)``, an independent budget for jobs declaring ``slots: 0``. GPU capacity
  and CPU capacity are separate concerns: CPU work must not block a GPU, but it still
  needs its own ceiling, since the GPU slots are otherwise the pool's only admission
  control (N worker threads would all run CPU jobs at once and thrash the box). The
  ceiling is the policy's ``cpu_cap``, a quarter of the usable cores by default.
- Memory gate — a GPU must show ``memory.free`` (minus the policy's ``reserve_mem_mib``)
  at least ``mem_mib`` on ``mem_checks`` consecutive ``nvidia-smi`` readings, taken
  ``mem_interval_s`` apart, before we dispatch to it. A first reading with ample headroom
  takes a fast path and skips the settle.
- Memory accounting — every granted slot records the job's ask in an ``.ask`` file
  beside its slot lock. The asks of the jobs holding slots on a GPU bound what this queue
  folder commits there (``mem_budget_mib``), and an ask whose memory has not shown up in
  the GPU's free reading yet is held against that reading for up to ``startup_hold_s``,
  so two jobs are never admitted into the same free space. Only slots whose lock is still
  held count, so a record can never outlive its job.

The acquire-check-dispatch section is serialized per machine by one global flock
(``<prefix>_acquire.lock``) so two workers cannot both pass the memory gate for the same
headroom at once. The GPU policy is re-read on every acquire, so an edit to the policy file
takes effect on a running pool.

The lock ``prefix`` is machine-local and keyed on the queue folder (see
:func:`resolve_lock_prefix`): two queue folders on one machine cap independently, two
spellings of one folder share a cap, and the files are never shared with another account
by construction.

All ``nvidia-smi`` access is behind :func:`query_free_mem`, and the lock ``prefix`` and
``mem_query`` are constructor parameters, so a caller can supply its own lock directory
and its own source of free-memory readings.
"""

from __future__ import annotations

import abc
import contextlib
import fcntl
import glob
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from jobq import monitor, store, yielding
from jobq.io import atomic_write_json


def _as_number(value, default):
    """A number read from a record written by another process, or ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value

# The three slot-lock filename shapes, as suffixes after ``lock_prefix`` (see
# _slot_lock_file / _cpu_slot_lock_file / _try_group_slot). reap_leaked_locks matches
# open fds against this whitelist rather than "anything under the prefix", so a broad or
# root-overlapping custom JOBQ_LOCK_PREFIX (or any non-lock file under the prefix) can
# never draw unrelated descriptors into the audit.
_SLOT_LOCK_SUFFIX_RE = re.compile(
    r"_(?:gpu\d+_slot\d+|cpu_slot\d+|grp_[A-Za-z0-9._-]+_gpu\d+_slot\d+)\.lock$"
)

# How long reap_leaked_locks retries for the acquire flock before skipping its tick.
# Bounded so a slow nvidia-smi holding the flock (query_free_mem waits up to its own
# subprocess timeout, and the gate may run several) degrades to skipped audits instead of
# a hung supervisor; long enough that ordinary acquire churn cannot starve the auditor.
_AUDIT_FLOCK_DEADLINE_S = 10.0

# ---- defaults of the tunable policy keys -------------------------------------------
# Each is the value used when a policy file does not name the key (and the value the pool
# uses when there is no policy at all). They live here so the policy fields, the
# :class:`Tunables` view and the worker's module-level names all read the same number.
DEFAULT_OOM_MEM_FACTOR = 1.5
DEFAULT_OOM_MEM_FLOOR_MIB = 4096
DEFAULT_OOM_CEILING_HEADROOM_MIB = 2048
DEFAULT_OOM_MAX_REQUEUES = 8
DEFAULT_OOM_OWN_USAGE_FRACTION = 0.75
DEFAULT_TEMPFAIL_RETRY_S = 15 * 60.0
DEFAULT_TEMPFAIL_MAX_REQUEUES = 8
DEFAULT_PARK_DEFER_MAX_S = 600.0
DEFAULT_KILL_GRACE_S = 20.0
DEFAULT_POLL_S = 20.0
DEFAULT_GPU_WAIT_S = 5.0
DEFAULT_SUPERVISE_S = 30.0
DEFAULT_ORPHAN_CLAIM_GRACE_S = 120.0
DEFAULT_MEM_CHECKS = 2
DEFAULT_MEM_INTERVAL_S = 3.0
DEFAULT_MEM_FASTPATH_FACTOR = 2.0
DEFAULT_STARTUP_HOLD_S = 300.0
DEFAULT_RESERVE_AFTER_S = 120.0
DEFAULT_RESERVE_MAX_GPUS = 1

# The ``mps`` policy value meaning "use MPS when this machine can, and carry on when not".
MPS_AUTO = "auto"

# The upper bound on the slot-units of one GPU, whatever a cap says. The slot locks need a
# finite range of filenames, so this is the hard upper limit on how many jobs (weighted by
# their ``slots``) one GPU can hold, and it keeps probing the whole range on every
# admission cheap.
MAX_SLOT_UNITS_PER_GPU = 64


@dataclass(frozen=True)
class Tunables:
    """The policy knobs the worker pool reads, as one value object.

    Built by :meth:`GpuManager.tunables` from this machine's policy. The pool asks for this
    instead of reaching into the policy, so an implementation with no policy behind it can
    answer ``None`` and let the pool use its own defaults.
    """

    oom_mem_factor: float = DEFAULT_OOM_MEM_FACTOR
    oom_mem_floor_mib: int = DEFAULT_OOM_MEM_FLOOR_MIB
    oom_ceiling_headroom_mib: int = DEFAULT_OOM_CEILING_HEADROOM_MIB
    oom_max_requeues: int = DEFAULT_OOM_MAX_REQUEUES
    oom_own_usage_fraction: float = DEFAULT_OOM_OWN_USAGE_FRACTION
    tempfail_retry_s: float = DEFAULT_TEMPFAIL_RETRY_S
    tempfail_max_requeues: int = DEFAULT_TEMPFAIL_MAX_REQUEUES
    park_defer_max_s: float = DEFAULT_PARK_DEFER_MAX_S
    kill_grace_s: float = DEFAULT_KILL_GRACE_S
    orphan_claim_grace_s: float = DEFAULT_ORPHAN_CLAIM_GRACE_S


@dataclass
class GpuPolicy:
    """Parsed ``gpu_policy.<hostname>.json``: which GPUs, how many slots, memory floor, env.

    Attributes:
        gpus: The physical GPU indices this machine's pool is allowed to use.
        cap_per_gpu: How many slot-units may be held at once on each of those GPUs. ``None``
            (the key absent or null) takes the cap from the machine's share of the cores:
            :attr:`workers_per_gpu`, the same figure a pool sizes its threads from. Either
            way the cap is bounded above by :data:`MAX_SLOT_UNITS_PER_GPU`, the hard upper
            limit on the jobs one GPU can hold. See :attr:`slot_units`.
        free_mem_mib: Free memory a GPU must show per slot-unit; also the default per-job ask.
            It must be above zero when ``cap_per_gpu`` is absent, since memory is then the
            only thing limiting how many jobs land on a GPU.
        cpu_cap: Concurrent CPU-only jobs (``slots: 0``) admitted machine-wide. A separate
            budget from ``cap_per_gpu``: CPU work must not consume GPU capacity, but it
            still needs a ceiling of its own, since the GPU slots otherwise double as the
            pool's only admission control. Defaults to a quarter of the usable cores so a
            burst of CPU jobs cannot starve the GPU jobs' data loaders. Lowering it never
            preempts: running CPU jobs keep their slots and merely block new admissions
            until they finish.
        cores: Cores this process may run on, read when the policy is read.
        cpu_reserve: Cores held back for the OS, the worker pool and the janitors.
        cpu_per_gpu_job: Cores charged per held GPU slot (default 1.0). It is the one
            place that says how many cores a job needs: nothing is read out of ``env``,
            whose entries are passed to jobs and nothing else.
        reserve_mem_mib: MiB every policy GPU must still show free after an admission
            (default 0 = off). Unlike ``free_mem_mib`` this is never handed to a job: it caps
            the machine by memory so somebody else's run can always start here. Applied once per
            admission, machine-locally, re-read on every acquire.
        shared_perms: Create this machine's queue state world-writable and set ``umask 0`` in
            the pool (default false, so the user's umask is respected). Turn it on when the
            same account holds different uids on the machines sharing the queue folder, where
            cross-machine reads and writes have to ride on the 'other' permission bits.
        oom_patterns: Log-tail patterns that classify a failure as out-of-memory. Empty (the
            default) uses the built-in CUDA/PyTorch markers; a non-empty list replaces them.
        cwd_fallback: Directory to run a job in when the directory it asks for does not
            exist on this machine. Unset (the default) makes such a job fail with a message
            naming both the directory and this key.
        env: Environment entries added to every job this machine dispatches.
        mem_budget_mib: Cap on the memory this queue folder's own jobs may commit on one
            GPU (0 = unset, no budget). A job is admitted only if the asks of the jobs
            already holding slots on that GPU, plus its own ask, stay within the budget.
        startup_hold_s: Upper bound on how long a just-granted job's ask is held against
            its GPU's free memory while its allocation is not visible in ``nvidia-smi``
            yet (see :meth:`GpuManager.acquire`). Once it elapses the ask stops being held.
        mem_checks: Consecutive free-memory readings a GPU must pass before a job is
            dispatched to it (1 = a single reading, no settle).
        mem_interval_s: Seconds between those readings.
        mem_fastpath_factor: A first reading showing at least this multiple of the ask
            (and 4 GiB to spare) skips the remaining readings; 0 disables the fast path.
        oom_mem_factor: Multiplier applied to a job's ask on an out-of-memory requeue.
        oom_mem_floor_mib: Smallest ask an out-of-memory requeue may set.
        oom_ceiling_headroom_mib: MiB below a GPU's total that the escalation ladder
            stops at, since an idle GPU never reports its total as free.
        oom_max_requeues: Out-of-memory requeues a job gets before the exit is recorded
            as an ordinary failure.
        oom_own_usage_fraction: Fraction of its own ask a job must have held for an
            out-of-memory exit to raise its ask; below it the exit is read as pressure
            from neighbours on the GPU and the ask is kept.
        tempfail_retry_s: How long a job that exited 75 stays unclaimable.
        tempfail_max_requeues: Exits of 75 a job gets before the exit is recorded as a
            failure.
        park_defer_max_s: Longest a parked lower-priority job defers to a parked
            higher-priority one before trying the acquire anyway.
        reserve_after_s: How long a job waits for capacity before the pool holds a GPU for
            it (see :meth:`GpuManager.reserve_for_wait`); 0 turns the reservation off.
        reserve_max_gpus: How many GPUs of this machine may be held for waiting jobs at
            once, so reservations cannot stall the whole machine.
        machine_gpus: How many GPUs ``nvidia-smi`` lists on this machine, or ``None`` when
            that cannot be read. The cores are shared with everyone on the machine, so
            this, rather than the length of :attr:`gpus`, is what a fair share divides by.
        monitor_interval_s: Seconds between rounds of utilisation samples; ``0`` turns
            sampling off on this machine (see :mod:`jobq.monitor`).
        monitor_idle_util_pct: Below this utilisation, and below ``monitor_idle_mem_mib``
            of memory in use, a GPU's sample is recorded as idle.
        monitor_idle_mem_mib: The memory half of that idle test.
        monitor_idle_cpu_pct: Below this utilisation a processor sample is recorded as idle.
        monitor_cpu_sample_s: Seconds between the two ``/proc/stat`` readings a processor
            utilisation is the ratio of.
        monitor_keep_days: How long sample rows are kept before the pool drops them.
        heartbeat_stale_s: How old this machine's heartbeat may be before ``jobq status``
            says the machine has not been heard from.
        workers_per_gpu: The machine's share of the cores expressed per GPU::

                max(1, floor((cores - cpu_reserve)
                             / (machine_gpus * cpu_per_gpu_job)))

            with the length of :attr:`gpus` standing in for ``machine_gpus`` when the
            machine's GPU list cannot be read. It is both the default number of worker
            threads per policy GPU and the default cap on the slot-units of one GPU.
        kill_grace_s: Seconds between the termination signal and the kill signal when the
            pool force-kills a job's process group.
        poll_s: Seconds an idle worker thread waits before re-deriving the ready list.
        gpu_wait_s: Seconds a parked worker thread waits between acquire attempts.
        supervise_s: Supervisor tick (respawn, hot-scale, janitors).
        orphan_claim_grace_s: How old a claim directory with no owner record must be
            before any machine may reclaim it.

    The ``yield_*`` fields configure the opt-in GPU-yielding framework (see
    :mod:`jobq.yielding`). They are all defaulted and the feature is off unless
    ``yield_to_foreign`` is explicitly set true.

    Unknown keys in the JSON are ignored, so a policy file carrying keys this version does
    not know still loads.
    """

    gpus: list[int]
    cap_per_gpu: int | None
    free_mem_mib: int
    cpu_cap: int = 0  # 0 -> derived from the usable core count at load
    cores: int = 0  # the cores this process may run on, read when the policy is read
    cpu_reserve: int = 2
    cpu_per_gpu_job: float = 1.0
    # Whether the JSON named cpu_cap, so a caller can tell an explicit 0 (the CPU lane is
    # off) from an absent key (a quarter of the cores).
    cpu_cap_explicit: bool = False
    reserve_mem_mib: int = 0
    shared_perms: bool = False
    cwd_fallback: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    yield_to_foreign: bool = False
    yield_poll_s: float = 30.0
    yield_confirm_polls: int = 2
    yield_min_foreign_procs: int = 1
    yield_cooldown_s: float = 900.0
    yield_action: str = "kill"
    # Only consulted for yield_action == "drain_if_near_done": a live job whose estimated
    # progress is >= yield_drain_threshold is spared (runs to completion instead of being
    # killed+requeued); yield_drain_max_s caps how long a spared job may keep a yielded GPU;
    # yield_progress_regex (two numeric groups = current/total) opts into log-based progress.
    yield_drain_threshold: float = 0.8
    yield_drain_max_s: float = 1800.0
    yield_progress_regex: str | None = None
    # Patterns whose presence in a job's log tail classifies its failure as out-of-memory;
    # empty means the built-in markers (see ``worker.OOM_MARKERS``).
    oom_patterns: tuple[str, ...] = ()
    # Local (same-namespace) GPU procs owned by these uids count as foreign for yielding.
    # Policy JSON accepts ``yield_to_users`` (names) and/or ``yield_to_uids`` (ints); names
    # are resolved via ``pwd`` at load and silently skipped if unknown on this host.
    yield_to_uids: tuple[int, ...] = ()
    # CUDA MPS (see ensure_mps_daemon / mps_env): ``"auto"`` by default, meaning the pool
    # uses MPS when the control binary is there and the daemon answers, and runs without it
    # otherwise. ``true`` demands it, so a pool refuses to start when it cannot be used;
    # ``false`` never uses it.
    # ``mps_pipe_dir``/``mps_log_dir`` default to per-user /tmp paths when None.
    mps: bool | str = MPS_AUTO
    mps_pipe_dir: str | None = None
    mps_log_dir: str | None = None
    # Memory accounting (see GpuManager.acquire): a per-GPU budget on our own jobs' asks
    # and the upper bound on the start-up hold.
    mem_budget_mib: int = 0
    startup_hold_s: float = DEFAULT_STARTUP_HOLD_S
    # Memory gate timing.
    mem_checks: int = DEFAULT_MEM_CHECKS
    mem_interval_s: float = DEFAULT_MEM_INTERVAL_S
    mem_fastpath_factor: float = DEFAULT_MEM_FASTPATH_FACTOR
    # Out-of-memory handling.
    oom_mem_factor: float = DEFAULT_OOM_MEM_FACTOR
    oom_mem_floor_mib: int = DEFAULT_OOM_MEM_FLOOR_MIB
    oom_ceiling_headroom_mib: int = DEFAULT_OOM_CEILING_HEADROOM_MIB
    oom_max_requeues: int = DEFAULT_OOM_MAX_REQUEUES
    oom_own_usage_fraction: float = DEFAULT_OOM_OWN_USAGE_FRACTION
    # Temporary-failure (exit 75) handling.
    tempfail_retry_s: float = DEFAULT_TEMPFAIL_RETRY_S
    tempfail_max_requeues: int = DEFAULT_TEMPFAIL_MAX_REQUEUES
    # Pool timing.
    park_defer_max_s: float = DEFAULT_PARK_DEFER_MAX_S
    # Holding a GPU for a job that keeps losing the memory it waits for.
    reserve_after_s: float = DEFAULT_RESERVE_AFTER_S
    reserve_max_gpus: int = DEFAULT_RESERVE_MAX_GPUS
    kill_grace_s: float = DEFAULT_KILL_GRACE_S
    poll_s: float = DEFAULT_POLL_S
    gpu_wait_s: float = DEFAULT_GPU_WAIT_S
    supervise_s: float = DEFAULT_SUPERVISE_S
    orphan_claim_grace_s: float = DEFAULT_ORPHAN_CLAIM_GRACE_S
    # Utilisation sampling and the pool heartbeat (see jobq.monitor).
    monitor_interval_s: float = monitor.DEFAULT_MONITOR_INTERVAL_S
    monitor_idle_util_pct: float = monitor.DEFAULT_MONITOR_IDLE_UTIL_PCT
    monitor_idle_mem_mib: int = monitor.DEFAULT_MONITOR_IDLE_MEM_MIB
    monitor_idle_cpu_pct: float = monitor.DEFAULT_MONITOR_IDLE_CPU_PCT
    monitor_cpu_sample_s: float = monitor.DEFAULT_MONITOR_CPU_SAMPLE_S
    monitor_keep_days: float = monitor.DEFAULT_MONITOR_KEEP_DAYS
    heartbeat_stale_s: float = monitor.DEFAULT_HEARTBEAT_STALE_S
    # How many GPUs this machine has and what share of the cores each of them gets, both
    # read when the policy is read (see _machine_shape).
    machine_gpus: int | None = None
    workers_per_gpu: int = 1
    # The keys the JSON actually named, so a caller can tell "defaulted" from "asked for".
    declared: frozenset[str] = frozenset()

    @property
    def slot_units(self) -> int:
        """Slot-units probed per GPU: the cap, bounded by :data:`MAX_SLOT_UNITS_PER_GPU`.

        The cap is ``cap_per_gpu`` when the policy names one and :attr:`workers_per_gpu`
        otherwise, so a machine that names no cap still holds no more jobs per GPU than
        the cores it may use pay for.

        Every place that scans the slot locks — admission, the occupancy probe, the ask
        ledger — uses this, so the range of lock files stays finite and self-cleaning and
        the occupancy report says how many units are held.
        """
        cap = self.workers_per_gpu if self.cap_per_gpu is None else self.cap_per_gpu
        return max(1, min(cap, MAX_SLOT_UNITS_PER_GPU))

    @property
    def min_mem_ask_mib(self) -> int:
        """Smallest ask this machine grants; 0 when the policy names ``cap_per_gpu``.

        Where the policy names a cap, the number of jobs on a GPU is that cap's business
        and an ask of zero is a job that says it needs nothing. Where it does not, memory
        is what the machine is meant to be scheduled by, and an ask of zero passes the
        gate however full the GPU is, so ``free_mem_mib`` stands in for it.
        """
        return 0 if self.cap_per_gpu is not None else self.free_mem_mib

    def tunables(self) -> Tunables:
        """This policy's worker-facing knobs as a :class:`Tunables`."""
        return Tunables(
            oom_mem_factor=self.oom_mem_factor,
            oom_mem_floor_mib=self.oom_mem_floor_mib,
            oom_ceiling_headroom_mib=self.oom_ceiling_headroom_mib,
            oom_max_requeues=self.oom_max_requeues,
            oom_own_usage_fraction=self.oom_own_usage_fraction,
            tempfail_retry_s=self.tempfail_retry_s,
            tempfail_max_requeues=self.tempfail_max_requeues,
            park_defer_max_s=self.park_defer_max_s,
            kill_grace_s=self.kill_grace_s,
            orphan_claim_grace_s=self.orphan_claim_grace_s,
        )

    @classmethod
    def from_dict(cls, d: dict) -> GpuPolicy:
        if not isinstance(d, dict):
            raise PolicyError(f"gpu_policy must be a JSON object (got {type(d).__name__})")
        cap = _cap_per_gpu(d)
        cores = _usable_cores()
        gpus = _gpu_list(d)
        machine_gpus = machine_gpu_count()
        workers_per_gpu = _workers_per_gpu(
            cores=cores,
            reserve=_number(d, "cpu_reserve", 2, low=0, cast=int),
            gpus=machine_gpus if machine_gpus else len(gpus),
            cores_per_job=_positive(d, "cpu_per_gpu_job", 1.0),
        )
        return cls(
            gpus=gpus,
            cap_per_gpu=cap,
            free_mem_mib=_free_mem_mib(d, cap),
            cpu_cap=_cpu_cap(d),
            cores=cores,
            cpu_reserve=_number(d, "cpu_reserve", 2, low=0, cast=int),
            cpu_per_gpu_job=_positive(d, "cpu_per_gpu_job", 1.0),
            cpu_cap_explicit=d.get("cpu_cap") is not None,
            reserve_mem_mib=_non_negative(d, "reserve_mem_mib", 0),
            shared_perms=_bool(d, "shared_perms", False),
            cwd_fallback=_opt_path(d, "cwd_fallback"),
            oom_patterns=_oom_patterns(d),
            env=_env_map(d),
            yield_to_foreign=_bool(d, "yield_to_foreign", False),
            yield_poll_s=_number(d, "yield_poll_s", 30.0, low=0.0, low_exclusive=True),
            yield_confirm_polls=_number(d, "yield_confirm_polls", 2, low=1, cast=int),
            yield_min_foreign_procs=_number(
                d, "yield_min_foreign_procs", 1, low=1, cast=int
            ),
            yield_cooldown_s=_number(d, "yield_cooldown_s", 900.0, low=0.0),
            yield_action=_yield_action(d),
            yield_drain_threshold=_number(
                d, "yield_drain_threshold", 0.8, low=0.0, high=1.0
            ),
            yield_drain_max_s=_number(d, "yield_drain_max_s", 1800.0, low=0.0),
            yield_progress_regex=_yield_progress_regex(d),
            yield_to_uids=_resolve_yield_uids(d),
            mps=_mps_mode(d),
            mps_pipe_dir=_opt_path(d, "mps_pipe_dir"),
            mps_log_dir=_opt_path(d, "mps_log_dir"),
            mem_budget_mib=_non_negative(d, "mem_budget_mib", 0),
            startup_hold_s=_number(d, "startup_hold_s", DEFAULT_STARTUP_HOLD_S, low=0.0),
            mem_checks=_number(d, "mem_checks", DEFAULT_MEM_CHECKS, low=1, cast=int),
            mem_interval_s=_number(d, "mem_interval_s", DEFAULT_MEM_INTERVAL_S, low=0.0),
            mem_fastpath_factor=_number(
                d, "mem_fastpath_factor", DEFAULT_MEM_FASTPATH_FACTOR, low=0.0
            ),
            oom_mem_factor=_number(d, "oom_mem_factor", DEFAULT_OOM_MEM_FACTOR, low=1.0),
            oom_mem_floor_mib=_number(
                d, "oom_mem_floor_mib", DEFAULT_OOM_MEM_FLOOR_MIB, low=1, cast=int
            ),
            oom_ceiling_headroom_mib=_number(
                d, "oom_ceiling_headroom_mib", DEFAULT_OOM_CEILING_HEADROOM_MIB, low=0,
                cast=int,
            ),
            oom_max_requeues=_number(
                d, "oom_max_requeues", DEFAULT_OOM_MAX_REQUEUES, low=0, cast=int
            ),
            oom_own_usage_fraction=_number(
                d, "oom_own_usage_fraction", DEFAULT_OOM_OWN_USAGE_FRACTION,
                low=0.0, high=1.0,
            ),
            tempfail_retry_s=_number(
                d, "tempfail_retry_s", DEFAULT_TEMPFAIL_RETRY_S, low=0.0
            ),
            tempfail_max_requeues=_number(
                d, "tempfail_max_requeues", DEFAULT_TEMPFAIL_MAX_REQUEUES, low=0, cast=int
            ),
            park_defer_max_s=_number(
                d, "park_defer_max_s", DEFAULT_PARK_DEFER_MAX_S, low=0.0
            ),
            reserve_after_s=_number(
                d, "reserve_after_s", DEFAULT_RESERVE_AFTER_S, low=0.0
            ),
            reserve_max_gpus=_number(
                d, "reserve_max_gpus", DEFAULT_RESERVE_MAX_GPUS, low=0, cast=int
            ),
            kill_grace_s=_number(d, "kill_grace_s", DEFAULT_KILL_GRACE_S, low=0.0),
            poll_s=_number(d, "poll_s", DEFAULT_POLL_S, low=0.0, low_exclusive=True, high=3600.0),
            gpu_wait_s=_number(d, "gpu_wait_s", DEFAULT_GPU_WAIT_S, low=0.0, low_exclusive=True, high=3600.0),
            supervise_s=_number(
                d, "supervise_s", DEFAULT_SUPERVISE_S, low=0.0, low_exclusive=True, high=3600.0
            ),
            orphan_claim_grace_s=_number(
                d, "orphan_claim_grace_s", DEFAULT_ORPHAN_CLAIM_GRACE_S, low=0.0
            ),
            monitor_interval_s=_number(
                d, "monitor_interval_s", monitor.DEFAULT_MONITOR_INTERVAL_S, low=0.0
            ),
            monitor_idle_util_pct=_number(
                d, "monitor_idle_util_pct", monitor.DEFAULT_MONITOR_IDLE_UTIL_PCT,
                low=0.0, high=100.0,
            ),
            monitor_idle_mem_mib=_number(
                d, "monitor_idle_mem_mib", monitor.DEFAULT_MONITOR_IDLE_MEM_MIB,
                low=0, cast=int,
            ),
            monitor_idle_cpu_pct=_number(
                d, "monitor_idle_cpu_pct", monitor.DEFAULT_MONITOR_IDLE_CPU_PCT,
                low=0.0, high=100.0,
            ),
            monitor_cpu_sample_s=_number(
                d, "monitor_cpu_sample_s", monitor.DEFAULT_MONITOR_CPU_SAMPLE_S, low=0.0
            ),
            monitor_keep_days=_number(
                d, "monitor_keep_days", monitor.DEFAULT_MONITOR_KEEP_DAYS, low=0.0
            ),
            heartbeat_stale_s=_number(
                d, "heartbeat_stale_s", monitor.DEFAULT_HEARTBEAT_STALE_S,
                low=0.0, low_exclusive=True,
            ),
            machine_gpus=machine_gpus,
            workers_per_gpu=workers_per_gpu,
            declared=frozenset(str(k) for k in d),
        )


# Every accepted ``yield_action``. "drain" is placement-only (the marker stops new work
# from landing; running jobs are left alone), the other two act on the live jobs.
YIELD_ACTIONS = ("kill", "drain", "drain_if_near_done")


class PolicyError(ValueError):
    """The policy file cannot be used: a required key is missing, or a value is wrong.

    The single type every policy read can raise, so a caller can answer one question — is
    this machine's policy usable right now — without having to know which key was read in
    which order. Every message names the key and the value that was found.
    """


def _err(key: str, value, expectation: str, extra: str = "") -> PolicyError:
    """A :class:`PolicyError` naming the key, what was expected and what was found."""
    tail = f". {extra}" if extra else ""
    return PolicyError(f"gpu_policy {key!r} must be {expectation} (got {value!r}){tail}")


def _mps_mode(d: dict) -> bool | str:
    """The ``mps`` knob: ``true``, ``false`` or ``"auto"``, which is what a missing key means."""
    raw = d.get("mps")
    if raw is None:
        return MPS_AUTO
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.lower() == MPS_AUTO:
        return MPS_AUTO
    raise _err("mps", raw, f'true, false or "{MPS_AUTO}"')


def _bool(d: dict, key: str, default: bool) -> bool:
    """A boolean policy knob: only true and false, never a number or a string."""
    raw = d.get(key)
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise _err(key, raw, "true or false")
    return raw


def _usable_cores() -> int:
    """Cores this process may run on: its CPU affinity, falling back to the core count."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 4)


def _yield_action(d: dict) -> str:
    """The policy's ``yield_action``, validated: a typo must not read silently as "drain"."""
    raw = d.get("yield_action")
    if raw is None:
        return "kill"
    if not isinstance(raw, str) or raw not in YIELD_ACTIONS:
        raise _err("yield_action", raw, f"one of {', '.join(YIELD_ACTIONS)}")
    return raw


def _yield_progress_regex(d: dict) -> str | None:
    """The policy's ``yield_progress_regex``: a compiling pattern with two capture groups.

    Two groups are what the progress estimate reads (current and total), so a pattern
    without them can never produce a reading and the key would do nothing at all.
    """
    raw = d.get("yield_progress_regex")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise _err("yield_progress_regex", raw, "a string")
    try:
        compiled = re.compile(raw)
    except re.error as exc:
        raise _err("yield_progress_regex", raw, "a pattern that compiles", str(exc)) from exc
    if compiled.groups < 2:
        raise _err(
            "yield_progress_regex",
            raw,
            "a pattern with at least two capture groups",
            "the first is read as the work done and the second as the total.",
        )
    return raw


def _cap_per_gpu(d: dict) -> int | None:
    """``cap_per_gpu``: absent or null -> no cap on the job count, else a positive integer."""
    return _number(
        d,
        "cap_per_gpu",
        None,
        low=1,
        cast=int,
        extra="leave the key out to let memory alone decide how many jobs share a GPU.",
    )


def _free_mem_mib(d: dict, cap: int | None) -> int:
    """``free_mem_mib``, required, and above zero on a machine with no ``cap_per_gpu``."""
    if d.get("free_mem_mib") is None:
        raise PolicyError(
            "gpu_policy is missing the required key 'free_mem_mib': how much free memory a "
            "GPU must show before a job starts on it, and the amount a job asks for when "
            "neither it nor its queue names one"
        )
    v = _number(d, "free_mem_mib", 0, low=0, cast=int)
    if cap is None and v <= 0:
        raise _err(
            "free_mem_mib",
            v,
            "> 0 when 'cap_per_gpu' is not set",
            "memory is then the only limit on how many jobs share a GPU.",
        )
    return v


def _gpu_list(d: dict) -> list[int]:
    """``gpus``, required: a list of distinct non-negative integers (an empty list is fine)."""
    raw = d.get("gpus")
    if raw is None:
        raise PolicyError(
            "gpu_policy is missing the required key 'gpus': the physical GPU indices this "
            "machine's pool may use, for example [0, 1]"
        )
    if not isinstance(raw, list):
        raise _err("gpus", raw, "a list of GPU indices")
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise _err("gpus", item, "a non-negative integer GPU index")
        if item in out:
            raise _err("gpus", item, "listed once")
        out.append(item)
    return out


def _cpu_cap(d: dict) -> int:
    """``cpu_cap``: absent -> a quarter of the cores, explicit 0 -> the CPU lane is off.

    The two are distinct, so a machine can refuse CPU-lane work entirely.
    """
    raw = d.get("cpu_cap")
    if raw is None:
        return max(1, _usable_cores() // 4)
    return _number(d, "cpu_cap", 0, low=0, cast=int, extra="0 disables the CPU lane.")


def _positive(d: dict, key: str, default: float) -> float:
    """A strictly-positive float policy knob: a zero divisor must never pass silently."""
    return _number(
        d,
        key,
        default,
        low=0.0,
        low_exclusive=True,
        extra="it charges or divides cores in the worker-count formula.",
    )


def _non_negative(d: dict, key: str, default: int) -> int:
    """A >= 0 integer policy knob (0 = feature off); a negative value is a config error."""
    return _number(d, key, default, low=0, cast=int)


def _number(
    d: dict,
    key: str,
    default,
    *,
    low=None,
    high=None,
    low_exclusive: bool = False,
    cast=float,
    extra: str = "",
):
    """A numeric policy knob, validated: wrong type or out of range names the key.

    ``low``/``high`` are inclusive bounds unless ``low_exclusive``. A value the policy does
    not name reads as ``default`` without validation, so a default is trusted as written.
    A boolean is refused: it is an integer subclass and never a sensible number here.
    """
    raw = d.get(key)
    if raw is None:
        return default
    kind = "an integer" if cast is int else "a number"
    if isinstance(raw, bool) or isinstance(raw, (str, list, dict)):
        raise _err(key, raw, kind, extra)
    try:
        value = cast(raw)
    except (TypeError, ValueError) as exc:
        raise _err(key, raw, kind, extra) from exc
    if low is not None and (value <= low if low_exclusive else value < low):
        bound = f"> {low}" if low_exclusive else f">= {low}"
        raise _err(key, value, bound, extra)
    if high is not None and value > high:
        raise _err(key, value, f"<= {high}", extra)
    return value


def _opt_path(d: dict, key: str) -> str | None:
    """An optional path-like policy knob: a string, or absent/empty meaning unset."""
    raw = d.get(key)
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise _err(key, raw, "a path given as a string")
    return raw


def _str_list(d: dict, key: str) -> list[str]:
    """An optional list of non-empty strings; a bare string is refused, not split."""
    raw = d.get(key)
    if raw is None:
        return []
    if isinstance(raw, str) or not isinstance(raw, list):
        raise _err(key, raw, "a list of strings", "write a single entry as a one-item list.")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise _err(key, item, "a non-empty string")
        out.append(item)
    return out


def _oom_patterns(d: dict) -> tuple[str, ...]:
    """``oom_patterns``: a list of non-empty strings; a non-empty list replaces the built-ins."""
    return tuple(_str_list(d, "oom_patterns"))


def _env_map(d: dict) -> dict[str, str]:
    """``env``: a mapping of strings, with a number accepted as the string of that number."""
    raw = d.get("env")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise _err("env", raw, "a mapping of environment names to values")
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k:
            raise _err("env", k, "a non-empty environment variable name")
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise _err("env", {k: v}, "a string value")
        out[k] = str(v)
    return out


def _resolve_yield_uids(d: dict) -> tuple[int, ...]:
    """``yield_to_uids`` (ints) together with ``yield_to_users`` (names resolved via ``pwd``).

    A name this machine does not know is skipped: the same policy may be copied between
    machines, and a name that exists on only some of them is not an error here.
    """
    import pwd

    raw = d.get("yield_to_uids")
    if raw is not None and (isinstance(raw, str) or not isinstance(raw, list)):
        raise _err("yield_to_uids", raw, "a list of numeric user ids")
    uids: set[int] = set()
    for u in raw or []:
        if isinstance(u, bool) or not isinstance(u, int) or u < 0:
            raise _err("yield_to_uids", u, "a non-negative numeric user id")
        uids.add(u)
    for name in _str_list(d, "yield_to_users"):
        try:
            uids.add(pwd.getpwnam(name).pw_uid)
        except KeyError:
            continue
    return tuple(sorted(uids))


def _workers_per_gpu(*, cores: int, reserve: int, gpus: int, cores_per_job: float) -> int:
    """A GPU's share of the cores, in jobs: ``(cores - reserve) / (gpus * cores_per_job)``.

    The cores of a machine are shared with everyone on it, so the share is divided by
    every GPU the machine has, not only by the GPUs this policy selects: a pool that has
    two of eight GPUs must not size itself as though the whole machine were its own. The
    answer is never below one, since a share that rounds to nothing would run no job at all.
    """
    return max(1, int((cores - reserve) // (max(1, gpus) * cores_per_job)))


@dataclass(frozen=True)
class WorkerCount:
    """The number of worker threads a pool starts with, and every input behind it."""

    workers: int
    cores: int
    reserve: int
    cores_per_job: float
    machine_gpus: int | None
    policy_gpus: int
    workers_per_gpu: int


def default_worker_count(policy: GpuPolicy | None) -> WorkerCount:
    """How many worker threads to run when the caller names no number.

    A GPU's share of the cores (:func:`_workers_per_gpu`) times the GPUs this machine's
    policy selects, so a pool runs as many jobs at once as its GPUs and its share of the
    machine pay for. GPUs currently yielded to another user still count: the pool keeps
    the threads that GPU's work needs and admits them again when it comes back.

    A policy that selects no GPU at all runs only jobs that declare ``slots: 0``, so its
    thread count is the ``cpu_cap`` budget those jobs are admitted against — and a
    policy that selects no GPU and sets ``cpu_cap`` to zero allows no work at all, which
    is a thread count of zero rather than one thread that can never be admitted
    anything. ``jobq work`` refuses to start on a machine with no policy file, so a pool
    always passes one here; ``None`` answers for a caller asking what a machine would do
    before it has a policy, and is one thread per usable core less the default reserve.

    The cores are the ones this process may run on, so a pool confined to part of a
    machine sizes itself to that part. A job is charged ``cpu_per_gpu_job`` cores.
    """
    cores = _usable_cores() if policy is None else policy.cores
    reserve = 2 if policy is None else policy.cpu_reserve
    per_job = 1.0 if policy is None else policy.cpu_per_gpu_job
    if policy is None:
        per_gpu = max(1, int((cores - reserve) // per_job))
        return WorkerCount(per_gpu, cores, reserve, per_job, None, 0, per_gpu)
    per_gpu = policy.workers_per_gpu
    workers = per_gpu * len(policy.gpus) if policy.gpus else max(0, policy.cpu_cap)
    return WorkerCount(
        workers=workers if workers <= 0 else max(1, workers),
        cores=cores,
        reserve=reserve,
        cores_per_job=per_job,
        machine_gpus=policy.machine_gpus,
        policy_gpus=len(policy.gpus),
        workers_per_gpu=per_gpu,
    )


def policy_path(root: Path, hostname: str) -> Path:
    return Path(root) / f"gpu_policy.{hostname}.json"


def load_policy(root: Path, hostname: str) -> GpuPolicy:
    """Load this machine's GPU policy.

    Raises:
        FileNotFoundError: there is no policy file for this machine.
        PolicyError: the file exists but cannot be read, is not JSON, or names a value
            this code cannot use. That one type covers every such case, so a caller only
            has to decide what to do about an unusable policy, not about each way of
            being unusable.
    """
    p = policy_path(root, hostname)
    if not p.exists():
        raise FileNotFoundError(
            f"No GPU policy for this machine: {p} is missing. Create it, e.g.\n"
            f'  {{"gpus":[4,5,6,7],"cap_per_gpu":4,"free_mem_mib":13000,'
            f'"reserve_mem_mib":0,'
            f'"env":{{"OMP_NUM_THREADS":"2"}}}}'
        )
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise PolicyError(f"gpu_policy {p} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise PolicyError(f"gpu_policy {p} could not be read: {exc}") from exc
    return GpuPolicy.from_dict(raw)


# Where prefixes derived from the queue folder live. Machine-local by design, and a
# module attribute so a caller can point it somewhere else.
LOCK_BASE_DIR = "/tmp"


def root_lock_prefix(root: Path) -> str:
    """The machine-local slot-lock prefix derived from a queue root: ``/tmp/jobq_<12 hex>``.

    Keyed on the realpath, so the same root reached relatively, absolutely or through a
    symlink lands in one namespace, while two different roots on a machine cap independently.
    A hash rather than the path itself keeps the name short and free of ``/``; it also
    makes the file private in practice, since another account's root hashes elsewhere.
    """
    digest = hashlib.sha256(os.path.realpath(str(root)).encode()).hexdigest()[:12]
    return f"{LOCK_BASE_DIR}/jobq_{digest}"


def resolve_lock_prefix(
    root: Path, hostname: str, lock_prefix: str | None = None
) -> str:
    """The slot-lock prefix for ``root`` on this machine — the single resolution all callers use.

    Precedence, highest first: an explicit ``lock_prefix`` (``--lock-prefix``), the
    ``JOBQ_LOCK_PREFIX`` env var, then the prefix derived from the queue folder
    (:func:`root_lock_prefix`). It depends on nothing a machine may lack, so a machine
    with no policy file resolves the same prefix its pool would.
    """
    if lock_prefix:
        return lock_prefix
    env = os.environ.get("JOBQ_LOCK_PREFIX")
    if env:
        return env
    return root_lock_prefix(root)


def _mps_dirs(policy: GpuPolicy) -> tuple[str, str]:
    """Resolve (pipe_dir, log_dir), defaulting to per-user /tmp paths when unset."""
    base = f"/tmp/jobq_mps_{os.getuid()}"
    pipe = policy.mps_pipe_dir or f"{base}/pipe"
    log = policy.mps_log_dir or f"{base}/log"
    return pipe, log


def mps_env(policy: GpuPolicy) -> dict[str, str]:
    """The two MPS env vars (resolved pipe/log dirs); empty dict when ``mps`` is ``false``."""
    if policy.mps is False:
        return {}
    pipe, log = _mps_dirs(policy)
    return {"CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_LOG_DIRECTORY": log}


def ensure_mps_daemon(policy: GpuPolicy) -> tuple[bool, str]:
    """Ensure a CUDA MPS control daemon is up for this user; say whether MPS is usable.

    Probes ``nvidia-cuda-mps-control`` (via ``get_server_list``); if it does not respond,
    spawns the daemon with ``-d`` and re-probes once. Anything that goes wrong — the pipe
    or log directory cannot be created, the binary is missing, the spawn fails, the probe
    never answers — gives a false first value and a reason in the second, because the pool
    has to run fine without MPS. The reason is empty when MPS is usable.
    """
    try:
        pipe, log = _mps_dirs(policy)
        Path(pipe).mkdir(parents=True, exist_ok=True)
        Path(log).mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_LOG_DIRECTORY": log}

        def _probe() -> bool:
            try:
                r = subprocess.run(
                    ["nvidia-cuda-mps-control"],
                    input="get_server_list\n",
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10,
                )
            except FileNotFoundError:
                raise
            except (subprocess.SubprocessError, OSError):
                return False
            return r.returncode == 0

        if _probe():
            return True, ""
        # Spawn the daemon with CUDA_VISIBLE_DEVICES removed from its env: a device-restricted
        # MPS server renumbers/limits the GPUs its clients can see, which breaks per-job device
        # indexing (each job sets its own CUDA_VISIBLE_DEVICES against the full physical set).
        spawn_env = dict(env)
        spawn_env.pop("CUDA_VISIBLE_DEVICES", None)
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True,
            text=True,
            env=spawn_env,
            timeout=10,
        )
        if _probe():
            return True, ""
        return False, "the CUDA MPS daemon did not come up after being started"
    except FileNotFoundError:
        return False, "nvidia-cuda-mps-control is not on this machine"
    except Exception as exc:  # noqa: BLE001 — MPS is an optimisation, never a requirement
        return False, f"the CUDA MPS daemon could not be set up ({exc})"


def _mps_control(policy: GpuPolicy, command: str, timeout: float = 10.0):
    """Send one command to this user's MPS control daemon; ``None`` if it cannot be run."""
    pipe, log = _mps_dirs(policy)
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_LOG_DIRECTORY": log}
    try:
        return subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def _mps_pipe_is_ours(policy: GpuPolicy) -> bool:
    """Whether the policy's MPS pipe directory exists and belongs to the current user.

    A pipe directory owned by somebody else addresses somebody else's daemon, and asking
    that one to quit would stop their work. An absent directory is not ours either: there
    is nothing of ours running behind it.
    """
    pipe, _log = _mps_dirs(policy)
    try:
        return os.stat(pipe).st_uid == os.getuid()
    except OSError:
        return False


def mps_daemon_running(policy: GpuPolicy) -> bool:
    """Whether a daemon of this user answers on this policy's pipe directory.

    False when the pipe directory is not this user's: that daemon is not ours to report on
    or to act upon.
    """
    if not _mps_pipe_is_ours(policy):
        return False
    r = _mps_control(policy, "get_server_list")
    return r is not None and r.returncode == 0


def stop_mps_daemon(policy: GpuPolicy) -> bool:
    """Ask this user's MPS control daemon to quit; False when none of ours answered.

    Only reaches the daemon addressed by this policy's pipe directory, and only when that
    directory belongs to the current user, so another person's daemon is left alone.
    """
    if not _mps_pipe_is_ours(policy):
        return False
    if not mps_daemon_running(policy):
        return False
    _mps_control(policy, "quit", timeout=30.0)
    return not mps_daemon_running(policy)


def query_gpu_indices() -> list[int]:
    """Physical GPU indices ``nvidia-smi`` reports on this machine (empty when it is absent)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    idx: list[int] = []
    for line in out.strip().splitlines():
        try:
            idx.append(int(line.strip()))
        except ValueError:
            continue
    return idx


def query_free_mem() -> dict[int, int]:
    """GPU index -> free memory (MiB) via ``nvidia-smi``: the only free-memory reading taken.

    The query carries a timeout because it runs inside the machine-wide acquire flock,
    where a wedged ``nvidia-smi`` would otherwise stall every worker thread on the
    machine indefinitely.
    """
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    free: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx, mem = (x.strip() for x in line.split(","))
        free[int(idx)] = int(mem)
    return free


def query_total_mem() -> dict[int, int]:
    """GPU index -> total memory (MiB) via ``nvidia-smi``, injectable like the free query."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    total: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx, mem = (x.strip() for x in line.split(","))
        total[int(idx)] = int(mem)
    return total


_TOTAL_MEM_CACHE: dict[int, int] | None = None


def total_mem(query=None) -> dict[int, int]:
    """Cached GPU index -> total memory (MiB); a GPU's capacity cannot change under us.

    An ``nvidia-smi`` failure yields an empty map (callers must treat that as "no ceiling
    known") and is not cached, so a transient failure is retried.
    """
    global _TOTAL_MEM_CACHE
    if _TOTAL_MEM_CACHE is None:
        try:
            _TOTAL_MEM_CACHE = (query or query_total_mem)()
        except (subprocess.SubprocessError, OSError, ValueError):
            return {}
    return dict(_TOTAL_MEM_CACHE)


def _clear_total_mem_cache() -> None:
    """Drop the :func:`total_mem` cache, so the next call queries again."""
    global _TOTAL_MEM_CACHE
    _TOTAL_MEM_CACHE = None


# Sentinel for "this process has not asked yet", so that "asked, and the answer was that
# the list cannot be read" is remembered rather than asked again on every policy read.
_UNREAD = object()
_MACHINE_GPU_COUNT: object = _UNREAD


def machine_gpu_count(query=None) -> int | None:
    """How many GPUs this machine has, whoever may use them; ``None`` when unreadable.

    Every GPU ``nvidia-smi`` lists, not only the ones a policy selects, because the cores
    are shared with everyone on the machine and a fair share of them is a share per GPU.
    Asked once per process, including when the answer is that the list cannot be read:
    the number of GPUs does not change under a running pool, and the policy this figure
    belongs to is re-read on every admission, which would otherwise run ``nvidia-smi``
    again each time.
    """
    global _MACHINE_GPU_COUNT
    if _MACHINE_GPU_COUNT is _UNREAD:
        found = (query or query_gpu_indices)()
        _MACHINE_GPU_COUNT = len(found) if found else None
    return _MACHINE_GPU_COUNT  # type: ignore[return-value]


def _clear_machine_gpu_count_cache() -> None:
    """Drop the :func:`machine_gpu_count` cache, so the next call queries again."""
    global _MACHINE_GPU_COUNT
    _MACHINE_GPU_COUNT = _UNREAD


def _close_lock(fd: int | None) -> None:
    """Release a lock and close its descriptor, whatever either of them does.

    The close is attempted even when the release fails: a descriptor left open is a slot
    gone until the pool restarts, while the kernel releases the lock with the descriptor
    anyway.
    """
    if fd is None:
        return
    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(fd)


def _thread_alive(tid: object) -> bool:
    """Whether a thread of this process with that identifier is still running.

    A record without an identifier is not pinned to a thread and counts as live, so
    anything written by hand or by an older shape is left alone.
    """
    if isinstance(tid, bool) or not isinstance(tid, int):
        return True
    return any(t.ident == tid for t in threading.enumerate())


def _mark_ask_released(path: str) -> None:
    """Note on a slot's ask record that its job has given the slot back.

    The record stays until its start-up window has closed (see
    :meth:`GpuManager._read_asks`), so a job that finished inside the window another
    job's footprint is measured over is still counted as having been on the GPU. It
    is removed outright when it cannot be rewritten, which only loses that one
    neighbour from the measurement.
    """
    try:
        with open(path) as fh:
            rec = json.load(fh)
        if not isinstance(rec, dict):
            raise ValueError(path)
        rec["released"] = True
        atomic_write_json(path, rec, indent=None)
    except (OSError, ValueError):
        with contextlib.suppress(OSError):
            os.unlink(path)


@dataclass
class GpuSlot:
    """Held capacity on physical GPU ``gpu``. ``release()`` drops the flocks (idempotent).

    ``_fds`` holds one lock fd per consumed slot-unit (a weighted job holds several).
    ``_group_fd`` is the additional per-cap-group slot lock when the queue declared its own
    ``cap_per_gpu`` (a stage ceiling on top of the global policy ceiling).
    """

    gpu: int | None
    slot: int
    _fds: list[int] = field(default_factory=list)
    _group_fd: int | None = None
    # The owning manager's live-fd registry (see GpuManager._live_lock_fds); release()
    # deregisters so the leak auditor knows these fds are legitimately gone. Mutations go
    # under the manager's ``_reg_lock`` so they serialize with the auditor's audit pass.
    _registry: set[int] | None = None
    _reg_lock: threading.Lock | None = None
    # Memory-ask records written beside the slot locks while this slot is held (see
    # GpuManager._write_ask). release() stamps them with the moment the slot was given
    # back rather than removing them: a GPU's committed memory is the sum over the
    # records of slots whose lock is still held, and a released record counts for
    # nothing there, but the footprint measurement still needs to know that this job was
    # granted on the GPU inside the window it is measuring over (see
    # GpuManager.measure_footprint). The record is removed once that window has closed.
    _ask_paths: list[str] = field(default_factory=list)
    # The grant this slot came from: when it happened and what the GPU showed free at
    # that moment. Together they are the baseline of the footprint measurement (see
    # GpuManager.measure_footprint), which is what a job that reports no peak of its own
    # contributes to its queue's learned memory request.
    granted_ts: float = 0.0
    free_at_grant: int = 0
    # What the job asked for, which is its share when a measurement covers several jobs.
    mem_mib: int = 0
    measured_mib: int | None = None
    # Whether the measurement was divided among jobs granted within one window.
    measured_shared: bool = False

    @property
    def is_cpu_only(self) -> bool:
        """True for a ``slots=0`` slot: it holds a CPU-budget lock, never GPU capacity."""
        return self.gpu is None

    def release(self) -> None:
        reg = self._registry
        with self._reg_lock if self._reg_lock is not None else contextlib.nullcontext():
            for fd in self._fds:
                _close_lock(fd)
                if reg is not None:
                    reg.discard(fd)
            self._fds = []
            if self._group_fd is not None and reg is not None:
                reg.discard(self._group_fd)
            _close_lock(self._group_fd)
            self._group_fd = None
            for path in self._ask_paths:
                _mark_ask_released(path)
            self._ask_paths = []


class GpuInterface(abc.ABC):
    """What the worker pool needs from a GPU manager.

    :class:`GpuManager` is the implementation for a real machine; another implementation
    subclasses this to get the same surface. Only the three placement methods are abstract
    — the rest describe optional facts about the machine and answer "nothing configured"
    by default, so an implementation only spells out what it has to say.
    """

    @abc.abstractmethod
    def acquire(
        self,
        mem_mib: int,
        *,
        slots: int = 1,
        cap_per_gpu: int | None = None,
        cap_group: str | None = None,
        gpus: tuple[int, ...] | None = None,
    ) -> GpuSlot | None:
        """Reserve capacity for one job, or return ``None`` while none is available.

        ``gpus`` restricts the job to those GPU numbers, for a queue tied to some of a
        machine's GPUs; ``None`` leaves the choice to this machine's policy. The caller
        passes it only for a queue that names GPUs.
        """

    @abc.abstractmethod
    def env(self) -> dict[str, str]:
        """Environment entries to add to every job this machine runs."""

    @abc.abstractmethod
    def default_mem_mib(self) -> int:
        """The memory ask a job inherits when neither it nor its queue names one."""

    def min_mem_ask_mib(self) -> int:
        """Smallest ask this machine grants; ``0`` when nothing has to stand in for a zero ask."""
        return 0

    def group_has_room(self, group: str | None, cap: int) -> bool:
        """Whether a cap-group slot is free on some GPU (True = no group ceiling here)."""
        return True

    def cpu_has_room(self) -> bool:
        """Whether a CPU-lane slot is free (True = no CPU ceiling here)."""
        return True

    def cwd_fallback(self) -> str | None:
        """Directory to run a job in when its own is unreachable here."""
        return None

    def oom_patterns(self) -> tuple[str, ...]:
        """Log-tail patterns classifying a failure as out-of-memory; empty = the built-ins."""
        return ()

    def mem_ceiling_mib(self) -> int:
        """Largest grantable memory ask on this machine; ``0`` means unknown (no ceiling)."""
        return 0

    def idle_mem_ceiling_mib(self) -> int:
        """Largest ask an idle GPU here could grant; ``0`` means unknown (no ceiling)."""
        return 0

    def reap_leaked_locks(self) -> list[str]:
        """Close slot locks this process holds that no live slot owns; return their paths."""
        return []

    def tunables(self) -> Tunables | None:
        """This machine's :class:`Tunables`, or ``None`` when there is no policy behind it."""
        return None

    def policy_gpus(self) -> tuple[int, ...] | None:
        """The GPUs this machine selects; ``None`` when there is no policy to read."""
        return None

    def _policy_or_none(self) -> GpuPolicy | None:
        """This machine's policy, or ``None`` when there is none or it cannot be used."""
        return None

    def reserve_for_wait(
        self,
        mem_mib: int,
        *,
        slots: int = 1,
        job_key: str,
        priority: int = 0,
        waited_s: float = 0.0,
        gpus: tuple[int, ...] | None = None,
        displaced: list | None = None,
    ) -> int | None:
        """Hold a GPU for a job that has waited too long, and say which; ``None`` = none."""
        return None

    def release_reservation(self, job_key: str | None = None) -> int | None:
        """Give back the GPU this thread holds for a waiting job; ``None`` = it held none."""
        return None

    def release_dead_reservations(self, waiting_tids=None) -> list[int]:
        """Give back the GPUs held for threads of this pool that are not waiting any more."""
        return []

    def reservation_holder(self, g: int) -> str | None:
        """The job a GPU is held for, or ``None`` when it is held for none."""
        return None

    def reservations(self) -> dict[int, dict]:
        """GPUs held for waiting jobs on this machine, by GPU index."""
        return {}

    def request_fits_somewhere(
        self, mem_mib: int, slots: int = 1, gpus: tuple[int, ...] | None = None
    ) -> bool:
        """Whether some GPU here could ever grant this request (True = nothing says no)."""
        return True

    def measure_footprint(self, slot: GpuSlot) -> int | None:
        """The memory a granted job took while it started up, when that can be measured."""
        return None


class GpuManager(GpuInterface):
    """Acquires (slot-lock + memory-gated) GPU slots for this machine.

    ``acquire(mem_mib)`` returns a :class:`GpuSlot` for the first eligible GPU, or ``None``
    if none is currently free (the worker then WAITs and retries). Round-robin starts from a
    pid-derived offset so co-launched workers spread out without needing shared state.
    """

    def __init__(
        self,
        root: Path,
        hostname: str,
        *,
        lock_prefix: str | None = None,  # resolved via resolve_lock_prefix when absent
        mem_query=query_free_mem,
        total_query=None,
        mem_checks: int | None = None,
        mem_interval_s: float | None = None,
        mem_fastpath_factor: float | None = None,
        clock=time.time,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        # Resolved once, for this manager's lifetime (see resolve_lock_prefix): the rest of
        # the policy hot-reloads, but moving a live pool's lock namespace mid-run would drop
        # every cap it is already enforcing. Slot locks only coordinate this user's own pools, and only
        # when their GPU sets overlap; co-existence with a foreign user's jobs on those GPUs
        # is handled by the free-mem gate + yield watchdog, not by locks.
        self.lock_prefix = resolve_lock_prefix(self.root, hostname, lock_prefix)
        self.mem_query = mem_query
        self.total_query = total_query
        # Each of the three gate knobs: an explicit constructor value wins, otherwise the
        # policy's (hot-reloaded on every acquire, like the rest of it).
        self.mem_checks = mem_checks
        self.mem_interval_s = mem_interval_s
        # Injectable wall clock, so a caller can age a start-up hold without sleeping.
        self.clock = clock
        # Freshest free-memory reading per GPU, written by the gate (see _mem_ok).
        self._last_free: dict[int, int] = {}
        # Highest free reading seen per GPU, which is what a GPU of this machine shows
        # when it is idle: the driver holds some of the total back, so the total itself
        # is never free (see idle_mem_ceiling_mib).
        self._free_seen: dict[int, int] = {}
        # Admission fast path: the consecutive-check settle exists for tight fits, where a
        # lagging smi reading could over-admit. When the first check shows at least
        # ``mem_fastpath_factor`` x the ask (and 4 GiB to spare) the remaining checks are
        # skipped. Each settle sleeps under the machine-wide acquire flock, so keeping it for
        # every admission would cap the whole machine at a few admissions a minute.
        # 0 disables the fast path.
        self.mem_fastpath_factor = mem_fastpath_factor
        # Leak audit state: an exception thrown between the flock and the GpuSlot
        # construction would otherwise leak slot fds and shrink the machine until a pool
        # restart. Every fd handed out inside a returned GpuSlot is registered here and
        # deregistered by GpuSlot.release(); reap_leaked_locks() closes what's left over.
        self._reg_lock = threading.Lock()
        self._live_lock_fds: set[int] = set()
        self._leak_suspects: dict[int, str] = {}
        # Serializes every short flock-to-register/close window that runs outside the
        # acquire flock (CPU-slot grabs, group_has_room probes, _probe scans) with the
        # leak auditor's snapshot+close pass: two-strike keys on (fd number, lock path)
        # and cannot distinguish descriptor generations, so without this a recurring
        # transient at a reused fd number could be closed as a "leak". The seconds-long
        # windows inside acquire() are covered by the acquire flock instead — this lock
        # must never wrap them, or CPU probes would block behind GPU admission. Ordering:
        # acquire flock -> _audit_lock -> _reg_lock (never reversed).
        self._audit_lock = threading.Lock()
        # The policy problem currently being reported, so an unusable policy is logged once
        # per outage instead of once per admission attempt. Guarded by ``_reg_lock``.
        self._policy_outage: str | None = None
        # The same for a free-memory query that will not answer: the pool retries every
        # few seconds, and a GPU that stays unreadable would otherwise fill the log.
        self._mem_outage: str | None = None

    def _policy(self) -> GpuPolicy:  # re-read every time (hot-reload of pins)
        return load_policy(self.root, self.hostname)

    def _policy_or_none(self) -> GpuPolicy | None:
        """This machine's policy, or ``None`` when there is none or it cannot be used.

        An unusable policy is reported once per outage rather than on every call: the pool
        re-reads the file on every admission, and a file left invalid for an hour would
        otherwise fill the log. A later successful read arms the report again.
        """
        try:
            policy = self._policy()
        except FileNotFoundError:
            self._policy_outage = None
            return None
        except PolicyError as exc:
            message = str(exc)
            with self._reg_lock:
                new = self._policy_outage != message
                self._policy_outage = message
            if new:
                logger.error(
                    "this machine's GPU policy cannot be used, so no job will be admitted "
                    "here until it is fixed: {}",
                    message,
                )
            return None
        with self._reg_lock:
            self._policy_outage = None
        return policy

    def _note_mem_outage(self, message: str | None) -> None:
        """Report a free-memory query that will not answer, once per outage.

        ``None`` says the last query answered, which arms the report again, so a GPU
        that goes quiet a second time is said a second time.
        """
        with self._reg_lock:
            new = message is not None and self._mem_outage != message
            self._mem_outage = message
        if new:
            logger.warning(
                "the free-memory query failed, so no job is admitted on this machine "
                "until it answers again: {}",
                message,
            )

    def env(self) -> dict[str, str]:
        """Environment entries from this machine's policy; empty when it cannot be read."""
        policy = self._policy_or_none()
        return {} if policy is None else policy.env

    def default_mem_mib(self) -> int:
        """The policy's ``free_mem_mib``; ``0`` when there is no usable policy to read."""
        policy = self._policy_or_none()
        return 0 if policy is None else policy.free_mem_mib

    def min_mem_ask_mib(self) -> int:
        """This machine's smallest grantable ask (see :attr:`GpuPolicy.min_mem_ask_mib`)."""
        policy = self._policy_or_none()
        return 0 if policy is None else policy.min_mem_ask_mib

    def mem_ceiling_mib(self) -> int:
        """Largest policy GPU's grantable memory — the highest ``mem_mib`` any job could get.

        Net of ``reserve_mem_mib``: the gate can never grant what the reserve holds back, so
        an OOM escalation clamped to this value cannot wedge a job in WAIT forever.
        ``0`` means unknown (no usable policy, or no totals readable), which callers must
        read as "no ceiling", never as "everything is over the ceiling".
        """
        policy = self._policy_or_none()
        if policy is None:
            return 0
        totals = total_mem(self.total_query)
        ceiling = max((totals.get(g, 0) for g in policy.gpus), default=0)
        return max(ceiling - policy.reserve_mem_mib, 0) if ceiling else 0

    def _note_free_seen(self, free: dict[int, int]) -> None:
        """Keep the highest free reading seen on each GPU of this machine."""
        for g, mib in free.items():
            try:
                value = int(mib)
            except (TypeError, ValueError):
                continue
            if value > self._free_seen.get(g, -1):
                self._free_seen[g] = value

    def idle_mem_ceiling_mib(self) -> int:
        """The largest request a GPU of this machine could grant while it is idle.

        An idle GPU never shows its whole total free — the driver holds some of it — so
        a request worked out from what jobs have used is bounded by what a GPU here has
        actually been seen to offer. When no reading has been taken yet the GPU's total
        less ``oom_ceiling_headroom_mib`` stands in for it. Either way what must stay
        free on the GPU comes off the top. ``0`` means unknown, which callers read as
        no ceiling.
        """
        policy = self._policy_or_none()
        if policy is None:
            return 0
        totals = total_mem(self.total_query)
        best = 0
        for g in policy.gpus:
            seen = self._free_seen.get(g)
            if seen is None:
                total = totals.get(g, 0)
                seen = total - policy.oom_ceiling_headroom_mib if total else 0
            best = max(best, seen)
        return max(best - policy.reserve_mem_mib, 0) if best > 0 else 0

    def _slot_lock_file(self, g: int, s: int) -> str:
        return f"{self.lock_prefix}_gpu{g}_slot{s}.lock"

    def _cpu_slot_lock_file(self, s: int) -> str:
        """CPU budget lives in its own lock namespace, independent of any GPU."""
        return f"{self.lock_prefix}_cpu_slot{s}.lock"

    def _take_lock(self, path: str) -> int | None:
        """Take one slot lock, creating its file; ``None`` when somebody else holds it.

        The file is created here and only here, so a slot that has never been used has no
        file at all. Nothing ever deletes these files, so an existing one whose lock is
        free is simply a slot that was used earlier.
        """
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return fd

    @staticmethod
    def _lock_is_free(path: str) -> bool:
        """Whether a slot lock is free, without creating anything.

        A slot with no file has never been taken, so it is free: asking the question must
        not leave a file behind, or a machine with no per-GPU cap would accumulate one
        file per probed slot per GPU. An existing file is answered by taking its lock
        briefly and letting it go again.
        """
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            return True
        except OSError:
            return False  # cannot tell: treat the slot as taken
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        finally:
            # The descriptor goes either way: on the free path the lock is let go again,
            # and on the taken path it was never ours to hold.
            _close_lock(fd)
        return True

    def _try_slot(self, g: int, s: int) -> int | None:
        """Take the lock of GPU slot ``s``, or ``None`` when a job on this machine holds it."""
        return self._take_lock(self._slot_lock_file(g, s))

    def _slot_is_free(self, g: int, s: int) -> bool:
        """Whether GPU slot ``s`` is free right now, without creating its file."""
        return self._lock_is_free(self._slot_lock_file(g, s))

    @staticmethod
    def _scan_indices(head: str, cap: int) -> list[int]:
        """Slot numbers to look at in one lock namespace whose file names start with ``head``.

        The cap's own range, plus every slot above it that has a lock file. A cap lowered
        while jobs are running leaves those jobs holding slots numbered at or above the
        new cap, and they are held capacity like any other: counting only the cap's range
        would admit a job onto a GPU that is already full, and would leave those jobs'
        memory asks out of the budget. Lock files are created only when a slot is taken,
        so this adds nothing for slots that were never used.
        """
        idx = set(range(max(0, cap)))
        pattern = re.compile(re.escape(head) + r"(\d+)\.lock$")
        for path in glob.glob(f"{glob.escape(head)}*.lock"):
            m = pattern.search(path)
            if m:
                idx.add(int(m.group(1)))
        return sorted(idx)

    def _gpu_slot_indices(self, g: int, policy: GpuPolicy) -> list[int]:
        """Slot numbers to look at on GPU ``g``."""
        return self._scan_indices(f"{self.lock_prefix}_gpu{g}_slot", policy.slot_units)

    def _held_count(self, paths: list[str]) -> int:
        """How many of these slot locks are held right now."""
        return sum(0 if self._lock_is_free(p) else 1 for p in paths)

    def effective_cpu_cap(self, policy: GpuPolicy) -> int:
        """How many CPU-lane slots may be held machine-wide right now.

        ``policy.cpu_cap`` itself — an explicit ``0`` means this machine admits no CPU-lane
        jobs at all (``cpu_has_room`` then reports the lane full, so the worker skips such
        queues instead of parking a thread).
        """
        return max(0, policy.cpu_cap)

    def _acquire_cpu_slot(self, policy: GpuPolicy) -> GpuSlot | None:
        """One of the machine-wide CPU slots (:meth:`effective_cpu_cap`), or None if all held.

        Deliberately independent of the GPU budget and of the memory gate: a CPU-only job
        neither consumes nor is blocked by GPU capacity, so it runs even when every GPU is
        full. Same flock mechanism as the GPU slots, so the ceiling holds across worker
        processes, not just threads.
        """
        cap = self.effective_cpu_cap(policy)
        head = f"{self.lock_prefix}_cpu_slot"
        with self._audit_lock:  # flock-to-register window must not overlap an audit pass
            if self._held_count(
                [self._cpu_slot_lock_file(s) for s in self._scan_indices(head, cap)]
            ) >= cap:
                return None  # every CPU-lane slot the cap allows is held
            for s in range(cap):
                path = self._cpu_slot_lock_file(s)
                if not self._lock_is_free(path):
                    continue  # asking costs no file; only the slot we take gets one
                fd = self._take_lock(path)
                if fd is None:
                    continue  # a peer took it between the question and the answer
                with self._reg_lock:
                    self._live_lock_fds.add(fd)
                return GpuSlot(
                    gpu=None,
                    slot=s,
                    _fds=[fd],
                    _registry=self._live_lock_fds,
                    _reg_lock=self._reg_lock,
                )
        return None

    def _try_group_slot(self, group: str, g: int, cap: int) -> int | None:
        """Grab one of ``cap`` per-GPU slots in the cap-group's own lock namespace.

        Queues sharing a ``cap_group`` must declare the same ``cap_per_gpu`` — the ceiling
        is the number of slots allowed in the namespace, so mixed values would give each
        queue a different effective ceiling in the same one. Slots held above the cap, by
        jobs that started while it was higher, count against it like any other.
        """
        if self._group_held(group, g, cap) >= cap:
            return None
        for s in range(cap):
            path = self._group_slot_lock_file(group, g, s)
            if not self._lock_is_free(path):
                continue  # asking costs no file; only the slot we take gets one
            fd = self._take_lock(path)
            if fd is not None:
                return fd
        return None

    def _group_held(self, group: str, g: int, cap: int) -> int:
        """How many slots of a cap group are held on GPU ``g``, whatever their number."""
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", group)
        head = f"{self.lock_prefix}_grp_{safe}_gpu{g}_slot"
        return self._held_count(
            [self._group_slot_lock_file(group, g, s) for s in self._scan_indices(head, cap)]
        )

    def _group_slot_lock_file(self, group: str, g: int, s: int) -> str:
        """Where one cap-group slot lock lives, with the group name made name-safe."""
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", group)
        return f"{self.lock_prefix}_grp_{safe}_gpu{g}_slot{s}.lock"

    def group_has_room(self, group: str, cap: int) -> bool:
        """Whether any policy GPU still has a free ``group`` stage slot (probe-and-release).

        Used by the worker to skip a capped queue instead of claiming a job and parking on
        it. Racy by design: room may vanish before the real acquire, and the thread then
        waits, which is rare rather than the steady state. No usable policy reports no
        room, which is the same answer admission itself gives.
        """
        policy = self._policy_or_none()
        if policy is None:
            return False
        yielded = yielding.yielded_gpus(self.root, self.hostname)
        with self._audit_lock:  # probe fds are never registered; keep audits out
            for g in policy.gpus:
                if g in yielded:
                    continue
                if self._group_held(group, g, cap) < cap:
                    return True
        return False

    def cpu_has_room(self) -> bool:
        """Whether a ``cpu_cap`` slot is currently free (probe-and-release).

        The CPU-lane twin of :meth:`group_has_room`: the worker skips a ``slots: 0`` queue
        instead of claiming a job and parking the thread on the cpu_cap budget. Without
        this, a burst of CPU-only submissions above ``cpu_cap`` wedges one pool thread per
        waiting job and starves GPU admission. Racy by design -- see group_has_room. No
        usable policy reports no room, which is the same answer admission itself gives.

        The question is answered by counting the slots that are held, exactly as
        :meth:`group_has_room` does, rather than by taking a slot and letting it go
        again: a probe that takes a slot holds the lane's last free one for as long as
        the probe lasts, so two threads probing at once can each refuse the other.
        """
        policy = self._policy_or_none()
        if policy is None:
            return False
        cap = self.effective_cpu_cap(policy)
        if cap <= 0:
            return False
        head = f"{self.lock_prefix}_cpu_slot"
        with self._audit_lock:  # probe fds are never registered; keep audits out
            held = self._held_count(
                [self._cpu_slot_lock_file(s) for s in self._scan_indices(head, cap)]
            )
        return held < cap

    def _gate(self, policy: GpuPolicy, name: str):
        """One memory-gate knob: the constructor override if given, else the policy's."""
        override = getattr(self, name)
        return getattr(policy, name) if override is None else override

    def _mem_ok(
        self,
        g: int,
        mem_mib: int,
        reserve: int = 0,
        policy: GpuPolicy | None = None,
        asks: list[dict] | None = None,
    ) -> bool:
        """Whether GPU ``g`` can take a ``mem_mib`` job and still show ``reserve`` MiB free.

        ``reserve`` is what must stay free on top of the job's own ask: this machine's policy
        ``reserve_mem_mib``. It is passed in by the caller so the policy keeps being
        re-read once per acquire (hot-reload).

        ``asks`` are the records of the jobs already holding slots on the GPU. The
        memory of a just-granted job may not be visible in a reading yet, so its ask is
        held against that reading (see :meth:`_startup_hold_mib`) — and the hold is
        worked out from the very reading the gate is about to judge, not from an earlier
        one: the gate settles for seconds between readings, and a hold computed against
        the older reading credits the GPU with memory that has been taken since.
        """
        policy = policy if policy is not None else self._policy()
        checks = self._gate(policy, "mem_checks")
        interval = self._gate(policy, "mem_interval_s")
        fastpath = self._gate(policy, "mem_fastpath_factor")
        for i in range(checks):
            if i:
                time.sleep(interval)
            try:
                free = self.mem_query().get(g, 0)
            except (subprocess.SubprocessError, OSError, ValueError):
                # A hung or broken smi (bounded by the query timeout) denies this
                # admission instead of raising through acquire while slot fds are held.
                logger.warning("nvidia-smi free-memory query failed; denying admission on GPU {}", g)
                return False
            # Freshest reading of this GPU, kept for the ask record's baseline: the
            # snapshot taken at the top of an acquire is older by however long the gate
            # settled, and an over-old baseline would over-credit the next job's hold.
            self._last_free[g] = free
            self._note_free_seen({g: free})
            hold = (
                0
                if not asks
                else self._startup_hold_mib(asks, free, policy.startup_hold_s)
            )
            headroom = free - reserve - hold
            if headroom < mem_mib:
                return False
            if (
                fastpath > 0
                and headroom >= mem_mib * fastpath
                and headroom >= mem_mib + 4096
            ):
                return True  # ample headroom: skip the settle sleep + re-check
        return True

    # ------------------- memory accounting -------------------

    def _ask_file(self, g: int, s: int) -> str:
        """Where a slot's memory-ask record lives (beside the slot lock, machine-local)."""
        return f"{self.lock_prefix}_gpu{g}_slot{s}.ask"

    def _write_ask(
        self,
        g: int,
        slots: list[int],
        mem_mib: int,
        free_now: int,
        *,
        granted_ts: float | None = None,
    ) -> list[str]:
        """Record a granted job's ask on its slots; return the files written.

        The whole ask goes on the job's first slot and zero on the rest, so a job holding
        several slot-units is counted once at its full ask. ``free_now`` is the GPU's free
        memory at the moment of the grant, the baseline the start-up hold measures the
        job's allocation against, and ``granted_ts`` is when the grant happened (the
        clock's current reading when the caller names none).

        Raises:
            OSError: a record could not be written. Everything written for this grant is
                removed first, so the caller sees either a complete set of records or
                none, and can roll the grant back.
        """
        written: list[str] = []
        stamp = float(self.clock()) if granted_ts is None else float(granted_ts)
        for i, s in enumerate(slots):
            path = self._ask_file(g, s)
            try:
                # Atomic, like every other record jobq writes: a reader takes this file
                # while admissions are in flight, and a half-written one would drop a
                # running job's ask out of the GPU's committed memory.
                atomic_write_json(
                    path,
                    {
                        "mib": int(mem_mib) if i == 0 else 0,
                        "granted_ts": stamp,
                        "free_at_grant": int(free_now),
                    },
                    indent=None,
                )
            except OSError:
                for done in written:
                    with contextlib.suppress(OSError):
                        os.unlink(done)
                raise
            written.append(path)
        return written

    def _read_asks(
        self, g: int, policy: GpuPolicy, *, include_released: bool = False
    ) -> list[dict]:
        """Ask records of the slots on GPU ``g`` whose lock is held right now.

        Heldness is what makes the ledger self-cleaning: a record whose slot is free
        belongs to a job that has finished (or to a process that died before it could
        say so) and is not counted against the GPU. Called under the acquire flock, so
        no admission of ours is in flight while it probes.

        A record of a job that gave its slot back is kept on disk until its start-up
        window has closed and then removed, because a footprint measured over that
        window has to know the job was on the GPU (see :meth:`measure_footprint`).
        ``include_released`` asks for those records as well; the memory budget and the
        start-up hold never see them, since that memory is gone from the GPU.
        """
        out: list[dict] = []
        now = float(self.clock())
        for s in self._gpu_slot_indices(g, policy):
            try:
                with open(self._ask_file(g, s)) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                rec = None
            held = not self._slot_is_free(g, s)
            if not isinstance(rec, dict):
                continue
            if held and not rec.get("released"):
                out.append(rec)
                continue
            if now - _as_number(rec.get("granted_ts"), 0.0) >= policy.startup_hold_s:
                with contextlib.suppress(OSError):
                    os.unlink(self._ask_file(g, s))
                continue
            if include_released:
                out.append(rec)
        return out

    @staticmethod
    def _ask_mib(rec: dict) -> int:
        try:
            return max(0, int(rec.get("mib", 0)))
        except (TypeError, ValueError):
            return 0

    def _startup_hold_mib(self, asks: list[dict], free_now: int, hold_s: float) -> int:
        """MiB to hold against a GPU's free memory for jobs that may not have loaded yet.

        A job's memory only appears in ``nvidia-smi`` once it has allocated it, so between
        the grant and that point its ask must be reserved or a second job is admitted into
        the same free space. The hold is what those jobs asked for minus what has already
        appeared, and what has appeared is the fall in the GPU's free memory since the
        earliest of their grants — the reading taken before any of them could allocate.
        It is one figure for the whole GPU, so jobs in their windows cannot each claim
        the same fall; measuring from the earliest grant is what lets a job that has
        finished loading stop being held while a later job is still in its window.

        The fall cannot be attributed: memory another process took looks exactly like the
        job loading, so it releases the hold early and a second job of ours can then be
        admitted into space the first one is still going to take. That is the limit of the
        method, and it is why the hold is an approximation rather than a promise. Memory
        another process released raises free and shrinks the fall, which raises the hold,
        the conservative direction. A job past ``hold_s`` leaves the set (and the baseline
        moves to the earliest job still in it), so a job that never allocates cannot
        reserve memory for the pool's lifetime.
        """
        now = float(self.clock())
        pending = []
        for rec in asks:
            try:
                granted = float(rec.get("granted_ts", 0.0))
            except (TypeError, ValueError):
                continue
            try:
                baseline = int(rec.get("free_at_grant", 0))
            except (TypeError, ValueError):
                # An unreadable baseline credits nothing (0 can only lower the fall), so
                # the ask stays held for its whole window.
                baseline = 0
            if now - granted < hold_s:
                pending.append((granted, baseline, self._ask_mib(rec)))
        if not pending:
            return 0
        total = sum(ask for _, _, ask in pending)
        # Earliest grant first; among grants at the same instant the lowest reading, which
        # credits the least.
        _, baseline, _ = min(pending, key=lambda p: (p[0], p[1]))
        return max(0, total - max(0, baseline - free_now))

    def _probe(self, policy: GpuPolicy) -> dict[int, int]:
        """Per-GPU count of slots currently held on this machine; non-destructive probe."""
        counts: dict[int, int] = {}
        with self._audit_lock:  # probe fds are never registered; keep audits out
            for g in policy.gpus:
                counts[g] = sum(
                    0 if self._slot_is_free(g, s) else 1
                    for s in self._gpu_slot_indices(g, policy)
                )
        return counts

    # ------------------- holding a GPU for a waiting job -------------------

    def _reserve_file(self, g: int) -> str:
        """Where a GPU's reservation record lives (beside the slot locks, machine-local)."""
        return f"{self.lock_prefix}_gpu{g}.reserve"

    def _reservation_is_mine(self, rec: dict) -> bool:
        """Whether this worker thread of this process made the reservation in ``rec``."""
        return (
            rec.get("pid") == os.getpid() and rec.get("tid") == threading.get_ident()
        )

    def _clear_reservation(self, g: int) -> None:
        """Remove a GPU's reservation record, whoever wrote it."""
        with contextlib.suppress(OSError):
            os.unlink(self._reserve_file(g))

    def _read_reservation(self, g: int) -> dict | None:
        """GPU ``g``'s reservation, or ``None``; a record whose pool is gone is removed.

        A reservation only means something while the pool that made it is still waiting,
        so one left behind by a pool that has ended is not a promise any more and must not
        keep the GPU out of use.
        """
        try:
            rec = json.loads(Path(self._reserve_file(g)).read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            self._clear_reservation(g)
            return None
        pid = rec.get("pid") if isinstance(rec, dict) else None
        if not isinstance(pid, int):
            self._clear_reservation(g)
            return None
        if pid != os.getpid() and (
            store.process_gone(pid, rec) or not store.pid_alive(pid)
        ):
            self._clear_reservation(g)
            return None
        return rec

    def _reservation_has_a_waiter(self, rec: dict) -> bool:
        """Whether the thread a reservation was made for is still there to be admitted.

        A GPU is held for one waiting job, and that job is one worker thread of this
        pool. A thread that ended without giving its GPU back — it crashed, or the pool
        was scaled down — leaves a record no admission can ever satisfy, and the GPU
        would take no work of this queue folder again until the pool restarted.
        """
        if rec.get("pid") != os.getpid():
            return True  # another pool's thread; only that pool can say
        return _thread_alive(rec.get("tid"))

    def reservation_holder(self, g: int) -> str | None:
        """The job GPU ``g`` is held for, or ``None`` when it is held for none."""
        rec = self._read_reservation(g)
        job = None if rec is None else rec.get("job")
        return job if isinstance(job, str) else None

    def reservations(self) -> dict[int, dict]:
        """GPUs of this machine currently held for a waiting job, by GPU index."""
        policy = self._policy_or_none()
        if policy is None:
            return {}
        out: dict[int, dict] = {}
        for g in policy.gpus:
            rec = self._read_reservation(g)
            if rec is not None:
                out[g] = rec
        return out

    @staticmethod
    def _allowed(policy: GpuPolicy, gpus: tuple[int, ...] | None) -> list[int]:
        """The policy GPUs a job may use: all of them, or the ones its queue names."""
        if gpus is None:
            return list(policy.gpus)
        wanted = set(gpus)
        return [g for g in policy.gpus if g in wanted]

    def request_fits_somewhere(
        self, mem_mib: int, slots: int = 1, gpus: tuple[int, ...] | None = None
    ) -> bool:
        """Whether a GPU open to this job could ever grant ``mem_mib``, empty of everything.

        Read from the GPUs' total memory less this machine's reserve, and from the
        budget this machine sets on what our jobs may ask for on one GPU. Totals that
        cannot be read answer yes: an unknown machine must not be reported as one where
        the job can never run.
        """
        policy = self._policy_or_none()
        if policy is None or not policy.gpus:
            return True
        totals = total_mem(self.total_query)
        open_to_it = self._allowed(policy, gpus)
        if not open_to_it:
            return False  # its queue names GPUs, and this policy selects none of them
        return any(
            self._could_ever_admit(policy, totals, g, mem_mib, slots)
            for g in open_to_it
        )

    def reserve_for_wait(
        self,
        mem_mib: int,
        *,
        slots: int = 1,
        job_key: str,
        priority: int = 0,
        waited_s: float = 0.0,
        gpus: tuple[int, ...] | None = None,
        displaced: list | None = None,
    ) -> int | None:
        """Hold one GPU for a job that keeps losing the memory it is waiting for.

        A large request can wait for ever while smaller ones take every piece of memory
        that frees up. Once a job has waited ``reserve_after_s``, the GPU where it would
        fit soonest — the most free memory after this machine's reserve, ties going to the
        GPU holding the fewest units — stops admitting any other job of this queue folder
        until this job is admitted or stops waiting. Jobs already running there are left
        alone, so the GPU frees up as they end.

        Only a job that is waiting for memory is held a GPU (see
        :meth:`_waiting_on_memory`): one that needs more than the next job to end on the
        GPU would give back. A job the next free slot there would admit is waiting for
        a turn, not for memory, and holding a whole GPU out of use for it would only
        make the queue run one job at a time.

        Only a request some GPU could grant if it held none of this queue folder's jobs
        is worth reserving for; a request no GPU can ever meet reserves nothing, since a
        GPU held out of use would never be given back.

        ``gpus`` names the GPUs the job's queue is tied to, so a GPU its tie does not
        allow is never held for it.

        A job that already holds a GPU keeps it: the answer is that GPU, whatever the
        readings say now. Nothing else of this queue folder is admitted there meanwhile,
        so the GPU drains as the jobs already on it end, and giving it back for a moment
        would hand that room straight to the next small job.

        At most ``reserve_max_gpus`` GPUs are held at once across the whole machine,
        whichever queue holds them, so the machine cannot be stalled by reservations.
        When they are all taken, the highest priority wins, then the longest wait: a job
        that beats the weakest reservation takes that GPU over there and then, and the
        job that held it goes back to waiting like any other. One GPU carries one
        reservation: the record is written under the acquire lock, over the record of the
        job whose GPU was taken, so two jobs cannot hold one GPU between them.

        ``displaced`` is a list the key of a job whose GPU was taken is appended to, for
        a caller that reports the change.

        Returns the GPU index held for this job (its own reservation if it already has
        one), or ``None`` when nothing was reserved.
        """
        policy = self._policy_or_none()
        if policy is None or not policy.gpus:
            return None
        acq_fd: int | None = None
        try:
            acq_fd = os.open(
                f"{self.lock_prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644
            )
            fcntl.flock(acq_fd, fcntl.LOCK_EX)  # no admission of ours may be in flight
            yielded = yielding.yielded_gpus(self.root, self.hostname)
            for g in policy.gpus:
                if g in yielded:
                    self._clear_reservation(g)  # a GPU we promised away holds nothing
            held = self._adopt_reservation(policy, job_key, yielded)
            if held is not None:
                return held
            if policy.reserve_after_s <= 0 or policy.reserve_max_gpus <= 0:
                return None
            if waited_s < policy.reserve_after_s:
                return None
            try:
                free = self.mem_query()
            except (subprocess.SubprocessError, OSError, ValueError):
                return None  # no readings, no decision
            self._note_free_seen(free)
            totals = total_mem(self.total_query)
            if not self._waiting_on_memory(policy, free, mem_mib, gpus, yielded):
                return None
            live: dict[int, dict] = {}
            for g in policy.gpus:  # every reservation on the machine counts against the cap
                if g in yielded:
                    continue
                rec = self._read_reservation(g)
                if rec is not None:
                    live[g] = rec
            candidates: list[tuple[int, int, int]] = []
            for g in self._allowed(policy, gpus):
                if g in yielded or not self._could_ever_admit(
                    policy, totals, g, mem_mib, slots
                ):
                    continue
                asks = self._read_asks(g, policy)
                headroom = free.get(g, 0) - policy.reserve_mem_mib
                # What the GPU would show free with none of our jobs on it.
                empty = headroom + sum(self._ask_mib(a) for a in asks)
                if empty < mem_mib:
                    continue
                candidates.append((-headroom, len(asks), g))
            if not candidates:
                return None
            candidates.sort()
            target = next((g for _h, _n, g in candidates if g not in live), None)
            if target is None or len(live) >= policy.reserve_max_gpus:
                target = self._reservation_to_take_over(
                    live, [g for _h, _n, g in candidates], priority, waited_s
                )
            if target is None:
                return None
            taken = live.get(target)
            if taken is not None and displaced is not None:
                displaced.append(taken.get("job"))
            self._write_reservation(target, mem_mib, job_key, priority, waited_s)
            return target
        finally:
            _close_lock(acq_fd)

    def _waiting_on_memory(
        self,
        policy: GpuPolicy,
        free: dict[int, int],
        mem_mib: int,
        gpus: tuple[int, ...] | None,
        yielded,
    ) -> bool:
        """Whether memory, rather than a turn, is what this job is waiting for.

        The question asked of each GPU the job may use is what the next job to end
        there would leave it: what the GPU shows free, less what must stay free on it,
        plus the largest single request held there. A job that would be admitted with
        that much is waiting for a turn — for a slot under the cap, for room in its
        stage group, or for nothing at all when the GPU would already take it — and
        holding a whole GPU for it would only make the queue run one job at a time.
        Only a job that needs more than the next free slot can give back is waiting for
        memory, and only on a GPU where that is so on every GPU it may use.

        Counting the largest request rather than the reading alone is also what keeps a
        momentary dip from qualifying a job. A GPU carrying a job larger than this one
        answers no whatever the reading does while jobs are allocating and freeing, and
        a GPU carrying jobs of this one's own size answers no as long as one of them is
        there to end. Requests are used rather than measured use for the same reason:
        the ask is recorded at admission and does not move while the job loads.
        """
        open_to_it = [g for g in self._allowed(policy, gpus) if g not in yielded]
        if not open_to_it:
            return False
        for g in open_to_it:
            headroom = free.get(g, 0) - policy.reserve_mem_mib
            biggest = max(
                (self._ask_mib(a) for a in self._read_asks(g, policy)), default=0
            )
            if headroom + biggest >= mem_mib:
                return False
        return True

    def _could_ever_admit(
        self,
        policy: GpuPolicy,
        totals: dict[int, int],
        g: int,
        mem_mib: int,
        slots: int = 1,
    ) -> bool:
        """Whether GPU ``g`` could admit the job with none of this folder's jobs on it.

        Three things have to be true of an empty GPU: it has room for the job's weight
        in slot-units, the budget this machine sets on what our jobs together may ask
        for on one GPU covers the request, and the GPU's own memory less what must
        stay free on it does too. A GPU that could never admit the job is no use to it:
        held, it would never be given back. A total that cannot be read is no evidence
        against the GPU.

        The weight is the one :meth:`acquire` would use, which is the job's ``slots``
        capped at the GPU's whole budget: a job heavier than a GPU takes the whole
        GPU rather than being refused.
        """
        if max(1, min(slots, policy.slot_units)) > policy.slot_units:
            return False
        if policy.mem_budget_mib and mem_mib > policy.mem_budget_mib:
            return False
        total = totals.get(g)
        return total is None or total - policy.reserve_mem_mib >= mem_mib

    def _adopt_reservation(
        self, policy: GpuPolicy, job_key: str, yielded: set[int] | frozenset[int]
    ) -> int | None:
        """The GPU this pool already holds for ``job_key``, taken over by this thread.

        A job that waits again — on another worker thread of the same pool, after its
        claim came back round — keeps the GPU that was held for it rather than starting
        the wait for one over. The thread is written into the record so the admission
        path knows which waiter the GPU is open to.
        """
        for g in policy.gpus:
            if g in yielded:
                continue
            rec = self._read_reservation(g)
            if rec is None or rec.get("pid") != os.getpid() or rec.get("job") != job_key:
                continue
            if not self._reservation_is_mine(rec):
                rec["tid"] = threading.get_ident()
                with contextlib.suppress(OSError):
                    atomic_write_json(self._reserve_file(g), rec)
            return g
        return None

    @staticmethod
    def _reservation_to_take_over(
        live: dict[int, dict], candidates: list[int], priority: int, waited_s: float
    ) -> int | None:
        """The reserved GPU this waiter outranks, or ``None`` when it outranks none."""
        held = [(g, rec) for g, rec in live.items() if g in candidates]
        if not held:
            return None
        gpu, rec = min(
            held,
            key=lambda item: (
                _as_number(item[1].get("priority"), 0),
                _as_number(item[1].get("waited_s"), 0.0),
            ),
        )
        theirs = (
            _as_number(rec.get("priority"), 0),
            _as_number(rec.get("waited_s"), 0.0),
        )
        return gpu if (priority, waited_s) > theirs else None

    def _write_reservation(
        self, g: int, mem_mib: int, job_key: str, priority: int, waited_s: float
    ) -> None:
        """Record that GPU ``g`` is held for this thread's waiting job."""
        rec = {
            "gpu": g,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "host": self.hostname,
            "job": job_key,
            "mem_mib": int(mem_mib),
            "priority": int(priority),
            "waited_s": float(waited_s),
            "since_utc": store.now_iso(),
            **store.process_identity(),
        }
        with contextlib.suppress(OSError):
            atomic_write_json(self._reserve_file(g), rec)

    def release_reservation(self, job_key: str | None = None) -> int | None:
        """Give back the GPU this thread holds, if it holds one; returns its index."""
        policy = self._policy_or_none()
        if policy is None:
            return None
        for g in policy.gpus:
            rec = self._read_reservation(g)
            if rec is None or not self._reservation_is_mine(rec):
                continue
            if job_key is not None and rec.get("job") != job_key:
                continue
            self._clear_reservation(g)
            return g
        return None

    def release_dead_reservations(self, waiting_tids=None) -> list[int]:
        """Give back the GPUs this pool holds for threads that are not waiting any more.

        A worker thread gives its GPU back on the way out of its wait, but a thread
        that ends without getting there — it crashed, or the pool was scaled down —
        leaves the GPU promised to nobody, and nothing of this queue folder starts
        there until the pool restarts. The supervisor asks this on every tick.

        ``waiting_tids`` are the threads that are parked waiting for capacity right now;
        a reservation held for a thread that is still alive but that is not among
        them is given back too. Left out, only a thread that has ended counts.
        """
        policy = self._policy_or_none()
        if policy is None:
            return []
        freed: list[int] = []
        for g in policy.gpus:
            rec = self._read_reservation(g)
            if rec is None or rec.get("pid") != os.getpid():
                continue
            gone = not _thread_alive(rec.get("tid"))
            if waiting_tids is not None:
                gone = gone or rec.get("tid") not in waiting_tids
            if gone:
                self._clear_reservation(g)
                freed.append(g)
        return freed

    # ------------------- what a job took while it started -------------------

    def measure_footprint(self, slot: GpuSlot) -> int | None:
        """How far a GPU's free memory fell since this slot was granted, when it can be told.

        The reading is taken at the end of the job's start-up window, while the job is
        still running. When this job is the only one of this queue folder granted on the
        GPU within a window, the whole fall is its own.

        When several of ours were granted on the GPU within one window of each other,
        their memory appears together and no reading separates them. A neighbour that
        has already finished counts too — its memory was on the GPU for part of the
        window, so leaving it out would hand this job the whole fall. The fall is then
        measured from the free reading taken at the earliest of those grants — before any
        of them could allocate — and divided among them in proportion to what they asked
        for. Such a figure is marked shared on the slot, and the queue only learns from
        shared figures once enough of them agree (see
        :func:`jobq.store.note_peak_mem`). A job whose neighbours have not finished
        allocating when its own window ends gets too small a share, which can only
        understate the need, never overstate it.

        What another user's process allocated or released on the GPU counts here too, so
        the figure is the GPU's behaviour during the window rather than a measurement of
        the job, and a job that allocates more later peaks above it.

        Returns the MiB attributed to this job, also stored on the slot, or ``None`` when
        there is nothing to say.
        """
        if slot is None or slot.gpu is None or not slot.granted_ts:
            return None
        policy = self._policy_or_none()
        if policy is None:
            return None
        window = policy.startup_hold_s
        acq_fd: int | None = None
        try:
            acq_fd = os.open(
                f"{self.lock_prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644
            )
            fcntl.flock(acq_fd, fcntl.LOCK_EX)
            try:
                free = self.mem_query().get(slot.gpu)
            except (subprocess.SubprocessError, OSError, ValueError):
                return None
            group = [
                rec
                for rec in self._read_asks(slot.gpu, policy, include_released=True)
                if abs(_as_number(rec.get("granted_ts"), 0.0) - slot.granted_ts) <= window
                and self._ask_mib(rec) > 0
            ]
        finally:
            _close_lock(acq_fd)
        if free is None:
            return None
        mine = self._ask_mib({"mib": slot.mem_mib})
        shared = len(group) > 1
        if not shared:
            baseline = int(slot.free_at_grant)
        else:
            # The reading taken at the earliest grant of the group, before any of them
            # could allocate; among grants at the same instant the lowest, which credits
            # the least.
            earliest = min(
                group,
                key=lambda rec: (
                    _as_number(rec.get("granted_ts"), 0.0),
                    _as_number(rec.get("free_at_grant"), 0),
                ),
            )
            baseline = int(_as_number(earliest.get("free_at_grant"), slot.free_at_grant))
        fall = baseline - int(free)
        if fall <= 0:
            return None
        if shared:
            total = sum(self._ask_mib(rec) for rec in group)
            if total <= 0 or mine <= 0:
                return None
            fall = fall * mine // total
            if fall <= 0:
                return None
        slot.measured_mib = fall
        slot.measured_shared = shared
        return fall

    def acquire(
        self,
        mem_mib: int,
        *,
        slots: int = 1,
        cap_per_gpu: int | None = None,
        cap_group: str | None = None,
        gpus: tuple[int, ...] | None = None,
    ) -> GpuSlot | None:
        """Try to reserve ``slots`` slot-units with ``>= mem_mib`` free, or ``None``.

        Placement is least-loaded-first: GPUs are tried in ascending order of currently-held
        slots (breadth-first spread — a small batch lands 1/GPU), with a pid-derived rotation
        as the tie-break. Caps are ceilings, not fill targets.

        ``slots`` is the job's weight against the global per-GPU budget
        (``policy.slot_units`` units): a heavy job may consume several units, all-or-nothing
        on one GPU, so e.g. with an 8-unit budget six weight-1 jobs + one weight-2 job fill
        the GPU exactly. A weight above the budget is clamped to it ("take the whole GPU")
        so the job can still run.

        ``cap_per_gpu``/``cap_group`` add a second, stage-level ceiling on top: the job must
        also win one of ``cap_per_gpu`` slots in the ``cap_group`` lock namespace on that GPU
        (queues sharing a ``cap_group`` co-cap). The group ceiling counts jobs, not units.

        ``gpus`` confines the job to the GPUs its queue is tied to: the ones both in
        that tie and in this machine's policy. A job that uses no GPU is unaffected by
        it, since it takes no GPU at all.

        ``slots=0`` declares a fully CPU-bound job. It takes a slot from the separate
        ``policy.cpu_cap`` budget in its own lock namespace, never from GPU capacity, and
        returns ``gpu=None``. Two independent budgets, because the two concerns are
        independent: charging a CPU-only job a GPU slot-unit would block a GPU job, while
        exempting it entirely would remove its only ceiling — the GPU slots double as the
        pool's admission control, so N worker threads could otherwise all run CPU jobs at
        once and thrash the box.

        Strictly opt-in: a job must declare ``slots: 0`` itself. Nothing infers CPU-ness from
        the command line, because a job that touches CUDA while claiming no GPU slot would
        silently oversubscribe the GPU.
        """
        policy = self._policy_or_none()
        if policy is None:  # no usable policy -> admit nothing here until it is fixed
            return None
        if slots == 0:
            return self._acquire_cpu_slot(policy)
        open_to_it = self._allowed(policy, gpus)
        if not open_to_it:
            return None
        need = max(1, min(slots, policy.slot_units))
        # Opening and flocking the machine-wide lock sits inside the clean-up scope: a failure
        # to take the lock must not leave its descriptor open, since the pool retries this
        # every few seconds and would run out of descriptors.
        acq_fd: int | None = None
        try:
            acq_fd = os.open(
                f"{self.lock_prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644
            )
            fcntl.flock(acq_fd, fcntl.LOCK_EX)  # serialize acquire-check-dispatch per machine
            offset = os.getpid() % len(open_to_it)
            rotated = open_to_it[offset:] + open_to_it[:offset]
            counts = self._probe(policy)
            # Sorting is stable, so the rotation above breaks ties between equal loads.
            order = sorted(rotated, key=lambda g: counts[g])
            # GPUs yielded to a foreign user are off-limits — skip them before the memory
            # snapshot (opt-in; re-read here so a flip/reclaim takes effect on the next
            # acquire). Empty set when the feature is off or the marker file is missing.
            yielded = yielding.yielded_gpus(self.root, self.hostname)
            # One snapshot up front: GPUs that are externally memory-occupied are skipped
            # without slot probing or gate sleeps (bounds time under the acquire lock).
            try:
                snapshot = self.mem_query()
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                # A hung or broken smi (bounded by the query timeout) denies this
                # admission, exactly as it does inside the memory gate, rather than
                # throwing out of acquire while the caller holds a claim.
                self._note_mem_outage(str(exc))
                return None
            self._note_mem_outage(None)
            self._note_free_seen(snapshot)
            for g in order:
                if g in yielded:
                    self._clear_reservation(g)  # a GPU promised away holds nothing
                    continue
                reserved = self._read_reservation(g)
                if (
                    reserved is not None
                    and not self._reservation_is_mine(reserved)
                    and self._reservation_has_a_waiter(reserved)
                ):
                    # Held for another waiting job: nothing of ours goes on this GPU
                    # until that job is admitted or stops waiting.
                    continue
                if snapshot.get(g, 0) < mem_mib:
                    continue
                if counts[g] + need > policy.slot_units:
                    # The GPU already holds the cap's worth of units, or more than it
                    # after a lowered cap, so nothing more goes on it until they end.
                    continue
                held: list[tuple[int, int]] = []  # (slot index, fd)
                group_fd: int | None = None
                # Any exception below (an nvidia-smi hiccup in _mem_ok, a filesystem error)
                # must roll the flocked fds back before propagating: a leaked fd is a
                # slot-unit gone until the pool dies.
                try:
                    for s in range(policy.slot_units):
                        fd = self._try_slot(g, s)
                        if fd is not None:
                            held.append((s, fd))
                            if len(held) == need:
                                break
                    if len(held) < need:  # not enough free units -> all-or-nothing rollback
                        for _, fd in held:
                            _close_lock(fd)
                        continue
                    if cap_per_gpu is not None:
                        group_fd = self._try_group_slot(
                            cap_group or "default", g, cap_per_gpu
                        )
                        if group_fd is None:  # stage ceiling reached on this GPU
                            for _, fd in held:
                                _close_lock(fd)
                            continue
                    # What this queue folder's own running jobs asked for on this GPU:
                    # a hard per-GPU budget, plus the start-up hold for the asks whose
                    # memory may not be visible in the free reading yet.
                    asks = self._read_asks(g, policy)
                    committed = sum(self._ask_mib(a) for a in asks)
                    if (
                        policy.mem_budget_mib
                        and committed + mem_mib > policy.mem_budget_mib
                    ):
                        for _, fd in held:
                            _close_lock(fd)
                        _close_lock(group_fd)
                        group_fd = None
                        continue
                    if self._mem_ok(
                        g, mem_mib, policy.reserve_mem_mib, policy, asks=asks
                    ):
                        # Late yield re-check: the memory gate sleeps (~mem_interval_s) while we
                        # hold the slot, and the watchdog can yield this GPU during that sleep.
                        # Re-read the marker fresh here (not the pre-loop snapshot) and, if the GPU
                        # became yielded, roll back exactly like the other abandon paths
                        # rather than dispatch onto a GPU we promised to vacate.
                        if g in yielding.yielded_gpus(self.root, self.hostname):
                            for _, fd in held:
                                _close_lock(fd)
                            _close_lock(group_fd)
                            continue
                        # The ask is recorded before the slot is handed out, against the
                        # free reading this admission was decided on. A crash before this
                        # point leaves no record and no held slot, since the kernel drops
                        # the flocks with the process.
                        free_at_grant = self._last_free.get(g, snapshot.get(g, 0))
                        granted_ts = float(self.clock())
                        try:
                            ask_paths = self._write_ask(
                                g,
                                [s for s, _ in held],
                                mem_mib,
                                free_at_grant,
                                granted_ts=granted_ts,
                            )
                        except OSError as exc:
                            # Granting without the record would let the next admission
                            # ignore this job's memory entirely, so the grant is undone
                            # and the job waits instead.
                            logger.warning(
                                "could not record the memory ask on gpu {} ({}); denying "
                                "this admission rather than granting it unrecorded",
                                g,
                                exc,
                            )
                            for _, fd in held:
                                _close_lock(fd)
                            _close_lock(group_fd)
                            return None
                        with self._reg_lock:
                            self._live_lock_fds.update(fd for _, fd in held)
                            if group_fd is not None:
                                self._live_lock_fds.add(group_fd)
                        return GpuSlot(
                            gpu=g,
                            slot=held[0][0],
                            _fds=[fd for _, fd in held],
                            _group_fd=group_fd,
                            _registry=self._live_lock_fds,
                            _reg_lock=self._reg_lock,
                            _ask_paths=ask_paths,
                            granted_ts=granted_ts,
                            free_at_grant=int(free_at_grant),
                            mem_mib=int(mem_mib),
                        )
                    for _, fd in held:  # not enough memory -> next GPU
                        _close_lock(fd)
                    _close_lock(group_fd)
                except Exception:
                    for _, fd in held:
                        _close_lock(fd)
                    _close_lock(group_fd)
                    raise
            return None
        finally:
            _close_lock(acq_fd)

    def reap_leaked_locks(self) -> list[str]:
        """Close slot-lock fds this process holds that no live :class:`GpuSlot` owns.

        A slot lock this process holds open without a matching entry in
        ``_live_lock_fds`` holds a slot-unit that nothing can give back, so the
        supervisor calls this to find
        such descriptors among ``/proc/self/fd`` and close them. An fd is closed only when
        the previous call saw the same (fd, path) pair unowned as well and it still
        resolves to that path, since fd numbers are reused. Returns the lock paths closed,
        for the master log.

        The pass holds the acquire flock and ``_audit_lock`` so that no acquire of this
        process is in flight: a thread inside ``acquire`` holds its slot lock before it
        registers it, and reading the fd table in that window would take a live grant for
        a leak. The wait for the flock is bounded by ``_AUDIT_FLOCK_DEADLINE_S``, because
        this runs on the supervisor thread and the memory gate can hold that flock for as
        long as ``nvidia-smi`` takes; a slow query then costs a skipped pass rather than a
        supervisor that stops respawning workers.
        """
        prefix = self.lock_prefix
        try:
            acq_fd = os.open(
                f"{prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644
            )
        except OSError:
            return []  # can't serialize with acquires -> skip this tick, never guess
        try:
            deadline = time.monotonic() + _AUDIT_FLOCK_DEADLINE_S
            while True:
                try:
                    fcntl.flock(acq_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        return []  # acquire wedged or hot; audit again next tick
                    time.sleep(0.05)
            with self._audit_lock:
                try:
                    fd_names = os.listdir("/proc/self/fd")
                except OSError:
                    return []
                seen: dict[int, str] = {}
                for s in fd_names:
                    try:
                        fd = int(s)
                        path = os.readlink(f"/proc/self/fd/{fd}")
                    except (ValueError, OSError):
                        continue
                    if not path.startswith(prefix):
                        continue
                    if not _SLOT_LOCK_SUFFIX_RE.fullmatch(path[len(prefix):]):
                        continue
                    seen[fd] = path
                closed: list[str] = []
                with self._reg_lock:
                    unowned = {
                        fd: p for fd, p in seen.items() if fd not in self._live_lock_fds
                    }
                    strikes = {
                        fd: p
                        for fd, p in unowned.items()
                        if self._leak_suspects.get(fd) == p
                    }
                    self._leak_suspects = unowned
                    for fd, path in strikes.items():
                        try:  # fd numbers get reused: close only if it still is that lock
                            if os.readlink(f"/proc/self/fd/{fd}") != path:
                                continue
                        except OSError:
                            continue
                        _close_lock(fd)
                        del self._leak_suspects[fd]
                        closed.append(path)
                return closed
        finally:
            _close_lock(acq_fd)

    def cwd_fallback(self) -> str | None:
        """This machine's ``cwd_fallback``; ``None`` when unset or the policy is unusable."""
        policy = self._policy_or_none()
        return None if policy is None else policy.cwd_fallback

    def oom_patterns(self) -> tuple[str, ...]:
        """This machine's ``oom_patterns``; empty when unset or the policy is unusable."""
        policy = self._policy_or_none()
        return () if policy is None else policy.oom_patterns

    def shared_perms(self) -> bool:
        """Whether this machine's policy asks for world-writable queue state."""
        policy = self._policy_or_none()
        return False if policy is None else policy.shared_perms

    def tunables(self) -> Tunables | None:
        """This machine's tunable knobs; ``None`` when the policy is missing or unusable."""
        policy = self._policy_or_none()
        return None if policy is None else policy.tunables()

    def reserve_mem_mib(self) -> int:
        """This machine's post-admission memory reserve; 0 when off or unreadable."""
        policy = self._policy_or_none()
        return 0 if policy is None else policy.reserve_mem_mib

    def cap_per_gpu(self) -> int | None:
        """This machine's per-GPU slot cap, or ``None`` when its policy names none."""
        policy = self._policy_or_none()
        return None if policy is None else policy.cap_per_gpu

    def policy_gpus(self) -> tuple[int, ...] | None:
        """The GPUs this machine selects; ``None`` when its policy cannot be read."""
        policy = self._policy_or_none()
        return None if policy is None else tuple(policy.gpus)

    def occupancy(self) -> dict[int, int]:
        """Best-effort per-GPU count of slots held right now (probes each slot lock).

        A slot is "held" if we cannot take its non-blocking lock (some worker on this
        machine owns it). Every held slot is counted, including one numbered at or above
        the current cap, so the number can be above the cap after it was lowered. No
        usable policy gives an empty map, so a report never fails on one.
        """
        policy = self._policy_or_none()
        return {} if policy is None else self._probe(policy)
