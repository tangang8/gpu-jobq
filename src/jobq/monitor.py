"""Utilisation sampling: GPU, CPU and slot rows written into the queue folder.

A pool knows what its own jobs are doing, but not what the machine as a whole was doing
while they ran, and nothing in the queue folder says whether a machine sat idle for a day.
This module writes three plain CSV files per machine under ``<queue folder>/monitor/`` —
one row per GPU per sample, one row per sample for the processor, and one row per sample
for the pool's slot use — so a later ``jobq usage`` can put idle GPUs and finished jobs
side by side over any window.

Beside each CSV file sits a log of the same name ending in ``.log``, one line per sample
saying the same thing in words — which GPUs were idle and what the busy ones were doing —
for a person who wants to ``tail`` a file rather than read columns. The CSV files are what
``jobq usage`` reads; the logs are read by nothing but people.

One more log, ``fleet_slots.log``, has one line per sample for every machine with a policy
file at once: the slots each live pool holds against its capacity, and which machines have
no pool running. Every sampler offers a line and the first to arrive in an interval writes
it, so the log has one line per interval however many machines sample.

The readings themselves come from ``nvidia-smi`` and from ``/proc``, both behind small
functions a caller can replace: :func:`query_gpu_samples` takes the GPU readings, and the
processor readings are read from :data:`PROC_DIR`, so a test can point them at files of
its own. Sampling runs on its own thread, so a reading that hangs or fails delays nothing
a job waits on; a failure is logged once per outage and the thread carries on.

Rows are appended, never rewritten, and a row older than the keep window is dropped once a
day, header kept. Each file has a lock file of its own beside it, so an append and the
daily trim of the same file cannot overlap. Every file is readable by anything that reads CSV, which is the point of
the format: the machine that samples and the machine that reads need share nothing but the
queue folder.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from jobq import store
from jobq.io import atomic_write_text, file_lock

# Where the processor readings are read from. A module attribute so a caller can point the
# reader at files it writes itself.
PROC_DIR = "/proc"

# The defaults of the monitoring policy keys (see docs/policy.md).
DEFAULT_MONITOR_INTERVAL_S = 300.0
DEFAULT_MONITOR_IDLE_UTIL_PCT = 5.0
DEFAULT_MONITOR_IDLE_MEM_MIB = 1024
DEFAULT_MONITOR_IDLE_CPU_PCT = 10.0
DEFAULT_MONITOR_CPU_SAMPLE_S = 5.0
DEFAULT_MONITOR_KEEP_DAYS = 30
DEFAULT_HEARTBEAT_STALE_S = 180.0

GPU_COLUMNS = (
    "timestamp",
    "gpu_index",
    "util_pct",
    "mem_used_mib",
    "mem_total_mib",
    "power_w",
    "idle",
    "our_jobs",
)
CPU_COLUMNS = (
    "timestamp",
    "ncpu",
    "util_pct",
    "busy_cores",
    "load1",
    "load5",
    "load15",
    "mem_used_mib",
    "mem_total_mib",
    "idle",
)
SLOT_COLUMNS = (
    "timestamp",
    "node",
    "used",
    "slots",
    "wait",
    "gpus",
    "yielded",
    "cap_per_gpu",
    "live",
)

# One trim pass a day is enough for a keep window measured in days, and it reads and
# rewrites the whole file, so it is deliberately rare.
TRIM_INTERVAL_S = 86400.0


def monitor_dir(root: Path) -> Path:
    """The directory holding the sample files of every machine."""
    return Path(root) / "monitor"


def gpu_csv_path(root: Path, hostname: str) -> Path:
    return monitor_dir(root) / f"gpu.{hostname}.csv"


def cpu_csv_path(root: Path, hostname: str) -> Path:
    return monitor_dir(root) / f"cpu.{hostname}.csv"


def slots_csv_path(root: Path, hostname: str) -> Path:
    return monitor_dir(root) / f"slots.{hostname}.csv"


def fleet_log_path(root: Path) -> Path:
    """The one log every machine's sampler writes: the slots of all of them, a line a sample."""
    return monitor_dir(root) / "fleet_slots.log"


def log_path(csv_path: Path) -> Path:
    """The log written beside one sample file: the same name, ending in ``.log``."""
    return Path(csv_path).with_suffix(".log")


def now_stamp() -> str:
    """The current time as ISO 8601 with the UTC offset, the stamp every row carries."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_stamp(text: object) -> datetime | None:
    """One row's timestamp as a datetime, or ``None`` when it cannot be read."""
    if not isinstance(text, str) or not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def sample_lock_path(path: Path) -> Path:
    """The lock held while one sample file is appended to or trimmed.

    Appending and trimming are not one operation: the trim reads the whole file and
    writes it back, so a row appended in between would be dropped by it. One lock per
    file, so the three files never wait for each other.
    """
    return Path(path).with_name(Path(path).name + ".lock")


def append_row(path: Path, columns: tuple[str, ...], row: dict) -> None:
    """Append one row to a CSV file, writing the header when the file is new or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(sample_lock_path(path)):
        fresh = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(columns))
            if fresh:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in columns})


def append_line(path: Path, line: str) -> None:
    """Append one line to a log, under the lock its daily trim takes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(sample_lock_path(path)), open(path, "a") as fh:
        fh.write(line + "\n")


def trim_old_lines(path: Path, keep_days: float, *, now: datetime | None = None) -> int:
    """Drop lines older than ``keep_days`` from one log. Returns the count.

    A line starts with its timestamp; one whose start cannot be read as a time is kept.
    """
    if keep_days <= 0 or not Path(path).exists():
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(days=keep_days)
    with file_lock(sample_lock_path(path)):
        try:
            lines = Path(path).read_text().splitlines()
        except OSError:
            return 0
        kept = []
        for line in lines:
            stamp = parse_stamp(line.split(" ", 1)[0])
            if stamp is None or stamp >= cutoff:
                kept.append(line)
        dropped = len(lines) - len(kept)
        if dropped:
            atomic_write_text(path, "".join(line + "\n" for line in kept))
    return dropped


def read_rows(path: Path) -> list[dict]:
    """Every row of one sample file as a dict, or an empty list when there is none."""
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    return [row for row in csv.DictReader(io.StringIO(text)) if row.get("timestamp")]


def trim_old_rows(path: Path, keep_days: float, *, now: datetime | None = None) -> int:
    """Drop rows older than ``keep_days`` from one file, keeping its header. Returns the count.

    A file with no rows left keeps its header, so a reader always finds the columns.
    """
    if keep_days <= 0 or not Path(path).exists():
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(days=keep_days)
    with file_lock(sample_lock_path(path)):
        try:
            text = Path(path).read_text()
        except OSError:
            return 0
        reader = csv.DictReader(io.StringIO(text))
        columns = tuple(reader.fieldnames or ())
        if not columns:
            return 0
        kept: list[dict] = []
        dropped = 0
        for row in reader:
            stamp = parse_stamp(row.get("timestamp"))
            if stamp is not None and stamp < cutoff:
                dropped += 1
                continue
            kept.append(row)
        if not dropped:
            return 0
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(columns))
        writer.writeheader()
        for row in kept:
            writer.writerow({c: row.get(c, "") for c in columns})
        # Written to one side and renamed into place, so a reader on any machine sees
        # either the whole file before the trim or the whole file after it.
        atomic_write_text(path, out.getvalue())
    return dropped


# --------------------------- the readings ---------------------------


@dataclass(frozen=True)
class GpuReading:
    """One GPU's reading: utilisation, memory and power as ``nvidia-smi`` reports them."""

    index: int
    util_pct: float
    mem_used_mib: int
    mem_total_mib: int
    power_w: float | None


def query_gpu_samples() -> dict[int, GpuReading]:
    """GPU index -> reading, from ``nvidia-smi``; the only GPU reading the sampler takes."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    readings: dict[int, GpuReading] = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        readings[index] = GpuReading(
            index=index,
            util_pct=_float_or(parts[1], 0.0),
            mem_used_mib=int(_float_or(parts[2], 0.0)),
            mem_total_mib=int(_float_or(parts[3], 0.0)),
            power_w=_float_or(parts[4], None),
        )
    return readings


def _float_or(text: str, default):
    """A number a GPU reading holds, or ``default`` for a field the driver leaves out."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _proc_path(name: str) -> Path:
    return Path(PROC_DIR) / name


def read_cpu_times() -> tuple[float, float] | None:
    """Busy and total processor jiffies from the ``cpu`` line of ``/proc/stat``.

    Busy is everything but idle and iowait, which is what a utilisation between two
    readings is the ratio of.
    """
    try:
        first = _proc_path("stat").read_text().splitlines()[0]
    except (OSError, IndexError):
        return None
    fields = first.split()
    if not fields or fields[0] != "cpu":
        return None
    try:
        values = [float(x) for x in fields[1:]]
    except ValueError:
        return None
    if len(values) < 5:
        return None
    total = sum(values)
    idle = values[3] + values[4]
    return total - idle, total


def cpu_utilisation(sample_s: float, *, sleep=time.sleep) -> float | None:
    """Processor utilisation as a percentage, from two readings ``sample_s`` apart.

    ``None`` when either reading cannot be taken or no time passed between them.
    """
    first = read_cpu_times()
    if first is None:
        return None
    sleep(max(0.0, sample_s))
    second = read_cpu_times()
    if second is None:
        return None
    busy = second[0] - first[0]
    total = second[1] - first[1]
    if total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * busy / total))


def read_loadavg() -> tuple[float, float, float] | None:
    """The three load averages from ``/proc/loadavg``."""
    try:
        fields = _proc_path("loadavg").read_text().split()
    except OSError:
        return None
    try:
        return float(fields[0]), float(fields[1]), float(fields[2])
    except (IndexError, ValueError):
        return None


_MEMINFO_LINE = re.compile(r"^(\w+):\s+(\d+)")


def read_meminfo() -> tuple[int, int] | None:
    """Memory used and total in MiB from ``/proc/meminfo`` (used = total less available)."""
    try:
        text = _proc_path("meminfo").read_text()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        m = _MEMINFO_LINE.match(line)
        if m:
            values[m.group(1)] = int(m.group(2))
    total_kib = values.get("MemTotal")
    available_kib = values.get("MemAvailable")
    if total_kib is None or available_kib is None:
        return None
    total = total_kib // 1024
    return max(0, total - available_kib // 1024), total


def usable_cores() -> int:
    """Cores this process may run on: its CPU affinity, falling back to the core count."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


# --------------------------- the sampler ---------------------------


@dataclass(frozen=True)
class MonitorConfig:
    """What the sampler reads out of this machine's policy."""

    interval_s: float = DEFAULT_MONITOR_INTERVAL_S
    idle_util_pct: float = DEFAULT_MONITOR_IDLE_UTIL_PCT
    idle_mem_mib: int = DEFAULT_MONITOR_IDLE_MEM_MIB
    idle_cpu_pct: float = DEFAULT_MONITOR_IDLE_CPU_PCT
    cpu_sample_s: float = DEFAULT_MONITOR_CPU_SAMPLE_S
    keep_days: float = DEFAULT_MONITOR_KEEP_DAYS


def config_from_policy(policy) -> MonitorConfig:
    """The sampler's settings from a policy, or the defaults when there is no policy."""
    if policy is None:
        return MonitorConfig()
    return MonitorConfig(
        interval_s=policy.monitor_interval_s,
        idle_util_pct=policy.monitor_idle_util_pct,
        idle_mem_mib=policy.monitor_idle_mem_mib,
        idle_cpu_pct=policy.monitor_idle_cpu_pct,
        cpu_sample_s=policy.monitor_cpu_sample_s,
        keep_days=policy.monitor_keep_days,
    )


@dataclass(frozen=True)
class SlotState:
    """What the pool's own admission holds right now, for the slots row."""

    used: int = 0
    slots: int = 0
    wait: int = 0
    gpus: int = 0
    yielded: int = 0
    cap_per_gpu: int = 0
    live: bool = True


class Sampler:
    """Appends one round of GPU, CPU and slot rows every ``interval_s``.

    The three callables are how the sampler learns what it cannot read itself: the GPU
    readings, how many of this queue folder's jobs hold a slot on each GPU, and the pool's
    own slot state. Each defaults to something a machine with no pool can answer.
    """

    def __init__(
        self,
        root: Path,
        hostname: str,
        *,
        config: MonitorConfig | None = None,
        gpus: tuple[int, ...] = (),
        gpu_query=None,
        occupancy=None,
        slot_state=None,
        sleep=None,
    ) -> None:
        self.root = Path(root)
        self.hostname = hostname
        self.config = config or MonitorConfig()
        self.gpus = tuple(gpus)
        # The readings are looked up when they are taken, not when the sampler is built,
        # so a caller that replaces one of these module functions is heard.
        self._gpu_query = gpu_query or (lambda: query_gpu_samples())
        self._occupancy = occupancy or (lambda: {})
        self._slot_state = slot_state or (lambda: SlotState(live=False))
        self._sleep = sleep or (lambda seconds: time.sleep(seconds))
        # One line per outage rather than one per sample, and the next trim pass.
        self._failing = False
        self._next_trim = 0.0

    def sample_once(self) -> None:
        """Take one round of readings and append the rows they produce.

        The processor reading is taken first, because it is the ratio between two
        readings a few seconds apart: the stamp every row of the round carries is taken
        after that wait, so it is when the round was taken rather than when it started.

        A processor reading that cannot be taken costs the round its processor row and
        nothing else. The GPU rows and the pool's slot row say what the machine and the
        pool were doing, which is the whole point of sampling, and a machine whose
        ``/proc`` cannot be read is exactly one worth having those rows for.
        """
        util = cpu_utilisation(self.config.cpu_sample_s, sleep=self._sleep)
        timestamp = now_stamp()
        self._write_gpu_rows(timestamp)
        if util is None:
            logger.warning(
                "the processor readings under {} could not be taken; this round has no "
                "processor row",
                PROC_DIR,
            )
        else:
            self._write_cpu_row(timestamp, util)
        self._write_slots_row(timestamp)
        append_fleet_line(self.root, timestamp, interval_s=self.config.interval_s)

    def _write_gpu_rows(self, timestamp: str) -> None:
        readings = self._gpu_query()
        held = self._occupancy()
        path = gpu_csv_path(self.root, self.hostname)
        words: list[str] = []
        idle_gpus = 0
        for g in self.gpus:
            reading = readings.get(g)
            if reading is None:
                continue
            idle = (
                reading.util_pct < self.config.idle_util_pct
                and reading.mem_used_mib < self.config.idle_mem_mib
            )
            jobs = int(held.get(g, 0))
            if idle:
                idle_gpus += 1
                words.append(f"{g}:idle")
            else:
                words.append(f"{g}:busy({_number_text(reading.util_pct)}%,{reading.mem_used_mib}MiB)")
            append_row(
                path,
                GPU_COLUMNS,
                {
                    "timestamp": timestamp,
                    "gpu_index": g,
                    "util_pct": _number_text(reading.util_pct),
                    "mem_used_mib": reading.mem_used_mib,
                    "mem_total_mib": reading.mem_total_mib,
                    "power_w": "" if reading.power_w is None else _number_text(reading.power_w),
                    "idle": 1 if idle else 0,
                    "our_jobs": jobs,
                },
            )
        if words:
            line = f"{timestamp} {idle_gpus}/{len(words)} idle | {' '.join(words)}"
        elif self.gpus:
            line = f"{timestamp} no reading for any GPU in this machine's policy"
        else:
            return  # no GPUs to sample, so nothing to say about them
        append_line(log_path(path), line)

    def _write_cpu_row(self, timestamp: str, util: float) -> None:
        ncpu = usable_cores()
        load = read_loadavg() or (0.0, 0.0, 0.0)
        mem_used, mem_total = read_meminfo() or (0, 0)
        path = cpu_csv_path(self.root, self.hostname)
        idle = util < self.config.idle_cpu_pct
        busy_cores = round(util * ncpu / 100.0)
        append_row(
            path,
            CPU_COLUMNS,
            {
                "timestamp": timestamp,
                "ncpu": ncpu,
                "util_pct": _number_text(util),
                "busy_cores": busy_cores,
                "load1": _number_text(load[0]),
                "load5": _number_text(load[1]),
                "load15": _number_text(load[2]),
                "mem_used_mib": mem_used,
                "mem_total_mib": mem_total,
                "idle": 1 if idle else 0,
            },
        )
        append_line(
            log_path(path),
            f"{timestamp} {'idle' if idle else 'busy'} | util {_number_text(util)}% "
            f"(~{busy_cores}/{ncpu} cores) load "
            f"{'/'.join(_number_text(value) for value in load)} "
            f"mem {mem_used}/{mem_total}MiB",
        )

    def _write_slots_row(self, timestamp: str) -> None:
        state = self._slot_state()
        path = slots_csv_path(self.root, self.hostname)
        append_row(
            path,
            SLOT_COLUMNS,
            {
                "timestamp": timestamp,
                "node": self.hostname,
                "used": state.used,
                "slots": state.slots,
                "wait": state.wait,
                "gpus": state.gpus,
                "yielded": state.yielded,
                "cap_per_gpu": state.cap_per_gpu,
                "live": 1 if state.live else 0,
            },
        )
        append_line(
            log_path(path),
            f"{timestamp} {state.used}/{state.slots} slots used, {state.wait} waiting | "
            f"{state.gpus} gpus, {state.yielded} yielded, cap {state.cap_per_gpu} per gpu | "
            f"{'pool live' if state.live else 'no pool'}",
        )

    def maybe_trim(self, *, now: float | None = None) -> None:
        """Drop rows past the keep window, at most once a day."""
        clock = time.monotonic() if now is None else now
        if clock < self._next_trim:
            return
        self._next_trim = clock + TRIM_INTERVAL_S
        for path in (
            gpu_csv_path(self.root, self.hostname),
            cpu_csv_path(self.root, self.hostname),
            slots_csv_path(self.root, self.hostname),
        ):
            trim_old_rows(path, self.config.keep_days)
            trim_old_lines(log_path(path), self.config.keep_days)
        trim_old_lines(fleet_log_path(self.root), self.config.keep_days)

    def tick(self) -> None:
        """One pass: sample, trim, and swallow whatever went wrong, logged once per outage."""
        try:
            self.sample_once()
            self.maybe_trim()
        except Exception as exc:  # noqa: BLE001 — a reading must never end the thread
            if not self._failing:
                self._failing = True
                logger.warning("utilisation sampling failed and will be retried: {}", exc)
            return
        if self._failing:
            self._failing = False
            logger.info("utilisation sampling is working again")

    def run(self, stop: threading.Event) -> None:
        """Sample every ``interval_s`` until ``stop`` is set; an interval of 0 returns at once."""
        if self.config.interval_s <= 0:
            return
        while True:
            self.tick()
            if stop.wait(self.config.interval_s):
                return


# --------------------------- the fleet log ---------------------------

# A machine's latest slots row speaks for it while it is no older than this many of the
# sampling intervals; past that the machine's pool is alive but is not sampling.
FLEET_ROW_FRESH_INTERVALS = 3.0
# A second sampler arriving within this share of an interval of the last fleet line adds
# nothing: the line it would write describes the same moment.
FLEET_LINE_MIN_GAP = 0.5


def _heartbeat_stale_s(root: Path, hostname: str) -> float:
    """How old a machine's heartbeat may be, from its policy file, or the default."""
    try:
        raw = json.loads((Path(root) / f"gpu_policy.{hostname}.json").read_text())
        value = float(raw.get("heartbeat_stale_s", DEFAULT_HEARTBEAT_STALE_S))
    except (OSError, ValueError, TypeError, AttributeError):
        return DEFAULT_HEARTBEAT_STALE_S
    return value if value > 0 else DEFAULT_HEARTBEAT_STALE_S


def _int_or_zero(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return 0


def fleet_cell(
    root: Path, hostname: str, *, now: datetime, fresh_s: float
) -> tuple[str, int, int]:
    """One machine's part of a fleet line, and the used and total slots it adds to the sum.

    A machine whose pool has no fresh heartbeat is down and offers no slots. One whose
    pool is alive is described by the last slots row it wrote, when that row is recent.
    """
    if not store.heartbeat_state(root, hostname, _heartbeat_stale_s(root, hostname))["fresh"]:
        return f"{hostname} 0/0 (down)", 0, 0
    row = latest_slots_row(root, hostname)
    stamp = parse_stamp(row.get("timestamp")) if row else None
    if row is None or stamp is None or (now - stamp).total_seconds() > fresh_s:
        return f"{hostname} ?/? (no samples)", 0, 0
    used, slots = _int_or_zero(row, "used"), _int_or_zero(row, "slots")
    cell = f"{hostname} {used}/{slots}"
    yielded, wait = _int_or_zero(row, "yielded"), _int_or_zero(row, "wait")
    if yielded:
        cell += f" ({yielded} yielded)"
    if wait:
        cell += f" +{wait}w"
    return cell, used, slots


def fleet_line(root: Path, timestamp: str, *, interval_s: float) -> str | None:
    """The fleet line for one moment, or ``None`` when no machine has a policy file."""
    hosts = machines_with_a_policy(root)
    if not hosts:
        return None
    now = parse_stamp(timestamp) or datetime.now(UTC)
    fresh_s = max(interval_s, 1.0) * FLEET_ROW_FRESH_INTERVALS
    cells, total_used, total_slots = [], 0, 0
    for host in hosts:
        cell, used, slots = fleet_cell(root, host, now=now, fresh_s=fresh_s)
        cells.append(cell)
        total_used += used
        total_slots += slots
    return f"{timestamp} {total_used}/{total_slots} | {' | '.join(cells)}"


def append_fleet_line(root: Path, timestamp: str, *, interval_s: float) -> bool:
    """Append this moment's fleet line unless another sampler just wrote one. True if written.

    Every machine's sampler calls this each round. The check and the append happen under
    the log's lock, so of the samplers arriving together exactly one writes.
    """
    line = fleet_line(root, timestamp, interval_s=interval_s)
    if line is None:
        return False
    path = fleet_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = parse_stamp(timestamp) or datetime.now(UTC)
    with file_lock(sample_lock_path(path)):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            lines = []
        last = parse_stamp(lines[-1].split(" ", 1)[0]) if lines else None
        if last is not None and (now - last).total_seconds() < interval_s * FLEET_LINE_MIN_GAP:
            return False
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return True


def _number_text(value: float) -> str:
    """A reading as text: whole numbers without a decimal point, the rest with one."""
    return str(int(value)) if float(value).is_integer() else f"{float(value):.1f}"


# --------------------------- reading it back ---------------------------


def machines_with_a_policy(root: Path) -> list[str]:
    """Every machine that has a policy file in the queue folder, in name order."""
    names = []
    for path in sorted(Path(root).glob("gpu_policy.*.json")):
        name = path.name[len("gpu_policy.") : -len(".json")]
        if name:
            names.append(name)
    return names


def _within(rows: list[dict], cutoff: datetime | None) -> list[dict]:
    """The rows whose timestamp is at or after ``cutoff`` (all of them when there is none)."""
    if cutoff is None:
        return rows
    out = []
    for row in rows:
        stamp = parse_stamp(row.get("timestamp"))
        if stamp is not None and stamp >= cutoff:
            out.append(row)
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numbers(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key, "")))
        except (TypeError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class GpuUsage:
    """What one GPU did over the window."""

    gpu_index: int
    util_pct: float | None
    idle_share: float | None
    unused_share: float | None
    mem_used_mib: float | None
    samples: int

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "util_pct": self.util_pct,
            "idle_share": self.idle_share,
            "unused_share": self.unused_share,
            "mem_used_mib": self.mem_used_mib,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class MachineUsage:
    """What one machine did over the window, and when it was last heard from."""

    hostname: str
    gpus: list[GpuUsage]
    cpu_util_pct: float | None
    slots_used: float | None
    slots_capacity: float | None
    last_sample_utc: str | None
    last_sample_age_s: float | None
    jobs_per_hour: float | None
    jobs_done: int

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname,
            "gpus": [g.to_dict() for g in self.gpus],
            "cpu_util_pct": self.cpu_util_pct,
            "slots_used": self.slots_used,
            "slots_capacity": self.slots_capacity,
            "last_sample_utc": self.last_sample_utc,
            "last_sample_age_s": self.last_sample_age_s,
            "jobs_per_hour": self.jobs_per_hour,
            "jobs_done": self.jobs_done,
        }


def jobs_finished_per_machine(root: Path, cutoff: datetime | None) -> dict[str, int]:
    """Successful results per machine whose end time falls in the window.

    A queue folder can hold hundreds of thousands of results, nearly all of them older
    than any window worth asking about, and reading every one of them to find the recent
    few is what makes ``jobq usage`` slow enough to look broken. A result is written once
    and never touched again, so its file's modification time cannot be older than the
    moment it records: a file the filesystem says predates the window is skipped without
    being read, and so is a whole queue whose results directory has not changed since
    before it.
    """
    counts: dict[str, int] = {}
    cutoff_epoch = None if cutoff is None else cutoff.timestamp()
    for name in store.list_queues(root):
        rdir = store.queue_dir(root, name) / "results"
        try:
            if cutoff_epoch is not None and rdir.stat().st_mtime < cutoff_epoch:
                continue  # nothing has finished in this queue since the window began
            with os.scandir(rdir) as it:
                entries = [e for e in it if e.name.endswith(".json")]
        except OSError:
            continue
        for entry in entries:
            try:
                if cutoff_epoch is not None and entry.stat().st_mtime < cutoff_epoch:
                    continue
            except OSError:
                continue
            state, rec = store.read_json_dict(Path(entry.path))
            if state != "ok" or rec.get("rc") != 0:
                continue
            stamp = parse_stamp(rec.get("end_utc"))
            if cutoff is not None and (stamp is None or stamp < cutoff):
                continue
            node = rec.get("node") or "an unrecorded machine"
            counts[node] = counts.get(node, 0) + 1
    return counts


def machine_usage(
    root: Path,
    hostname: str,
    *,
    cutoff: datetime | None,
    window_s: float | None,
    jobs_done: int,
    now: datetime | None = None,
) -> MachineUsage:
    """One machine's row of the usage table, over the window."""
    moment = now or datetime.now(UTC)
    gpu_rows = _within(read_rows(gpu_csv_path(root, hostname)), cutoff)
    cpu_rows = _within(read_rows(cpu_csv_path(root, hostname)), cutoff)
    slot_rows = _within(read_rows(slots_csv_path(root, hostname)), cutoff)
    per_gpu: dict[int, list[dict]] = {}
    for row in gpu_rows:
        try:
            index = int(row.get("gpu_index", ""))
        except (TypeError, ValueError):
            continue
        per_gpu.setdefault(index, []).append(row)
    gpus = [
        GpuUsage(
            gpu_index=index,
            util_pct=_mean(_numbers(rows, "util_pct")),
            idle_share=_mean(_numbers(rows, "idle")),
            unused_share=_mean([1.0 if v == 0 else 0.0 for v in _numbers(rows, "our_jobs")]),
            mem_used_mib=_mean(_numbers(rows, "mem_used_mib")),
            samples=len(rows),
        )
        for index, rows in sorted(per_gpu.items())
    ]
    stamps = [
        parse_stamp(row.get("timestamp"))
        for row in (*gpu_rows, *cpu_rows, *slot_rows)
    ]
    latest = max([s for s in stamps if s is not None], default=None)
    return MachineUsage(
        hostname=hostname,
        gpus=gpus,
        cpu_util_pct=_mean(_numbers(cpu_rows, "util_pct")),
        slots_used=_mean(_numbers(slot_rows, "used")),
        slots_capacity=_mean(_numbers(slot_rows, "slots")),
        last_sample_utc=latest.isoformat(timespec="seconds") if latest else None,
        last_sample_age_s=(moment - latest).total_seconds() if latest else None,
        jobs_per_hour=(jobs_done / (window_s / 3600.0)) if window_s else None,
        jobs_done=jobs_done,
    )


def usage_report(
    root: Path, *, window_s: float | None = None, now: datetime | None = None
) -> list[MachineUsage]:
    """The usage of every machine that has a policy file, over the window."""
    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(seconds=window_s) if window_s else None
    finished = jobs_finished_per_machine(root, cutoff)
    return [
        machine_usage(
            root,
            host,
            cutoff=cutoff,
            window_s=window_s,
            jobs_done=finished.get(host, 0),
            now=moment,
        )
        for host in machines_with_a_policy(root)
    ]


def recent_gpu_summary(
    root: Path, hostname: str, *, window_s: float, now: datetime | None = None
) -> tuple[float | None, int, int]:
    """Mean utilisation, idle GPUs and GPUs seen, over the last ``window_s`` of samples.

    A GPU counts as idle when its most recent sample in the window says so.
    """
    moment = now or datetime.now(UTC)
    rows = _within(read_rows(gpu_csv_path(root, hostname)), moment - timedelta(seconds=window_s))
    latest: dict[int, dict] = {}
    for row in rows:
        try:
            latest[int(row.get("gpu_index", ""))] = row
        except (TypeError, ValueError):
            continue
    idle = sum(1 for row in latest.values() if row.get("idle") == "1")
    return _mean(_numbers(rows, "util_pct")), idle, len(latest)


USAGE_COLUMNS = (
    "machine",
    "gpu",
    "util",
    "idle",
    "none of yours",
    "mem used",
    "cpu",
    "slots",
    "last sample",
)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}%"


def _share(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _mib(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _age(seconds: float | None) -> str:
    if seconds is None:
        return "no samples"
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _slots_text(usage: MachineUsage) -> str:
    if usage.slots_used is None or usage.slots_capacity is None:
        return "-"
    return f"{usage.slots_used:.1f} of {usage.slots_capacity:.0f}"


def usage_rows(report: list[MachineUsage]) -> list[list[str]]:
    """The usage table as rows of text, the header first.

    One row per GPU, with the machine's own figures on its first row; a machine that has
    written no samples still gets a row, so a machine nobody is watching is visible.
    """
    rows = [list(USAGE_COLUMNS)]
    for usage in report:
        machine_cells = [
            _pct(usage.cpu_util_pct),
            _slots_text(usage),
            _age(usage.last_sample_age_s),
        ]
        if not usage.gpus:
            rows.append([usage.hostname, "-", "-", "-", "-", "-", *machine_cells])
            continue
        for n, gpu in enumerate(usage.gpus):
            rows.append(
                [
                    usage.hostname if n == 0 else "",
                    str(gpu.gpu_index),
                    _pct(gpu.util_pct),
                    _share(gpu.idle_share),
                    _share(gpu.unused_share),
                    _mib(gpu.mem_used_mib),
                    *(machine_cells if n == 0 else ["", "", ""]),
                ]
            )
    return rows


def render_usage(report: list[MachineUsage]) -> list[str]:
    """The usage table as printable lines, columns as wide as their widest cell."""
    rows = usage_rows(report)
    widths = [max(len(row[i]) for row in rows) for i in range(len(USAGE_COLUMNS))]
    return ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]


def render_jobs_per_hour(report: list[MachineUsage], window_text: str | None) -> list[str]:
    """The lines under the table: successful jobs an hour, per machine, over the window."""
    window = f" in the last {window_text}" if window_text else " over the recorded window"
    lines = [f"jobs finished per hour{window}:"]
    for usage in report:
        rate = "-" if usage.jobs_per_hour is None else f"{usage.jobs_per_hour:.2f}"
        lines.append(f"  {usage.hostname}: {rate} ({usage.jobs_done} finished)")
    return lines


def latest_slots_row(root: Path, hostname: str) -> dict | None:
    """The last slots row this machine wrote, or ``None`` when it has written none."""
    rows = read_rows(slots_csv_path(root, hostname))
    return rows[-1] if rows else None
