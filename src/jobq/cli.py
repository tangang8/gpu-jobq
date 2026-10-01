"""CLI for the file-based GPU job queue: ``jobq <command>``.

Submit jobs into named queues inside a queue folder, then run one foreground worker pool
per machine; the pool claims, GPU-schedules, runs and records each job. Several machines can
drain one queue folder safely — a worker only ever acts on its own machine's claims and pids.

The queue folder comes from one settings file, ``jobq_paths.toml``, in the current
directory or the nearest directory above it; there is no default, and a relative folder in
that file is resolved against the file's own directory rather than the current one.

Subcommands: ``config`` / ``init`` / ``submit`` / ``work`` / ``status`` / ``requeue`` /
``resume`` / ``priority`` / ``pin`` / ``park`` / ``unpark`` / ``reset-mem-floor`` /
``release`` / ``stop`` / ``wait`` / ``usage`` / ``monitor`` / ``mps-stop``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from loguru import logger

from jobq import gpu as gpu_mod
from jobq import io as jobq_io
from jobq import monitor as monitor_mod
from jobq import settings, store, yielding
from jobq import summary as summary_mod
from jobq.gpu import (
    GpuManager,
    GpuPolicy,
    PolicyError,
    load_policy,
    resolve_lock_prefix,
)
from jobq.gpu import policy_path as gpu_policy_path
from jobq.io import atomic_write_json
from jobq.model import parse_jobs_file
from jobq.store import QueueMetaUnreadable
from jobq.summary import QueueSummary, SinceError, parse_since, render_table
from jobq.worker import run_pool

app = typer.Typer(add_completion=False, help=__doc__)

LOCK_PREFIX_HELP = (
    "Prefix for GPU slot flock files (default: /tmp/jobq_<hash of the resolved root>). "
    "Point it at the pool's prefix when the pool was started with a non-default one."
)


# How far back the machine lines of `jobq status` look for GPU samples.
MACHINE_SAMPLE_WINDOW_S = 600.0

# The window `jobq usage` reports on when none is asked for. Every recorded sample would
# mean a machine that has been idle since last month drags this week's figures down, and
# the jobs-an-hour rate would be divided by a window nobody named.
DEFAULT_USAGE_WINDOW_TEXT = "24h"


def configure_loguru() -> None:
    """Print INFO+ logs to stderr in a compact format."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def fail(message: str, *args) -> typer.Exit:
    """Report an expected problem in one sentence and end the command.

    Every command uses this instead of letting an error travel out of the program: a
    traceback tells the reader about this code, and what they need is the one thing that
    is wrong and what to do about it.
    """
    logger.error(message, *args)
    return typer.Exit(code=2)


def out(message: str, *args) -> None:
    """Print one line of a report on standard output.

    What ``jobq status``, ``jobq usage`` and ``jobq config`` have to say is the answer to
    the question that was asked, so it goes where the table goes and can be piped,
    paged and read alongside it. Only what went wrong goes to the log stream.
    """
    print(message.format(*args) if args else message, flush=True)


def no_queue_folder_message() -> str:
    """The one sentence saying how to name the queue folder."""
    return (
        f'the queue folder is not set: create {settings.FILE_NAME} with '
        f'{settings.QUEUE_FOLDER_KEY} = "/path/to/queue/folder" in this directory or in '
        'a directory above it'
    )


def queue_folder_source() -> tuple[Path, Path] | None:
    """The queue folder and the settings file naming it, or ``None`` when no file does.

    Deliberately without a default and with nothing on the command line to override it:
    acting on the wrong queue folder is worse than refusing to act, and a folder written
    down once in a file is read the same way by every command.
    """
    try:
        return settings.queue_folder_source()
    except settings.SettingsError as exc:
        raise fail("{}", exc) from exc


def resolve_root() -> Path:
    """The queue folder the settings file names; otherwise exit."""
    found = queue_folder_source()
    if found is None:
        raise fail("{}", no_queue_folder_message())
    return found[0]


def check_queue_name(name: str) -> str:
    """Return the queue name, or end the command with one sentence naming it.

    A queue is a directory inside the queue folder, so a name that is not one plain
    component names something else entirely; it is refused before anything is read or
    written.
    """
    try:
        return store.check_queue_name(name)
    except store.InvalidQueueName as exc:
        raise fail("{}", exc) from exc


def check_queue_names(names: list[str]) -> list[str]:
    """Return the queue names, ending the command at the first one that cannot be used."""
    return [check_queue_name(n) for n in names]


def require_queue(root: Path, queue: str) -> None:
    """End the command unless ``queue`` is a queue under ``root``."""
    check_queue_name(queue)
    if not store.queue_exists(root, queue):
        raise fail(
            "there is no queue named {!r} under {}; jobq status lists the queues there",
            queue,
            root,
        )


def read_meta_or_fail(root: Path, queue: str):
    """A queue's settings, or a one-sentence message and a non-zero exit."""
    try:
        return store.read_meta(root, queue)
    except QueueMetaUnreadable as exc:
        raise fail("{}", exc) from exc


def apply_shared_perms(root: Path) -> None:
    """Honour this machine's ``shared_perms`` before a command writes anything.

    The pool reads the same key at startup. A command that writes into the queue folder
    without it would create files the same account cannot rewrite from another machine,
    which is the whole point of the setting.
    """
    try:
        policy = load_policy(root, store.this_host())
    except (FileNotFoundError, PolicyError):
        return
    if policy.shared_perms:
        jobq_io.set_shared_perms(True)
        os.umask(0)


def confirm_or_exit(lines: list[str], question: str, *, yes: bool) -> None:
    """Say what a destructive command is about to do, and get a yes before doing it.

    ``jobq requeue``, ``jobq reset-mem-floor`` and ``jobq release --force`` throw work
    away — results, escalated memory floors, a claim a running job may still be behind —
    and none of them can be undone. Each therefore lists what it has found before it
    touches anything. ``--yes`` answers in advance, and a command with no terminal to
    ask from needs it, so nothing is destroyed by a script that meant to report.
    """
    for line in lines:
        out(line)
    if yes:
        return
    if not sys.stdin.isatty():
        raise fail("{} — pass --yes to confirm; nothing was changed", question)
    if not typer.confirm(question):
        out("Nothing was changed")
        raise typer.Exit(code=0)


def _claim_age_str(start_utc: str | None) -> str:
    """Humanize a claim's age from its owner ``start_utc`` ('?' when unparsable)."""
    if not start_utc:
        return "?"
    try:
        start = datetime.fromisoformat(start_utc)
    except ValueError:
        return "?"
    secs = int((datetime.now(UTC) - start).total_seconds())
    if secs < 3600:
        return f"{secs // 60}m"
    return f"{secs / 3600:.1f}h"


def _parse_env(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--env K=V`` options into a dict."""
    env: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise typer.BadParameter(f"--env expects K=V, got {item!r}")
        k, v = item.split("=", 1)
        env[k.strip()] = v
    return env


def check_cap_group(root: Path, queue: str, cap: int | None, group: str | None) -> None:
    """Refuse a cap group that already exists in this root with a different ceiling.

    Queues sharing a ``cap_group`` share one lock namespace, and the ceiling enforced there
    is the slot-file range each queue probes — so disagreeing values would give each queue
    its own effective ceiling in a namespace that is supposed to hold one.
    """
    if cap is None:
        return
    group = group or queue
    for other in store.list_queues(root):
        if other == queue:
            continue
        defaults = store.read_meta(root, other).defaults
        other_cap = defaults.get("cap_per_gpu")
        if other_cap is None:
            continue
        if str(defaults.get("cap_group") or other) != group:
            continue
        if int(other_cap) != int(cap):
            raise typer.BadParameter(
                f"cap_group {group!r} already exists in {root} with cap_per_gpu "
                f"{int(other_cap)} (queue {other!r}); queues sharing a cap group must "
                f"declare the same cap, got {int(cap)}"
            )


@app.command()
def config(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print the folder and the settings file as a JSON object, for scripts.",
    ),
) -> None:
    """Print the settings file in use and the queue folder it names.

    It only reads: the queue folder is set by writing the file, and nothing on the command
    line changes it, so every command in a directory acts on the same folder.
    """
    configure_loguru()
    found = queue_folder_source()
    if found is None:
        raise fail("{}", no_queue_folder_message())
    folder, path = found
    if as_json:
        out("{}", json.dumps({"queue_folder": str(folder), "settings_file": str(path)}))
        return
    out("The queue folder is {}, from {}", folder, path)


@app.command()
def init(
    gpus: str = typer.Option(
        None, "--gpus", help="Comma list of GPU indices to use (default: every GPU found)."
    ),
    cap_per_gpu: int = typer.Option(
        None,
        help=(
            "Hard limit on the slot-units held per GPU. Left out, the policy names no "
            "cap and memory alone decides how many jobs share a GPU."
        ),
    ),
    free_mem_mib: int = typer.Option(
        None,
        "--free-mem-mib",
        help=(
            "The memory a job asks for when it names none, in MiB. Left out, a quarter "
            "of the smallest GPU's total, rounded down to a round number."
        ),
    ),
    cpu_cap: int = typer.Option(
        None,
        "--cpu-cap",
        help="How many jobs that use no GPU (slots: 0) may run at once on this machine. "
        "Left out, the policy does not name it and a quarter of the usable cores applies. "
        "Pass 0 to refuse such jobs here.",
    ),
    mps: bool = typer.Option(
        None,
        "--mps/--no-mps",
        help="Demand a shared MPS daemon, or refuse one. Left out, the policy says "
        f'"{gpu_mod.MPS_AUTO}": the pool uses MPS when this machine can and runs without '
        "it when it cannot.",
    ),
    shared_perms: bool = typer.Option(
        False,
        "--shared-perms",
        help="Create the files jobq writes so that anyone can write them. Use it when "
        "your account has a different numeric user id on the machines sharing the queue "
        "folder. It is written into the policy and applied to the files init writes.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing policy file."),
) -> None:
    """Write this machine's GPU policy file into the queue folder (creating the folder).

    Every commonly needed key is written out explicitly, at conservative values, so the
    file reads as the whole configuration of this machine rather than a fragment.
    """
    configure_loguru()
    if cpu_cap is not None and cpu_cap < 0:
        raise typer.BadParameter(
            f"--cpu-cap must be 0 or more (got {cpu_cap}); 0 means this machine runs no "
            "jobs that declare slots: 0"
        )
    if free_mem_mib is not None and free_mem_mib <= 0:
        raise typer.BadParameter(
            f"--free-mem-mib must be a positive number of MiB (got {free_mem_mib}); it is "
            "the memory a job asks for when it names none"
        )
    root = resolve_root()
    host = store.this_host()
    if shared_perms:
        jobq_io.set_shared_perms(True)
        os.umask(0)
    else:
        apply_shared_perms(root)
    path = gpu_policy_path(root, host)
    if path.exists() and not force:
        raise fail(
            "{} already exists; edit it, or pass --force to overwrite it with a fresh "
            "conservative policy", path,
        )
    if gpus:
        try:
            indices = [int(x.strip()) for x in gpus.split(",") if x.strip()]
        except ValueError as exc:
            raise typer.BadParameter(f"--gpus expects a comma list of integers: {gpus!r}") from exc
        if not indices:
            raise typer.BadParameter("--gpus named no GPU indices")
    else:
        indices = gpu_mod.query_gpu_indices()
        if not indices:
            raise fail(
                "no GPUs found: nvidia-smi is not available (or reported none) on {}. "
                "Pass --gpus 0,1,... to write a policy anyway.", host,
            )
    totals = gpu_mod.total_mem()
    smallest = min((totals.get(g, 0) for g in indices), default=0)
    if free_mem_mib is None:
        free_mem_mib = _modest_share(smallest) if smallest else 8000
    elif smallest and free_mem_mib > smallest:
        logger.warning(
            "free_mem_mib {} MiB is above the smallest GPU's total of {} MiB, so no job "
            "using the default ask can ever start on that GPU",
            free_mem_mib,
            smallest,
        )
    policy = {
        "gpus": indices,
        **({"cap_per_gpu": cap_per_gpu} if cap_per_gpu is not None else {}),
        "free_mem_mib": free_mem_mib,
        "reserve_mem_mib": 0,
        "mem_budget_mib": 0,
        "startup_hold_s": 300,
        **({"cpu_cap": cpu_cap} if cpu_cap is not None else {}),
        "mps": gpu_mod.MPS_AUTO if mps is None else mps,
        "yield_to_foreign": False,
        "shared_perms": shared_perms,
        "env": {},
    }
    try:
        store.ensure_dir(Path(root))
        atomic_write_json(path, policy)
    except OSError as exc:
        raise fail("could not write into the queue folder {}: {}", root, exc) from exc
    logger.info("Wrote {}:", path)
    for k, v in policy.items():
        logger.info("  {}: {}", k, v)
    tail = (
        "state the memory each job needs, with mem_mib in the jobs file or --mem-mib on "
        "the queue: it is the one setting that lets jobq place jobs well. A queue that "
        "states none is asked for what its finished jobs turned out to use."
    )
    if smallest:
        fits = smallest // free_mem_mib
        logger.info(
            "free_mem_mib is {} MiB, so {} using the default ask fit on the smallest GPU "
            "({} MiB); " + tail,
            free_mem_mib,
            "one job" if fits == 1 else f"{fits} jobs",
            smallest,
        )
    else:
        logger.info(
            "free_mem_mib is {} MiB, which is what a job asks for when it names none; "
            + tail,
            free_mem_mib,
        )
    if cap_per_gpu is None:
        try:
            cap = load_policy(root, host).slot_units
        except (FileNotFoundError, PolicyError):
            cap = None
        if cap is not None:
            logger.info(
                "the policy names no cap_per_gpu, so up to {} slot-unit(s) may be held "
                "on each GPU: this machine's share of the cores, per GPU. Add "
                "cap_per_gpu for a hard limit of your own.",
                cap,
            )


def _modest_share(total_mib: int) -> int:
    """A quarter of a GPU, rounded down to a multiple of 500 MiB (at least 500).

    A modest starting point rather than a whole GPU: a job that needs more says so with
    its own ``mem_mib``, and a queue whose jobs report what they use is asked for what
    they turned out to need. Starting at a whole GPU instead would let one job of
    unknown appetite hold a GPU that nothing else may share.
    """
    return max(500, (total_mib // 4) // 500 * 500)


def _differing_queue_options(
    root: Path, queue: str, requested: dict, requested_defaults: dict
) -> list[str]:
    """Options of an existing queue the caller asked to change; empty when none.

    A queue's settings are written once, when it is created, and later submits only
    append jobs. An option that disagrees with the stored one is therefore a request this
    command cannot honour, and reporting it beats applying the jobs under settings the
    caller did not ask for. Only options the caller actually passed are compared.
    """
    meta = store.read_meta(root, queue)
    stored = {
        "priority": meta.priority,
        "depends_on": list(meta.depends_on),
        "node": meta.runs_on_text,
        "strict_deps": meta.strict_deps,
    }
    diffs = [
        f"{k}: queue has {stored[k]!r}, this submit asks for {v!r}"
        for k, v in requested.items()
        if v is not None and stored[k] != v
    ]
    diffs += [
        f"{k}: queue has {meta.defaults.get(k)!r}, this submit asks for {v!r}"
        for k, v in requested_defaults.items()
        if v is not None and meta.defaults.get(k) != v
    ]
    return diffs


def parse_ties(specs: list[str] | None, gpus: str | None = None) -> tuple:
    """The machine tie a command line asks for, as :class:`~jobq.store.MachineTie` entries.

    ``specs`` are written as ``machine``, ``machine:0,1`` or ``machine:4-7``; ``gpus``
    names GPU numbers without a machine, which tie the queue to those numbers on
    whichever machine runs it. A form that cannot be used ends the command with a
    message naming it.
    """
    ties = []
    for spec in specs or []:
        try:
            ties.append(store.MachineTie.parse(spec))
        except store.InvalidMachineTie as exc:
            raise fail("{}", exc) from exc
    if gpus:
        try:
            ties.append(store.MachineTie(machine=None, gpus=store.parse_gpu_list(gpus)))
        except store.InvalidMachineTie as exc:
            raise fail("{}", exc) from exc
    return tuple(ties)


def note_machines_without_a_policy(root: Path, ties: tuple) -> None:
    """Say which machines of a tie have no policy file in this queue folder.

    Not a refusal: a machine may be named before it is set up, and the name is the
    hostname that machine will report for itself.
    """
    for tie in ties:
        if tie.machine and not gpu_policy_path(root, tie.machine).exists():
            logger.info(
                "there is no policy file for machine {!r} in {}, so nothing claims from "
                "this queue there until that machine runs jobq init",
                tie.machine, root,
            )


@app.command()
def submit(
    queue: str = typer.Argument(..., help="Queue name to submit into (created if new)."),
    jobs_file: Path = typer.Option(..., "--jobs-file", help="JSONL or plain-lines jobs file."),
    template: str = typer.Option(
        None, help="cmd template for plain lines: {line}=whole line, {0},{1},..=|-split fields."
    ),
    depends_on: str = typer.Option(
        None, help="Comma list of queues that must be complete before this one runs."
    ),
    strict_deps: bool = typer.Option(
        False,
        "--strict-deps",
        help="Block while any job of a dependency has failed (default: a failed job counts as finished).",
    ),
    node: list[str] = typer.Option(
        None,
        "--node",
        help="Machine this queue runs on, repeatable, each optionally with the GPUs of "
        "it to use: --node gpu-host-1:0,1 --node gpu-host-2 (default: any machine).",
    ),
    priority: int = typer.Option(None, help="Higher runs first among ready queues (default 0)."),
    max_consecutive_failures: int = typer.Option(
        None,
        "--max-consecutive-failures",
        help="Pause this queue once its N most recently finished jobs all failed for "
        f"their own reasons (default {store.DEFAULT_MAX_CONSECUTIVE_FAILURES}; 0 never "
        "pauses). Resume it with jobq resume.",
    ),
    mem_mib: int = typer.Option(None, help="Queue-default free-MiB gate for its jobs."),
    cap_per_gpu: int = typer.Option(
        None, help="Stage ceiling: max jobs of this queue per GPU (on top of the policy cap)."
    ),
    cap_group: str = typer.Option(
        None, help="Cap-group name (default: queue name). Queues sharing a group co-cap."
    ),
    slots: int = typer.Option(
        None,
        help="Queue-default job weight in global per-GPU slot-units (heavy jobs take >1). "
        "Use 0 for a queue whose jobs use no GPU at all: such a job takes a slot from the "
        "separate gpu_policy "
        "cpu_cap budget instead of GPU capacity, so it never blocks a GPU job but is still "
        "admission-controlled. Only declare 0 if the jobs never touch CUDA -- they run with "
        "CUDA_VISIBLE_DEVICES emptied.",
    ),
    cwd: Path = typer.Option(
        None,
        help="Directory to run these jobs in (default: the directory submit was run from).",
    ),
    env: list[str] = typer.Option(None, "--env", help="Queue-default env, repeatable K=V."),
    dry_run: bool = typer.Option(False, help="Print the expanded jobs and exit."),
) -> None:
    """Expand a jobs file and append the jobs to a queue (its meta is created on first submit)."""
    configure_loguru()
    root = resolve_root()
    check_queue_name(queue)
    dependencies = (
        check_queue_names([d.strip() for d in depends_on.split(",")]) if depends_on else None
    )
    ties = parse_ties(node)
    note_machines_without_a_policy(root, ties)
    apply_shared_perms(root)
    try:
        text = jobs_file.read_text()
    except OSError as exc:
        raise fail("could not read the jobs file {}: {}", jobs_file, exc) from exc
    try:
        jobs = parse_jobs_file(text, template=template)
    except ValueError as exc:
        raise fail("{}", exc) from exc
    defaults: dict = {}
    if mem_mib is not None:
        defaults["mem_mib"] = mem_mib
    if cap_per_gpu is not None:
        defaults["cap_per_gpu"] = cap_per_gpu
    if cap_group is not None:
        defaults["cap_group"] = cap_group
    if slots is not None:
        defaults["slots"] = slots
    if max_consecutive_failures is not None:
        defaults["max_consecutive_failures"] = max_consecutive_failures
    # The queue's default working directory. Recorded absolutely, so a pool started
    # anywhere runs the jobs where they were submitted from.
    defaults["cwd"] = str(Path(cwd).resolve() if cwd is not None else Path.cwd())
    env_d = _parse_env(env)
    if env_d:
        defaults["env"] = env_d
    # Appending to a queue cannot change its settings, so options that disagree with the
    # stored ones are reported and nothing is written. Only what the caller passed counts:
    # the working directory, which submit always records, is compared solely when --cwd
    # named it.
    requested = {
        "priority": priority,
        "depends_on": dependencies,
        "node": store.ties_text(ties) if ties else None,
        "strict_deps": strict_deps or None,
    }
    requested_defaults = {
        "mem_mib": mem_mib,
        "slots": slots,
        "cap_per_gpu": cap_per_gpu,
        "cap_group": cap_group,
        "max_consecutive_failures": max_consecutive_failures,
        "env": env_d or None,
        "cwd": defaults["cwd"] if cwd is not None else None,
    }

    def _report_conflict(diffs: list[str]) -> None:
        logger.error(
            "queue {!r} already exists and its settings are written once; this submit "
            "would change:", queue,
        )
        for d in diffs:
            logger.error("  {}", d)
        logger.error(
            "nothing was submitted. Drop those options to append jobs, or submit into "
            "a new queue."
        )

    if store.queue_exists(root, queue):
        read_meta_or_fail(root, queue)  # report unusable settings instead of raising
        diffs = _differing_queue_options(root, queue, requested, requested_defaults)
        if diffs:
            _report_conflict(diffs)
            raise typer.Exit(code=2)
    # A zero-job queue is never "complete", so creating one parks every dependent forever.
    # Refuse to create one here; appending nothing to an existing queue stays a no-op.
    if not jobs and not store.queue_exists(root, queue):
        logger.error(
            "refusing to create queue {!r}: {} expanded to no jobs (a meta-only queue can "
            "never complete and would park any dependent queue forever)",
            queue, jobs_file,
        )
        raise typer.Exit(code=2)
    # Validate before creating meta: a rejected batch must not leave a meta-only (zero-job)
    # queue behind, which would wedge the pool and any dependent queues.
    try:
        check_cap_group(root, queue, cap_per_gpu, cap_group)
        store.check_new_jobs(root, queue, jobs)
    except QueueMetaUnreadable as exc:
        raise fail("{}", exc) from exc
    except ValueError as exc:
        raise fail("{}", exc) from exc
    if dry_run:
        # After every check a real submit makes (zero-job refusal, cap-group conflict,
        # duplicate keys), so the flag is a usable preflight. Nothing is written.
        logger.info("Would submit {} job(s) to queue {!r}:", len(jobs), queue)
        for j in jobs:
            logger.info("  [{}] {}", j.jobkey, j.cmd)
        return
    # Creating the queue, comparing this submit against its settings and appending the jobs
    # happen under one lock, so of two first submissions carrying different settings one
    # creates the queue and the other is refused having written nothing.
    try:
        store.submit_jobs(
            root,
            queue,
            jobs,
            depends_on=dependencies or [],
            ties=ties,
            priority=priority or 0,
            strict_deps=strict_deps,
            defaults=defaults,
            differing_options=lambda meta: _differing_queue_options(
                root, queue, requested, requested_defaults
            ),
            precheck=lambda r: check_cap_group(r, queue, cap_per_gpu, cap_group),
        )
    except store.QueueSettingsConflict as exc:
        _report_conflict(exc.diffs)
        raise typer.Exit(code=2) from exc
    except typer.BadParameter as exc:
        raise fail("{}", exc.message) from exc
    except ValueError as exc:
        raise fail("{}", exc) from exc
    except OSError as exc:
        raise fail("could not write into the queue folder {}: {}", root, exc) from exc
    logger.info("Submitted {} job(s) to queue {!r} under {}", len(jobs), queue, root)


@app.command()
def work(
    workers: int = typer.Option(
        None,
        help="Number of worker threads (the per-GPU slot cap is global). Left out, the "
        "pool divides the cores it may use, less the policy's cpu_reserve, by the cores "
        "a job is assumed to need, and says in its log what it chose.",
    ),
    queues: str = typer.Option(
        None,
        help="Comma list restricting which queues to drain (recommended on a queue folder "
        "holding queues you do not want this pool to claim from).",
    ),
    lock_prefix: str = typer.Option(None, "--lock-prefix", help=LOCK_PREFIX_HELP),
    no_monitor: bool = typer.Option(
        False,
        "--no-monitor",
        help="Do not append utilisation samples to the queue folder while this pool runs.",
    ),
) -> None:
    """Run the foreground worker pool on this machine (keep it in a terminal that stays open)."""
    configure_loguru()
    root = resolve_root()
    if workers is not None and workers < 1:
        raise fail("--workers must be at least 1 (got {}); leave it out to let the pool "
                   "choose", workers)
    q = (
        check_queue_names([x.strip() for x in queues.split(",") if x.strip()])
        if queues
        else None
    )
    # Before the pool lock, the pid file or any claim: a pool without this machine's
    # policy has no GPUs to offer, so its threads would claim jobs from the shared queue
    # and then wait for capacity that never arrives, holding work other machines can run.
    try:
        load_policy(root, store.this_host())
    except FileNotFoundError as exc:
        raise fail(
            "this machine ({}) has no GPU policy, so there is nothing for a pool to run "
            "on: {} is missing. Run jobq init --gpus <indices> --free-mem-mib <MiB> on "
            "this machine to write it, then start the pool again. Add --shared-perms if "
            "the same account has different numeric user ids on the machines sharing "
            "this queue folder.",
            store.this_host(),
            gpu_policy_path(root, store.this_host()),
        ) from exc
    except PolicyError as exc:
        raise fail("{}", exc) from exc
    prefix = resolve_lock_prefix(root, store.this_host(), lock_prefix)
    logger.info(
        "worker pool starting: node={} threads={} root={} queues={} lock_prefix={}",
        store.this_host(),
        workers if workers is not None else "(chosen from the cores available)",
        root,
        q or "(all)",
        prefix,
    )
    try:
        run_pool(
            root,
            workers=workers,
            queues=q,
            lock_prefix=lock_prefix,
            monitor_samples=not no_monitor,
        )
    except PolicyError as exc:
        # The pool reads this machine's policy before it starts anything; an unusable one
        # is the same kind of one-sentence problem as every other error here.
        raise fail("{}", exc) from exc
    except RuntimeError as exc:
        raise fail("{}", exc) from exc
    except OSError as exc:
        raise fail("could not write into the queue folder {}: {}", root, exc) from exc


# How many successfully finished jobs must have reported a peak before the suggestion
# below is worth printing, how much room it leaves above the largest reading, and the step
# it rounds up to.
MEM_SUGGESTION_MIN_JOBS = 5
MEM_SUGGESTION_HEADROOM = 1.15
MEM_SUGGESTION_STEP_MIB = 500


def suggested_mem_mib(largest_peak_mib: int) -> int:
    """A ``mem_mib`` to try: the largest peak plus room, rounded up to a round number."""
    with_room = largest_peak_mib * MEM_SUGGESTION_HEADROOM
    steps = math.ceil(with_room / MEM_SUGGESTION_STEP_MIB)
    return int(steps * MEM_SUGGESTION_STEP_MIB)


def _smallest_gpu_mib(root: Path) -> int:
    """The total memory of the smallest GPU this machine's policy names; 0 if unknown."""
    totals = gpu_mod.total_mem()
    if not totals:
        return 0
    try:
        gpus = load_policy(root, store.this_host()).gpus
    except (FileNotFoundError, PolicyError):
        gpus = list(totals)
    sizes = [totals[g] for g in gpus if g in totals]
    return min(sizes) if sizes else 0


def _mem_suggestion(root: Path, results: list[dict]) -> str | None:
    """One sentence suggesting a ``mem_mib`` from what this queue's jobs actually used.

    Only advice, and only once the queue has enough finished jobs to have shown its real
    appetite: nothing is changed by printing it. Jobs that failed are left out, since a
    job that died early says nothing about how much memory the work needs.
    """
    peaks = [
        int(r["peak_mem_mib"])
        for r in results
        if r.get("rc") == 0 and r.get("peak_mem_mib") is not None
    ]
    if len(peaks) < MEM_SUGGESTION_MIN_JOBS:
        return None
    largest = max(peaks)
    suggestion = suggested_mem_mib(largest)
    sentence = (
        f"{len(peaks)} finished jobs reported their peak memory; the largest was "
        f"{largest} MiB, so try mem_mib {suggestion}"
    )
    smallest = _smallest_gpu_mib(root)
    if smallest and suggestion:
        fits = smallest // suggestion
        sentence += (
            f", which fits {fits} such job on this machine's smallest GPU "
            f"({smallest} MiB)"
            if fits == 1
            else f", which fits {fits} such jobs on this machine's smallest GPU "
            f"({smallest} MiB)"
        )
    return sentence + "."


def _print_table(summaries: list[QueueSummary]) -> None:
    """Print the overview table on standard output, one line per queue."""
    for line in render_table(summaries):
        print(line)
    # The rest of the report goes to the log stream: flush so the reader sees the table
    # first, whichever stream the two are piped to.
    sys.stdout.flush()


def _hidden_line(hidden: int, since_text: str | None) -> str:
    """The sentence under the table saying which complete queues were left out and why."""
    word = "queue" if hidden == 1 else "queues"
    window = f" with no activity in the last {since_text}" if since_text else ""
    return (
        f"{hidden} complete {word}{window} are not shown; use --all to list every queue"
        if hidden != 1
        else f"{hidden} complete {word}{window} is not shown; use --all to list every queue"
    )


def _report_root(
    root: Path,
    queue: str | None = None,
    lock_prefix: str | None = None,
    include_complete: bool = False,
    since_s: float | None = None,
    since_text: str | None = None,
) -> None:
    """Print the overview table for ``root`` (one queue if named) + this machine's state."""
    # A directory that holds queue state under a name jobq cannot use is reported, so it
    # is not simply invisible; no pool claims from it and no command writes into it.
    for odd in store.list_invalid_queue_dirs(root):
        out(
            "directory {!r} under {} holds queue state, but its name is not a valid queue "
            "name, so it is not used: {}",
            odd, root, store.queue_name_reason(odd),
        )
    # A queue whose own directories are links is refused by every command, so it is named
    # here rather than being simply missing from the table.
    for unsafe, reason in store.list_unsafe_queues(root):
        out(
            "queue {!r} under {} is not used, and no pool claims from it: {}",
            unsafe, root, reason,
        )
    if queue is None:
        all_names = store.list_queues(root)
        if not all_names:
            out("No queues under {}", root)
            _report_machines(root)
            return
        summaries = summary_mod.queue_summaries(
            root, include_complete=include_complete, since=since_s
        )
        _print_table(summaries)
        hidden = len(all_names) - len(summaries)
        if hidden > 0:
            print(_hidden_line(hidden, since_text), flush=True)
        for s in summaries:
            if s.state == summary_mod.UNREADABLE:
                # The table says a row could not be read; the sentence says what about it.
                try:
                    store.read_meta(root, s.name)
                except QueueMetaUnreadable as exc:
                    logger.error("{}", exc)
                continue
            if s.state == summary_mod.COMPLETE:
                continue  # a queue with no work left has nothing to size
            advice = _mem_suggestion(root, store.load_results(root, s.name))
            if advice:
                out("queue {!r}: {}", s.name, advice)
        _report_node(root, lock_prefix)
        return
    if not store.queue_exists(root, queue):
        out("queue {!r}: does not exist", queue)
        _report_node(root, lock_prefix)
        return
    _print_table([summary_mod.queue_summary(root, queue)])
    for name in [queue]:
        # with_deferred is opt-in (a sidecar read per pending job): a human status call can
        # afford it, the worker's claim loop cannot.
        st = store.queue_status(root, name, with_deferred=True)
        try:
            meta = store.read_meta(root, name)
        except QueueMetaUnreadable as exc:
            logger.error("{}", exc)
            continue
        out(
            "queue {!r} (runs on: {}, priority {}, waits for: {}): "
            "{} pending, {} running, {} done, {} failed, {} in total",
            name,
            meta.runs_on_text,
            meta.priority,
            ", ".join(meta.depends_on) if meta.depends_on else "nothing",
            st["pending"], st["running"], st["done"], st["failed"], st["total"],
        )
        _report_tie_here(root, meta)
        if meta.parked:
            because = f": {meta.parked_reason}" if meta.parked_reason else ""
            out(
                "    PARKED{} (no pool claims from it; bring it back with: jobq unpark {})",
                because, name,
            )
        learned = store.read_learned_mem(root, name)
        if learned is not None:
            out(
                "    learned memory per job: {} MiB, from the largest of {} peak(s) "
                "({} MiB): {} reported by the jobs, {} measured from the GPU",
                learned.get("request_mib"), learned.get("jobs"), learned.get("peak_mib"),
                learned.get("reported"), learned.get("measured"),
            )
        pause = store.read_pause(root, name)
        if pause is not None:
            keys = ", ".join(str(k) for k in (pause.get("keys") or [])) or "-"
            out(
                "    PAUSED since {} after {} consecutive job failures: {} "
                "(resume with: jobq resume {})",
                pause.get("since_utc", "?"), pause.get("limit", "?"), keys, name,
            )
        if st.get("deferred"):
            # Deferred jobs are pending but unclaimable until their tempfail retry window
            # passes, which otherwise reads as a stalled queue.
            until = st.get("deferred_until")
            when = (
                datetime.fromtimestamp(until, UTC).strftime("%H:%M:%SZ")
                if until
                else "?"
            )
            out(
                "    deferred={} (tempfail retry; earliest claimable {})", st["deferred"], when
            )
        results = store.load_results(root, name)
        peaks = sorted(
            r["peak_mem_mib"] for r in results if r.get("peak_mem_mib") is not None
        )
        if peaks:
            out(
                "    peak_mem_mib (n={}): median={} max={}",
                len(peaks), peaks[len(peaks) // 2], peaks[-1],
            )
        advice = _mem_suggestion(root, results)
        if advice:
            out("    {}", advice)
        for r in st["running_jobs"]:
            age = _claim_age_str(r.get("start_utc"))
            gpu = "no GPU" if r["gpu"] is None else f"gpu {r['gpu']}"
            line = (
                f"    running {r['key']} on {r['node'] or 'an unrecorded machine'}, "
                f"{gpu}, process {r['pid']}, for {age}"
            )
            if r["node"] not in (None, store.this_host()):
                line += (
                    f"  [claim from another machine — {_pool_heartbeat_text(root, r['node'])}; "
                    f"if that machine crashed, run: jobq release {name} '{r['key']}' --force]"
                )
            elif r.get("pool_gone"):
                # The claim outlived the pool that made it, so the counts alone read as
                # work in progress: say on the job's own line what will happen to it.
                where = r["node"] or store.this_host()
                line += (
                    "  [left by a pool that has ended; it goes back in the queue when a "
                    f"pool starts on {where}"
                )
                line += (
                    "; its own process is still running and that pool ends it first]"
                    if r.get("job_running")
                    else "]"
                )
            out(line)
    _report_node(root, lock_prefix)


def _heartbeat_stale_s(root: Path, host: str) -> float:
    """How old that machine's heartbeat may be before it counts as not heard from.

    A machine's own policy decides it: a machine whose supervisor ticks every few
    minutes is not silent at the same age as one that ticks every few seconds. When its
    policy cannot be read this machine's own setting stands in, since that is the
    closest thing to a shared expectation the queue folder has.
    """
    for candidate in (host, store.this_host()):
        try:
            return load_policy(root, candidate).heartbeat_stale_s
        except (FileNotFoundError, PolicyError):
            continue
    return monitor_mod.DEFAULT_HEARTBEAT_STALE_S


def _pool_heartbeat_text(root: Path, host: str) -> str:
    """Whether a machine's pool is alive by its heartbeat, in one phrase."""
    state = store.heartbeat_state(root, host, _heartbeat_stale_s(root, host))
    if state["fresh"]:
        return f"its pool is alive (heartbeat {int(state['age_s'])}s old)"
    if state["heartbeat_utc"]:
        return f"its pool has not been heard from since {state['heartbeat_utc']}"
    return "its pool has not been heard from at all"


def _machine_line(root: Path, host: str) -> str:
    """One machine's line of the status report: its pool, and what its samples say."""
    line = f"machine {host}: {_pool_heartbeat_text(root, host)}"
    util, idle, n_gpus = monitor_mod.recent_gpu_summary(
        root, host, window_s=MACHINE_SAMPLE_WINDOW_S
    )
    if n_gpus:
        line += (
            f"; last 10 minutes {util:.0f}% utilisation over {n_gpus} gpu(s), "
            f"{idle} idle"
        )
    latest = monitor_mod.latest_slots_row(root, host)
    if latest:
        line += f"; slots {latest.get('used', '?')} of {latest.get('slots', '?')}"
    return line


def _report_machines(root: Path) -> None:
    """One line per machine that has a policy file in the queue folder."""
    for host in monitor_mod.machines_with_a_policy(root):
        out("{}", _machine_line(root, host))


def _report_tie_here(root: Path, meta) -> None:
    """Say why this machine cannot work on a queue whose tie rules it out.

    The two ways a tie rules this machine out read very differently on the machine you
    are standing on, so each is said in its own words: the queue belongs to other
    machines, or it belongs here but names GPUs this machine's policy does not select.
    """
    host = store.this_host()
    if not meta.allows_machine(host):
        out(
            "    this machine ({}) is not among the machines this queue runs on, so no "
            "pool here claims from it", host,
        )
        return
    named = meta.gpus_on(host)
    if named is None:
        return
    try:
        policy_gpus = set(load_policy(root, host).gpus)
    except (FileNotFoundError, PolicyError):
        return
    usable = sorted(g for g in named if g in policy_gpus)
    if usable:
        out(
            "    on this machine ({}) its jobs run on GPU {}",
            host, ", ".join(str(g) for g in usable),
        )
        return
    out(
        "    this queue names GPU {} on this machine ({}), and its policy selects {}, "
        "so no pool here claims from it and no worker waits on it",
        ", ".join(str(g) for g in named),
        host,
        ", ".join(str(g) for g in sorted(policy_gpus)) or "no GPU at all",
    )


def _pool_leaving_note(root: Path, host: str) -> str:
    """What the pool has been asked to do, when it has been asked to leave.

    The two requests in the queue folder are visible to any terminal on this machine. A
    pool draining because a signal reached its own terminal is not: that is why the note
    says a stop request when the file is there and a signal otherwise, for a pool whose
    log shows it is draining.
    """
    if store.stop_now_path(root, host).exists():
        return " (ending its running jobs and putting them back in the queue)"
    if store.stop_path(root, host).exists():
        return " (draining after a stop request; running jobs finish, no new job is claimed)"
    if store.pool_draining_after_signal(root, host):
        return " (draining after a signal; running jobs finish, no new job is claimed)"
    return ""


def _report_node(root: Path, lock_prefix: str | None = None) -> None:
    """Print what can be said about this machine: its pool, its slots, its policy."""
    # Is a pool running here, and which log is it writing? Only answerable for this machine:
    # a pid recorded by another machine says nothing about this one.
    pool = store.pool_state(root, store.this_host())
    if pool["alive"]:
        log = pool["log"]
        folder = Path(log).parent if log else None
        if folder is not None and folder != root / "logs":
            out("this machine ({}) pool logs are in {}", store.this_host(), folder)
        out(
            "this machine ({}) pool: RUNNING pid={} log={}{}",
            store.this_host(), pool["pid"],
            Path(log).name if log else "(unknown)",
            _pool_leaving_note(root, store.this_host()),
        )
    else:
        out("this machine ({}) pool: not running", store.this_host())
    if gpu_mod.mps_daemon_running(_node_policy_or_default(root)):
        out("this machine ({}) MPS daemon: running (stop it with: jobq mps-stop)",
                    store.this_host())
    # Resolved outside the occupancy try, whose FileNotFoundError arm means the far more
    # common "no policy here".
    prefix = resolve_lock_prefix(root, store.this_host(), lock_prefix)
    if prefix is not None:
        try:
            gm = GpuManager(root, store.this_host(), lock_prefix=prefix)
            occ = gm.occupancy()
            if occ:
                # An opt-in memory reserve caps the machine instead of the slots -> say so,
                # since idle-looking slots are then expected.
                reserve = gm.reserve_mem_mib()
                # The count is what is held, which can be above the cap when the cap was
                # lowered while those jobs were running; they keep their slots until they
                # end, and nothing new is admitted meanwhile.
                cap = gm.cap_per_gpu()
                over = [g for g, n in occ.items() if cap is not None and n > cap]
                out(
                    "this machine ({}) slot occupancy: {}{}{}",
                    store.this_host(),
                    ", ".join(f"gpu{g}={n}" for g, n in sorted(occ.items())),
                    f" reserve_mem_mib={reserve}" if reserve else "",
                    (
                        f" (above the cap of {cap} on "
                        + ", ".join(f"gpu{g}" for g in sorted(over))
                        + "; those jobs keep their slots until they end)"
                        if over
                        else ""
                    ),
                )
            held = gm.reservations()
            if held:
                out(
                    "this machine ({}) GPUs held for waiting jobs: {}",
                    store.this_host(),
                    ", ".join(
                        f"gpu{g} for {rec.get('job', 'an unnamed job')} "
                        f"({rec.get('mem_mib', '?')} MiB, since {rec.get('since_utc', '?')})"
                        for g, rec in sorted(held.items())
                    ),
                )
        except FileNotFoundError:
            pass  # no policy on this machine -> nothing to report
        except PolicyError as exc:
            logger.error("this machine's GPU policy cannot be used: {}", exc)
        except PermissionError as err:
            logger.warning(
                "could not read GPU slot locks under {} ({}). Pass --lock-prefix matching "
                "the prefix the worker pool was started with.",
                prefix,
                err,
            )
    # Yielded GPUs (opt-in framework): silent when the feature is off / marker file absent.
    yielded = yielding.read_yielded(root, store.this_host())
    if yielded:
        out(
            "this machine ({}) yielded GPUs: {}",
            store.this_host(),
            ", ".join(
                f"{g} (foreign={v.get('foreign')}, since {v.get('first_seen')})"
                for g, v in sorted(yielded.items())
            ),
        )
        # The watchdog is the only thing that ever gives a yielded GPU back, and it
        # cannot decide anything while the policy does not load, so the GPUs stay off
        # limits until it does. That reads as an idle machine unless it is said.
        try:
            load_policy(root, store.this_host())
        except (FileNotFoundError, PolicyError) as exc:
            out(
                "this machine ({}) cannot read its GPU policy ({}), so those GPUs stay "
                "yielded until it can",
                store.this_host(),
                exc,
            )
    _report_machines(root)


@app.command()
def status(
    queue: str = typer.Argument(None, help="One queue (default: all queues under the root)."),
    lock_prefix: str = typer.Option(None, "--lock-prefix", help=LOCK_PREFIX_HELP),
    all_: bool = typer.Option(
        False, "--all", help="Include queues with no work left (they are left out by default)."
    ),
    since: str = typer.Option(
        None,
        "--since",
        help="Also include complete queues touched within this window, written as "
        f"{summary_mod.SINCE_FORMS}. Queues with work left are always shown.",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print the queue summaries as one JSON document and nothing else."
    ),
) -> None:
    """One table of the queues (state, counts, memory, time left) + this machine's pool.

    With a queue name, the detailed view of that one queue follows its row of the table.
    """
    configure_loguru()
    if queue is not None:
        check_queue_name(queue)
    since_s = None
    if since is not None:
        try:
            since_s = parse_since(since)
        except SinceError as exc:
            raise fail("{}", exc) from exc
    root_path = resolve_root()
    if as_json and queue is not None and not store.queue_exists(root_path, queue):
        # The same sentence the table path gives, so the two views of a name that is not
        # there do not have to be read differently. It goes to the log stream, because
        # --json prints the document and nothing else.
        logger.info("queue {!r}: does not exist", queue)
        print(json.dumps({"root": str(root_path), "queue": queue, "queues": []}, indent=2))
        return
    if as_json:
        if queue is not None:
            try:
                store.read_meta(root_path, queue)
            except QueueMetaUnreadable as exc:
                logger.error("{}", exc)
        summaries = (
            [summary_mod.queue_summary(root_path, queue)]
            if queue is not None
            else summary_mod.queue_summaries(
                root_path, include_complete=all_, since=since_s
            )
        )
        print(json.dumps(
            {
                "root": str(root_path),
                "since": since,
                "since_s": since_s,
                "include_complete": all_,
                "queues": [s.to_dict() for s in summaries],
            },
            indent=2,
        ))
        return
    _report_root(root_path, queue, lock_prefix, all_, since_s, since)


@app.command()
def usage(
    since: str = typer.Option(
        None,
        "--since",
        help="Window to report on, written as "
        f"{summary_mod.SINCE_FORMS}. Left out, the last {DEFAULT_USAGE_WINDOW_TEXT}.",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print the same figures as one JSON document and nothing else."
    ),
) -> None:
    """What the machines of this queue folder did: GPUs, processor, slots, jobs an hour."""
    configure_loguru()
    window_text = since or DEFAULT_USAGE_WINDOW_TEXT
    try:
        window_s = parse_since(window_text)
    except SinceError as exc:
        raise fail("{}", exc) from exc
    root = resolve_root()
    report = monitor_mod.usage_report(root, window_s=window_s)
    if as_json:
        print(json.dumps(
            {
                "root": str(root),
                "since": window_text,
                "since_s": window_s,
                "machines": [m.to_dict() for m in report],
            },
            indent=2,
        ))
        return
    if not report:
        out("No machine under {} has a policy file, so none is sampled", root)
        return
    for line in monitor_mod.render_usage(report):
        print(line)
    for line in monitor_mod.render_jobs_per_hour(report, window_text):
        print(line)
    sys.stdout.flush()


@app.command()
def monitor(
    once: bool = typer.Option(
        False, "--once", help="Take one round of samples and return, instead of sampling on."
    ),
) -> None:
    """Sample this machine's utilisation in the foreground, for a machine with no pool.

    A pool samples while it runs, so this refuses to start beside one.
    """
    configure_loguru()
    root = resolve_root()
    host = store.this_host()
    if store.pool_state(root, host)["alive"]:
        raise fail(
            "a worker pool is running on this machine ({}) for {}, and a pool samples "
            "utilisation itself; there is nothing for jobq monitor to add",
            host, root,
        )
    try:
        policy = load_policy(root, host)
    except (FileNotFoundError, PolicyError) as exc:
        logger.info("sampling without a policy on this machine ({}): {}", host, exc)
        policy = None
    config = monitor_mod.config_from_policy(policy)
    sampler = monitor_mod.Sampler(
        root,
        host,
        config=config,
        gpus=tuple(policy.gpus) if policy is not None else (),
        occupancy=lambda: GpuManager(root, host).occupancy() if policy is not None else {},
        slot_state=lambda: _watching_slot_state(root, host, policy),
    )
    if once:
        sampler.tick()
        return
    if config.interval_s <= 0:
        raise fail(
            "monitor_interval_s is 0 in this machine's policy, which turns sampling off; "
            "set it to the number of seconds between samples"
        )
    logger.info(
        "sampling {} every {:.0f}s into {}; stop it with Ctrl-C",
        host, config.interval_s, monitor_mod.monitor_dir(root),
    )
    stop = threading.Event()
    try:
        sampler.run(stop)
    except KeyboardInterrupt:
        stop.set()


def _watching_slot_state(root: Path, host: str, policy) -> monitor_mod.SlotState:
    """The slots row of a machine that is watched but runs no pool: capacity, nothing used."""
    gpus = list(policy.gpus) if policy is not None else []
    yielded = yielding.read_yielded(root, host) or {}
    yielded_here = sum(1 for g in gpus if g in yielded)
    cap = policy.slot_units if policy is not None else 0
    return monitor_mod.SlotState(
        used=0,
        slots=max(0, len(gpus) - yielded_here) * cap,
        wait=0,
        gpus=len(gpus),
        yielded=yielded_here,
        cap_per_gpu=cap,
        live=False,
    )


@app.command()
def requeue(
    queue: str = typer.Argument(..., help="Queue to requeue jobs in."),
    all_: bool = typer.Option(False, "--all", help="Requeue done jobs too (default: failed only)."),
    reset_oom: bool = typer.Option(
        False,
        "--reset-oom",
        help="Clear each requeued job's OOM-requeue counter, so a job that exhausted the "
        "worker's OOM backstop gets its retries back (the counter is preserved by default).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List the jobs this would requeue and change nothing."
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before deleting the results."),
) -> None:
    """Make terminal jobs pending again so a worker retries them (failed jobs by default).

    This deletes the result of every job it requeues, which is what makes the job pending
    again: its exit code, timings and log path are gone from the queue, though the log
    file itself stays. It also resumes the queue if a run of failures had paused it. The
    escalated per-job mem_mib floor is always kept — ``jobq reset-mem-floor`` is the one
    way to clear it.
    """
    configure_loguru()
    root = resolve_root()
    require_queue(root, queue)
    apply_shared_perms(root)
    targets = store.requeue_targets(root, queue, failed_only=not all_)
    if dry_run:
        out("Would requeue {} job(s) in {!r}:", len(targets), queue)
        for jk, _attempt in targets:
            out("  {}", jk)
        return
    if not targets:
        out("Nothing to requeue in {!r}", queue)
        return
    word = "result" if len(targets) == 1 else "results"
    confirm_or_exit(
        [
            f"this deletes {len(targets)} {word} in {queue!r}, so those jobs run again "
            "from the beginning:",
            *[f"  {jk}" for jk, _a in targets],
        ],
        "Continue?",
        yes=yes,
    )
    try:
        keys = store.requeue(root, queue, failed_only=not all_, reset_oom=reset_oom)
    except OSError as exc:
        raise fail("could not write into the queue folder {}: {}", root, exc) from exc
    out("Requeued {} job(s) in {!r}: {}", len(keys), queue, keys or "-")
    # The dep gate is a point-in-time check: a dependent queue that already drained while
    # this one was complete is not invalidated by this requeue, so surface it.
    if keys:
        for dq in store.list_queues(root):
            try:
                depends_on = store.read_meta(root, dq).depends_on
            except QueueMetaUnreadable as exc:
                logger.warning("{}", exc)
                continue
            if queue not in depends_on:
                continue
            st = store.queue_status(root, dq)
            if st["running"] or st["done"] or st["failed"]:
                logger.warning(
                    "dependent queue {!r} already ran against the pre-requeue state of "
                    "{!r} (done={} failed={} running={}); if its jobs consumed this "
                    "queue's outputs, requeue it too once this queue re-drains",
                    dq, queue, st["done"], st["failed"], st["running"],
                )


@app.command()
def resume(
    queue: str = typer.Argument(..., help="Queue to un-pause."),
) -> None:
    """Clear a queue's failure pause (and its run of failures) so pools claim from it again.

    The only way out of a pause: nothing resumes a queue on a timer, on a new submission or
    on a later success. ``jobq requeue`` runs this too.
    """
    configure_loguru()
    root = resolve_root()
    require_queue(root, queue)
    apply_shared_perms(root)
    if store.resume_queue(root, queue):
        logger.info("Resumed queue {!r}; pools will claim from it again", queue)
    else:
        logger.info("Queue {!r} was not paused; its failure run is cleared", queue)


def _change_setting(root: Path, queue: str, change, what: str) -> None:
    """Run one queue-settings change and report the value before and after it."""
    require_queue(root, queue)
    apply_shared_perms(root)
    try:
        before, after = change()
    except QueueMetaUnreadable as exc:
        raise fail("{}", exc) from exc
    except OSError as exc:
        raise fail("could not write into the queue folder {}: {}", root, exc) from exc
    logger.info("queue {!r} {}: was {}, is now {}", queue, what, before, after)


@app.command()
def priority(
    queue: str = typer.Argument(..., help="Queue whose priority to set."),
    value: int = typer.Argument(..., help="Higher runs first among the ready queues."),
) -> None:
    """Set a queue's priority. Pools take it into account on their next pass."""
    configure_loguru()
    root = resolve_root()
    _change_setting(
        root,
        queue,
        lambda: store.set_queue_priority(root, queue, value),
        "priority",
    )


@app.command()
def pin(
    queue: str = typer.Argument(..., help="Queue to tie to machines, or to untie."),
    machines: list[str] = typer.Argument(
        None,
        help="Machines that may claim this queue, each optionally with the GPUs of it "
        "to use: gpu-host-1:0,1 gpu-host-2:4-7 other.",
    ),
    any_machine: bool = typer.Option(
        False, "--any", help="Untie the queue, so any machine may claim it."
    ),
    gpus: str = typer.Option(
        None,
        "--gpus",
        help="GPU numbers to use on whichever machine runs the queue, such as 0,1 or 4-7.",
    ),
    add: str = typer.Option(
        None, "--add", help="Add one machine (MACHINE or MACHINE:GPUS), keeping the rest."
    ),
    remove: str = typer.Option(
        None,
        "--remove",
        help="Remove one machine from the tie, keeping the rest. A machine name only: "
        "to stop using some of its GPUs, add it again with the ones to keep.",
    ),
) -> None:
    """Tie a queue to machines and to the GPUs of them, or untie it with --any.

    A pool claims from a queue only when its own hostname is among the machines, or the
    queue names none. A queue that names GPUs on a machine is admitted only to those
    GPUs there; the machine policy's own GPU list, yielded GPUs, memory and the caps all
    still apply, so the GPUs it can really use are the ones in both lists. Jobs that use
    no GPU follow the machine part and ignore the GPUs.
    """
    configure_loguru()
    root = resolve_root()
    chosen = [bool(machines or gpus), any_machine, bool(add), bool(remove)]
    if sum(1 for c in chosen if c) != 1:
        raise fail(
            "say what to tie {!r} to in one way: name the machines (with --gpus for the "
            "GPUs to use), or --any, or --add MACHINE, or --remove MACHINE",
            queue,
        )
    require_queue(root, queue)
    # Each of these reads the tie, changes it and writes it back inside one lock, so two
    # people adding a machine at the same moment do not write over each other.
    if remove:
        # The whole machine goes or it stays: a MACHINE:GPUS here reads as "stop using
        # those GPUs" and would silently remove the machine entirely instead.
        if ":" in remove:
            raise fail(
                "--remove takes a machine name, not {!r}: it removes the whole machine "
                "from the tie. To change which of its GPUs the queue uses, add it again "
                "with --add {}",
                remove,
                remove,
            )
        try:
            gone = store.MachineTie.parse(remove).machine
        except store.InvalidMachineTie as exc:
            raise fail("{}", exc) from exc
        change = lambda: store.remove_queue_tie(root, queue, gone)  # noqa: E731
    elif add:
        one = parse_ties([add])[0]
        note_machines_without_a_policy(root, (one,))
        change = lambda: store.add_queue_tie(root, queue, one)  # noqa: E731
    else:
        ties = () if any_machine else parse_ties(machines, gpus)
        note_machines_without_a_policy(root, ties)
        change = lambda: store.set_queue_ties(root, queue, ties)  # noqa: E731
    _change_setting(root, queue, change, "machines")


@app.command()
def park(
    queue: str = typer.Argument(..., help="Queue to set aside."),
    reason: str = typer.Option(None, "--reason", help="Why it is set aside."),
) -> None:
    """Set a queue aside: no pool claims from it, on any machine, until jobq unpark.

    The jobs it is already running finish normally, and queues that depend on it keep
    waiting. Parking is separate from the pause a run of failures causes and from the
    machine a queue is tied to; a queue can be parked as well as either of those.
    """
    configure_loguru()
    root = resolve_root()
    _change_setting(
        root,
        queue,
        lambda: store.set_queue_parked(root, queue, True, reason=reason),
        "parked",
    )


@app.command()
def unpark(
    queue: str = typer.Argument(..., help="Queue to bring back."),
) -> None:
    """Bring a parked queue back, so pools claim from it again."""
    configure_loguru()
    root = resolve_root()
    _change_setting(
        root,
        queue,
        lambda: store.set_queue_parked(root, queue, False),
        "parked",
    )


@app.command("reset-mem-floor")
def reset_mem_floor_cmd(
    queue: str = typer.Argument(..., help="Queue whose attempts sidecars to edit."),
    job: list[str] = typer.Option(
        None, "--job", help="Restrict to these jobkeys (repeatable; default: every job)."
    ),
    above: int = typer.Option(
        None,
        "--above",
        help="Only clear floors strictly above this many MiB — the wedge case, where a "
        "co-tenant OOM escalated a floor past what any GPU can grant.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before clearing the floors."),
) -> None:
    """Clear escalated per-job mem_mib floors in any job state (no requeue, results untouched)."""
    configure_loguru()
    root = resolve_root()
    require_queue(root, queue)
    apply_shared_perms(root)
    floors = store.read_mem_floors(
        root, queue, jobkeys=list(job) if job else None, above_mib=above
    )
    if not floors:
        out("No escalated mem_mib floor to clear in {!r}", queue)
        return
    word = "floor" if len(floors) == 1 else "floors"
    confirm_or_exit(
        [
            f"this clears {len(floors)} escalated memory {word} in {queue!r}, so those "
            "jobs ask for what their queue asks for again:",
            *[f"  {jk} {mib} MiB" for jk, mib in floors],
        ],
        "Continue?",
        yes=yes,
    )
    cleared = store.reset_mem_floors(
        root, queue, jobkeys=list(job) if job else None, above_mib=above
    )
    for jk, old in cleared:
        out("{} {} -> cleared", jk, old)
    out("Cleared mem_mib floor on {} job(s) in {!r}", len(cleared), queue)


@app.command()
def release(
    queue: str = typer.Argument(..., help="Queue holding the claim."),
    key: str = typer.Argument(..., help="Job key (or jobkey) whose claim to drop."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Releases a claim made by another machine too — only do this when the owning "
        "machine is confirmed dead or idle, or the job may run twice.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before releasing the claim."),
) -> None:
    """Force-remove a stuck claim, ignoring the machine guard (needs --force; never automatic)."""
    configure_loguru()
    root = resolve_root()
    require_queue(root, queue)
    if not force:
        raise fail("release ignores the machine safety guard; pass --force to confirm")
    apply_shared_perms(root)
    try:
        owner = store.read_owner(root, queue, store.sanitize_jobkey(key)) or {}
    except store.InvalidJobName as exc:
        raise fail("{}; nothing was released", exc) from exc
    where = owner.get("node") or "an unrecorded machine"
    confirm_or_exit(
        [
            f"this releases the claim on {key!r} in {queue!r}, held by {where} since "
            f"{owner.get('start_utc', 'an unrecorded moment')}; if that machine is still "
            "running the job, it will run twice.",
        ],
        "Continue?",
        yes=yes,
    )
    try:
        existed = store.force_release(root, queue, key)
    except store.InvalidJobName as exc:
        raise fail("{}; nothing was released", exc) from exc
    out("{} claim for {!r} in {!r}", "Released" if existed else "No", key, queue)


@app.command()
def stop(
    clear: bool = typer.Option(False, "--clear", help="Remove both stop requests instead of setting one."),
    now: bool = typer.Option(
        False,
        "--now",
        help="End this machine's running jobs at once and put them back in the queue.",
    ),
    yes: bool = typer.Option(False, "--yes", help="Answer the --now question with yes."),
) -> None:
    """Ask this machine's workers to finish and exit, or with --now to end their jobs at once."""
    configure_loguru()
    root = resolve_root()
    apply_shared_perms(root)
    host = store.this_host()
    p = store.stop_path(root, host)
    now_p = store.stop_now_path(root, host)
    if clear:
        p.unlink(missing_ok=True)
        now_p.unlink(missing_ok=True)
        logger.info("Cleared stop request {} (and {})", p, now_p)
        return
    if now:
        _stop_now(root, host, p, now_p, yes=yes)
        return
    try:
        store.ensure_dir(p.parent)
        p.touch(mode=jobq_io.lock_file_mode())
    except OSError as exc:
        raise fail("could not write into the queue folder {}: {}", root, exc) from exc
    logger.info("Set stop file {} (workers exit after current jobs)", p)


def _running_here(root: Path, host: str) -> int:
    """How many jobs this machine holds a claim on right now."""
    n = 0
    for name in store.list_queues(root):
        try:
            jobs = store.queue_status(root, name)["running_jobs"]
        except (OSError, store.QueueMetaUnreadable, store.QueuePathUnsafe):
            continue
        n += sum(1 for j in jobs if j.get("node") == host)
    return n


def _stop_now(root: Path, host: str, stop_file: Path, now_file: Path, *, yes: bool) -> None:
    """Ask the pool on this machine to end its running jobs and leave.

    The request needs a pool to answer it: without one, nothing is written, because a
    request left in the queue folder would end the jobs of the next pool to start. The
    stop file is written alongside the request, so a pool that reads only the stop file
    still finishes and leaves.
    """
    if not store.pool_state(root, host)["alive"]:
        logger.info(
            "no worker pool is running on {}; nothing was asked to end and no request was "
            "left behind",
            host,
        )
        return
    n = _running_here(root, host)
    word = "job" if n == 1 else "jobs"
    if not yes:
        question = (
            f"this ends {n} running {word} on {host}; "
            + ("it starts" if n == 1 else "they start")
            + " again from the beginning. Continue?"
        )
        if not sys.stdin.isatty():
            raise fail(
                "{} needs --yes when it is not run from a terminal; nothing was changed",
                question,
            )
        if not typer.confirm(question):
            logger.info("Nothing was changed")
            return
    try:
        store.ensure_dir(stop_file.parent)
        stop_file.touch(mode=jobq_io.lock_file_mode())
        now_file.touch(mode=jobq_io.lock_file_mode())
    except OSError as exc:
        raise fail("could not write into the queue folder {}: {}", root, exc) from exc
    logger.info(
        "Asked the pool on {} to end its {} running {} and put them back in the queue",
        host, n, word,
    )


def _node_policy_or_default(root: Path) -> GpuPolicy:
    """This machine's policy, or one carrying only defaults when the machine has no policy file.

    Enough for the MPS commands, which need the pipe/log directories and nothing else.
    """
    try:
        return load_policy(root, store.this_host())
    except (FileNotFoundError, ValueError):
        return GpuPolicy(gpus=[], cap_per_gpu=None, free_mem_mib=0)


@app.command("mps-stop")
def mps_stop() -> None:
    """Stop the MPS daemon this user's pools started on this machine (no-op when none runs)."""
    configure_loguru()
    found = queue_folder_source()
    if found is None:
        raise fail(
            "mps-stop reads this machine's policy to learn which directory the MPS pipe "
            "lives in, so it needs the queue folder. Run it before removing the queue "
            "folder. {}",
            no_queue_folder_message(),
        )
    policy = _node_policy_or_default(found[0])
    if not gpu_mod.mps_daemon_running(policy):
        logger.info("No MPS daemon of this user is running on {}", store.this_host())
        return
    if gpu_mod.stop_mps_daemon(policy):
        logger.info("Stopped the MPS daemon on {}", store.this_host())
    else:
        logger.warning(
            "asked the MPS daemon on {} to quit, but it is still answering; a client may "
            "still be attached", store.this_host(),
        )


@app.command()
def wait(
    queue: str = typer.Argument(..., help="Queue to block on."),
    poll_s: float = typer.Option(60.0, "--poll-s", help="Seconds between completion checks."),
) -> None:
    """Block until a queue is complete (every job terminal); for chaining from a shell."""
    configure_loguru()
    root = resolve_root()
    require_queue(root, queue)
    while not store.queue_complete(root, queue):
        st = store.queue_status(root, queue)
        logger.info(
            "waiting on {!r}: pending={} running={} done={} failed={}",
            queue, st["pending"], st["running"], st["done"], st["failed"],
        )
        time.sleep(poll_s)
    logger.info("queue {!r} complete", queue)


if __name__ == "__main__":
    app()
