# Queue folder and state files

[README](../README.md) · [Usage](usage.md) · [Recovery](troubleshooting.md)

The `queue_folder` in `jobq_paths.toml` points to the folder shared by all pools.
Each queue is a direct child directory with `meta.json` and `jobs.jsonl`.
Capacity locks live separately on each worker machine.

```text
<root>/
├── gpu_policy.<hostname>.json
├── worker.<hostname>.lock            # held by the pool for its whole life
├── worker.<hostname>.pid
├── worker.<hostname>.json
├── .worker.<hostname>.lock.rmw     # serializes rewrites of worker.<hostname>.json
├── stop.<hostname>                 # present when a stop is requested
├── stop.now.<hostname>             # present when that stop is to end running jobs
├── workers.<hostname>              # optional live worker-count target
├── yielded.<hostname>.json
├── .submit.lock                    # serializes submissions and settings changes
├── monitor/
│   ├── gpu.<hostname>.csv
│   ├── gpu.<hostname>.log
│   ├── cpu.<hostname>.csv
│   ├── cpu.<hostname>.log
│   ├── slots.<hostname>.csv
│   ├── slots.<hostname>.log
│   ├── fleet_slots.log             # one line per sample for all machines together
│   └── <name>.lock                 # one per file, appends against the daily trim
├── logs/
│   └── <stamp>_worker.<hostname>.log
└── <queue>/
    ├── meta.json
    ├── jobs.jsonl
    ├── claims/<jobkey>/owner.json
    ├── results/<jobkey>.json
    ├── attempts/<jobkey>.json
    ├── logs/<stamp>_<jobkey>.a<attempt>.<hostname>.log
    ├── failures.json
    ├── failures.unreadable.<stamp>.json
    ├── paused.json
    ├── complete.state.json
    ├── mem_learned.json           # the memory this queue's jobs turned out to need
    ├── .jobs.lock
    ├── .attempts.lock
    ├── .claims.lock
    ├── .mem_learned.lock
    └── .failures.lock
```

Many files are created only when needed. The stored format version is `1` in
`meta.json`; a missing `format_version` is interpreted as version 1. jobq
accepts any numeric version it finds and reports one it cannot read as a
number, so this field is not a compatibility guarantee.

A queue's machine tie is stored in one field or two. A queue tied to exactly
one machine with no GPU list has that machine's name in `node` and no
`machines` key, so a reader of that one field sees the whole tie. Anything
fuller — several machines, or GPUs named
on them — is written in `machines`, a list of objects with a `machine` name
(`null` for GPU numbers that apply to whichever machine runs the queue) and a
list of GPU numbers; `node` then carries the one machine name when the tie
names one, and is `null` otherwise. When `machines` is present it is the field
that counts:

```json
"node": null,
"machines": [
  {"machine": "gpu-host-1", "gpus": [0, 1]},
  {"machine": "gpu-host-2", "gpus": [4, 5, 6, 7]}
]
```

A queue folder written before the optional files and keys here still reads and
drains: a `meta.json` with no `machines` key is a queue tied by `node` alone, a
queue with no `mem_learned.json` asks the machine default until one of its jobs
has a known peak, a `meta.json` with no `parked` key is a queue that is not
parked, and a result with no `measured_mem_mib` is one nothing was measured
for. A `node` starting with `PARKED` is read as parked, with the text after the
word as the reason; that convention applies to `node` only.

## How job states are determined

| Files for a job | Reported state |
| --- | --- |
| No claim or result | `pending` |
| Claim directory, no result | `running` — claimed, possibly still waiting for capacity |
| Result with `rc == 0` | `done` |
| Nonzero or unreadable result | `failed` |

A result takes precedence over a leftover claim. A deferred retry remains
pending, but its attempts record prevents claiming until `not_before`.
A queue is complete when it has at least one job and every job has a result.
Complete does not mean successful.

A claim is created with atomic `mkdir`. Its owner record is written afterward
and records the hostname, pool PID, start time, GPU, process identity, and
`run_id`, an identifier of this run of this job. The identifier is written
before the job starts, and the job's environment carries it as `JOBQ_RUN_ID`,
so the next pool on that machine can find the run's processes however they were
rearranged. A job whose identifier cannot be written into the claim is not
started and the claim is given back.

Once the job's process exists, the record also carries `job_pid`, `job_pgid`,
`job_pid_start` and `job_boot_id`: the job runs in its own session, so those
are the quick way for the next pool on that machine to find and end a job whose
pool was killed, and they are what `jobq status` reads to say whether the job's
own process is still running. A claim written without any of these keys is
simply reclaimed when its pool is gone. See
[recovering after a crash](troubleshooting.md#recover-after-a-crash).

The queue's directory and the `claims`, `results`, `attempts` and `logs`
directories inside it must be real directories, not symbolic links: the names
under them are built from a job's key, and a link would put the entries
somewhere else. A queue with a link in one of those places is refused by every
command, named by `jobq status`, and not claimed from. The queue folder itself
and any directory above it may be reached through links.

## Machine files in the root

| File | Purpose and lifecycle |
| --- | --- |
| `gpu_policy.<hostname>.json` | Machine configuration written by `init` or edited by you. Removing it blocks new admissions on that machine. |
| `worker.<hostname>.lock` | Held with `flock` by the pool on this machine for its whole life, taken before anything else. A second pool on the same machine and queue folder is refused while the first lives, whatever the pid file holds. The kernel drops the lock when the process ends. The file itself is never removed. |
| `worker.<hostname>.pid` | Pool PID marker, the human-readable record of which process the pool is. Normal shutdown and an interrupted shutdown both remove it. |
| `worker.<hostname>.json` | Pool log path, boot ID, process start identity used for local liveness checks, and `heartbeat_utc`, rewritten on every supervisor tick. Removed at normal pool exit. |
| `stop.<hostname>` | Persistent stop request: the pool finishes its running jobs and exits. `jobq stop --clear` removes it. |
| `stop.now.<hostname>` | Request to end the running jobs at once, written by `jobq stop --now` beside the stop file. The pool removes it when it exits; `jobq stop --clear` removes it too. |
| `workers.<hostname>` | Optional positive worker-count target, checked by the supervisor. Persists across runs. Removing it does not reset a running pool's target. |
| `yielded.<hostname>.json` | GPUs withheld from admission by the yield watchdog. The watchdog updates and clears entries. |
| `logs/<stamp>_worker.<hostname>.log` | Pool activity, including claims, waits, starts, outcomes, retries, and exits. A job put back because a GPU was yielded is logged under `YIELD REQUEUE`; one put back because the pool was interrupted under `INTERRUPT REQUEUE`; a job that outlived its pool under `ENDED SURVIVOR`. |

### Pool log verbs

Every line of `logs/<stamp>_worker.<hostname>.log` begins with a UTC timestamp
and then one of these verbs.

| Verb | What the line says |
| --- | --- |
| `ALL QUEUES COMPLETE` | The pool left and every queue in the folder is complete. |
| `CLAIM` | A job was claimed by this pool. |
| `CLAIM KEPT` | Claim recovery looked at a claim and left it where it is, with the reason. |
| `DRAIN` | A nearly finished job on a yielded GPU is spared and left to finish. |
| `DRAIN REQUEST` | A signal asked the pool to finish its jobs and claim nothing new. |
| `END` | A job finished, with its exit code and any peak-memory reading. |
| `END NOW` | `jobq stop --now` asked the pool to end its running jobs. |
| `ENDED SURVIVOR` | A job that outlived the pool that started it was ended. |
| `INTERRUPT` | The pool ended one running job for `jobq stop --now`. |
| `INTERRUPT REQUEUE` | A job ended that way is pending again. |
| `KILL` | A running job was killed so its GPU could be yielded. |
| `OOM BACKSTOP` | A job used up its out-of-memory retries, so the exit counts as a failure. |
| `OOM CEILING` | A job ran out of memory while already asking for all a GPU here can grant. |
| `OOM REQUEUE` | A job that ran out of memory is pending again with a higher request. |
| `PAUSE` | A queue stopped being claimed after a run of consecutive job failures, with the time the pause ends. `PAUSE LAPSED <queue>` says the pause has ended and this worker tries one job while the queue stays paused for every other worker. |
| `POOL EXIT` | The pool left with work outstanding, and why. |
| `QUEUE COMPLETE` | Every job of one queue is terminal. |
| `REAP` | Debris was removed: a leftover temporary file, an orphan claim of this pool, or a leaked slot lock. |
| `RECLAIM` | A yielded GPU is available to this pool again. |
| `RELEASE` | A GPU that was held for a waiting job is free for other jobs again. |
| `RESERVE` | A GPU is held for a job that has waited too long for memory. |
| `RESPAWN` | A worker thread that ended without returning was started again. |
| `RESUME` | A queue's failure pause has been cleared, so it is claimed from again. |
| `RUN ID` | A job was handed back unrun because its claim carries no run identifier. |
| `SCALE` | The live worker target changed, following `workers.<hostname>`. |
| `SIDECAR` | A job was made terminal because its attempts record could not be updated. |
| `START` | A job's process started, with the GPU it was granted. |
| `STEAL` | A claim left behind by a pool that is gone was handed back to the queue. |
| `TEMPFAIL` | A job exited 75 and is deferred until its retry time. |
| `TEMPFAIL BACKSTOP` | A job did that too often, so the exit counts as a failure. |
| `WAIT` | A worker is waiting for capacity, the pool says why its workers are leaving, or a queue's jobs are left for other machines because their working directory is not reachable here. |
| `WORKERS` | The thread count the pool chose at startup and the figures behind it. |
| `YIELD` | A GPU was yielded to another user, with the foreign process count. |

`jobq status` can verify a pool's process only on the machine where the command
runs; remote PID numbers are not useful for local process checks. For the other
machines it reads `heartbeat_utc` instead, and says that a machine's pool is
alive or has not been heard from since a given time. The heartbeat is
information: a machine's claims are still recovered by that machine's own pool.

## Monitoring files

The `monitor/` directory holds three CSV files per machine, appended by the
pool (or by `jobq monitor` on a machine that runs no pool) every
`monitor_interval_s` seconds. Each file starts with a header row and is only
ever appended to; rows older than `monitor_keep_days` are dropped once a day,
header kept.

| File | One row per | Columns |
| --- | --- | --- |
| `gpu.<hostname>.csv` | GPU in this machine's policy, per sample | `timestamp,gpu_index,util_pct,mem_used_mib,mem_total_mib,power_w,idle,our_jobs` |
| `cpu.<hostname>.csv` | sample | `timestamp,ncpu,util_pct,busy_cores,load1,load5,load15,mem_used_mib,mem_total_mib,idle` |
| `slots.<hostname>.csv` | sample | `timestamp,node,used,slots,wait,gpus,yielded,cap_per_gpu,live` |

The timestamp is ISO 8601 with the UTC offset. The GPU readings come from
`nvidia-smi`; `idle` is `1` when utilisation is below `monitor_idle_util_pct`
and memory in use is below `monitor_idle_mem_mib`, and `our_jobs` counts this
queue folder's jobs holding a slot on that GPU. The processor utilisation is
the ratio of two `/proc/stat` readings `monitor_cpu_sample_s` apart,
`busy_cores` is that share of `ncpu`, the load averages come from
`/proc/loadavg`, the memory from `/proc/meminfo` as total less available, and
`idle` is `1` below `monitor_idle_cpu_pct`. The slots row holds what the pool's
jobs hold (`used`), the capacity behind it (`slots`, the policy GPUs less the
yielded ones times the cap in force), how many claimed jobs wait for capacity
(`wait`), and `live`, which is `1` for a pool and `0` for `jobq monitor`.

Beside each CSV file is a log of the same name ending in `.log`, with one line
per sample saying the same thing in words, for reading with `tail`:

```text
gpu.<hostname>.log    2026-10-03T02:41:54+00:00 6/8 idle | 0:idle 1:idle 2:idle 3:idle 4:idle 5:idle 6:busy(75%,30719MiB) 7:busy(97%,30719MiB)
cpu.<hostname>.log    2026-10-03T02:41:54+00:00 busy | util 36.9% (~18/48 cores) load 19.4/20.1/21.7 mem 159002/740466MiB
slots.<hostname>.log  2026-10-03T02:41:54+00:00 8/40 slots used, 0 waiting | 8 gpus, 0 yielded, cap 5 per gpu | pool live
```

A busy GPU's entry gives its utilisation and its memory in use. The logs are trimmed with the CSV
files, by `monitor_keep_days`. `jobq usage` and `jobq status` read the CSV
files only.

`fleet_slots.log` is the one file all machines share. Each line gives the slots
in use against the slots on offer across every machine with a policy file, and
then each machine's own figures:

```text
2026-10-03T02:41:54+00:00 13/80 | gpu-host-1 8/40 | gpu-host-2 5/40 (1 yielded) +2w | gpu-host-3 0/0 (down)
```

A machine is `(down)`, and offers no slots, when its pool's heartbeat is older
than its `heartbeat_stale_s`, which is the test `jobq status` applies. A machine
whose pool is alive is described by the last slots row it wrote: `(N yielded)`
counts its yielded GPUs and `+Nw` its claimed jobs waiting for capacity. A live
pool with no slots row in the last three sampling intervals, such as one started
with `--no-monitor`, shows as `?/? (no samples)` and is left out of the sum.
Every sampler offers a line each round and the first one in an interval writes
it, so the log has one line per interval however many machines sample.

## Files inside a queue

| File | Purpose | Supported operation |
| --- | --- | --- |
| `meta.json` | Queue name, creation time, priority, dependencies, machine tie (`node`, and `machines` for the fuller form), whether it is parked and why, defaults, format version | Set at first submission; `priority`, `pin`, `park` and `unpark` change one setting each afterwards |
| `mem_learned.json` | The request learned for a queue that states no memory: `request_mib`, the largest peak behind it, and how many peaks were reported by the jobs or measured from the GPU | Written under `.mem_learned.lock` when a result with a peak is recorded; removing it starts the queue over at the machine default |
| `jobs.jsonl` | Submitted job objects, in submission order | Append through `jobq submit` |
| `claims/<jobkey>/owner.json` | Ownership while a worker waits or runs | Automatic cleanup when safe; `release --force` for a verified dead job |
| `results/<jobkey>.json` | Terminal exit code, machine, GPU, timestamps, log path, attempt, optional memory readings (`peak_mem_mib` and `peak_mem_alloc_mib` as the job reported them, `measured_mem_mib` as the pool measured it from the GPU) and failure reason | `requeue` removes selected results to make jobs pending again |
| `attempts/<jobkey>.json` | Attempt counter, OOM retries, exit-75 retries, memory floor, retry time | `requeue`, `requeue --reset-oom`, or `reset-mem-floor`, depending on the field |
| `failures.json` | Consecutive counted failures and their keys | `resume` or `requeue` clears the streak; a success also clears it |
| `paused.json` | Queue pause: `since_utc`, `until_utc` (absent when the pause does not end by itself), `keys` of the failures behind it, `limit`, and `probe_utc` (when one job was last let through after the pause ended) | At `until_utc` one worker rewrites `until_utc` to now plus `failure_pause_s`, sets `probe_utc`, and tries one job; any job success removes the file; `resume` or `requeue` removes it at once |
| `failures.unreadable.<stamp>.json` | Unreadable failure state preserved for inspection | Inspect before removing |
| `complete.state.json` | Completion cache based on jobs-file size and result count | Automatically checked; removing it only costs a fresh completion scan |
| `logs/<stamp>_<jobkey>.a<attempt>.<hostname>.log` | Combined stdout/stderr plus a job header, one file per attempt so the out-of-memory classification reads this attempt's output alone | Read directly; a result that has a log names it in `log` |
| `.jobs.lock`, `.attempts.lock`, `.failures.lock`, `.mem_learned.lock`, `.claims.lock` | Serialize state updates; their contents are unused | Leave in place while any process accesses the queue |

The root also holds `.submit.lock`, which serializes creating a queue, checking
a submission against its settings and appending its jobs.

The jobkey is derived from the job's key. Separators and whitespace become
`__`; a long key is shortened to fit the filename limit in bytes, on a
character boundary, and a hash of the whole key is appended so the name stays
unique. A key whose derived log file name would still not fit on this machine
is rejected at submit.

Job logs use second-resolution timestamps and are opened for writing. Rapid
retries of the same job on the same machine within one second can reuse a log
path; do not treat this naming scheme as a guaranteed archive of every attempt.
Terminal results also represent the current recorded outcome, not a full
execution history.

## Machine-local capacity files

The default prefix is `/tmp/jobq_<hash>`, with 12 hex characters derived from
the resolved queue-root path. Symlink aliases of the same root resolve to the
same namespace; distinct roots normally have separate namespaces.

| Suffix after the prefix | Purpose |
| --- | --- |
| `_gpu<g>_slot<s>.lock` | One GPU slot unit, held with `flock` |
| `_gpu<g>_slot<s>.ask` | Memory request, grant time and free-memory baseline associated with an active slot |
| `_gpu<g>.reserve` | A GPU held for one waiting job: the pool, the thread, the job's key, its request and when the hold began |
| `_cpu_slot<s>.lock` | One CPU-only job slot |
| `_grp_<group>_gpu<g>_slot<s>.lock` | One job slot within a queue cap group |
| `_acquire.lock` | Serialize GPU admission checks |

The existence of a `.lock` file does not mean its capacity is occupied;
the held kernel lock is what matters. Locks are released when descriptors
close or the pool process dies. Orphaned job processes can outlive those locks.
An `.ask` record whose slot lock is not held is removed when inspected.

`--lock-prefix` and `JOBQ_LOCK_PREFIX` can override the prefix; see
[policy precedence](policy.md#environment-files-and-permissions).
Use the same prefix for `status` that the pool uses. Shared namespaces need
compatible limits across pools and must stay local to each machine.

MPS uses `/tmp/jobq_mps_<uid>/pipe` and `/tmp/jobq_mps_<uid>/log` by default.
These belong to the daemon, which can outlive the pool. Stop its clients and
use `jobq mps-stop` before cleaning those directories.

## Cleanup and manual edits

Prefer the CLI for state changes. Removing a result can rerun a job; removing
a live claim can run it twice; removing attempts records loses retry limits
and memory floors. Deleting `jobs.jsonl` or `meta.json` loses the queue's work
or configuration. Never unlink a lock file while another process may hold it:
a new file at the same path would represent a different lock.

Before manual state maintenance, request a stop on every relevant machine
and wait for every pool and its job processes to finish. `jobq stop` itself
does not wait. Retain logs and results you still need; deleting a log leaves
any result pointing to it with a missing file.

Atomic state writes use temporary files containing `.tmp.` in their names,
then rename them into place. Logs and PID/stop markers use other write paths.
Pool startup removes matching temporary files older than one hour from the
root and its immediate queue directories; it does not recursively sweep every
claims/results/attempts directory. A recent temporary file may belong to an
active write, so do not assume it is abandoned.
