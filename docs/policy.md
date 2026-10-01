# Machine policy reference

[README](../README.md) · [Usage](usage.md) · [Tuning](tuning.md)

Each worker machine reads
`gpu_policy.<hostname>.json` in the queue folder, where `<hostname>` is the value returned
by Python's `socket.gethostname()`. Queue defaults live separately in
`<queue>/meta.json`.

Create the policy with `jobq init`. To replace an existing policy with fresh
initial values, use `jobq init --force`; this overwrites your edits.

## A small policy

```json
{
  "gpus": [0, 1],
  "free_mem_mib": 12000,
  "cap_per_gpu": 2,
  "reserve_mem_mib": 2048,
  "env": {"OMP_NUM_THREADS": "2"}
}
```

This allows up to two ordinary, one-slot jobs on each listed GPU, provided the
memory checks pass. A job without a memory override requests 12000 MiB, and
admission also leaves 2048 MiB of headroom. These numbers are an example; choose
requests from your own workload's [measured memory use](tuning.md).

`gpus` and `free_mem_mib` are required. Other keys have the defaults below.
Use JSON booleans (`true`/`false`), numbers for numeric settings, and lists where
specified. Unknown keys are ignored, so check spelling carefully. Memory values
are in MiB; timing values are in seconds.

## When edits take effect

| Settings | When read |
| --- | --- |
| GPU list, memory checks, CPU and GPU caps | On capacity checks; changes affect subsequent admissions |
| `env`, `cwd_fallback` | When starting a job |
| Retry and kill settings | When the relevant action runs |
| Yield settings | On watchdog polls; the policy must exist when the pool starts |
| `shared_perms` | At pool startup |
| Monitoring settings | At pool startup |
| `mps`, `mps_pipe_dir`, `mps_log_dir` | At pool startup |
| `poll_s`, `gpu_wait_s`, `supervise_s` | At pool startup |
| Initial automatic worker count | At pool startup; later changes use `workers.<hostname>` |

Lowering `cap_per_gpu`, `cpu_cap` or a queue's cap group never stops a job that
is already running: it keeps the slot it took until it ends. Those held slots
are counted whatever their number, so a GPU holding the new cap's worth or
more takes no further job, and the memory asks recorded on it stay in the
`mem_budget_mib` accounting. While that lasts, `jobq status` reports the true
number of slots held, which can be above the cap, and says so.

A job already waiting for capacity keeps the memory request calculated when
it was claimed. Changing `free_mem_mib`, a queue default, or a stored OOM floor
does not recompute that waiting request. Drain and restart the pool when you
need those jobs to be claimed with a new request. Running jobs keep their
existing environment and allocation.

To restart cleanly: run `jobq stop` on that machine, wait for the pool to exit,
then run `jobq stop --clear` and `jobq work` with the desired options.

During normal policy validation failures, a running pool reports the problem
and stops admitting jobs until the file is fixed. It does not terminate jobs
already running. A missing policy file is refused at startup: `jobq work` exits
non-zero without taking the pool lock or claiming anything, and names the file
to write and the `jobq init` command that writes it. `jobq status` reports queue
state and any policy problem it encounters.

## GPU selection and memory

| Key | Default | Meaning |
| --- | --- | --- |
| `gpus` | Required | List of distinct, nonnegative physical GPU indices. `[]` permits CPU-only work only. |
| `free_mem_mib` | Required | Memory request when the job, its queue and the queue's learned value all name none. Must be positive when the policy names no `cap_per_gpu`; may be `0` with one. A nonpositive job request falls back to this value on a machine whose policy names no cap. |
| `cap_per_gpu` | The workers-per-GPU figure below | Positive number of slot units per GPU. Omitted or `null`, the cap is this machine's workers per GPU, worked out when the policy is read. A fixed ceiling of 64 slot units per GPU bounds any cap. |
| `reserve_mem_mib` | `0` | Extra free memory required in addition to the incoming request and startup holds. This is an admission check, not an enforced reservation. |
| `mem_budget_mib` | `0` | Maximum sum of running jobs' memory requests plus the incoming request, per GPU in this lock namespace. `0` disables this limit. |
| `startup_hold_s` | `300` | How long recent requests whose allocations are not yet visible count against free memory. `0` disables the hold. |
| `mem_checks` | `2` | Number of consecutive passing free-memory readings required; at least `1`. The fast path can finish earlier. |
| `mem_interval_s` | `3` | Delay between memory readings; nonnegative. |
| `mem_fastpath_factor` | `2` | Skip further readings when headroom is at least this multiple of the request and at least request + 4096 MiB. `0` disables the shortcut. |

`jobq init` writes `free_mem_mib` as a quarter of the smallest selected GPU's
total memory, rounded down to a multiple of 500 MiB, when it can read that
total; otherwise it uses 8000 MiB. It prints how many jobs that fits on a GPU,
and recommends stating the memory per job at submit time instead of relying on
it. It does not write a slot cap unless you pass `--cap-per-gpu`, and prints the
cap that applies without one. Use an explicit cap of `1` if you require one job
at a time per GPU.

The request of one job is decided in this order: the job's own `mem_mib`, then
the queue's `--mem-mib`, then the value learned for that queue, then this key.
An out-of-memory retry raises the request above any of them.

A queue that states no memory learns one from its own jobs. Once at least one
job of the queue has a known peak, the others request 110% of the largest known
peak, rounded up to a multiple of 100 MiB, and never more than the largest GPU
in this policy can grant. A job's peak is the one it reported with
`JOBQ_PEAK_GPU_MEM_MIB=`; failing that, it is the footprint the pool measured
from the GPU itself, which is the fall in that GPU's free memory over the
job's start-up window, recorded only when no other job of this queue folder was
granted on that GPU during the window. Failed jobs and jobs that ran out of
memory never lower the learned value. It is kept in the queue folder and shown
by `jobq status`; see [the memory default](tuning.md#the-memory-a-job-asks-for).

All memory admission checks must pass. Neither the request nor the budget
limits what the command actually allocates. The startup accounting is an
estimate based on whole-GPU readings, with [known limits](tuning.md#memory-accounting-limits).

## CPU capacity and worker count

| Key | Default | Meaning |
| --- | --- | --- |
| `cpu_cap` | `max(1, usable cores // 4)` | Concurrent CPU-only jobs on this machine. Explicit `0` disables CPU-only admission. |
| `cpu_reserve` | `2` | Cores excluded from the calculated budgets; nonnegative. |
| `cpu_per_gpu_job` | `1.0` | Assumed cores per held GPU slot unit; positive. |

Usable cores means the process's CPU affinity where available, with CPU count
as fallback. Reducing `cpu_cap` does not terminate existing jobs. These
settings count jobs; they do not set CPU affinity or force application thread
counts.

The cores of a machine are shared with everyone on it, so the pool takes a
share of them per GPU rather than the whole machine:

```text
workers per GPU = max(1, floor((usable cores - cpu_reserve)
                               / (GPUs on the machine * cpu_per_gpu_job)))
```

GPUs on the machine means every GPU `nvidia-smi` lists there, not only the ones
`gpus` selects. When that list cannot be read, the number of GPUs in this
policy stands in and the pool says so on its `WORKERS` line.

Without `work --workers`, the pool runs `workers per GPU` threads for each GPU
in this policy, counting GPUs currently yielded to another user. A policy that
selects no GPU at all runs only `slots: 0` work, so its thread count is the
`cpu_cap` budget. The `WORKERS` line records the number chosen and every input to it.

`cores_per_job` is `cpu_per_gpu_job` and nothing else; nothing in `env`
affects it. The same workers-per-GPU figure is the default
`cap_per_gpu`. This is separate from `cpu_cap`. See
[changing the worker count](tuning.md#worker-threads).

## Yielding to other GPU users

| Key | Default | Meaning |
| --- | --- | --- |
| `yield_to_foreign` | `false` | Stop admitting work on GPUs where foreign activity is detected. |
| `yield_poll_s` | `30` | Positive interval between detection passes. |
| `yield_confirm_polls` | `2` | Consecutive positive sightings before yielding; at least `1`. |
| `yield_min_foreign_procs` | `1` | Minimum foreign-process count that triggers a sighting. |
| `yield_cooldown_s` | `900` | Observed quiet time before a yielded GPU becomes available again. |
| `yield_action` | `"kill"` | `"kill"`, `"drain"`, or `"drain_if_near_done"`; see below. |
| `yield_drain_threshold` | `0.8` | Minimum estimated progress to spare a job under `drain_if_near_done`, from `0` to `1`. |
| `yield_drain_max_s` | `1800` | Maximum drain time from the recorded yield, checked on polls while foreign activity persists. |
| `yield_progress_regex` | Unset | Regex with at least two capture groups: completed work, then total work. |
| `yield_to_users` | `[]` | Local usernames to count as foreign even when visible. Unknown usernames are skipped. |
| `yield_to_uids` | `[]` | Same override using nonnegative numeric UIDs. |

Actions apply to this pool's jobs on the yielded GPU:

- `kill`: terminate them and put them back in the queue.
- `drain`: let them finish; block new admissions on that GPU.
- `drain_if_near_done`: spare jobs whose estimated progress meets the threshold;
  terminate and requeue the others.

Progress comes from the last usable regex match in the log, then elapsed time
relative to the median duration of at least five successful jobs in the queue.
If neither estimate is available, progress is zero. For output such as
`step 80/100`, a JSON pattern is `"step ([0-9]+)/([0-9]+)"`.

Detection compares NVIDIA's compute-process counts with locally attributable
GPU processes. It is a heuristic, not a reliable map of all other users.
Visible local processes normally count as local unless selected by the user/UID
overrides. Container visibility, device attribution, and unreadable process
information can affect the estimate. If information is unavailable, the
watchdog preserves its current decision. Turning yielding off clears its GPU
markers on the next successful watchdog pass.

## Retries and process termination

| Key | Default | Meaning |
| --- | --- | --- |
| `oom_patterns` | Built-in CUDA/PyTorch markers | List of case-sensitive text fragments in the log tail. A nonempty list replaces the built-ins; an empty list uses them. |
| `oom_mem_factor` | `1.5` | Multiplier for an increased OOM memory request; at least `1`. |
| `oom_mem_floor_mib` | `4096` | Minimum request when increasing an OOM floor; at least `1`. |
| `oom_ceiling_headroom_mib` | `2048` | Headroom below the largest selected GPU's total, after subtracting `reserve_mem_mib`, when computing the OOM ceiling. |
| `oom_max_requeues` | `8` | Automatic OOM retries before recording failure; `0` disables retries. |
| `oom_own_usage_fraction` | `0.75` | Below this fraction of its request, a job may be treated as suffering pressure from other jobs, keeping its request unchanged. If the log gives a failed allocation size, own use plus that allocation must also fit the old request. |
| `tempfail_retry_s` | `900` | Delay before retrying exit code `75`. |
| `tempfail_max_requeues` | `8` | Automatic exit-75 retries before recording failure. |
| `park_defer_max_s` | `600` | How long a worker waiting for capacity defers to eligible higher-priority waiters before also trying. Read when entering the wait. |
| `reserve_after_s` | `120` | How long a claimed job waits for capacity before one GPU is held for it; `0` turns holding off. |
| `reserve_max_gpus` | `1` | How many GPUs of this machine may be held for waiting jobs at once. |
| `kill_grace_s` | `20` | Time allowed after `SIGTERM` before attempting `SIGKILL` for a job being terminated by the pool. |
| `orphan_claim_grace_s` | `120` | Minimum age before reclaiming a claim with no owner record; also requires two sightings at least five seconds apart. |

A job whose request is larger than the memory smaller jobs keep taking would
otherwise wait for ever. Once such a job has waited `reserve_after_s`, and its
request is one that some GPU in this policy could grant if that GPU held none
of this queue folder's jobs, one GPU is held for it: the GPU where it would
fit soonest, meaning the most free memory after `reserve_mem_mib`, ties going
to the GPU holding the fewest slot units.

A GPU is held only for a job that is waiting for memory, which means it needs
more than the next job to end would give back: on every GPU it may use, what
the GPU shows free less `reserve_mem_mib`, plus the largest single request
held there, is still short of its own request. A job merely queued behind
`cap_per_gpu` or a `cap_group` is waiting for a slot, and the next job to end
gives it one, so no GPU is held for it.

While a GPU is held, no other job of this queue folder starts on it, whatever
its priority; the jobs already running there run to the end, and the other
GPUs are unaffected. One GPU carries one reservation, and at most
`reserve_max_gpus` GPUs are held on a machine whichever queues hold them, so
holding cannot stall it. Among several waiting jobs the highest priority holds
a GPU, then the longest wait, and a job that outranks the holder takes the
GPU over there and then, leaving that job to wait like any other. The hold
ends when the job is admitted
anywhere, when it stops waiting for any reason, when the GPU is yielded, or
when the pool that made it has ended; a request no GPU could ever grant never
holds one, and the pool log says so once. The hold is recorded beside the slot
locks, so a second pool on the machine honours it, and the pool log has
`RESERVE` and `RELEASE` lines. See
[a job waiting for a large request](troubleshooting.md#a-job-waiting-for-a-large-request).

Retry counts and memory floors live with the job in the shared queue folder,
so they survive pool restarts. Exhausted automatic retries produce failed
results but do not count toward the queue's consecutive-failure pause.
The pause threshold is a queue setting, configured by
`submit --max-consecutive-failures` (default `5`; `0` disables it), not a policy
key. See [retry and recovery commands](troubleshooting.md).

## Pool timing

These three settings require a restart. Each must be greater than `0` and no
more than `3600`.

| Key | Default | Meaning |
| --- | --- | --- |
| `poll_s` | `20` | Idle-worker delay before checking ready queues again. |
| `gpu_wait_s` | `5` | Delay between capacity attempts for a claimed, waiting job. |
| `supervise_s` | `30` | Supervisor interval for worker recovery, scaling, and cleanup. |

## Environment, files, and permissions

| Key | Default | Meaning |
| --- | --- | --- |
| `env` | `{}` | Variables added to jobs, before queue and per-job overrides. |
| `cwd_fallback` | Unset | Working directory to use if the requested directory is unavailable. |
| `shared_perms` | `false` | Enable world-writable queue-state creation and set the pool's umask to `0`. Child jobs inherit that umask. Requires restart. |

A note on `OMP_NUM_THREADS`, which policies often set in `env`. The letters
stand for OpenMP. It limits the number of processor threads certain numerical
libraries use, for example the processor-side calculations of PyTorch and
NumPy. It matters mainly for processor work such as analysis and has little
effect on GPU training. It does not limit a job's data loading, its tokenizers
or the processes it starts. jobq passes it to jobs and does not act on it: the
cores a job is charged come from `cpu_per_gpu_job`, and nothing reads anything
out of `env`.

Locks use `/tmp/jobq_<root-hash>`, derived from the queue folder. Explicit
`work --lock-prefix` (or `status --lock-prefix`) takes precedence over
`JOBQ_LOCK_PREFIX`, then that default. Keep capacity locks local to the
machine; the queue state itself is what belongs on shared storage. All pools
sharing a lock namespace need compatible GPU indices and limits.

`shared_perms` is for a trusted shared location when the same person's numeric
UID differs between machines. It broadens access to both jobq state and files
created by child jobs under the inherited umask. It does not repair every
existing file's permissions or bypass inaccessible parent directories.

## Monitoring

| Key | Default | Meaning |
| --- | --- | --- |
| `monitor_interval_s` | `300` | Seconds between rounds of utilisation samples. `0` turns sampling off on this machine. |
| `monitor_idle_util_pct` | `5` | Below this GPU utilisation, and below `monitor_idle_mem_mib` in use, a GPU's sample counts as idle. From `0` to `100`. |
| `monitor_idle_mem_mib` | `1024` | The memory half of that idle test. |
| `monitor_idle_cpu_pct` | `10` | Below this processor utilisation, a processor sample counts as idle. From `0` to `100`. |
| `monitor_cpu_sample_s` | `5` | Seconds between the two `/proc/stat` readings a processor utilisation is the ratio of. |
| `monitor_keep_days` | `30` | How long sample rows are kept; the pool drops older rows once a day. |
| `heartbeat_stale_s` | `180` | How old a machine's heartbeat may be before `jobq status` says it has not been heard from. Above `0`. |

The pool samples while it runs, on a thread of its own, and writes the rows
into `monitor/` in the queue folder; `jobq work --no-monitor` turns sampling
off for one run without editing the policy. A machine that should be watched
without running jobs runs `jobq monitor`. See
[the sample files](queue-folder.md#monitoring-files) and
[`jobq usage`](usage.md#watch-utilisation).

## CUDA MPS

| Key | Default | Meaning |
| --- | --- | --- |
| `mps` | `"auto"` | `"auto"` uses an MPS daemon when this machine can, `true` demands one, `false` never uses one. |
| `mps_pipe_dir` | `/tmp/jobq_mps_<uid>/pipe` | MPS pipe directory. |
| `mps_log_dir` | `/tmp/jobq_mps_<uid>/log` | MPS log directory. |

All three settings require a pool restart. With `"auto"`, the pool starts or
reuses this user's daemon when the control binary is there and answers, and
logs `MPS enabled`; when it cannot, it logs one line saying why and runs
without MPS. With `true` the pool refuses to start in that case, naming the
reason, which is what a machine that must use MPS wants. With `false` the pool
never looks for a daemon. The daemon survives pool exit because
other pools may use it. After its clients finish, `jobq mps-stop` asks this
user's daemon on the current machine to stop. It does not stop worker pools.
