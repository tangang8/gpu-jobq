# Tuning throughput

[README](../README.md) · [Policy](policy.md) · [Troubleshooting](troubleshooting.md)

Measure completed jobs per hour while changing one setting at a time. A fuller
GPU is useful only if the workload finishes sooner.

## Start with a baseline

For a new machine, `jobq init --cap-per-gpu 1` provides an explicit one-job
limit on each GPU. Without such a cap the limit is this machine's workers per
GPU, described under [worker threads](#worker-threads). The default memory
request is a quarter of the smallest selected GPU, so several jobs share a GPU
from the start; state the memory each job needs rather than relying on it.

Run representative jobs and record their duration and peak memory. One job
per GPU is a useful starting point; whether sharing helps depends on the work.
Even a job running alone can run out of memory.

## Measure memory use

Any job can report a peak by printing an integer number of MiB near the end of
its output:

```text
JOBQ_PEAK_GPU_MEM_MIB=7200
```

For PyTorch, add this near application startup:

```python
from jobq.report import maybe_register

maybe_register()
```

The helper registers an exit hook only when `JOBQ_REPORT_PEAK_MEM=1`, which
jobq sets by default. It reports PyTorch's peak reserved and allocated CUDA
memory if CUDA was initialized. It imports PyTorch lazily; missing PyTorch or
reporting errors produce no reading. The environment running your script must
have jobq installed to import this helper.

The worker reads the tail of the log. Abrupt termination may prevent an exit
hook from running. The helper's reading excludes the CUDA context and non-PyTorch
allocations, so allow extra space above it.

Find readings in `jobq status`, the pool's `END` lines, and result fields
`peak_mem_mib` / `peak_mem_alloc_mib`. Retry attempts without a terminal result
may still have readings in the job and pool logs.

After at least five successful jobs report peaks, `jobq status` suggests:

```text
suggested MiB = largest successful peak × 1.15, rounded up to 500 MiB
```

This is advice only. It does not update policy or account for every source of
memory use. The displayed fit estimate uses GPU total memory and does not
subtract reserves, other users' allocations, or other admission limits.

## The memory a job asks for

State the memory each job needs. It is the one setting that lets jobq place
jobs well, and a queue or job that states it is never affected by what other
jobs used. The request is decided in this order:

1. the job's own `mem_mib`;
2. the queue's `--mem-mib`;
3. the value learned for that queue, when there is one;
4. this machine's policy `free_mem_mib`.

`jobq init` writes `free_mem_mib` as a quarter of the smallest selected GPU,
rounded down to a multiple of 500 MiB: a modest start rather than a whole GPU.

A queue that states no memory learns one from its own finished jobs. Once at
least one job of the queue has a known peak, the rest of the queue asks for
110% of the largest known peak, rounded up to a multiple of 100 MiB and never
above what a GPU in the policy grants while it is idle. An idle GPU never
shows its whole total free, since the driver holds some of it back, so the
ceiling is the highest free reading this machine has taken, or the GPU's total
less `oom_ceiling_headroom_mib` when no reading has been taken yet, and in
either case less `reserve_mem_mib`. A request the ceiling cut down is marked as
capped in `jobq status QUEUE`.

A peak the job reported with `JOBQ_PEAK_GPU_MEM_MIB=` is used first; failing
that, the pool's own measurement of how far the GPU's free memory fell while
the job was starting up. When one job of this queue folder was granted on the
GPU in that window, the whole fall is its own. When several were granted
within one window of each other, their memory appears together: the fall is
measured from the free reading taken at the earliest of those grants and
divided among them in proportion to what they asked for. Such a figure is
marked shared, and the queue only learns from shared figures once three of them
lie within 20% of the smallest of the group, the largest of that group being the
one used. Jobs that failed or ran out of memory never lower the value; a larger
peak raises it. `jobq status QUEUE` prints the learned value and how many
peaks, reported or measured, are behind it, and the queue overview's memory
column marks it as learned.

The measurement is weaker than a reported peak: it cannot see memory a job
takes later, and it counts whatever else happened on the GPU during the
window. A shared figure is weaker again, since the split follows the requests
rather than what each job really took, and a job whose neighbours are still
loading when its own window ends gets too small a share. Report peaks from the
jobs, or state the request, when it matters.

## Choose requests and concurrency limits

Use a request above the largest representative peak, then compare throughput
at different concurrency levels. A machine policy could contain:

```json
{
  "gpus": [0, 1],
  "free_mem_mib": 12000,
  "cap_per_gpu": 3,
  "reserve_mem_mib": 2048
}
```

Each ordinary job requests 12000 MiB unless overridden, and at most three
slot units may be held on each GPU. The reserve and actual free-memory readings
may allow fewer jobs. Edit `free_mem_mib` for jobs using the machine default;
use job `mem_mib` or a new queue's `--mem-mib` for a workload-specific request.

| Control | What it limits |
| --- | --- |
| Job `mem_mib` / queue `--mem-mib` | Memory requested by one job |
| Policy `free_mem_mib` | Request for jobs with no job, queue or learned value |
| Policy `cap_per_gpu` | Held slot units per GPU across this lock namespace |
| Queue `--cap-per-gpu` | Jobs per GPU in that queue's cap group |
| Policy `mem_budget_mib` | Sum of running requests plus the incoming request per GPU |
| Policy `reserve_mem_mib` | Extra free memory required at admission |
| `work --workers` | Jobs that pool threads can run or wait on simultaneously |

An OOM retry can add a persistent memory floor above the job's configured
request. Changing `free_mem_mib` will not lower that floor. See
[resetting memory floors](troubleshooting.md#out-of-memory-retries).

A job with `slots: 2` uses two units of the policy cap on one GPU, without
multiplying its memory request. With `cap_per_gpu: 4`, it leaves two units for
other jobs. A weight above the policy cap is clamped to that cap. Without an
explicit policy cap, the internal range is 64 units.

The queue cap counts each job once, whatever its slot weight. For example,
a queue with `--cap-per-gpu 2` can place two jobs on a GPU only if their combined
slot weight and memory also fit the machine's limits.

## Worker threads

More workers allow more jobs to wait for or use capacity; they do not bypass
GPU or CPU limits. The cores of a machine are shared with everyone on it, so
the pool takes a share of them per GPU:

```text
workers per GPU = max(1, floor((usable cores - cpu_reserve)
                               / (GPUs on the machine * cpu_per_gpu_job)))
workers         = workers per GPU * GPUs in this machine's policy
```

GPUs on the machine means every GPU `nvidia-smi` lists, not only the selected
ones; when that cannot be read, the policy's own GPUs stand in and the pool
says so. Yielded GPUs still count as this pool's. A policy selecting no GPU
runs `cpu_cap` threads. With 48 usable cores, 2 reserved and 8 GPUs on the
machine, a GPU gets 5: a policy with all 8 GPUs runs 40 workers, one with 2
GPUs runs 10, and either way a GPU takes at most 5 slot units unless
`cap_per_gpu` says otherwise. The pool logs the count and every input to it on
its `WORKERS` line.

Choose a fixed initial count with `jobq work --workers 8`. To change a running
pool's target on the current machine:

```bash
python - <<'PY'
import socket
from pathlib import Path
from jobq.io import atomic_write_text

root = Path("/shared/me/jobq")  # the queue_folder from jobq_paths.toml
atomic_write_text(root / f"workers.{socket.gethostname()}", "8\n")
PY
```

The supervisor reads this file at `supervise_s` intervals. Threads above a
reduced target exit after their current work; adding workers starts more
threads. The file persists and can override `--workers` on later runs. Removing
it leaves a running pool at its current target. Use positive integers.

CPU-only jobs use a separate `cpu_cap`, but still need available worker
threads. Set application thread counts to match the cores you assume a job
takes; the scheduler counts jobs and does not enforce CPU usage.

`OMP_NUM_THREADS`, which policies often set in `env`, is one such thread count:
the letters stand for OpenMP, and it limits the processor threads certain
numerical libraries use, such as the processor-side calculations of PyTorch and
NumPy. It matters mainly for processor work and has little effect on GPU
training, and it does not limit data loading, tokenizers or the processes a job
starts. jobq passes it to jobs and does not act on it: the cores a job is
charged come from `cpu_per_gpu_job` alone.

## See idle GPUs and finished jobs together

Concurrency settings are worth changing when you can see what they cost.
`jobq usage --since 24h` puts both halves of that question in one table: for
each GPU, how busy it was, how often it was idle, and how often none of your
jobs was on it; under the table, the successful jobs each machine finished per
hour over the same window. A machine with idle GPUs and a low rate is one to
raise `cap_per_gpu` or lower `mem_mib` on; one whose GPUs stay busy while the
rate falls is one to leave alone or give fewer jobs at once.

Take a reading before and after a change, over windows of the same length and
with the same kind of work in the queue, since the rate depends on what the
jobs are. The samples come from the pool itself, so they cost nothing to keep:
see [watching utilisation](usage.md#watch-utilisation).

## MPS, and measuring what it gives you

jobq enables CUDA's Multi-Process Service for jobs sharing a GPU wherever the
machine allows it: the policy's `mps` defaults to `"auto"`, so a pool starts or
reuses this user's daemon when the control binary answers, and runs without it,
saying why, when it does not. To measure what it is worth on your workload,
drain and restart the pool with `"mps": false` and compare throughput with the
same jobs. Set `"mps": true` on a machine that must use it: the pool then
refuses to start rather than run without it.

MPS does not increase the GPU's memory or bypass jobq's admission checks.
The pool does not stop the daemon when it exits. After its clients finish,
use `jobq mps-stop`. It reads this machine's policy to learn which directory the
MPS pipe lives in, so it needs a `jobq_paths.toml` and says so when
there is none; run it before removing the queue folder. See
[MPS settings](policy.md#cuda-mps).

## Memory-accounting limits

Admission combines whole-GPU free-memory readings with the requests recorded
for jobs holding capacity locks. It cannot enforce future allocations.

- Slow or staged allocation: a job can allocate more after another job has
  started. `startup_hold_s` protects recent requests only for a limited period.
- Other GPU users: their allocations can change between readings, or look
  like one of your jobs has finished allocating and release a startup hold early.
- Overestimated requests: `mem_budget_mib` counts requests, so an oversized
  request can block useful work even when actual GPU use is low.
- Independent roots: default locks are separate for each queue root. Their
  recorded requests are not combined unless you deliberately share a local
  lock namespace with compatible settings.

If jobs fail while loading, inspect their real memory use and loading time
before raising concurrency. If capacity waits persist, the pool's `WAIT` lines
and [troubleshooting guide](troubleshooting.md) help identify the constraint.
