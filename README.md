# jobq

Run a queue of shell commands on one or more GPU machines. Each machine runs a
worker pool; the pools share jobs through a folder they can all read and write.
There is no server or database to manage.

jobq is intended for one person using several Linux machines. It assigns one
GPU per job, supports CPU-only jobs, and records commands, logs, and results
as files. Memory checks and a limit on jobs per GPU control when jobs start.
They do not enforce a job's actual memory usage or provide isolation between users.

## Install

Requirements:

- Linux, Python 3.11 or newer, and Bash.
- NVIDIA GPUs and a working `nvidia-smi` on machines running GPU jobs.
- For multiple machines, a shared filesystem that supports atomic directory
  creation, atomic rename, and cross-machine file locking (`flock`).
- Your job's code, data, and software environment available on every machine
  that may run it. jobq does not copy them or install job dependencies.

From the directory containing this README, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jobq --help
```

`uv sync` installs from the lock file, so you get the versions jobq is tested
with. With pip instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
jobq --help
```

Install jobq on each worker machine. Activate the environment your commands need
before starting a pool, or use an absolute executable path in each command.
PyTorch is optional; jobq itself does not require it.

## Run your first queue

**1. Name the queue folder.** It is the folder that holds your queues, their
jobs, results and logs, and one policy file per machine. Write it in a file
named `jobq_paths.toml` in your project directory; jobq finds that file from
any directory inside the project. Copy `jobq_paths.example.toml` from this
repository and set the path, or write the file yourself:

```toml
# The folder holding the queues, their jobs, results and logs.
queue_folder = "/shared/me/jobq"
```

Use a path on a shared filesystem for several machines, or a local path for
one. Add `jobq_paths.toml` to the project's `.gitignore`, since the folder is
yours rather than the project's. `jobq config` prints the file in use and the
folder it names.

**2. Initialize this machine:**

```bash
jobq init
```

`init` writes `gpu_policy.<hostname>.json` inside the queue folder, selecting
every detected GPU. To use only some of them:

```bash
# Use this instead of the init command above.
jobq init --gpus 0,1
```

Every machine that runs a pool needs this file: `jobq work` refuses to start
without it and says which command writes it.

**3. Create a jobs file.** It is a text file with one job per line, each line a
JSON object with a unique name, `key`, and the shell command to run, `cmd`. A
line may also carry `mem_mib`, `slots`, `env` and `cwd` for that job alone; the
[usage guide](docs/usage.md) has the full format. This example checks placement
without requiring a training script. Run it from a directory that all worker
machines can access:

```bash
cat > jobs.jsonl <<'JOBS'
{"key":"check-1","cmd":"echo job=$JOBQ_JOB_KEY gpu=$CUDA_VISIBLE_DEVICES; hostname"}
{"key":"check-2","cmd":"echo job=$JOBQ_JOB_KEY gpu=$CUDA_VISIBLE_DEVICES; hostname"}
JOBS
```

For a sweep, write the lines from a shell loop:

```bash
for seed in 1 2 3; do
  echo "{\"key\":\"train-$seed\",\"cmd\":\"python train.py --seed $seed\"}"
done > jobs.jsonl
```

Or list one value per line and let jobq build the command:

```bash
printf '1\n2\n3\n' > seeds.txt
jobq submit demo --jobs-file seeds.txt --template 'python train.py --seed {line}'
```

Here and below, `demo` stands for the name of your queue.

**4. Submit and run:**

```bash
jobq submit demo --jobs-file jobs.jsonl --mem-mib 8000
jobq work
```

`--mem-mib` states the GPU memory one job of the queue needs: a queue holds
jobs of one kind, so the amount belongs to the queue, and a single job that
needs more carries its own `mem_mib` on its line.

The pool runs in the foreground and exits when the queues it can work on have
finished. Keep it in a persistent terminal, such as a `tmux` session, for long
jobs. Use `--workers N` to set the number of jobs the pool runs at once, and
`--queues a,b` to limit the pool to some queues.

Jobs run through `bash -c`, in the directory recorded at the queue's first
submission. Use `--cwd /shared/path/to/project` when creating the queue to
choose another directory. Each GPU job gets `CUDA_VISIBLE_DEVICES` set to its
assigned physical GPU; inside a CUDA application that device is normally `cuda:0`.

**5. Check the results** from another terminal in the same project:

```bash
jobq status demo
cat /shared/me/jobq/demo/results/check-1.json
```

The result contains the exit code (`rc`), machine, GPU, timestamps, and job log
path. Job output is under `demo/logs/` in the queue folder; pool activity is
under `logs/` there. `done` means exit code 0; `failed` means a nonzero or
unreadable result.

## Use several machines

On each machine, use a `jobq_paths.toml` naming the same shared folder, run
`jobq init`, and start `jobq work`. Each machine has its own policy and may run
one pool per queue folder. Hostnames must distinguish the worker machines. When the same
account has different numeric user ids on different machines, run
`jobq init --shared-perms` on each machine so files created on one machine can
be written from the other; see the [policy reference](docs/policy.md).

Pools claim jobs using atomic directory creation. A claimed job may still be
waiting for GPU capacity. Work can run again after retries or recovery, so
commands should tolerate being restarted. There is no exactly-once execution
guarantee; see [recovery and known limitations](docs/troubleshooting.md).

## Tell jobs how much memory they need

State the GPU memory each job needs. It is the most useful setting you can
give jobq. A job starts on a GPU only when that GPU has at least the requested
amount free, so an accurate request lets several jobs share a GPU without
running out of memory.

| Where | How | Applies to |
| --- | --- | --- |
| When creating a queue | `jobq submit demo --jobs-file jobs.jsonl --mem-mib 8000` | Every job in the queue |
| In the jobs file | `"mem_mib": 12000` on that job's line | That job |
| In the machine policy | `free_mem_mib` | Jobs whose queue and job line name no amount |

When neither the queue nor the job names an amount, jobq starts from the
policy's `free_mem_mib` and then learns from the queue itself: once a job of
the queue has a known peak, later jobs request 110% of the largest peak seen,
capped at what an idle GPU on the machine can grant. A job reports its peak by
printing `JOBQ_PEAK_GPU_MEM_MIB=<number>`; otherwise jobq measures how much GPU
memory appeared while the job loaded, which is less exact. See
[tuning](docs/tuning.md). This is a fallback. A stated amount is more reliable.

A job that runs out of memory is put back in the queue and requests more on its
next attempt. That recovers from a request that was too low, but the failed
attempt is lost work, and it can also bring down another job on the same GPU.

A job that has waited two minutes for memory reserves one GPU and keeps it
until it starts: no other job of yours starts there in the meantime. This keeps
a large job from waiting indefinitely behind a stream of small ones.

## Defaults

| Setting | Default | Change it with |
| --- | --- | --- |
| Jobs a pool runs at once | Workers per GPU, times the GPUs in this machine's policy | `jobq work --workers N` |
| Workers per GPU | (usable CPU cores - 2) / (all GPUs on the machine x the cores charged per job), rounded down | `cpu_reserve`, `cpu_per_gpu_job` in the policy |
| Jobs per GPU | The workers-per-GPU figure, and never more than memory allows | `cap_per_gpu` in the policy |
| Memory per job | A quarter of the smallest GPU, then 110% of the queue's largest known peak, capped at what an idle GPU grants | `--mem-mib`, `mem_mib`, `free_mem_mib` |
| Jobs that use no GPU, at once | A quarter of the usable CPU cores | `cpu_cap` in the policy |
| Failures in a row before a queue is paused | 5 | `jobq submit --max-consecutive-failures N` |
| How long a failure pause lasts | 900 seconds | `failure_pause_s` in the policy |
| Yielding to other users | Off | `yield_to_foreign` in the policy |
| NVIDIA MPS | On where the machine allows it | `mps` in the policy |
| Queue priority | 0; higher runs first | `jobq submit --priority N`, `jobq priority` |
| Utilisation sampling | Every 300 seconds | `monitor_interval_s` in the policy, `jobq work --no-monitor` |

Example: a machine with 48 cores and 8 GPUs gives 5 workers per GPU. A policy
that lists all 8 GPUs runs up to 40 jobs at once; one that lists 2 GPUs runs up
to 10. CPU cores are shared with everyone on the machine, which is why the
figure depends on all of its GPUs and not only on yours.

## Let jobs share a GPU with MPS

Where the machine allows it, use NVIDIA's Multi-Process Service: it lets
several of your jobs compute on one GPU at the same time instead of taking
turns on it. jobq turns it on by itself when the machine can, which is what the
policy's `mps` default of `"auto"` means, and runs without it, saying why, when
it cannot. `jobq status` says whether the daemon is running on this machine.
Set `mps` to `false` to keep it off, or to `true` on a machine that must use
it; see the [policy reference](docs/policy.md).

## Share a machine with other people

jobq does not coordinate with other users. It limits what your own jobs take,
using the machine policy file, `gpu_policy.<hostname>.json`:

| Policy key | Effect |
| --- | --- |
| `gpus` | The GPUs your jobs may use on this machine |
| `mem_budget_mib` | The most memory your jobs together may request on one GPU |
| `reserve_mem_mib` | Memory that must remain free on a GPU after one of your jobs starts |
| `yield_to_foreign` | Give up a GPU when another user's process appears on it |

With `yield_to_foreign` enabled, jobq stops placing jobs on a GPU once it sees
another user's process there. `yield_action` decides what happens to your jobs
already running on that GPU:

| `yield_action` | Running jobs |
| --- | --- |
| `kill` | Stopped immediately and returned to the queue |
| `drain` | Left to finish |
| `drain_if_near_done` | Nearly finished jobs are left to finish; the rest are stopped and returned to the queue |

A job stopped for a yield is not counted as failed. jobq uses the GPU again
after it has seen no other user's process there for `yield_cooldown_s`, which
defaults to 15 minutes. Most policy edits take effect while a pool is running;
the [policy reference](docs/policy.md) lists every key.

## Watch utilisation

A running pool samples its machine every five minutes — each GPU's
utilisation and memory, the processor, and the slots the pool holds — and
appends the readings to `monitor/` in the queue folder. It is on by default;
`jobq work --no-monitor` turns it off for one run. `jobq usage --since 7d`
reads what every machine wrote and prints one table of GPUs, processor and
slots, with the jobs each machine finished per hour beneath it. A machine that
should be watched without running jobs runs `jobq monitor`, which samples the
same way in the foreground.

## Everyday commands

| Task | Command |
| --- | --- |
| Inspect every queue and this machine's pool | `jobq status` |
| See how busy the machines were, and jobs per hour | `jobq usage --since 7d` |
| Sample a machine that runs no pool | `jobq monitor` |
| Preview a submission | `jobq submit demo --jobs-file more.jsonl --dry-run` |
| Add new, uniquely named jobs | `jobq submit demo --jobs-file more.jsonl` |
| Retry failed jobs and clear a failure pause (asks first; `--yes` in scripts) | `jobq requeue demo` |
| Clear a pause without retrying finished jobs | `jobq resume demo` |
| Wait until all jobs have results, including failures | `jobq wait demo` |
| Change a queue's priority | `jobq priority demo 5` |
| Set a queue aside, and bring it back | `jobq park demo --reason "..."`, `jobq unpark demo` |
| Tie a queue to machines or GPUs | `jobq pin demo gpu-host-1 gpu-host-2:0,1`, `jobq pin demo --any` |
| Let this machine's active jobs finish, then exit | `jobq stop` |
| End this machine's active jobs now and return them to the queue | `jobq stop --now` |
| Allow a stopped machine to work again | `jobq stop --clear`, then `jobq work` |

Queue settings are saved on the first submission. Later submissions append jobs;
they cannot change the queue's settings. Use `jobq priority`, `jobq pin`, and
`jobq park` to change priority, machine and GPU ties, and the parked state of
an existing queue. For scripts, note that `jobq wait` does not signal job
failures with a failing exit status. Use strict dependencies when the next
queue requires success.

## Stop a pool

| Action | Running jobs | Pool |
| --- | --- | --- |
| `jobq stop` | Finish normally | Claims nothing new; exits when they finish |
| Ctrl-C or `kill <pid>`, any number of times | Finish normally | Same as `jobq stop` |
| `jobq stop --now` | Ended and returned to the queue | Exits |
| `kill -9 <pid>` | Left running | Ends at once; the next pool on that machine ends those jobs before it runs them again |

`jobq stop --now` asks for confirmation; pass `--yes` in scripts. A job returned
to the queue starts again from the beginning, so commands should tolerate that.

Every command reads the queue folder from `jobq_paths.toml`, so they all act on
the folder of the project you are in. Each command has `--help`.

## Documentation

| Guide | What you will find |
| --- | --- |
| [Submitting and running jobs](docs/usage.md) | Job fields, templates, environment, dependencies, priorities, and CPU-only work |
| [Machine policy reference](docs/policy.md) | Every policy key, defaults, and when edits take effect |
| [Tuning throughput](docs/tuning.md) | Memory reporting, concurrency, CPU budgets, and MPS |
| [Troubleshooting and recovery](docs/troubleshooting.md) | Waiting jobs, retries, pauses, stopping, and stale claims |
| [Queue folder reference](docs/queue-folder.md) | State files, logs, local locks, and cleanup |
| [Development](docs/development.md) | Set-up, source map, and the execution path |

## Acknowledgements

The design and the original implementation of this queue are the author's. Much of the
code, the tests, the trials and the documentation in this repository were written with
Claude (Anthropic) working under the author's direction; the author reviewed and is
responsible for all of it.

## License

[MIT](LICENSE).
