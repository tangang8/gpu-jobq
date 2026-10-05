# Submitting and running jobs

[README](../README.md) · [Policy](policy.md) · [Troubleshooting](troubleshooting.md)

Name the queue folder in a `jobq_paths.toml` file, as below; every command
reads it from there.

## Where the queue folder comes from

jobq reads the queue folder from one file, `jobq_paths.toml`, in the current
directory or in the nearest directory above it. Nothing on the command line
changes it, so every command run inside a project acts on the same folder:

```toml
# The folder holding the queues, their jobs, results and logs.
queue_folder = "/shared/me/jobq"
```

Keys jobq does not know are ignored. A file that cannot be parsed stops the
command with a message naming the file. A relative path is taken from the
directory of the file that holds it, and a leading `~` is expanded; an absolute
path is clearest. With no such file, a command that needs the folder says so
and how to write one.

The same file may name where the utilisation samples go:

```toml
# Optional. Left out, the samples go into monitor/ beside this file.
monitor_folder = "/shared/me/jobq-monitor"
```

Left out, they go into `monitor/` in the directory of the file, which is the
project itself. `monitor_folder` is read as `queue_folder` is: a relative path
is taken from the directory of the file, and a leading `~` is expanded.

`jobq config` prints the file in use, the queue folder it names and the folder
the samples go into, and writes nothing; `jobq config --json` prints the same
three paths as a JSON object for scripts.

The queue folder belongs to you rather than to the project, so a project that
keeps a `jobq_paths.toml` usually adds that file to its `.gitignore` and tells
its readers to write their own. The `monitor/` directory beside it holds this
machine's readings and belongs in the `.gitignore` too.

## Set up this machine

`jobq init` writes `gpu_policy.<hostname>.json` into the queue folder, creating
the folder when it is not there, and writes every commonly needed key out
explicitly so the file reads as this machine's whole configuration. Run it once
per machine. `jobq init --force` overwrites an existing policy.

`jobq work` requires that file: on a machine without it the pool refuses to
start and names the command to write it, so it cannot claim jobs it has no
capacity to admit. The read-only commands, `jobq status` and `jobq usage` among
them, work on a machine with no policy.

| Option | What it sets |
| --- | --- |
| `--gpus 0,1` | The GPU indices this machine's pool may use. Left out, every GPU found. |
| `--cap-per-gpu N` | A hard limit on the slot units held per GPU. Left out, the policy names no cap and memory alone decides how many jobs share a GPU. |
| `--free-mem-mib N` | The memory a job asks for when nothing else names one, in MiB. Left out, a quarter of the smallest GPU's total, rounded down to a multiple of 500 MiB. |
| `--cpu-cap N` | How many jobs that use no GPU (`slots: 0`) may run at once here. Left out, the policy does not name it and a quarter of the usable cores applies. `0` refuses such jobs on this machine. |
| `--mps` / `--no-mps` | Demand a shared MPS daemon, or refuse one. Left out, the policy says `"auto"`: the pool uses MPS where this machine can and runs without it where it cannot. |
| `--shared-perms` | Create the files jobq writes so that anyone can write them, for an account whose numeric user id differs between the machines sharing the queue folder. |

`init` prints how many jobs the memory it chose fits on a GPU and the slot cap
that applies, and it recommends stating the memory per job at submit time
rather than relying on the default. The [policy reference](policy.md) documents
every key, including the ones `init` does not write.

## Jobs files

A JSONL file contains one job object per line. Blank lines and lines starting
with `#` are ignored. This example shows all per-job fields:

```json
{"key":"train-seed-1","cmd":"python train.py --seed 1","mem_mib":12000,"slots":1,"cwd":"/shared/me/project","env":{"OMP_NUM_THREADS":"2"}}
```

| Field | Meaning | When omitted |
| --- | --- | --- |
| `key` | Human-readable identifier, unique within the queue | The original JSON line becomes the key; an explicit short key is easier to use |
| `cmd` | Command passed to `bash -c` | Required |
| `mem_mib` | GPU memory request in MiB; use a positive integer for GPU jobs | Queue `--mem-mib`, then policy `free_mem_mib` |
| `slots` | Weight on one GPU, or `0` for CPU-only work | Queue `--slots`, then `1` |
| `cwd` | Working directory; use an absolute path | Queue `--cwd`, or the directory of the first submission |
| `env` | Environment variables added or overridden for this job | Queue and machine values still apply |

`slots: 2` reserves two units of capacity on one GPU. It does not request two
GPUs. jobq does not schedule a multi-GPU allocation for a single job.

A key becomes a filename called its jobkey: pipes, slashes, backslashes,
and whitespace runs become `__`. For example, `train|seed-1` becomes
`train__seed-1`. Two keys that produce the same jobkey are rejected, including
duplicates in the same submission. Empty keys and keys starting with a dot are
rejected. A key longer than a filename allows is shortened by bytes, without
splitting a character, and a hash of the whole key is appended so the name stays
unique; a key whose log file name would still not fit is rejected at submit,
with a message naming it. Short keys such as `train-seed-1` read best.

Queue names are directory names directly under the root, so a name is one path
component of letters, digits, underscores, hyphens and dots, at most 64
characters, and it may not start with a dot. Anything else is rejected with a
message naming it, and nothing is created. Simple names such as `train`, `eval`
or `experiment-01` are the intent.

Commands execute with the pool user's permissions. Submit trusted commands
and job files. Keep the job in the foreground until all its work finishes;
background children are not separately tracked as jobs.

## Templates for repeated commands

For a parameter sweep, plain lines can be shorter than JSON. Save this as
`seeds.txt`:

```text
small|1
small|2
large|1
```

Expand and inspect the commands before submitting:

```bash
jobq submit sweep --jobs-file seeds.txt \
  --template 'python train.py --model {0} --seed {1}' --dry-run

jobq submit sweep --jobs-file seeds.txt \
  --template 'python train.py --model {0} --seed {1}'
```

The full line becomes the key. `{line}` inserts the whole line; `{0}`, `{1}`,
and later indices insert fields separated by `|`. This is Python string
formatting, with no automatic shell quoting. Double literal braces as `{{`
and `}}`. A line starting with `{` is always parsed as JSON, even when a
template is supplied.

`--dry-run` expands jobs and checks duplicate keys and conflicting stored
settings without creating the queue. It does not run commands, validate their
inputs, or check whether their resource requests fit a machine.

## Queue defaults and appending work

Set shared values when you first create the queue:

```bash
jobq submit train --jobs-file train.jsonl \
  --cwd /shared/me/project --mem-mib 12000 --slots 1 \
  --env OMP_NUM_THREADS=2 --env PYTHONUNBUFFERED=1
```

Later, append only new keys:

```bash
jobq submit train --jobs-file more-train.jsonl
```

Appending keeps the original queue settings, including its working directory,
even if the new submission happens elsewhere. An explicitly supplied setting
that disagrees with the stored one is rejected. To use different defaults,
create a new queue or supply supported overrides in each new job object.
Creating the queue, comparing a submission against its settings and appending
the jobs happen under one lock, so of two first submissions that ask for
different settings exactly one creates the queue and the other is rejected
without writing anything.

Submitting the same file twice is an error, not a way to retry it. Use
`jobq requeue` for completed jobs. A new queue must contain at least one job.

## Environment and working directory

Ordinary environment variables are merged in this order; later entries win:

1. The environment inherited by `jobq work`.
2. The machine policy's `env`.
3. MPS connection variables, when MPS started successfully.
4. Queue variables from `submit --env`.
5. Per-job `env`.

The worker then sets these placement and job identity variables:

| Variable | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Assigned physical GPU index, or empty for CPU-only jobs |
| `JOBQ_GPU` | Same physical index, or empty for CPU-only jobs |
| `JOBQ_QUEUE` | Queue name |
| `JOBQ_JOB_KEY` | Original job key |
| `JOBQ_NODE` | Worker hostname |
| `JOBQ_ATTEMPT` | Stored attempt counter, starting at `0` |
| `JOBQ_RUN_ID` | Identifier of this run of this job, written into the claim before the job starts |

Every process the job starts inherits `JOBQ_RUN_ID`, and that is how a pool
starting on this machine later finds processes left over from a run whose pool
was killed. Do not unset it in a job's command.

`JOBQ_REPORT_PEAK_MEM` defaults to `1` unless already set. It enables the
optional [memory-reporting helper](tuning.md#measure-memory-use); it does not
make an arbitrary command report memory automatically. The attempt counter
is bookkeeping for recorded retries, not a unique execution identifier after
an abrupt crash.

Each attempt writes its own log, `<queue>/logs/<stamp>_<jobkey>.a<attempt>.<hostname>.log`
in the queue folder, so the output of one attempt is never mixed with another's
and the out-of-memory check reads the attempt it is judging. A result names the
log it belongs to in its `log` field.

The shell is noninteractive. Do not rely on your interactive shell startup
files activating an environment. For example, use
`/shared/me/project/.venv/bin/python train.py` in `cmd` when appropriate.

If the requested working directory is unavailable, jobq tries the machine
policy's `cwd_fallback`. If neither is usable, the job is not run on this
machine and stays pending for other machines; the pool log says so once per
queue with a `WAIT` line. Use absolute paths in per-job `cwd`; relative paths are interpreted
from the pool's working directory. A fallback does not translate paths inside
your command or copy missing files.

## Dependencies, priority, and machine selection

Create a second queue that waits for training:

```bash
jobq submit evaluate --jobs-file evaluate.jsonl \
  --depends-on train --strict-deps
```

| Option | Behavior |
| --- | --- |
| `--depends-on train,prepare` | Wait for every job in each named queue to have a result |
| `--strict-deps` | Also require zero failed jobs in each dependency |
| `--priority 10` | Prefer this ready queue over queues with lower priority; default is `0` |
| `--node gpu-host-1` | Only a pool with this exact hostname may claim the queue. Repeatable, and each one may name the GPUs to use there: `--node gpu-host-1:0,1 --node gpu-host-2:4-7` |
| `work --queues train,evaluate` | Limit this pool to the named queues |

The option is spelled `--node` because that is also the name of the field the
tie is stored in and of the `JOBQ_NODE` variable a job receives; everywhere
else this documentation calls it a machine.

By default, failed jobs count as finished for dependencies. Use `--strict-deps`
when downstream work requires successful upstream outputs. Missing or empty
dependency queues cannot complete. Dependency cycles are not rejected at
submission and leave their queues waiting.

A worker that holds a claim while it waits for capacity re-reads its queue's
settings on every poll. If the queue is parked, if a run of failures pauses it,
or if its tie stops naming this machine or the GPUs the job was going to use,
the worker hands the claim back and the job is pending again for whichever
machine the queue now belongs to. That is not an attempt and not a failure.

Priorities affect selection and workers waiting for capacity; they do not
interrupt running jobs. Equal priorities are ordered by queue creation time.
The pool may claim several jobs before GPU space becomes available.

Dependency checks happen before a job is selected. Adding or requeuing upstream
work does not undo downstream jobs that have already started or finished.
Requeue downstream work yourself when its inputs need to be regenerated.

## Change a queue that already exists

A queue's settings are written once, at its first submission, but four of them
can be changed afterwards. Each command changes only its own setting, leaves
every other one as it was, prints the old value and the new one, and refuses a
queue that does not exist.

```bash
jobq priority train 10                    # pools use it on their next pass
jobq pin train gpu-host-1                 # only that machine may claim from it
jobq pin train gpu-host-1:0,1 gpu-host-2:4-7      # those machines, and those GPUs of them
jobq pin train --gpus 0,1                 # those GPU numbers, on whichever machine runs it
jobq pin train --add third:2              # one more entry, keeping the others
jobq pin train --remove gpu-host-2            # that machine goes, the others stay
jobq pin train --any                      # untie it again
jobq park train --reason "waiting for new data"
jobq unpark train
```

`--remove` takes a machine name and removes the whole entry; to change which of a
machine's GPUs the queue uses, add it again with `--add machine:GPUS`.

A machine may carry the GPUs of it the queue may use, written after a colon as single
numbers and ranges: `gpu-host-1:0,1`, `gpu-host-2:4-7`, `gpu-host-2:0,4-6`. A machine with no list
means every GPU there. A queue tied to GPUs is admitted only to those GPUs on that
machine; everything else still applies, so the GPUs it can really use are the ones both
in the tie and in that machine's policy, minus any yielded to another user. Jobs that
use no GPU (`slots: 0`) follow the machine part of the tie and ignore the GPU part.

A tie naming GPUs that a machine's policy does not select means the jobs of the queue
that take a GPU cannot run there: pools on that machine do not claim those jobs and no
worker waits on them, and `jobq status QUEUE` run there says so, naming the GPUs asked
for and the ones selected. Jobs with `"slots": 0` take no GPU and follow only the
machine part of the tie, so a pool there still claims them, whatever the queue's own
default is.
`jobq pin` prints a note when a named machine has no policy file in the queue folder,
which is what a machine that has not yet run `jobq init` looks like; the name is still
accepted.

A parked queue is claimed from by no pool, on any machine, until it is
unparked. The jobs it is already running finish normally, and queues that
depend on it keep waiting. Parking is separate from the pause a run of failures
causes and from the machine a queue is tied to: a queue can be parked as well
as either of those, and unparking does not clear a pause.

A failure pause stops queues being selected; it cannot stop the jobs that were
already claimed when it was set. Those run to the end and record their results,
so with many workers a few more failures than the limit can occur before the
queue stops. A job that used up its out-of-memory or exit-75 retries counts
towards the pause like any other failed job; only a fault of the pool itself
does not (see the [troubleshooting guide](troubleshooting.md)).

A parked queue does not keep a pool alive. A pool exits once every queue it
could work on is complete, counting neither parked queues, nor queues holding
no jobs of their own, nor queues tied to other machines, nor queues whose
dependencies can never be met: a dependency that is parked, that holds no jobs,
that is not a queue in this folder at all, that sits in a ring of queues
waiting on each other, or that itself waits on one of those. Its
last log line names each queue it leaves behind and the reason. A queue waiting
on one tied to another machine does keep the pool running, since the machine it
is tied to can still finish it. `ALL QUEUES COMPLETE` is written only when
every queue in the folder is complete. A queue a run of failures has paused
keeps the pool running, and so does a queue waiting on one, since `jobq resume`
can put them back to work at any moment.

A queue whose machine tie starts with `PARKED` is read as parked as well, with
whatever follows the word as the reason; `jobq unpark` on such a queue removes
the tie.

## CPU-only jobs

For an entire CPU-only queue:

```bash
jobq submit prepare --jobs-file prepare.jsonl --slots 0
```

Or set `"slots": 0` on individual jobs. These jobs use the machine's `cpu_cap`
budget and receive an empty `CUDA_VISIBLE_DEVICES`. They still occupy a worker
thread; CPU and GPU jobs share the pool's thread count. Declaring a separate
CPU queue lets a saturated CPU budget be skipped during queue selection.

A machine with no GPU can use this policy, written to
`gpu_policy.<hostname>.json` in the queue folder:

```json
{
  "gpus": [],
  "free_mem_mib": 1,
  "cpu_cap": 4
}
```

Then run `jobq work --queues prepare --workers 4`. The normal `jobq init`
autodetection needs a GPU; write the policy above yourself for a CPU-only machine.

## Limit one queue or a group of queues

```bash
jobq submit heavy-a --jobs-file a.jsonl --cap-per-gpu 2 --cap-group heavy
jobq submit heavy-b --jobs-file b.jsonl --cap-per-gpu 2 --cap-group heavy
```

Together, these queues may hold two jobs per GPU in that lock namespace.
Queues sharing a group must declare the same cap. Without `--cap-group`, the
group name defaults to the queue name.

The queue cap counts jobs; the machine policy cap counts slot units.
Both must allow a GPU job to start. Memory checks apply as well. See
[tuning](tuning.md) for examples.

## The queue overview

`jobq status` without a queue name prints one table of the queues in the queue
folder, then the lines about this machine:

```text
queue   state                    pending  running  done  failed  priority  runs on      memory per job  time per job     time left   last activity
train   running                  7        1        4     0       5         any machine  20000           about 15m (n=4)  about 1.9h  just now
eval    ready                    5        0        0     0       1         any machine  unknown         unknown          unknown     just now
report  waiting for train, eval  3        0        0     0       0         any machine  unknown         unknown          unknown     just now
flaky   paused                   2        0        0     0       0         any machine  unknown         unknown          unknown     just now
1 complete queue is not shown; use --all to list every queue
```

The rows come in the order a pool on this machine takes the queues, so the
first row is what runs next: the queues that can be worked on here now, by
priority and then by creation time, followed by the queues tied to another
machine, the queues waiting for a dependency, the queues a run of failures has
paused, the queues set aside with `jobq park`, the queues holding no jobs, and
the queues whose settings could not be read. A queue counts as one this machine
can work on when this machine is among its machines, or it names none, and when
it names GPUs here at least one of them is a GPU this machine's policy selects.

| Column | What it says |
| --- | --- |
| queue | The queue name, never shortened. |
| state | `running`, `ready`, `only on` the machine the queue is tied to, `waiting for` the dependencies named after it, `paused`, `parked`, `empty` for a queue holding no jobs, `complete`, or `unreadable`. `jobq status QUEUE` gives the reason a queue is parked. |
| pending | Jobs with no result and no claim, saying how many of them are parked for a later retry. |
| running | Jobs a pool holds a claim on right now. |
| done | Jobs whose result is an exit code of zero. |
| failed | Jobs whose result is anything else. |
| priority | The queue's priority; higher is taken first. |
| runs on | The machines the queue is tied to, written as they are typed (`gpu-host-1:0,1 gpu-host-2:4-7`), or `any machine`. The column is as wide as it needs to be. |
| memory per job | One figure: the queue's `mem_mib`, or the value learned from its finished jobs marked as learned, or the `free_mem_mib` of the machine's policy marked as the machine default. The jobs' own values follow it only when they are not all equal to it, then `slots` when it is not one, and `no GPU` for a queue whose jobs use none. |
| time per job | The median run time of the queue's finished jobs and how many they are. |
| time left | The rough estimate below. |
| last activity | How long ago a job of the queue finished or started, or work was submitted to it. |

The estimate comes only from the queue's own finished jobs, and it is only a
rough figure. jobq takes the median run time of the successful jobs among the
200 most recently written results, counts one such run for every pending job
and the remainder of one for every running job, and divides that by the number
of the queue's jobs running right now. It knows nothing about what the commands
do or about the other queues, so it is always shown rounded, as `about 2h`,
`about 25m` or `under 1m`. Until three of the queue's jobs have finished
successfully there is no median and the column says `unknown`; while none of
its jobs is running there is no rate to divide by and it says `not running`.

Complete queues are left out of the table and counted in the line below it.
`jobq status --all` lists them too. `jobq status --since 24h` adds the complete
queues that had a job finish or start, or work submitted, within that window;
queues with work left are always listed, whatever the window says. A window is
written as a number and a unit, such as `30m`, `6h`, `24h`, `1d` or `2d12h`:

```bash
jobq status --since 2d12h
```

`jobq status --json` prints the same summaries as one JSON document and nothing
else on standard output. `jobq status QUEUE` prints that queue's row and then
its detailed view: its running jobs, why it is paused, and its deferred jobs.
Reading the overview writes nothing into the queue folder, so it is safe to run
from any machine at any time.

## Watch utilisation

While a pool runs it samples this machine every `monitor_interval_s` seconds —
each GPU's utilisation and memory, the processor, and how many slots the pool
holds — and appends the readings to the monitor folder: `monitor/` beside
`jobq_paths.toml`, in the project you run jobq from, unless `monitor_folder` in
that file names another place. The files are plain CSV, so they can be opened,
plotted or tailed directly. It is on by default; `jobq work --no-monitor` turns
it off for one run, and `monitor_interval_s: 0` in the policy turns it off on
that machine.

`jobq usage` reads the sample files of every machine that has a policy file and
prints one table. It reads them from the monitor folder of the project it is
run in, so it sees another machine's samples only where both machines write to
the same folder: either the project directory is on a filesystem they share, or
`monitor_folder` on each machine names a shared folder.

```text
machine     gpu  util  idle  none of yours  mem used  cpu  slots     last sample
gpu-host-1  0    72%   8%    12%            18400     41%  5.6 of 8  just now
gpu-host-1  1    64%   9%    20%            17100
gpu-host-2  -    -     -     -              -         -    -         no samples
jobs finished per hour in the last 7d:
  gpu-host-1: 4.20 (706 finished)
  gpu-host-2: 0.00 (0 finished)
```

The GPU columns are averages over the window: utilisation, the share of
samples the GPU counted as idle, the share of samples with none of your jobs
on it, and the memory in use. The machine columns are its processor
utilisation, the slots its pool held against the capacity behind them, and how
long ago it last wrote a sample. A machine that has written none still gets a
row, so a machine nobody is watching is visible. Under the table, the
successful jobs each machine finished per hour over the same window.

```bash
jobq usage --since 7d
jobq usage --since 24h --json
```

The window is written as for `jobq status --since`; left out, it is the last 24
hours, which the line under the table names. `--json` prints the same figures as
one document.

A machine that should be watched without running jobs runs `jobq monitor`,
which samples in the foreground and writes the same three files, with a slots
row saying nothing is used and no pool is live. It refuses to start where a
pool is already running on this machine and queue folder, since that pool
samples. `jobq monitor --once` takes a single round of samples and returns,
which suits a machine that is sampled from a timer rather than by a process
that stays up.

`jobq status` also prints one line per machine that has a policy file: whether
its pool is alive by its heartbeat or has not been heard from since a given
time, and, where samples exist, utilisation and idle GPUs over the last ten
minutes and the slots of its latest sample.

## Commands that ask before they act

Three commands throw something away, so each lists what it is about to do and
asks before doing it.

| Command | What it lists | Answering in advance |
| --- | --- | --- |
| `jobq requeue QUEUE` | The jobs whose results it would delete to make them pending again | `--yes`, and `--dry-run` to list them and change nothing |
| `jobq reset-mem-floor QUEUE` | The escalated per-job memory floors it would clear | `--yes` |
| `jobq release QUEUE KEY --force` | The claim it would remove, and which machine owns it | `--yes` |

A command run without a terminal to ask from needs `--yes`, so a script cannot
delete results by accident. `jobq release` needs `--force` as well before it
touches a claim another machine made; releasing a claim whose job is still
running can run that job twice.

Every command writes its output on standard output, so `jobq status --json`
and `jobq usage --json` can be piped straight into another program.

## Stopping a pool

There are three ways to bring a pool to an end, and they differ in what
happens to the jobs it is running.

```bash
jobq stop            # finish the running jobs, claim nothing new, then exit
jobq stop --now      # end the running jobs at once and put them back in the queue
```

`jobq stop` writes a request into the queue folder and returns at once; the
pool finishes the jobs it is running, records their results and exits. Use it
for an orderly end, and `jobq stop --clear` followed by `jobq work` to start
working again.

An interrupt or a termination signal, such as Ctrl-C in the pool's own terminal
or `kill <pid>`, asks for the same thing: the pool claims nothing new and lets
its running jobs finish. It prints one line saying how many jobs that is.
Sending the signal again does not end them.

`jobq stop --now` is the abrupt one: the pool ends each running job with a
termination signal, waits `kill_grace_s`, then the kill signal, and puts those
jobs back in the queue as pending with their attempt counter raised. No result
is recorded for them and nothing is counted as a failure, so they run again
from the beginning on the next pool. It asks for confirmation, or takes
`--yes`, and it needs a pool running on this machine to answer it. It can be
run from any terminal on that machine, including after a plain `jobq stop` or a
signal, which it upgrades.

`jobq status` says on this machine's pool line whether the pool is draining,
and why, or ending its jobs.
