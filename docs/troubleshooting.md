# Troubleshooting and recovery

[README](../README.md) · [Usage](usage.md) · [Queue files](queue-folder.md)

Start with `jobq status QUEUE`. It reports shared queue counts, but pool
liveness, slot occupancy, and MPS status describe the machine where you run
status. Look at the result's `log` field for job output and the pool log
under `logs/` in the queue folder for scheduling decisions.

## Jobs are waiting

| Symptom | Check or action |
| --- | --- |
| `pending` jobs, no active pool | Start `jobq work`; a pool exits after completing its eligible queues and does not remain as a submission service. |
| `PAUSED` queue | Fix the error shown in failed job logs, then use `requeue` or `resume` below. Status shows how long the pause has left; when it ends one job is tried; a success lifts the pause, a failure extends it by `failure_pause_s`. |
| Pool log `WAIT queue <name>: working directory ... is not reachable on this machine` | The directory the queue's or job's `cwd` names is missing here and `cwd_fallback` is unset or missing too. The jobs stay pending for machines where the directory exists; create the directory or set `cwd_fallback`, then restart the pool here. |
| `deferred` jobs | Exit-75 retries wait until `not_before`; status prints the earliest retry time. Unreadable attempt records can also defer claims. |
| A claimed job says `running`, but no GPU is assigned | It may be waiting for capacity. A claim is acquired before a GPU slot. A CPU-only job also has no GPU assignment. |
| `WAIT` for capacity | Compare the job's request and OOM floor with free memory, reserves, budgets, yielded GPUs, and both machine and queue caps. |
| GPU idle, CPU-only work waiting | Check `cpu_cap` and available worker threads. |
| Queue never selected | Check `--queues`, the queue's machine, whether it is parked (`jobq unpark` brings it back), dependencies, and the machine policy. |
| Dependency never finishes | Check missing/empty queues, cycles, pauses, and failed jobs under `--strict-deps`. |
| Changed a setting, waiting job still uses old request | The request is fixed at claim time. Stop and restart the pool to recalculate it. |
| Policy missing or invalid | Create or repair `gpu_policy.<hostname>.json`. Existing pools cannot admit work without a usable policy. |
| `jobq work` exits saying this machine has no GPU policy | Run `jobq init --gpus <indices> --free-mem-mib <MiB>` on this machine, then start the pool again. The pool refuses to start without a policy, so it never claims work it could not admit. |
| Pool exits immediately | Check the stop marker, and whether any incomplete queue matches this machine and `--queues`. A parked queue, a queue holding no jobs of its own, a queue tied to other machines or to GPUs this policy does not select, and a queue whose dependencies can never be met are all set aside rather than waited for, so they do not hold the pool open; its last log line names each of them and why. |
| Nothing is admitted on a machine whose GPUs look free | Check whether `nvidia-smi` answers. A free-memory query that fails denies every admission here until it answers again, and the pool log says so once per outage. |

A request larger than anything this machine can grant waits indefinitely.
There is no automatic timeout for a job that cannot be placed.

## A job waiting for a large request

A job asking for much of a GPU can lose every piece of memory that frees up to
smaller jobs arriving behind it. Once it has waited `reserve_after_s` (120
seconds by default; `0` turns this off), the pool holds one GPU for it: the
GPU where it would fit soonest. No other job of this queue folder starts on
that GPU until this one does, while the jobs already running there run to
their end, so the GPU empties and the waiting job starts.

Only a job that is waiting for memory is held a GPU. The test applied to each
GPU the job may use is what the next job to end there would leave it: what the
GPU shows free, less `reserve_mem_mib`, plus the largest single request held
there. A job that would be admitted with that much is waiting for a turn — for
a slot under the cap, for room in its `cap_group`, or for nothing at all when
the GPU would take it as it is — and holding a whole GPU for it would only
make the queue run one job at a time. Only a job needing more than the next
free slot can give back, on every GPU it may use, is held one.

Counting the largest request is also what keeps a momentary reading from
qualifying a small job: a GPU carrying a job larger than it answers no
whatever the reading does while jobs are allocating and freeing. A GPU is
further only held for a request that GPU could admit with none of this queue
folder's jobs on it: within `mem_budget_mib`, within the GPU's total less
`reserve_mem_mib`, and among the GPUs the queue's tie allows.

Once a job holds a GPU it keeps it until it is admitted or stops waiting, and
one GPU carries one reservation. A waiting job of a higher priority takes a
GPU over there and then; the job that held it goes back to waiting like any
other. So the pool log has one `RESERVE` line naming the GPU and the job and
one `RELEASE` line when the hold ends, the `RELEASE` for a GPU taken over
naming the job that took it, and `jobq status` lists the GPUs held and what
they are held for.

| Symptom | Check or action |
| --- | --- |
| A large job waits while small ones keep starting | Check `reserve_after_s` in this machine's policy; `0` means no GPU is ever held. |
| No GPU is held for a job that waits | A GPU is held only for a job waiting for memory. A job with room on a GPU as soon as one job there ends is waiting for a slot, and the cap or the worker count is what to look at. |
| The log says the request is more than any GPU can grant | Lower the job's `mem_mib` or run it where a larger GPU is. A request no GPU can meet never holds one, since the GPU would never be given back. |
| A whole machine looks blocked by holds | At most `reserve_max_gpus` GPUs, one by default, are held at a time across the whole machine whichever queue holds them, and only one job holds a GPU. |
| A GPU is held with nothing starting there | The hold ends when the job is admitted anywhere, when it stops waiting, when the GPU is yielded, or when the pool that made it has ended. A record left by a pool that has ended is ignored and removed. |

## Failures and pauses

An ordinary nonzero exit creates a failed result. Five consecutive counted
job failures pause the queue by default. The count follows completion order,
and a successful job clears the current streak and any pause. The pause blocks new
selection, and it cannot stop the jobs that were already claimed when it was
set: those run to the end and record their results, so with many workers a few
more failures than the limit can occur before the queue stops.

The pause lasts `failure_pause_s` seconds of the policy of the machine whose
job failure set it (default `900`; `0` keeps it until `resume` or `requeue`).
When it ends one worker on one machine tries one job, and its pool log has a line
`PAUSE LAPSED <queue>: the failure pause ended; trying one job again`; every
other worker treats the queue as paused meanwhile. A success lifts the pause,
and a failure extends it by `failure_pause_s` of the trying machine's policy.

Set the threshold when creating the queue:

```bash
jobq submit train --jobs-file train.jsonl --max-consecutive-failures 3
```

Use `0` to disable automatic pausing. An out-of-memory exit that is requeued at
a higher memory ask, an exit 75 that is retried later, and a job killed to yield
a GPU do not extend the streak: none of them is an outcome. A job that runs out
of those retries, or that runs out of memory while already asking for all a GPU
can grant, does extend it — the job failed for a reason of its own, and a queue
whose jobs all do that is what the pause is for. Only a fault of the pool itself
(it could not open the log, could not update the attempts record) records a failed result without extending the
streak; those results carry `not_failure_reason`.

After fixing the cause:

```bash
# Retry failed results and clear the queue's pause/streak.
jobq requeue train

# Or only clear the pause/streak, leaving failed results as they are.
jobq resume train
```

`requeue --all` also retries successful jobs. Requeueing removes terminal
result records, including their timings and log references; the underlying
log files remain. Save results you need before requeueing. A record with an
unreadable attempts sidecar is skipped rather than resetting unknown counters.

`jobq requeue`, `jobq reset-mem-floor` and `jobq release --force` each list
what they are about to throw away and ask before doing it. `--yes` answers in
advance, and a command run without a terminal to ask from needs it, so a script
cannot delete results by accident. `jobq requeue --dry-run` lists the jobs it
would hand back and changes nothing.

Restart a pool if it has already exited. Check any dependent queues before
retrying upstream work: their existing outputs are not invalidated automatically.

## Out-of-memory retries

A nonzero exit with a recognized OOM fragment in the log tail may be retried.
The worker normally increases the memory request by a factor of `1.5`, with
an initial floor of 4096 MiB and a ceiling based on the largest allowed GPU.
The factor applies to the request the job was actually admitted with, not to
whatever its queue asks for now. When log details suggest pressure from other
jobs, the request can stay the same. The retry limit defaults to eight
requeues. Hitting the limit or the computed ceiling records a failed result,
and that result counts towards the queue's failure pause like any other.

An exit of 75 is answered before the log is classified, so a job that asks to
be retried later is never read as an out-of-memory exit because an earlier
stage of it printed one.

The OOM counter and increased memory floor survive requeueing and restarts.
After correcting the cause, these commands address different parts of that
state:

```bash
# Retry failed jobs with a fresh OOM retry budget; keep memory floors.
jobq requeue train --reset-oom

# Clear escalated floors above 24000 MiB; do not change results or claims.
jobq reset-mem-floor train --above 24000

# Clear a particular job's floor using its sanitized jobkey.
jobq reset-mem-floor train --job train__seed-1
```

A terminal job still needs `requeue` after its floor is cleared. A job already
waiting for capacity retains its calculated request until it is claimed again.
A floor learned on a larger GPU may be too large for another machine.

## Exit code 75: retry later

A job can exit with code `75` to request another attempt. By default the worker
waits 900 seconds, then makes it claimable again, up to eight requeues. These
values come from `tempfail_retry_s` and `tempfail_max_requeues`.

The temporary-failure counter persists. `requeue --reset-oom` resets only the
OOM counter; it does not reset this counter or the deferral time. After
exhaustion, fix the underlying issue before retrying. The CLI has no dedicated
command to reset the exit-75 counter.

## Stop and restart a pool

Run this on the machine whose pool you want to stop:

```bash
jobq stop
```

It writes `stop.<hostname>` and returns immediately. Running jobs are allowed
to finish; workers waiting for capacity release their claims and exit. Wait
for the pool process itself to finish. The marker persists until cleared:

```bash
jobq stop --clear
jobq work
```

Clearing the marker does not start a new process. For a restart, wait for the
old pool to exit before clearing it. A marker cleared while the pool is still
draining puts the pool back to claiming jobs.

An interrupt or a termination signal asks for the same orderly end. Ctrl-C in
the pool's terminal, or `kill <pid>`, makes the pool claim nothing further and
let the jobs it is running finish and record their results; it prints and logs
one line saying how many jobs that is and that `jobq stop --now` ends them at
once. Sending the signal again prints a reminder of the same and changes
nothing. No signal ends a running job.

To end the running jobs at once, use:

```bash
jobq stop --now          # asks for confirmation
jobq stop --now --yes    # without the question, for scripts
```

Each running job's processes get a termination signal, then the kill signal
after `kill_grace_s`. Those jobs go back in the queue as pending with their
attempt counter raised by one, no result is recorded for them and nothing is
counted as a failure; they start again from the beginning under the next pool.
The pool then exits. The request travels through the queue folder, so it works
from any terminal on that machine, and it upgrades a drain already under way,
whether that drain came from `jobq stop` or from a signal. It needs a pool
running on this machine: without one it says so and leaves nothing behind that
could end the jobs of the next pool to start. `jobq stop --clear` removes both
requests.

`jobq status` says on this machine's pool line whether the pool is draining,
and whether that came from a stop request or a signal, or whether it is ending
its jobs. The pool log's last `POOL EXIT` line says which of the three
happened. To stop all machines, repeat the stop operation on each one. MPS has
its own lifecycle; use `jobq mps-stop` after its clients finish if you also
want to stop that daemon.

## A machine has not been heard from

On the machine where `jobq status` runs, whether a pool is alive comes from
the pool lock, `worker.<hostname>.lock`: a pool holds it for its whole life and
the kernel drops it however that process ends, so it answers without depending
on any file the pool wrote. The pid file supplies the number to show, and
stands in only where the lock cannot answer.

For every other machine there is the heartbeat. A running pool rewrites
`heartbeat_utc` in `worker.<hostname>.json` on every supervisor tick, so
`jobq status` can say for each machine whether its pool is alive or has not
been heard from since a given time, and can say the same on a job whose claim
belongs to another machine. A heartbeat older than `heartbeat_stale_s` (180
seconds by default), or a machine that has never written one, is what "not
heard from" means.

A machine falls silent when its pool ended, when the machine or its network
did, or when it cannot write into the queue folder. Check whether a pool is
running there and what its last log lines say; a machine reachable only
through the queue folder still shows its last sample under `jobq usage`.

The heartbeat is information and nothing else. The claims of a silent machine
are recovered by that machine's own pool when it next starts, as below, so
there is nothing to clear from here; use `jobq release QUEUE KEY --force` only
for a job you have verified is dead.

## Recover after a crash

Claims record the pool PID, hostname, boot ID, and process start time. A pool
recovers the claims of a pool of its own machine that is gone in a pass of its
own, at pool start and on every supervisor tick, over every queue in the folder
that has claims, whatever that queue's state: paused, parked, waiting on a
dependency, tied elsewhere, outside this pool's `--queues`, or with settings
this account cannot read. A claim nobody hands back counts as work in progress
until somebody does.

A claim the pass looks at and leaves where it is gets a `CLAIM KEPT` line in
the pool log naming the reason: its surviving job could not be ended, its owner
record is there but cannot be read, or its attempts record is. Claims with no
owner record at all have a grace period and a repeated-sighting check before
any machine reclaims them.

A job runs in its own session, so killing only the pool leaves the job running.
Two things on the claim let the next pool find what is left. Every claim
carries a run identifier, written when the claim is taken and before the job
starts; the job's process and everything it starts carry it in the environment
as `JOBQ_RUN_ID`. Once the job's process exists, the claim also records that
process, its process group, when it started and the machine's boot ID.

The next pool on the same machine finds the claim, sees that the pool is gone, and
ends what the run left running: every process on this machine whose environment
carries the run identifier, and the process the pool started together with its
process group. A process id is only a number, and the machine hands it out
again once the process behind it has ended, so the recorded process is signalled
only when the process there is the one the claim describes: the same boot, the
start time the claim recorded, and the process group it recorded. A claim
written without a start time is matched on the group alone, and one that
matches on neither is left to the run identifier, which names the run itself.
The group is signalled once something in it has been recognised as this run's.
Each gets a termination signal,
then the kill signal after `kill_grace_s`. Only then does the job go back in
the queue with its attempt counter raised by one; the pool log records this
under `ENDED SURVIVOR`. That is not counted as a job failure. This covers a
child that outlived the shell the pool started, a child that moved to another
process group or its own session, and a claim whose job process was never
recorded.

The search for processes carrying the run identifier reads the machine's
process table, so it happens only when a pool finds a claim whose pool is gone, never
while jobs are being started or finished. Processes that have ended are passed
over without their environment being read, and only this user's processes are
looked at.

If those processes cannot be ended, for example because they belong to another
user, the claim stays where it is, the reason is logged once for that claim,
and the job is not started again on this machine. A claim written without a run
identifier and without a job process is simply reclaimed. A full machine reboot
ends its old processes, but interrupted commands may still have produced
partial output.

A pool asked by `jobq stop --now` to end its jobs does the same to the jobs it
is running itself: it ends them, puts them back in the queue as pending with
their attempt counter raised, releases its slots, removes its pid file and logs
`POOL EXIT` saying its jobs were ended and put back. Each such job is logged
under `INTERRUPT` when it is ended and `INTERRUPT REQUEUE` when it is back in
the queue, which is separate from the `YIELD REQUEUE` of a job ended to yield a
GPU. No result is recorded for such a job. A job whose process starts while
this is happening is ended by the worker thread that started it, and a worker
that had not yet started its job hands the claim back unrun, so no job process
of that pool is left running when it exits.

While such a claim is still on disk, `jobq status` counts the job as running
and says on its line that it was left by a pool that has ended and goes back in
the queue when a pool starts on that machine, and whether the job's own process
is still running. Recovery happens once per claim, however many worker threads
the next pool has.

A pool on another machine does not automatically reclaim an owned remote
claim. If the owning machine will not return, first establish that the original job is
not running, then release that exact job key:

```bash
jobq release train train-seed-1 --force
```

`release` removes the claim; it does not kill the process or delete a result.
Releasing a live job can cause duplicate execution. It accepts only the key (or
jobkey) of a job submitted to that queue, and removes exactly that job's claim
directory: an empty key, `.`, `..`, a name holding a path separator, a name that
resolves outside the claims directory through a link, and a key that names no
job of the queue are all rejected with a message, and nothing is removed.

Commands should handle retries deliberately, for example by using job-specific
output paths and writing completed outputs atomically. jobq records outcomes;
it cannot roll back a command's side effects or guarantee exactly-once execution.

## Waiting from a script

```bash
jobq wait train --poll-s 10
jobq status train
```

`wait` ends once all jobs have results, including failures. Its successful exit
status is not a success check for the jobs. A paused or unplaceable queue can
make it wait indefinitely. For dependent jobq work that needs successful
inputs, submit with `--depends-on train --strict-deps`.
