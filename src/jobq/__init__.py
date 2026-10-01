"""jobq: a file-based GPU job queue for one user, drained by worker pools on several machines.

A queue folder on a shared filesystem holds the queues; each machine runs one worker pool that
claims jobs from it, schedules them onto that machine's GPUs and records the outcome. Jobs are
shell command strings run with ``bash -c``. Cross-machine coordination is by atomic claim
directories; per-machine behaviour comes from ``gpu_policy.<hostname>.json`` in the queue folder.
"""

from jobq.model import Job, parse_jobs_file, sanitize_jobkey

__all__ = ["Job", "parse_jobs_file", "sanitize_jobkey"]
