"""Optional peak-GPU-memory reporting a job can opt into.

The worker reads a job's peak memory from its log: it parses the last line of the form::

    JOBQ_PEAK_GPU_MEM_MIB=<int> [JOBQ_PEAK_GPU_MEM_ALLOC_MIB=<int>]

Any job may print that line by whatever means it likes. For a PyTorch job,
:func:`maybe_register` registers an exit hook that prints it from torch's own high-water
marks; the worker sets ``JOBQ_REPORT_PEAK_MEM=1`` in every job's environment, so a job that
calls ``maybe_register()`` at startup reports automatically.

torch is imported lazily and every failure is swallowed: a measurement must never fail a
job, and a job without torch installed simply reports nothing.

Caveat on the number: it is the torch caching allocator's per-device high-water mark, taken
as the maximum over the process's visible devices rather than their sum. It excludes the
CUDA context and any non-torch allocation, so the process's true GPU footprint is higher.
"""

from __future__ import annotations

import atexit
import os
import sys

ENV_VAR = "JOBQ_REPORT_PEAK_MEM"
PEAK_PREFIX = "JOBQ_PEAK_GPU_MEM_MIB"
PEAK_ALLOC_PREFIX = "JOBQ_PEAK_GPU_MEM_ALLOC_MIB"


def peak_mib() -> tuple[int, int] | None:
    """``(reserved_mib, allocated_mib)`` maxed over devices, or ``None`` if CUDA is unused."""
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        n = torch.cuda.device_count()
        reserved = max(torch.cuda.max_memory_reserved(i) for i in range(n))
        alloc = max(torch.cuda.max_memory_allocated(i) for i in range(n))
    except Exception:  # noqa: BLE001 — never fail a job over a measurement
        return None
    return int(reserved) // (1 << 20), int(alloc) // (1 << 20)


def report_peak() -> None:
    """Print the peak line to stderr; silent when there is nothing to report."""
    try:
        peak = peak_mib()
        if peak is None:
            return
        print(
            f"{PEAK_PREFIX}={peak[0]} {PEAK_ALLOC_PREFIX}={peak[1]}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # noqa: BLE001 — including failures during interpreter teardown
        pass


def maybe_register(env: dict | None = None) -> bool:
    """Register the exit hook iff ``JOBQ_REPORT_PEAK_MEM=1``; return whether it was."""
    if (env if env is not None else os.environ).get(ENV_VAR) != "1":
        return False
    atexit.register(report_peak)
    return True
