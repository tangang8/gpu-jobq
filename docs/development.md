# Development

[README](../README.md)

## Set up

From the repository root, with Python 3.11 or newer, uv installs the package
and its development dependencies from the lockfile:

```bash
uv sync --extra dev
uv run ruff check .
```

The test suite runs serially or across processes with `pytest-xdist`; each
test works in its own temporary folder, so both give the same result:

```bash
uv run pytest -q
uv run pytest -q -n auto
```

Run commands with `uv run` or activate `.venv`. Without uv, use a virtual
environment and pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
```

## Source map

| Module | Responsibility |
| --- | --- |
| [`cli.py`](../src/jobq/cli.py) | Typer commands, submission options, status formatting, operator actions |
| [`settings.py`](../src/jobq/settings.py) | The `jobq_paths.toml` file naming the queue folder, and its search |
| [`model.py`](../src/jobq/model.py) | `Job`, jobkey conversion, JSONL and template expansion |
| [`store.py`](../src/jobq/store.py) | Queue metadata, claims, results, attempts, pauses, completion checks |
| [`io.py`](../src/jobq/io.py) | Atomic replacement writes, `flock`, shared-permission mode |
| [`gpu.py`](../src/jobq/gpu.py) | Policy parsing, GPU/CPU capacity locks, memory admission, MPS |
| [`worker.py`](../src/jobq/worker.py) | Pool supervision, queue selection, command execution, retry decisions |
| [`yielding.py`](../src/jobq/yielding.py) | Foreign-process estimates, yield markers, progress estimates |
| [`monitor.py`](../src/jobq/monitor.py) | Utilisation sampling, the sample files, and the usage report |
| [`report.py`](../src/jobq/report.py) | Optional PyTorch peak-memory reporting |
| [`summary.py`](../src/jobq/summary.py) | Read-only per-queue summaries, the time estimate, the status table |

The execution path is: parse submission → append jobs → select a ready queue
→ create a claim → acquire capacity → run `bash -c` → record a result or retry
state → release the claim and capacity. Claims precede GPU allocation, so a
claimed job may still be waiting.

Claims and results are shared state; capacity locks are machine-local.
`GpuInterface` and injectable query functions let scheduling run without
calling real NVIDIA tools. The state format is described in
[queue-folder.md](queue-folder.md).
