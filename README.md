# Python camera CV pipeline

Three spawned stages — Capture, Processing and Reporter — coordinated by one
Supervisor. The original assessment simulators in `provided/` are unchanged.
The verified deployment target is Linux/x86_64 with Python 3.10 or newer; CI
checks Python 3.10, 3.12 and 3.13 on Ubuntu 24.04.

## Install

From this directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
```

The lock contains the complete runtime/quality environment and approved Linux
wheel hashes for the CI interpreters. For a runtime-only install, use
`python -m pip install -e .`; direct runtime and build versions are pinned in
`pyproject.toml`. Install the matching Ubuntu `python3-venv` package if your
interpreter lacks ensurepip. Select `.venv/bin/python` in your IDE.

## Run

```bash
cv-pipeline --help
cv-pipeline --print-config
cv-pipeline --duration 20 --output results-new.jsonl --summary accounting-new.json
cv-pipeline --config configs/default.json --duration 60
```

Output files must be **new**: existing files, symlink components and path
traversal are rejected. Owned files use private permissions. Their parent
folders must exist. Without an output path, Reporter writes JSONL to stdout;
service logs go to stderr. Ctrl+C/SIGTERM requests graceful drain. Duration
starts after all stages are ready and is limited to one day when specified.

`configs/default.json` is a complete example; partial JSON overrides work too.
Configuration files must be regular files up to 64 KiB without duplicate keys.
Camera IDs are unique ASCII identifiers of 1–64 letters, digits, `_`, `-` or `.`.

## Buffering and policies

- Three stages do not imply three slots. With `slots_per_camera: null`, raw
  credits per camera are `ceil(batch_size / camera_count) + 1`. Defaults give
  eight total slots / 189.8 MiB; batch 16 gives twenty slots / 474.6 MiB.
  Any explicit positive slot count is accepted within the memory budget.
- `memory_budget_mib` defaults to 1024. Validation estimates raw frames,
  camera temporaries, batch/scratch storage and process overhead and checks
  nested cgroup limits and their available memory, with a safety margin.
  Runtime samples proportional resident memory (PSS) and fails on observed
  over-budget usage; cgroups remain the hard limit. The allocator checks tmpfs capacity and reserves its
  backing before touching it. This is not a guarantee against host OOM.
- No raw credit means an explicit drop-newest. Processing drops inputs older
  than `max_frame_age_ms` before preprocessing. This age limit does not cap
  receipt-to-output latency. FIFO/per-camera order and fair polling are preserved.
- Defaults resize RGB 4K frames to 640x480 using INTER_AREA, then grayscale.
  RGB scratch and a bounded grayscale pool are reused. Each grayscale destination
  stays exclusively owned until its inference call finishes.
  OpenCV uses two threads by default and one process owns all inference calls.
  `processing_overlap: true` enables one preparer thread and one queued batch;
  inference stays on the Processing main thread. The pool then holds at most
  three batches. `frame_policy: "latest"` selects the newest available queued
  descriptor per camera and accounts replaced descriptors as `dropped_superseded`.
  It does not guarantee retaining camera frames dropped before admission.
- Batch size is 1–16, default 4, with a 50 ms first-frame collection deadline.
  Missing cameras do not block partial batches. Batching happens in Processing.
- Normal drain checks captured = reported + explicit drops and processed =
  reported. `--summary` saves final accounting independently of best-effort logs.
- Expected disconnects recreate that camera with bounded jittered backoff.
  An unexpected camera error/stall fails the pipeline. All cameras unavailable
  beyond `availability_timeout_s` (default 30 s) also fail the pipeline.
  This timeout must exceed the slowest camera interval plus 500 ms. Camera
  operation state and timestamps are read as one consistent watchdog snapshot.
- `output_max_bytes` defaults to 256 MiB per segment (one by default), including stdout; quota
  exhaustion fails explicitly before writing the overflowing record. Service
  logs have a separate 16 MiB best-effort limit. Use deployment storage retention.
  Diagnostic transport/preparation failures are counted without synchronous
  stderr fallback. Malformed log messages are counted while later records drain.
- `output_flush_records` (default 1) and `output_flush_interval_ms` (default 100)
  bound buffered emission. EOS always flushes. File buffering can be batched;
  a borrowed line-buffered stdout may still flush on each newline.
- `output_durable: true` adds fsync for owned result files. The default flush
  is not durable storage acknowledgement; neither mode provides replay or
  exactly-once delivery after an emergency kill.

A partial tuning configuration can enable the optional modes:

```json
{
  "processing_overlap": true,
  "preprocessing_threads": 4,
  "frame_policy": "latest",
  "output_flush_records": 16
}
```

Save it as a JSON file and pass its path with `--config`. These settings trade CPU
and buffering latency for throughput/freshness; compare them on the target host.

## Metrics and failure safety

Results include successful-read frame ID, reconnect epoch, receipt clocks,
brightness/detection, actual batch size and latency intervals. Frame IDs can have
explicit drop gaps. Receipt time is not a hardware exposure timestamp.
`capture_to_report_ms` ends at serialization start; it excludes the write itself.
Reporter summaries also include `delivery_latency_ms_recent`, sampled after
write/flush (and configured fsync) completes. Diagnostic loss counters and memory
observations are included in final accounting. Inference latency is repeated per participating result and must not be summed
across frames as device compute time.

Reporter emits completed windows of at least ten seconds. Recent latency rings
are bounded to 2048 samples per camera. Raw pixels use fixed shared memory;
all cross-stage messages use schema-checked JSON datagrams, not pickle. Unnamed
`eventfd` counters bound outstanding records; socket byte limits can apply earlier.
Free slots use camera-local shared byte states plus eventfd availability, so
startup needs no prefilled datagrams or named semaphores. Stop flags are monotonic shared bytes.

Workers receive SIGKILL on Supervisor death. Normal worker crashes/stalls trigger
parent-owned bounded termination and pool cleanup. Pool names are unlinked
after readers attach, so kernel mapping lifetime reclaims steady-state storage
even if the whole tree and tracker are killed. Benchmark/test launchers
isolate process groups and clean only pools owned by that launched Supervisor
if its resource tracker was also killed. Same-UID workers remain mutually trusted;
this is not an isolation sandbox. Use an unprivileged service and a process-tree
manager for deployment.

## Validate

```bash
python -m unittest discover -s tests -v
ruff check src tests benchmarks
ruff format --check src tests benchmarks
mypy
python benchmarks/check_complexity.py
python benchmarks/run_benchmark.py --duration 20 --directory benchmark-output-new
python benchmarks/performance_suite.py --duration 20 --repeats 2 --directory artifacts/matrix-new
python benchmarks/soak_test.py --duration 300 --directory artifacts/soak-new
python benchmarks/ci_summary.py --directory artifacts/matrix-new
```

Benchmark directories must be fresh. The matrix covers batches 4/8/16, sparse
input, one CPU core, forced reconnects, slow inference, slow output, four OpenCV
threads, overlap, latest queued frames and buffered output. It records
FPS, p95 latency, drops, per-process RSS, effective configuration and reliable
final accounting. RSS contains shared mappings and must not be summed.

The soak command runs four sequential five-minute trials: serial and overlapped
processing, each with stable cameras and with repeated disconnects. This provides
comparisons with only one condition changed at a time. It records per-process CPU utilization
(100% means one core), steady-state RSS drift, complete FPS windows, recent p95,
drops and final cleanup. Acceptance requires all windows above 10 FPS, recent
delivery p95 below 2 seconds, and no more than 64 MiB of median RSS growth per
process between the early and final steady minute. Missing CPU/RSS observations
fail the measurement check. Steady-state sampling must cover the interval with
no gaps above two seconds and CPU data for at least 90% of observations. CPU
statistics exclude startup and samples after the requested workload duration.
A fixed 2001-bucket histogram additionally checks p95 in every delivery window,
including the final partial window; values at or above 2000 ms fail acceptance.
This short soak does not establish long-term leak
freedom. Full trial reports and `summary.json` remain in the new directory.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md), [CI.md](docs/CI.md) and
[VALIDATION.md](docs/VALIDATION.md) for module responsibilities, automated gates
and current measured evidence. C++ remains reserved in the sibling workspace.

Optional bounded output rotation uses `output_max_files` (default 1, maximum
1024) with an owned `output_path`. Each segment is limited by `output_max_bytes`;
subsequent files append `.0001`, `.0002`, etc. Records never span segments. Files
are private and create-only; existing files and symlinks are rejected. Old
segments are retained, and exhaustion of the total segment allowance fails the
pipeline. This bounds disk consumption without silently deleting results.
