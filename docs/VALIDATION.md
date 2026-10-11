# Validation and operating contract

Environment: 2026-10-11, Linux/x86_64, Intel i5-11400H, Python 3.10.12,
NumPy 2.2.6 and OpenCV 5.0.0. Supplied camera, inference and example files
are byte-for-byte unchanged. This file describes current behavior; old measurements
are identified separately rather than presented as repeated current validations.

## Automated checks

The review-hardening run passed all 133 tests in 71.1 seconds. Focused checks
also pass for sequence snapshots, CLI output overrides, bounded rotation,
full-window latency evidence, sparse sampling rejection and explicit failure
reasons. Ruff lint/format, mypy (30 application/benchmark modules), and the
cyclomatic complexity limit of 20 pass. Bandit reports no medium/high findings.
Full test output: `artifacts/review-improvements-tests.log`.
The earlier coverage measurement was 90%; coverage and dependency advisories
were not refreshed in the validator change; the preceding independent audit
refreshed dependency advisories with no known findings. Python 3.12/3.13 and installed-wheel
checks run in CI; hosted execution requires publishing the repository to GitHub.

Tests cover fair/partial batching, raw leases and rollback, explicit drops,
reconnects, slow sources, signals, crashes/hangs, malformed IPC/configuration,
private output/quotas, parent/group SIGKILL and shared-memory disposal. Current
regressions additionally cover independent worker cleanup after failures,
process-exit races, active camera release and watchdog expiry, deterministic
benchmark source policies, and versioned cleanup status validation. Recovered
join failures close process handles without hiding the original error; live
workers are never closed. Reporter lifetime includes all throughput windows,
and artifact validation rejects shorter summaries (with a 1 ms allowance for
older artifacts' initialization clock ordering).

The serializer uses the complete `ReportedResult` TypedDict. Benchmark evidence
requires every result field, matching UTC timestamp/receipt nanoseconds, strict
identity/detection/brightness values and finite nonnegative consistent timing
intervals. Throughput counters are bounded to signed 64-bit range before division;
oversized values and overflow become controlled failed reports. Reader-level
regressions cover omitted fields, invalid timestamps/timings and huge integers.

A fresh 12-second smoke benchmark after the retry/duration fixes passed with
320 results, serialization p95 429.8 ms and successful cleanup. Its full report
is `artifacts/retry-duration-performance/report.json`; this single run does not
replace the multi-condition performance matrix.

## Rotation and controlled short comparisons

Four sequential 12-second trials passed after the review hardening. Each used
16 KiB result segments with a bounded allowance, read all segments in order,
validated terminal summaries and reconciled frame accounting. These are short
integration checks, not statistically established performance regressions.

| Case | Complete-window FPS | Worst delivery-window p95 bound (ms) | Results | Segments |
| --- | ---: | ---: | ---: | ---: |
| serial | 25.02 | 420 | 324 | 9 |
| overlap | 36.29 | 486 | 472 | 12 |
| serial-reconnect | 24.76 | 442 | 320 | 9 |
| overlap-reconnect | 36.28 | 475 | 474 | 12 |

Reports: `artifacts/review-improvements/{case}/report.json`, aggregate
`artifacts/review-improvements/summary.json`. Histogram bounds use nearest-rank
p95 and millisecond ceilings; the 2000 ms bucket represents overflow. Existing
segments are never overwritten, records never span files and exhausting the
configured allowance fails explicitly. Rotation is opt-in through
`output_max_files`; stdout supports a single segment.

## Sustained load with strengthened gates

A fresh 180-second serial/stable trial passed the new sampling and delivery-window
checks: 17 complete windows, 4844 results, mean window FPS
26.80, minimum FPS 24.86, recent delivery p95
422.5 ms and worst window p95 upper bound
451 ms. Each process had 597–598 steady-state observations;
all sampling coverage gates passed. Largest median RSS growth was
0.152 MiB, peak observed pipeline PSS
391.2 MiB. Cleanup completed, worker PIDs disappeared
and every camera had zero in-flight frames. No finite cgroup limit was observed.

Evidence: `artifacts/review-improvements-soak/report.json` and `summary.json`.
Only the serial/stable case received this fresh three-minute check. The four
controlled rotation cases above received 12 seconds each. The complete four-case
3–5 minute matrix is available through the soak command and was not rerun here.

## Historical sustained load before stricter sampling/window gates

Two sequential 300-second capture trials passed. Each produced 29 complete FPS
windows, valid strict result records, reconciled frame/drop accounting and a
successful cleanup summary. Recorded worker PIDs no longer existed, shared-memory
names were removed and every camera had zero in-flight frames after drain.

| Case | Mean window FPS | Minimum window FPS | Recent delivery p95 (ms) | Results | Disconnects | Largest median RSS growth (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default, stable sources | 26.72 | 24.70 | 418.6 | 8044 | 0 | 0.719 |
| Overlap, repeated reconnects | 38.89 | 36.55 | 470.6 | 11701 | 120 | 0.715 |

Median RSS compares elapsed seconds 30–90 with the final minute. No process
exceeded the 64 MiB growth gate. Peak observed pipeline PSS was 392.2 / 394.5 MiB;
this measures the four monitored pipeline processes, not host-wide consumption.
CPU percentages below use 100% for one core and exclude the initial 30 seconds.

| Process | Default mean CPU (%) | Overlap/reconnect mean CPU (%) |
| --- | ---: | ---: |
| Supervisor | 3.97 | 3.43 |
| Capture | 110.27 | 94.69 |
| Processing | 56.72 | 78.46 |
| Reporter | 0.96 | 1.24 |

Across the entire emitted stream, serialization p95 was 422.4 / 470.5 ms and
maximum latency was 543.3 / 505.7 ms. The worst complete window's serialization
p95 was 458.8 / 480.7 ms. These per-record measurements exclude output write/flush;
the delivery p95 in the first table includes successful write/flush completion
and describes the latest 2048 results. Explicit drop percentages were 46.30% /
16.23%. The cases differ in both overlap and disconnect policy, so these figures
are not a controlled single-variable A/B comparison.

Full reports: `artifacts/validator-soak/{default,overlap-reconnect}/report.json`.
Aggregate CPU/memory/FPS evidence: `artifacts/validator-soak/summary.json`.
Whole-stream latency and cleanup checks: each trial's `whole-run-analysis.json`.
The current helper runs four controlled cases with stricter coverage and
all-window latency gates. Repeat with
`python benchmarks/soak_test.py --duration 300 --directory artifacts/soak-new`.
The five-minute interval shows short sustained stability, not multi-day leak
freedom. The helper is a local/manual check; hosted CI has not run it here.

## Resource and failure contract

The supported target is Linux/x86_64 with mutually trusted workers, an
unprivileged UID, sufficient shared memory/RAM/storage and an external process-tree
manager. Bounded JSON uses fixed constructors, structural/semantic validation,
finite numeric values and bounded payload size. It prevents executable IPC payloads
but does not sandbox native adapters or workers sharing the same UID.

Raw frames use camera-partitioned shared-memory slots and eventfd credits.
Raw leases end before inference. The pool is reserved before first page access,
unlinked after readers attach, and disposed after workers are reaped. A crash
requires disposal of the complete transport rather than reuse of lost credits.
Workers bind SIGKILL to parent death; launchers clean isolated process groups.
Shutdown attempts every worker independently and reports aggregated failures.

Camera CONNECTING, READING, PUBLISHING and RELEASING operations are watched.
Low-FPS reads include the inter-frame interval in their budget. All-camera outage
is bounded separately. STOPPED follows completion of camera release. Processing
publishes real progress after completed frame preparation, before inference and
after each result publication; recurring heartbeats do not hide blocked operations.

Memory estimates and sampled PSS are admission/observation checks, not hard
limits. Only deployment cgroups provide a hard memory boundary. Missing PSS or
finite cgroup observations are reported explicitly. Process RSS includes shared
mappings and must not be summed as physical usage.

Output files are private and create-only; path components reject symlinks and
traversal. Result and diagnostic quotas are independent. Optional owned-file
fsync strengthens durability. Summary paths are reserved before startup; failed
runs can leave an empty reservation. Essential cleanup precedes final accounting.
Schema version 1 includes exit_code, cleanup_completed and cleanup_errors. Summary
write/flush/close failures trigger truncation rollback; finalization diagnostics
use a fresh bounded channel. This is not atomic publication across arbitrary
process death or a promise of survival through host/storage failure.

## Performance acceptance

CI runs fifteen cases twice, sequentially, normally for 20 seconds each. Normal
benchmark sources do not disconnect randomly. The explicit disconnect case fails
cam_01 on read attempt 30 per connection with fixed retry delays; other cameras
stay connected. Source policy and effective configuration are recorded per report.
These benchmark-only overrides preserve production behavior and supplied files.

Each accepted trial needs a complete actual throughput window of at least ten
seconds, rates recomputed from counts, emitted-identity/window reconciliation,
a unique terminal Reporter summary, correct camera/batch totals, consistent
recent latency percentiles and final zero-in-flight accounting. Accounting must
use schema version 1 and successful cleanup with no errors. Older accounting
artifacts are rejected rather than silently accepted as current evidence.

Ordinary p95 delivery limits are 2000 ms; slow output permits 3000 ms. The large
output and batch-16/one-result-slot slow-output stress cases permit 5000 ms.
Input age is checked at preprocessing start and is not an output-latency SLA.
Percentiles describe bounded recent rings, not all historical frames. Short local
trials establish acceptance only, not statistical or hardware-independent speedups.

## Historical tuning evidence

The following sequential 12-second measurements predate the current deterministic
source policy. They remain tuning context, not evidence for the latest changes.
Raw reports are in ignored artifacts/final-review-performance/.

| Condition | Completed-window FPS | p95 delivery ms |
| --- | ---: | ---: |
| Default batch 4, two OpenCV threads | 24.87 | 422 |
| Batch 8 | 26.75 | 715 |
| Batch 16 | 26.87 | 885 |
| Four OpenCV threads | 27.90 | 380 |
| Latest | 24.25 | 272 |
| Combined: four threads, overlap, latest, 1 ms flush | 36.46 | 394 |

Batch 4 remains the baseline. More threads and bounded overlap are deployment
choices; larger batches increased tail latency for modest throughput gains.
Prior one-thread component medians with 15 measured iterations were 8.80 ms
simulator generation, 1.55 ms host copy, 15.21 ms area resize and 0.09 ms grayscale.
Scratch/pool reuse passed bitwise comparison on simulator and noise inputs.
Production keeps resize-before-grayscale with INTER_AREA. Direct camera-to-pool
capture requires a future read_into adapter contract; it is not currently offered.

Quarantine/restart, replay, cross-camera exposure synchronization, exactly-once
external delivery and hard latency guarantees require separate product requirements.
Host/cgroup OOM and uninterruptible kernel I/O need deployment-level controls.

## Current targeted runs

Two sequential 12-second trials passed acceptance, schema-version-1 accounting,
cleanup and zero-in-flight checks. These are local smoke measurements, not
a statistical comparison or execution of the full fifteen-case matrix.

| Workload | Completed-window FPS | p95 delivery ms | Disconnects |
| --- | ---: | ---: | ---: |
| normal | 24.67 | 432 | 0 |
| disconnect | 25.04 | 413 | 5 |

Reports are in `artifacts/cleanup-policy-performance/`; the complete test log is
`artifacts/final-cleanup-fixes-tests.log`. The ordinary workload had no random
faults; the controlled workload produced five cam_01 disconnects and recovered.
