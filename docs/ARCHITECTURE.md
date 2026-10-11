# Architecture and ownership

The required stages are three processes; raw-buffer capacity is a separate
memory/latency policy. Supervisor starts downstream consumers before Capture
and owns resource allocation, signals, health checks and final cleanup.

## Modules

| File | Responsibility and main interfaces |
| --- | --- |
| `__init__.py` | Side-effect-free package import. |
| `__main__.py` | Module execution delegates to CLI. |
| `cli.py` | Parse arguments, load overrides, print configuration or run Supervisor; no CV startup for help. |
| `config.py` | Frozen CameraConfig/PipelineConfig, automatic slot resolution, grouped validation policies, bounded regular-file JSON loading and duplicate-key rejection. |
| `messages.py` | Frozen frame/result/timing/health/EOS records, enums and the complete ReportedResult JSONL TypedDict; no pixels or live native objects. |
| `contracts.py` | Narrow camera, inference, clock, queue and stop Protocols for dependency injection. |
| `adapters.py` | Translate supplied camera disconnects, validate model output schemas and provide separated UTC/monotonic clocks. |
| `wire.py` | Fixed JSON record registry; recursive scalar/container encoding and schema/type validation without importing sender-supplied constructors. |
| `validation.py` | Semantic wire ranges, identity/counter/timing checks and complete external Reporter contract validation for benchmark evidence. |
| `ipc.py` | Atomic bounded datagrams, unnamed eventfd counters, camera-partitioned SlotCredits and monotonic shared-stop flags; spawn transfers private socket handles. |
| `transport.py` | TransportOwner reserves/unlinks the raw pool; FrameTransport publishes descriptors and camera-partitioned credits; FrameLease returns credits on exit. |
| `resources.py` | Discover nested v1/v2 memory cgroups and ancestors, sample process PSS and enforce the observed memory budget. |
| `lifecycle.py` | Linux parent-death SIGKILL with race check and isolated process-group cleanup, including owned pools after tracker death. |
| `capture.py` | One camera-owning thread per source, reconnect/backoff, admission/drop accounting and per-camera/all-camera availability monitoring. |
| `batching.py` | Persistent round-robin cursor, first-frame deadline and per-camera EOS; never waits for a particular missing camera. |
| `preprocessing.py` | Single-owner exact resize-then-grayscale transformer with reusable RGB scratch and optional caller-owned grayscale destinations. |
| `processing.py` | BatchProcessor separates prepare/infer; GrayscalePool bounds destinations; optional OverlappedBatches owns one preparer thread and one waiting batch. |
| `output.py` | Create-only private file output using directory descriptors/no-follow, byte quota and optional durable flush; never closes borrowed stdout. |
| `reporter.py` | JSONL serialization, completed throughput windows and bounded recent latency metrics; synchronous bounded-policy output. |
| `observability.py` | Thread-safe local counters and bounded latency rings; throughput uses monotonic elapsed time. |
| `logging_setup.py` | Nonblocking diagnostic publication and bounded listener shutdown; independent best-effort log byte limit. |
| `supervisor.py` | One-shot main-thread lifecycle, readiness/progress/exit checks, independent cleanup scopes and final accounting/summary persistence. |

The split follows ownership and failure boundaries. Pixel ownership, scheduling,
external adapters, wire schemas and output policy change for different reasons;
combining them would couple testing and recovery. Protocols describe behavior
without another inheritance framework. Helper classes exist for state that must
persist across calls: slot leases, round-robin cursors, backoff, health, scratch,
metrics and output byte accounting.

## Frame lifetime

1. Capture reads an owned 4K uint8 frame, records successful-read identity and
   receipt clocks, then attempts a camera-local raw credit.
2. No credit counts drop-newest. Otherwise copy all pixels into the reserved
   slot before publishing its small JSON descriptor. Failed publication returns
   the credit; timeout is distinct from an ordinary shutdown drop.
3. Processing collects descriptors fairly, enters validated leases and copies
   retained pixels through preprocessing into independent grayscale outputs.
4. Each raw lease returns its credit before inference, allowing Capture reuse.
   A batch error also returns collected but unvisited credits.
5. One inference owner returns validated results. Reporter writes small records,
   updates successful-emission counters and emits a final summary after FIFO EOS.

Explicit slot counts may prioritize memory/freshness over full batches. Auto
mode provisions the rounded-up batch share plus one credit per camera; it does
not guarantee a full batch when cameras are offline or input is sparse.

## Safety and recovery

All application IPC payloads use bounded JSON. Generic containers remain data,
never executable Python constructors. Data channels use unnamed eventfd credits; socket buffers independently limit
queued bytes. Free slots use partitioned byte states and eventfd availability,
not prefilled datagram records. There are no named queue semaphores or feeder threads to leak or deadlock on a
killed owner. Killing a participant can still lose an in-flight credit, so the
entire transport is discarded on failure rather than reused for worker restart.

Shared memory is random-named with its Supervisor PID prefix and private mode.
Capacity probing is followed by posix_fallocate on its existing CPython/Linux
file descriptor. ENOSPC is caught before page touching; the owner unlinks the name after Capture and Processing attach, keeping their
mappings alive. This makes steady-state whole-tree kill reclaim storage without
a surviving tracker. Startup cleanup still unlinks any unsealed pool. Resource
tracker handles parent-only death during startup; isolated launchers remove only their own UID/PID-prefixed pools if a forced
group kill also killed the tracker. No global `/dev/shm` sweep is used.

Signal callbacks record intent without IPC or logging. Workers ignore ordinary
signals while the parent drains, but bind a kernel SIGKILL parent-death policy.
A heartbeat proves liveness; active progress has an independent expiry. Kernel
uninterruptible I/O and host/cgroup OOM are deployment boundaries, not Python
cleanup guarantees. Configure an unprivileged UID, process-tree supervision,
memory/shm/storage limits and recovery policy.

FIFO/drop-newest and sequential preprocessing/inference remain defaults.
Optional latest-queued selection is performed only by the descriptor consumer:
it releases and accounts superseded raw leases, with bounded scans per camera.
Optional overlap uses one preparer thread, one waiting batch and one inference
owner; three fixed grayscale batches cover active, queued and preparing data.
Failure cancels queued work and a stalled thread leaves mapping disposal to the
Supervisor's process kill. No transformer or inference instance is shared between
callers. Pixel operation order and INTER_AREA remain unchanged.

Reporter buffers at most output_flush_records identities, flushes by count/time
and EOS, and accounts delivery only after flush succeeds. Per-result timestamps
still measure serialization start; separate summary percentiles measure completed
write/flush latency. A completed flush is not exactly-once delivery or durable
storage unless the owned-file fsync policy is enabled.

Memory estimates include a safety margin and overlapping buffers. Runtime PSS
observations apportion shared pages and include all live stage processes; nested
cgroup current usage and ancestor limits are reported. These are admission and
observation policies; deployment cgroups must enforce a hard limit. Exposure
synchronization, stage restart and replay remain outside the assessment contract.

## Processing types and review hardening

`PreparedFrame` names each retained identity and its monotonic preprocessing
boundaries. `PreparedBatch` owns detached grayscale destinations until inference
returns. `serial_results()` accepts raw descriptors; `overlapped_results()` accepts
only prepared batches. Both yield `list[ProcessedResult]` to `publish_results()`.
No generator switches between unrelated input types. All application and benchmark
functions are checked with mypy's `disallow_untyped_defs` policy.

The preparer publishes a lock-protected `PreparationPhase` and local observation
timestamp. The collector marks transport operations active and empty-source polls
idle. The main thread checks the preparer's independent observation deadline;
its own heartbeat cannot certify an unresponsive collector. Live empty polling
refreshes observations, while preparation must complete within the health budget.
Local close cancellation also interrupts the collector's polling loop. FIFO and
latest scans share EOS producer validation.

The batch deadline starts when the first descriptor is accepted, before latest
selection scans. Both the outer round-robin loop and each bounded latest scan
check that deadline before another receive;
one in-progress nonblocking receive or scheduling delay can still overrun it.
Capture read liveness allows the configured inter-frame interval plus 500 ms,
or the normal health timeout if larger. Connecting and publication retain the
normal timeout, so sparse input does not hide stalls in unrelated operations.
CameraState and immutable CameraObservation snapshots keep state and both clocks
consistent under a process-local lock, held only for updates/reads, never camera
I/O. Each operation begins its own budget. Availability configuration must exceed
the slowest camera's frame interval plus 500 ms, allowing a healthy slow source
to produce even when faster sources are disconnected.

Reporter computes receive timeouts from the next pending flush deadline, capped
by its ordinary 100 ms health poll. This bounds intentional waiting; scheduler
latency and blocking output can still exceed the requested interval.

Supervisor securely reserves the summary descriptor before pool allocation and
worker startup and rejects a shared result/summary path. Existing files remain
untouched; a later failed run may leave its newly reserved summary empty. Workers
count their final service-log attempt before publishing stage summaries. The parent
samples diagnostic losses after the bounded listener drain; reliable summary
`log_counts_final` is false if a blocked listener is still alive. The service-log
pipeline summary precedes that drain and explicitly marks its counts provisional.
Abandoned queued logs and a failed current write are counted as diagnostic losses.
The listener transitions through NEW/RUNNING/DRAINING/CLOSED or FAILED. Closing
a failed listener also discards and counts records queued after its thread exited.
`listener_drained` requires CLOSED; final loss counters can nevertheless be valid
after FAILED. The provisional service summary says `frames_drained` and does not
claim successful exit before the accounting file is written. If that write fails
after listener closure, a fresh bounded listener consumes the failure diagnostic.

Nested camera metrics and final accounting use distinct TypedDict contracts;
camera identities and counters are required, with optional latency percentiles.
Wire validation checks keys/types followed by nonnegative totals, matching camera
identities, bounded brightness/batches and ordered times/percentiles. Registered
classes are checked by identity, not just their names. TypedDict schema resolution
uses a bounded cache. Benchmark identity tracking retains only the latest frame/epoch and
count per configured camera, rejects repeated/backward frames, and reconciles
each camera's emitted results with final accounting. Legitimate dropped-frame
gaps and reconnect epochs remain valid.

## Role resources and diagnostic failures

The parent retains WorkerResources for allocation/accounting and compatibility
tests. Spawned targets receive CaptureResources, ProcessingResources or
ReporterResources. Reporter gets no raw memory specification or slot/frame
channels; Capture gets no result channel; Processing gets no capture stop signal.
Common ControlResources contain only health/log channels, abort and parent PID.
This reduces accidental coupling between trusted workers; it is not a security
sandbox or a restriction of individual socket endpoints to read-only/write-only.
Resource cleanup attempts every channel close and reports the first error as its
cause after all attempts. The independent mapping finally still closes/unlinks.

BoundedQueueHandler counts OSError and unexpected emit/preparation failures without
QueueHandler's synchronous stderr fallback. Malformed consumed log datagrams are
counted, leave the listener FAILED and do not prevent later valid records from
being consumed. Endpoint failure is also FAILED. A failed diagnostic listener
does not abort the application; bounded shutdown and finality remain explicit.

Cgroup mount roots and mountpoints decode octal mountinfo escapes in one pass.
Memory observations expose `pss_available`, `missing_pss_pids` and whether finite
cgroup limits were observed. Missing observations never constitute a hard memory
limit; external deployment cgroups remain necessary.

Publication refreshes and immediately transmits progress after every accepted
result, including within a backpressured batch. A degraded log listener keeps
exclusive consumption ownership until its thread exits; diagnostic failure alone
does not authorize replacement. Configuration reserves camera ID `total` for
aggregate FPS and enforces the verified Linux/x86_64 platform. Wire validation
rejects UTC timestamps beyond the safe ISO datetime range (year 9999). Reporter
flushes pending writes before closing throughput windows, which carry actual
`elapsed_s` and per-camera `counts` alongside the existing `fps` mapping.

Log timestamps are bounded on encode/decode. Formatting failures discard only
the offending record, whereas sink failures terminate the diagnostic consumer.
Serial preparation publishes completed-frame progress and gives inference a fresh
budget; overlap refreshes preparer observations after individual completed frames.
Cleanup attempts all independent resources before publishing final accounting.
Accounting records include `cleanup_completed` and `cleanup_errors`; a cleanup
error returns nonzero. Final summary write/flush/close uses a duplicate descriptor
for truncation rollback on failure, so a failed publication leaves no success bytes.
Finalization failures use a fresh bounded diagnostic channel after original IPC
is disposed. This transaction handles reported I/O errors, not arbitrary machine
crashes or durability without fsync.

Worker shutdown treats each process independently and aggregates failures after
all reap attempts. ProcessLookupError during kill is a normal process-exit race.
Camera watchdogs treat RELEASING as active work; STOPPED is published after
release completes, including exceptional completion. Accounting schema version 1
requires cleanup_completed and cleanup_errors to agree with exit_code.

## Review hardening

`PipelineConfig.__post_init__` snapshots camera/geometry sequences as tuples,
including direct API construction. Caller mutations cannot invalidate admission
checks. `resources.reserve_shared_memory` contains the CPython/Linux private
handle dependency and rejects unsupported handles before touching raw pages.

`ResultOutput` supports optional create-only numbered segments. The configured
byte limit applies per segment; total retained data is bounded by that limit
times `output_max_files`. No record is split and no existing file is replaced.
The default remains one segment. Rotation flushes (and optionally fsyncs) the
previous segment before transferring stream ownership.

Reporter retains recent exact samples for diagnostics and a fixed histogram for
each throughput window. Buckets are millisecond ceilings through 1999 ms plus
an overflow bucket at 2000 ms. Percentiles use the nearest-rank definition. Thus bucket ceilings are conservative near
the two-second acceptance boundary; the overflow value is not an exact latency.
Worst window p95 includes the final partial window and cannot forget earlier
slow periods. Benchmark artifact readers stream rotated segments in order.

Soak acceptance requires continuous steady-state RSS coverage, CPU coverage and
explicit reasons for all failed gates. Its four cases form a serial/overlap by
stable/reconnect comparison. CPU measurement begins when all stage readiness
records have been observed (within one polling interval), and excludes samples
beyond the requested duration. Deployment cgroups remain the hard memory limit.
