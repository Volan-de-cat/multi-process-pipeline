# Automated checks

GitHub Actions workflows live in the repository-root `.github/workflows/`.
Push the files to GitHub to activate them. Actions are pinned to reviewed full
commit SHAs; permissions are read-only, obsolete same-ref runs are cancelled,
and Bash pipefail prevents tee from hiding failed checks.

## Python CI

`python-ci.yml` runs on Python/workflow pushes, pull requests and manual dispatch.

- **Quality:** Ruff lint/format (including B/SIM rules), mypy checking all application and benchmark code with fully annotated functions,
  a cyclomatic complexity limit of 20 per function/method, Radon artifacts and
  source/wheel build using the locked local backend, followed by a real
  isolated installed-wheel pipeline smoke test.
- **Tests:** all unit, integration, regression and security tests on Python
  3.10/3.12/3.13, with subprocess branch coverage. XML/HTML are collected before
  enforcing the 80% threshold, so a low-coverage failure retains diagnostics.
- **Security:** Bandit rejects medium/high source findings; pip-audit checks
  installed package advisories. Fixed tmpfs capacity probes have narrow documented
  B108 suppressions; security rules are not disabled globally.

Each job installs `requirements.lock` with `--require-hashes`, then installs the
editable project with `--no-deps --no-build-isolation`. The lock includes direct
and transitive runtime/tool versions plus approved Linux wheel hashes.
`requirements-dev.txt` is a readable direct-tool list; CI uses the complete lock.
Update versions/hashes deliberately and rerun tests/audit. Known-advisory checks
cannot prove absence of native-library or unknown vulnerabilities.

Test/quality/security artifacts are retained for 14 days. SIGKILLed Python workers
cannot reliably save coverage; their shutdown behavior is tested independently.
Hosted jobs have timeouts and require enough `/dev/shm` for automatic raw pools.
Branch protection must be configured in GitHub to require these job results.

## Performance matrix

`python-performance.yml` runs on pushes to main/master, weekly schedule and
manual dispatch. Each of fifteen cases is run twice, sequentially, by
`benchmarks/performance_suite.py`. Manual trial duration is 12–60 seconds;
default is 20. Total workflow timeout is 40 minutes.

| Case | Workload | Minimum completed-window FPS | p95 limit |
| --- | --- | ---: | ---: |
| batch-4 / batch-8 / batch-16 | Default 50 nominal camera FPS, automatic pool | >10 | <2000 ms |
| low-input | Four sources at 2 FPS each | >5 | <2000 ms |
| sparse-camera | One source at 1 FPS, 700 ms health timeout | >0.5 | <2000 ms |
| four-threads / overlap / latest / buffered-output | Exact-pixel CPU parallelism, bounded overlap, newest queued frames, batched flush | >10 | <2000 ms |
| single-core | One allowed CPU, one OpenCV thread | >5 | <2000 ms |
| disconnect | cam_01 recreated after 30 reads; must observe reconnect | >10 | <2000 ms |
| slow-inference | Extra 200 ms per inference call | >8 | <2000 ms |
| slow-output | Extra 80 ms per output, result capacity 8 | >8 | <3000 ms |

Degraded-input/device cases have explicit acceptance limits instead of pretending
that every hardware condition must exceed the ordinary >10 FPS target. Every
case also requires exit code zero, complete Reporter/final accounting summaries,
and valid JSONL. Benchmark artifact readers validate consumed fields, finite
numbers, complete camera counters and agreement with emitted result counts.
Result identities must belong to configured cameras, increase monotonically per
camera and match each camera's final reported count; equal aggregate totals alone
cannot pass validation.
Malformed or truncated artifacts produce a failed report instead of aborting the
matrix; startup/drain allowances derive from the effective configuration. p95 gates use receipt-to-completed-flush latency, including
output work and configured buffering. An assertion failure makes the workflow fail and artifacts are
still uploaded. The matrix writes summary.json after each trial, retaining earlier
results if a later trial fails. Fault/slowdown injection lives only in the benchmark
helper; production has no testing switches and supplied simulators are unchanged.
The Markdown renderer validates rows independently: missing fields, wrong shapes,
duplicate keys and nonfinite numbers produce explicit diagnostic text while valid
rows and other sections remain available. Rendering never overrides failed
performance acceptance or changes its exit status.

Reports contain effective configuration, affinity, dependencies/CPU, throughput,
latency, drops, raw logs/results and per-process RSS. A separate final-accounting
file avoids relying on best-effort service logs. Directories are fresh; launchers
always clean isolated process trees. Performance artifacts are retained for 30 days
and a Markdown table is appended to the GitHub job summary. Matrix summaries also
include the preprocessing report when present. Preprocessing has a
separate warm-up/timing/pixel-error report. The accepted scratch-reuse and pooled-grayscale methods
must be bitwise equal to the reference on simulator and noise frames; experimental
quality-changing methods are recorded without being accepted as replacements.
The same report and job summary include isolated generation, host-copy, resize,
grayscale and inference timings for batches 1/4/8/16. Generation excludes camera
pacing and host-copy excludes IPC/shared-memory leases; the component medians
cannot be summed to predict concurrent pipeline throughput.

These are absolute acceptance checks, not a statistical regression guarantee.
Hosted CPU/load varies; use a controlled runner and comparable environments for
relative performance gates. Report RSS per process; use PSS/cgroup measurements
when budgeting physical memory instead of adding shared mappings repeatedly.

## Local commands

From python/ in the activated locked environment:

```bash
ruff check src tests benchmarks
ruff format --check src tests benchmarks
mypy
python benchmarks/check_complexity.py
python benchmarks/soak_test.py --duration 300 --directory artifacts/soak-new
coverage erase
coverage run -m unittest discover -s tests -v
coverage combine
coverage report
coverage xml
coverage html
bandit -r src -ll
pip-audit --local --skip-editable
python -m build --no-isolation
python benchmarks/performance_suite.py --duration 20 --repeats 2 --directory artifacts/matrix-new
python benchmarks/ci_summary.py --directory artifacts/matrix-new
```

Current checks and measured performance are recorded in
[VALIDATION.md](VALIDATION.md). Actual GitHub runs require pushing
this repository; local YAML parsing is not remote workflow execution.

The complexity budget of 20 covers both application and benchmark functions.
The fifteen-case performance matrix also exercises batch 16, a one-item result
queue, a one-second health budget and slow output. Benchmark acceptance requires
actual windows of at least ten seconds, recomputes FPS from counts, and matches
those counts against emitted result identities. Old throughput artifacts without
`elapsed_s` and `counts` are rejected by the updated acceptance reader.

Benchmark artifacts now require one terminal Reporter summary. Its camera counts,
batch histogram and ordered finite percentiles are checked against streamed results.
Delivery percentiles must not precede corresponding recent serialization percentiles.
No result, throughput row or second summary may follow the terminal record.
`ArtifactData` carries named bounded aggregates instead of a positional tuple.

The performance matrix also tests 3840x3840 grayscale output with batch 16 and
a one-second health budget, exercising completed-frame and inference transitions.
This case allows 2048 MiB estimated memory and 5000 ms p95 delivery latency.

Benchmark sources use an explicit policy recorded in each report: random camera
disconnects are disabled in ordinary trials; the disconnect workload fails cam_01
on read attempt 30 and uses fixed retry delays. Other cameras stay connected.
These overrides are confined to spawned benchmark workers; provided simulators
and production behavior are unchanged. Acceptance requires accounting schema
version 1, successful cleanup and an empty cleanup_errors list. Old accounting
artifacts require regeneration; silent legacy fallback is not supported.

The local soak now runs four controlled cases (12–20 minutes total), requires
representative sampling and checks all delivery windows as well as recent
latency. Each failure lists violated gates. Hosted short performance cases remain
sequential. Rotated result segments are streamed by the benchmark validator.
