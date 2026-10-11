"""Linux memory observations and cgroup hierarchy discovery.

Estimates are admission checks, not hard limits. PSS sums apportion shared pages
and are sampled at runtime; deployment cgroups provide the actual hard boundary.
Cgroup observations include ancestors and existing consumers of their budget.
"""

import os
import re
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TypedDict


class CgroupMemory(TypedDict):
    """One observed finite cgroup limit and its current consumption."""

    path: str
    limit_bytes: int
    current_bytes: int


class MemoryObservation(TypedDict):
    """Explicitly distinguish missing PSS/cgroup data from finite observations."""

    pss_bytes: int | None
    peak_pss_bytes: int
    pss_available: bool
    missing_pss_pids: list[int]
    cgroup_limits_observed: bool
    cgroups: list[CgroupMemory]


def mountinfo_path(value: str) -> Path:
    """Decode mountinfo's octal escapes in a single pass, including literal backslashes."""
    return Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value))


def cgroup_memory(proc: Path = Path("/proc"), mounts: Path | None = None) -> list[CgroupMemory]:
    """Read v1/v2 memory limits/current usage from the process group and ancestors.

    Paths are resolved against mountinfo roots, supporting nested host groups and
    cgroup namespaces. Missing or inaccessible observations remain unavailable.
    """
    try:
        groups = (proc / "self/cgroup").read_text().splitlines()
        mount_lines = (mounts or proc / "self/mountinfo").read_text().splitlines()
    except OSError:
        return []
    paths: dict[str, str] = {}
    for line in groups:
        _, controllers, group = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            paths["cgroup2" if not controllers else "cgroup"] = group
    observations: list[CgroupMemory] = []
    seen = set()
    for line in mount_lines:
        before, after = line.split(" - ", 1)
        kind, _, options = after.split()[:3]
        if kind not in paths or (kind == "cgroup" and "memory" not in options.split(",")):
            continue
        fields = before.split()
        root, mount = mountinfo_path(fields[3]), mountinfo_path(fields[4])
        group_path = Path(paths[kind])
        try:
            relative = group_path.relative_to(root)
        except ValueError:
            continue
        directory = mount / relative
        names = (
            ("memory.max", "memory.current")
            if kind == "cgroup2"
            else ("memory.limit_in_bytes", "memory.usage_in_bytes")
        )
        while True:
            if directory not in seen:
                seen.add(directory)
                try:
                    limit = (directory / names[0]).read_text().strip()
                    current = int((directory / names[1]).read_text().strip())
                    if limit.isdigit():
                        observations.append(
                            {
                                "path": str(directory),
                                "limit_bytes": int(limit),
                                "current_bytes": current,
                            }
                        )
                except (OSError, ValueError):
                    pass
            if directory == mount:
                break
            directory = directory.parent
    return observations


def process_pss(pid: int) -> int | None:
    """Return proportional resident bytes; inaccessible or exited processes give None."""
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


class MemoryMonitor:
    """Sample process PSS; reject observed over-budget usage without promising OOM prevention."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = budget_bytes
        self.peak_pss_bytes = 0
        self.last_pss_bytes: int | None = None

    def sample(self, pids: list[int]) -> MemoryObservation:
        values = [process_pss(pid) for pid in pids]
        self.last_pss_bytes = (
            sum(v for v in values if v is not None) if all(v is not None for v in values) else None
        )
        if self.last_pss_bytes is not None:
            self.peak_pss_bytes = max(self.peak_pss_bytes, self.last_pss_bytes)
            if self.last_pss_bytes > self.budget_bytes:
                raise RuntimeError("observed pipeline PSS exceeds memory_budget_mib")
        cgroups = cgroup_memory()
        return {
            "pss_bytes": self.last_pss_bytes,
            "peak_pss_bytes": self.peak_pss_bytes,
            "pss_available": self.last_pss_bytes is not None,
            "missing_pss_pids": [
                pid for pid, value in zip(pids, values, strict=True) if value is None
            ],
            "cgroup_limits_observed": bool(cgroups),
            "cgroups": cgroups,
        }


def reserve_shared_memory(memory: SharedMemory, size: int) -> None:
    """Reserve the exact CPython/Linux inode before writes can trigger SIGBUS.

    Keep the private implementation dependency at this boundary. Never reopen a
    shared-memory pathname: it may have been replaced between creation and use.
    """
    descriptor = getattr(memory, "_fd", None)
    if type(descriptor) is not int or descriptor < 0:
        raise RuntimeError("shared-memory backing reservation requires CPython/Linux")
    os.posix_fallocate(descriptor, 0, size)
