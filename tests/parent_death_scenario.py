"""Isolated parent-death probe with Linux subreaping and unconditional cleanup."""

import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = Path(sys.argv[1])
folder.mkdir(exist_ok=True)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "cannot become subreaper")
process = None
pool = None
try:
    with (folder / "service.log").open("w") as logs:
        process = subprocess.Popen(
            [sys.executable, "-m", "cv_pipeline", "--output", str(folder / "results.jsonl")],
            cwd=root,
            stderr=logs,
            stdout=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            text = (folder / "service.log").read_text()
            found = re.search(r"pool_created name=(\S+)", text)
            if found:
                pool = found.group(1)
            workers = {
                stage: int(pid)
                for stage, pid in re.findall(r"stage_ready stage=(\w+) pid=(\d+)", text)
            }
            if len(workers) == 3:
                break
            if process.poll() is not None:
                raise RuntimeError("startup failed")
            time.sleep(0.05)
        else:
            raise TimeoutError("readiness timeout")
        if len(sys.argv) > 2 and sys.argv[2] == "group":
            os.killpg(process.pid, signal.SIGKILL)
            from cv_pipeline.lifecycle import reap_group

            reap_group(process)
        else:
            process.kill()
        process.wait(timeout=3)
        time.sleep(1)
        states = {}
        for stage, pid in workers.items():
            path = Path(f"/proc/{pid}/status")
            states[stage] = (
                re.search(r"^State:\s+(.+)", path.read_text(), re.MULTILINE).group(1)
                if path.exists()
                else "gone"
            )
        result = {
            "parent_exit_code": process.returncode,
            "observation_delay_s": 1,
            "worker_states": states,
            "pool_still_exists": bool(pool and Path("/dev/shm", pool).exists()),
        }
        (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
finally:
    if process is not None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                time.sleep(0.05)
    if pool and Path("/dev/shm", pool).exists():
        from multiprocessing import shared_memory

        memory = shared_memory.SharedMemory(name=pool)
        memory.close()
        memory.unlink()
