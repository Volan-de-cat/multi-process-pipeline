"""Linux process-tree safety and bounded subprocess cleanup.

Workers die with their Supervisor even if it cannot execute Python cleanup.
Launchers isolate pipeline process groups; cleanup targets the group even after
its direct parent has exited. No worker-owned IPC lock is needed for termination.
"""

import ctypes
import os
import signal
import subprocess
from contextlib import suppress
from multiprocessing import shared_memory
from pathlib import Path


def bind_parent_death(parent_pid: int) -> None:
    """Install SIGKILL on parent death, closing the prctl/getppid startup race."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def reap_group(process: subprocess.Popen, timeout: float = 10) -> None:
    """Stop an isolated process group, then force-kill remaining descendants.

    Only call for a process launched with start_new_session=True. A disappeared
    direct parent is not proof that the rest of its group exited.
    """
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=3)
    # A forced group kill also kills resource_tracker. Remove only named pools
    # belonging to this launched Supervisor and this UID; never sweep global shm.
    for path in Path("/dev/shm").glob(f"cv_pipeline_{process.pid}_*"):  # nosec B108
        try:
            if path.stat().st_uid != os.geteuid():
                continue
            memory = shared_memory.SharedMemory(name=path.name)
            try:
                memory.unlink()
            finally:
                memory.close()
        except FileNotFoundError:
            pass
