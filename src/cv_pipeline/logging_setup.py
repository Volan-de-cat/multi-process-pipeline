"""Best-effort asynchronous diagnostics with bounded listener shutdown.

Workers and the parent only enqueue records. The lifecycle thread never writes
to stderr, including inside signal callbacks. Log handling is deliberately weaker
than result delivery: overflow or an unavailable sink must not prevent cleanup.
A Python thread cannot interrupt arbitrary stream.write; close bounds its wait
and permits a blocked daemon sink to finish later without retaining IPC owners.
"""

import logging
import os
import stat
import sys
import threading
import time
from enum import Enum
from logging.handlers import QueueHandler
from queue import Empty, Full
from typing import TextIO

from .contracts import MessageQueue


class BoundedQueueHandler(QueueHandler):
    """Nonblocking diagnostic publisher; expose the local discarded-record count."""

    def __init__(self, queue: MessageQueue[logging.LogRecord]) -> None:
        """Keep the standard LogRecord preparation but replace overflow behavior."""
        super().__init__(queue)
        self.dropped_records = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        """Send without waiting; count overflow rather than raising logging errors."""
        try:
            self.queue.put_nowait(record)
        except (Full, ValueError, OSError):
            self.dropped_records += 1

    def handleError(self, record: logging.LogRecord) -> None:
        """Count preparation/transport failures without synchronous stderr fallback.

        QueueHandler.emit delegates unexpected exceptions here. Diagnostic errors
        must never perform I/O on the producer or Supervisor lifecycle thread.
        """
        self.dropped_records += 1


def configure_worker_logging(queue: MessageQueue[logging.LogRecord]) -> None:
    """Route one worker's service logs to the parent; keep Reporter output separate."""
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(BoundedQueueHandler(queue))
    root.setLevel(logging.INFO)


class ListenerState(Enum):
    """Distinguish consumer liveness, output failure and completed queue cleanup."""

    NEW = "new"
    RUNNING = "running"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


class LogListenerOwner:
    """Parent-local consumer whose lifetime cannot block resource disposal."""

    def __init__(
        self,
        queue: MessageQueue[logging.LogRecord],
        stream: TextIO | None = None,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        """Accept an injectable stream for blocked/failing-output regression tests."""
        self._queue = queue
        self._max_bytes = max_bytes
        self._written = 0
        self.dropped_records = 0
        self._stream = sys.stderr if stream is None else stream
        self._formatter = logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(message)s")
        self._fd = None
        try:
            original = self._stream.fileno()
            if stat.S_ISREG(os.fstat(original).st_mode):
                # A duplicate shares the output offset, preserving child traceback
                # ordering. os.write bypasses TextIO/Handler locks at interpreter exit.
                self._fd = os.dup(original)
            else:
                # Reopening a pipe/terminal makes O_NONBLOCK private to our file
                # description; setting it on dup() would also change sys.stderr.
                self._fd = os.open(f"/proc/self/fd/{original}", os.O_WRONLY | os.O_NONBLOCK)
        except (AttributeError, OSError, ValueError):
            # Injectable streams without a descriptor retain bounded thread-close
            # behavior; only the default Linux descriptor sink is nonblocking.
            self._fd = None
        self._stop = threading.Event()  # Local only: no killed worker can own its lock.
        self._deadline: float | None = None
        self._thread: threading.Thread | None = None
        self.state = ListenerState.NEW

    def _consume(self) -> None:
        """Drain logs until the local deadline or an empty queue after shutdown."""
        try:
            self._drain()
        finally:
            self._discard_pending()
            if self.state is not ListenerState.FAILED:
                self.state = ListenerState.CLOSED
            # The consumer owns the descriptor even if close() stopped waiting.
            # A blocked custom stream has no descriptor to leak.
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def _discard_pending(self) -> None:
        """Count records abandoned after a bounded drain or a failed diagnostic sink.

        The parent closes after reaping producers. The kernel byte bound limits this
        nonblocking cleanup; a blocked sink leaves counts explicitly provisional.
        """
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return
            except ValueError:
                self.dropped_records += 1
                self.state = ListenerState.FAILED
                continue  # The malformed datagram has already been consumed.
            except OSError:
                self.state = ListenerState.FAILED
                return
            self.dropped_records += 1

    def _drain(self) -> None:
        """Poll complete records and discard diagnostic writes that would block."""
        while self._deadline is None or time.monotonic() < self._deadline:
            try:
                record = self._queue.get(timeout=0.05)
            except Empty:
                if self._stop.is_set():
                    return
                continue
            except ValueError:
                self.dropped_records += 1
                self.state = ListenerState.FAILED
                continue  # Keep consuming valid records after a malformed datagram.
            except OSError:
                self.state = ListenerState.FAILED
                return
            try:
                message = self._formatter.format(record) + "\n"
            except Exception:
                # A bad record does not imply the diagnostic sink is unusable.
                self.dropped_records += 1
                continue
            try:
                size = len(message.encode("utf-8", errors="backslashreplace"))
                if self._written + size > self._max_bytes:
                    self.dropped_records += 1
                    continue
                self._written += size
                if self._fd is None:
                    self._stream.write(message)
                    self._stream.flush()
                else:
                    data = message.encode("utf-8", errors="backslashreplace")
                    while data:
                        try:
                            written = os.write(self._fd, data)
                            data = data[written:]
                        except BlockingIOError:
                            self.dropped_records += 1
                            break  # Full diagnostic pipe: discard the remainder.
            except Exception:
                self.dropped_records += 1
                self.state = ListenerState.FAILED
                return  # Never propagate sink errors into pipeline ownership.

    @property
    def is_alive(self) -> bool:
        """Report consumer liveness independently of diagnostic failure state."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start one daemon consumer; all essential cleanup remains in the parent."""
        if self.state is not ListenerState.NEW:
            raise RuntimeError("listener is one-shot")
        self.state = ListenerState.RUNNING
        self._thread = threading.Thread(
            target=self._consume, name="service-log-listener", daemon=True
        )
        self._thread.start()

    def close(self, graceful: bool = True, timeout_s: float = 0.5) -> bool:
        """Request drain and wait at most timeout_s; return whether the thread exited.

        A blocked arbitrary sink may leave a daemon thread until its write returns.
        That limitation is explicit and never extends the Supervisor's deadline.
        """
        if self.state is ListenerState.RUNNING:
            self.state = ListenerState.DRAINING
        self._deadline = time.monotonic() + (timeout_s if graceful else 0)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            finished = not self._thread.is_alive()
            if finished:
                # Producers are now stopped. Count messages accepted after an early
                # sink failure, rather than confusing thread exit with queue drain.
                self._discard_pending()
            return finished
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        return True
