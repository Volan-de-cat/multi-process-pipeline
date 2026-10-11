"""Crash-safe control primitives for the verified Linux deployment.

Data, health and service logs use bounded JSON datagrams: killing a sender cannot leave a partial
message or a shared writer lock that blocks the Supervisor. Signals contain only
monotonic byte flags, never Condition waiters belonging to another process.
The decoder rejects arbitrary constructors; handles still belong to same-trust
workers, not a sandbox boundary.
"""

from __future__ import annotations

import os
import select
import socket
import time
from multiprocessing.context import BaseContext
from multiprocessing.reduction import DupFd
from queue import Empty, Full
from typing import Any

from .wire import decode, encode


class SharedStopSignal:
    """A monotonic, lock-free stop byte for same-host cooperative polling.

    Only transitions from zero to one are permitted. The byte does not publish frame
    memory: Queue operations still establish frame ownership. Raw ctypes byte reads
    and writes avoid recoverability assumptions about process-shared locks on the
    verified Linux/x86_64 target. No reset/reuse or general atomic counter is offered.
    """

    def __init__(self, context: BaseContext) -> None:
        """Allocate one shared byte from the same spawn context as worker queues."""
        self._value = context.RawValue("b", 0)

    def is_set(self) -> bool:
        """Read the stop byte without acquiring a lock owned by another process."""
        return bool(self._value.value)

    def set(self) -> None:
        """Request stop permanently; safe even if a former reader was killed."""
        self._value.value = 1

    def wait(self, timeout: float | None = None) -> bool:
        """Poll at ≤2 ms intervals until set or the monotonic deadline expires."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            time.sleep(0.002 if remaining is None else min(0.002, remaining))
        return True


class EventCredits:
    """Unnamed kernel counter; each successful read consumes exactly one credit.

    eventfd semaphore mode has constant storage, independent of outstanding credits.
    Spawn duplicates the descriptor explicitly. Participant death may lose ownership;
    the Supervisor disposes the entire transport on that failure path.
    """

    def __init__(self, count: int) -> None:
        if type(count) is not int or not 0 <= count < 2**32:
            raise ValueError("invalid event credit count")
        self._fd = os.eventfd(count, os.EFD_SEMAPHORE | os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    def __getstate__(self) -> dict[str, Any]:
        return {"descriptor": DupFd(self._fd)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._fd = state["descriptor"].detach()

    def take(self, block: bool, deadline: float | None) -> None:
        while True:
            try:
                os.eventfd_read(self._fd)
                return
            except BlockingIOError:
                remaining = None if deadline is None else deadline - time.monotonic()
                if not block or (remaining is not None and remaining <= 0):
                    raise Empty from None
                select.select([self._fd], [], [], remaining)

    def give(self) -> None:
        # Capacity is resource-bounded far below UINT64_MAX. A failed write is a
        # transport failure, never an empty receive or an silently discarded item.
        os.eventfd_write(self._fd, 1)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class SlotCredits:
    """Camera-partitioned free-slot bitmap with eventfd availability.

    One Capture thread reserves a partition. Processing returns published slots;
    Capture returns only its unpublished reservations. No slot has simultaneous
    writers. On Linux/x86_64 byte stores precede the kernel credit publication.
    This is an ownership protocol between trusted workers, not a general MPMC ring.
    Startup needs no socket messages regardless of pool size.
    """

    def __init__(self, context: BaseContext, start: int, count: int) -> None:
        self._start = start
        self._count = count
        self._states = context.RawArray("b", count)
        self._available = EventCredits(count)
        self._cursor = 0

    def get(self, block: bool = True, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        self._available.take(block, deadline)
        for offset in range(self._count):
            index = (self._cursor + offset) % self._count
            if self._states[index] == 0:
                self._states[index] = 1
                self._cursor = (index + 1) % self._count
                return self._start + index
        raise RuntimeError("slot bitmap and availability counter disagree")

    def put(self, item: int, block: bool = True, timeout: float | None = None) -> None:
        if type(item) is not int:
            raise ValueError("slot must be an integer")
        index = item - self._start
        if not 0 <= index < self._count:
            raise ValueError("slot outside its camera partition")
        if self._states[index] != 1:
            raise RuntimeError("slot released twice")
        self._states[index] = 0
        self._available.give()

    def get_nowait(self) -> int:
        return self.get(block=False)

    def put_nowait(self, item: int) -> None:
        self.put(item, block=False)

    def close(self) -> None:
        self._available.close()


class DatagramChannel:
    """Atomic JSON messages with a record ceiling and independent kernel byte limit.

    capacity bounds outstanding records, but socket buffers can fill sooner.
    Data publication always supplies a deadline. Free-slot storage uses SlotCredits
    and therefore never prefills this transport before consumers start.
    """

    MAX_MESSAGE_BYTES = 60_000

    def __init__(
        self, buffer_bytes: int = 256_000, *, capacity: int | None = None, context: Any = None
    ) -> None:
        self._closed = False
        self._credits = None
        self._receiver, self._sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            for endpoint in (self._receiver, self._sender):
                endpoint.setblocking(False)
                endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
                endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
            if capacity is not None:
                if type(capacity) is not int or capacity <= 0:
                    raise ValueError("capacity must be a positive integer")
                self._credits = EventCredits(capacity)
        except BaseException:
            self.close()
            raise

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        """Publish atomically; restore acquired credit on serialization/send failure."""
        payload = encode(item)
        if len(payload) > self.MAX_MESSAGE_BYTES:
            raise ValueError("control message exceeds datagram size limit")
        deadline = None if timeout is None else time.monotonic() + timeout
        if self._credits is not None:
            try:
                self._credits.take(block, deadline)
            except Empty:
                raise Full from None
        try:
            self._send(payload, block, deadline)
        except BaseException:
            if self._credits is not None:
                self._credits.give()
            raise

    def _send(self, payload: bytes, block: bool, deadline: float | None) -> None:
        while True:
            try:
                self._sender.send(payload)
                return
            except BlockingIOError:
                remaining = None if deadline is None else deadline - time.monotonic()
                if not block or (remaining is not None and remaining <= 0):
                    raise Full from None
                select.select([], [self._sender], [], remaining)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Consume one message; credit failures propagate independently of Empty."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                payload = self._receiver.recv(self.MAX_MESSAGE_BYTES + 1)
            except BlockingIOError:
                remaining = None if deadline is None else deadline - time.monotonic()
                if not block or (remaining is not None and remaining <= 0):
                    raise Empty from None
                select.select([self._receiver], [], [], remaining)
                continue
            if self._credits is not None:
                self._credits.give()
            if len(payload) > self.MAX_MESSAGE_BYTES:
                raise ValueError("oversized wire payload")
            return decode(payload)

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)

    def get_nowait(self) -> Any:
        return self.get(block=False)

    def close(self) -> None:
        if not self._closed:
            self._receiver.close()
            self._sender.close()
            if self._credits is not None:
                self._credits.close()
            self._closed = True
