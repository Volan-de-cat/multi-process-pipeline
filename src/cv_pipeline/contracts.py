"""Structural interfaces for testable external APIs and bounded IPC.

Protocols describe only behavior consumed by stages; they allocate no resources
and introduce no inheritance framework. Injected clocks/cameras make failure and
timeout policies testable without slow real sleeps. Concrete data queues and
atomic control channels both implement the narrow MessageQueue interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar

from .messages import InferenceResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

T = TypeVar("T")


class Camera(Protocol):
    """One-owner source; disconnect requires a new instance, not another read."""

    def read(self) -> NDArray[np.uint8]:
        """Return the next owned uint8 frame, or raise an expected disconnect."""
        ...

    def release(self) -> None:
        """Release native/simulated resources; adapters must permit repeated release."""
        ...


class InferenceEngine(Protocol):
    """One owner per Processing process; result order matches the input sequence."""

    def infer(self, frames: Sequence[NDArray[np.uint8]]) -> list[InferenceResult]:
        """Consume independent grayscale arrays and return one result per input."""
        ...


class Clock(Protocol):
    """Separate receipt labels from elapsed-time arithmetic on the same host."""

    def monotonic_ns(self) -> int:
        """Return nanoseconds for deadlines and cross-process elapsed intervals."""
        ...

    def utc_ns(self) -> int:
        """Return Unix UTC nanoseconds for human-readable receipt timestamps."""
        ...


class StopSignal(Protocol):
    """Monotonic cooperative signal; publication must remain safe after peer death."""

    def is_set(self) -> bool:
        """Read whether stop was requested without depending on another process."""
        ...

    def set(self) -> None:
        """Request stop permanently; never reset a running pipeline's flag."""
        ...

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to seconds; return true on stop, false on timeout."""
        ...


class MessageQueue(Protocol[T]):
    """Bounded trusted-message transport; Queue depth is not a correctness API."""

    def put(self, item: T, block: bool = True, timeout: float | None = None) -> None:
        """Publish a message or raise Full when its capacity/deadline is exhausted."""
        ...

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """Consume the next complete message, or raise Empty on its deadline."""
        ...

    def close(self) -> None:
        """Dispose only process-local handles after ownership ends."""
        ...

    def get_nowait(self) -> T:
        """Receive without waiting; raise Empty when unavailable."""
        ...

    def put_nowait(self, item: T) -> None:
        """Publish without waiting; raise Full when unavailable."""
        ...
