"""Safe create-only JSONL segments with bounded rotation and optional durable flush.

Every path component is opened relative to a checked directory descriptor.
Symlinks, traversal, existing files and special files are rejected; permissions
are private regardless of umask. Stdout is borrowed and never closed by us.
"""

import os
import sys
from pathlib import Path


class ResultOutput:
    """Count UTF-8 bytes and enforce storage policy for files and borrowed stdout."""

    def __init__(
        self, path: str | None, max_bytes: int, durable: bool = False, max_files: int = 1
    ) -> None:
        """Open private output; max_bytes applies per segment, max_files bounds retention.

        Rotation appends numbered suffixes without overwriting or deleting data.
        Borrowed stdout supports a single segment. Final summaries use one file.
        """
        if type(max_files) is not int or not 1 <= max_files <= 1024:
            raise ValueError("invalid output segment count")
        if max_files > 1 and path is None:
            raise ValueError("rotation requires owned file output")
        self._path = path
        self._max_files = max_files
        self._segment = 0
        self._owned = path is not None
        self._limit = max_bytes
        self._bytes = 0
        self._durable = durable
        self._stream = sys.stdout
        if path is not None:
            parts = Path(path).parts
            if not parts or ".." in parts or parts[-1] in ("/", "."):
                raise ValueError("output path must name a file without traversal")
            directory = os.open(
                "/" if Path(path).is_absolute() else ".", os.O_RDONLY | os.O_DIRECTORY
            )
            try:
                for part in parts[:-1]:
                    if part in ("/", "."):
                        continue
                    next_fd = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                    )
                    os.close(directory)
                    directory = next_fd
                fd = os.open(
                    parts[-1],
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
                try:
                    self._stream = os.fdopen(fd, "w", encoding="utf-8", buffering=8192)
                except BaseException:
                    os.close(fd)
                    raise
            finally:
                os.close(directory)

    def write(self, text: str) -> int:
        """Write an intact record, rotating before it when the current segment is full.

        Refuse oversized records and exhaustion of the segment allowance. Open the
        successor securely before retiring the current stream; failed creation
        leaves the original stream available for cleanup.
        """
        size = len(text.encode("utf-8"))
        if size > self._limit:
            raise OSError("record exceeds result output segment quota")
        if self._bytes + size > self._limit:
            if self._segment + 1 >= self._max_files:
                raise OSError("result output byte quota exceeded")
            # Secure create-only opening also applies to every numbered segment.
            successor = ResultOutput(
                f"{self._path}.{self._segment + 1:04d}", self._limit, self._durable
            )
            try:
                self.flush()
                self.close()
            except BaseException:
                successor.close()
                raise
            self._stream = successor._stream
            self._bytes = 0
            self._segment += 1
        result = self._stream.write(text)
        self._bytes += size
        return result

    def flush(self) -> None:
        """Flush Python buffering and optionally fsync an owned regular file."""
        self._stream.flush()
        if self._owned and self._durable:
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        """Close an owned file only; stdout lifetime belongs to the caller."""
        if self._owned:
            self._stream.close()

    def finish(self, text: str) -> None:
        """Publish a final summary and close it, rolling back bytes on any failure.

        Only single-segment owned files support this transaction. Keep a duplicate descriptor until
        write, flush and close complete so even a close error can invalidate success.
        A failed summary remains empty rather than claiming a successful lifecycle.
        """
        if not self._owned or self._max_files != 1:
            raise ValueError("final summary requires one owned output segment")
        rollback_fd = os.dup(self._stream.fileno())
        try:
            self.write(text)
            self.flush()
            self.close()
        except BaseException:
            # Closing first prevents buffered data from reappearing after truncation.
            try:
                self.close()
            finally:
                os.ftruncate(rollback_fd, 0)
            raise
        finally:
            os.close(rollback_fd)
