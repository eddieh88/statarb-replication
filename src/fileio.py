"""Crash-safe writes: an interrupted download must never look like a finished one.

The fetchers skip files that already exist, so a partially written file would be
treated as complete forever. Writing to a temporary name in the same directory and
renaming it with os.replace (atomic on the same filesystem) prevents that.
"""
import os
from contextlib import contextmanager


@contextmanager
def atomic_path(path):
    """Yield a temporary path; move it to `path` only if the block succeeds."""
    tmp = f"{path}.part"
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_bytes(path, data: bytes) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "wb") as fh:
            fh.write(data)
