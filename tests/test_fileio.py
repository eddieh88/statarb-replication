"""An interrupted write must leave nothing behind that looks like a finished file."""
import os
import pytest

from fileio import atomic_path, atomic_write_bytes


def test_atomic_write_creates_file_and_no_temp(tmp_path):
    dst = tmp_path / "file.parquet"
    atomic_write_bytes(dst, b"payload")
    assert dst.read_bytes() == b"payload"
    assert os.listdir(tmp_path) == ["file.parquet"]


def test_failed_write_leaves_no_file(tmp_path):
    dst = tmp_path / "file.parquet"
    with pytest.raises(RuntimeError):
        with atomic_path(dst) as tmp:
            with open(tmp, "wb") as fh:
                fh.write(b"half a file")
            raise RuntimeError("connection dropped")
    assert os.listdir(tmp_path) == []
