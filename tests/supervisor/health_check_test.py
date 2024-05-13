"""Check supervisor health checks."""

import io
import tempfile

from src.startup.health_check.log_file_check import (
    PriorUncorrectableECCError,
    tail,
    throw_on_file_error,
)


def test_log_file_check():
    """Checks parsing log files."""
    file_content = """[trainer4]:E0228 17:37:21.488888  9191 ProcessGroupNCCL.cpp:1405] [PG 3 Rank 4] NCCL watchdog thread terminated with exception: CUDA error: uncorrectable ECC error encountered"""
    file_object = io.StringIO(file_content)
    # TODO: Replace with pytest
    try:
        throw_on_file_error(file_object)
        assert False, "Expected PriorUncorrectableECCError"
    except PriorUncorrectableECCError:
        pass
    file_content = """Hello world\n[trainer4]:E0228 17:37:21.488888  9191 ProcessGroupNCCL.cpp:1405] [PG 3 Rank 4] NCCL watchdog thread terminated with exception: CUDA error: uncorrectable ECC error encountered\nHello\nWorld"""
    file_object = io.StringIO(file_content)
    try:
        throw_on_file_error(file_object)
        assert False, "Expected PriorUncorrectableECCError"
    except PriorUncorrectableECCError:
        pass
    file_content = """[trainer6]:I0228 14:32:00.476809  9076 ProcessGroupNCCL.cpp:1471] Pipe file /tmp/nccl_trace_rank14918.pipe has been opened, write to it to trigger NCCL Debug Dump.\nThis is not an error"""
    file_object = io.StringIO(file_content)
    # Should not raise an exception
    throw_on_file_error(file_object)


def test_tail():
    """Checks tailing log files."""
    file_content = """Hello world\n[trainer4]:E0228 17:37:21.488888  9191 ProcessGroupNCCL.cpp:1405] [PG 3 Rank 4] NCCL watchdog thread terminated with exception: CUDA error: uncorrectable ECC error encountered\nHello\nWorld"""
    with tempfile.NamedTemporaryFile() as f:
        f.write(file_content.encode())
        f.flush()
        lines = tail(f.name, 10)
        assert lines == file_content
        lines = tail(f.name, 1)
        assert lines == "World"


if __name__ == "__main__":
    test_log_file_check()
    test_tail()
