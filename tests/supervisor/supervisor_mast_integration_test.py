import subprocess
import signal
import time
import os
import socket
import tempfile

import unittest

# To run as a standalone test: pytest --capture=no tests/supervisor/supervisor_mast_integration_test.py
# Use -k to filter to run a specific test

# test will emulate a mast launcher running src.startup.supervise
# but using a fake trainer, and injecting some fake health test errors.

N = 4


def emulate_mast_launch(to_launch, multi_group=False):
    # for testing the mast launcher entrypoint
    def create_host(i):
        env = {
            "MAST_HPC_MAX_TASK_FAILURES": "1",
            "MAST_HPC_JOB_ATTEMPT_INDEX": "0",
            **os.environ,
        }
        env.setdefault("MAST_HPC_JOB_VERSION", "0")
        env.setdefault("MAST_HPC_ATTEMPT_INDEX", "0")
        env["TW_TASK_ID"] = str(i)
        env["MAST_HPC_TASK_GROUP_HOSTNAMES"] = socket.gethostname()
        if multi_group:
            if i is None:
                env["MAST_HPC_TASK_GROUP_SIZE"] = str(1)
                env["MAST_HPC_TASK_GROUP_NAME"] = "supervisor"
            else:
                env["MAST_HPC_TASK_GROUP_SIZE"] = str(N)
                env["MAST_HPC_TASK_GROUP_NAME"] = "hosts"
        else:
            env["MAST_HPC_TASK_GROUP_SIZE"] = str(N)
        env["HOSTNAME"] = (
            socket.gethostname()
            if i == 0 and not multi_group or i is None
            else f"fake_host_{i}"
        )
        env["TORCH_SUPERVISOR_TASK_SIZE"] = str(N)
        return subprocess.Popen(to_launch, env=env)

    hosts = [create_host(i) for i in range(N)]
    if multi_group:
        hosts.append(create_host(None))
    print("PIDS", [h.pid for h in hosts])
    r = hosts[0].wait()
    if r != 0:
        print(f"Host 0 exited with {r}, interrupting other hosts...")
        for c in hosts:
            c.send_signal(signal.SIGINT)
    return r

class SupervisorMastIntegrationTest(unittest.TestCase):
    def test_e2e_ok(self):
        with tempfile.TemporaryDirectory() as test_dir:
            os.environ["MAST_HPC_MAX_TASK_FAILURES"] = "3"
            os.environ["TORCH_SUPERVISOR_HEARTBEAT_LIVENESS"] = "3.0"
            os.environ["SUPERVISOR_HEALTH_CHECKS"] = "--testing"
            os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"] = f"/{test_dir}/reply"
            os.environ["PYTHONPATH"] = "."
            os.environ["DUMP_DIR"] = "/tmp"
            os.environ["SUPERVISOR_TRAIN_PATTERN"] = "--fake"
            os.environ["SUPERVISOR_STEPBEAT_TIMEOUT_INITIAL"] = "2"
            os.environ["SUPERVISOR_STEPBEAT_TIMEOUT"] = "2"

            rest = "python -m src.startup.supervise --nnodes 4 --failover 2 --max-attempts 2 --nproc-per-node 2"
            args = [*rest.split(" "), "python", f"{os.path.dirname(os.path.abspath(__file__))}/worker.py", "--path", test_dir]

            emulate_mast_launch(args)

            rest = "python -m src.startup.supervise --nnodes 4 --failover 2 --max-attempts 3 --nproc-per-node 2"
            args = [*rest.split(" "), "python", __file__, "--fake"]
            assert 0 == emulate_mast_launch(args)

            output_files = [f"rank_{i}_output.txt" for i in range(N)]
            self.assertTrue(set(os.listdir(test_dir)).issuperset(set(output_files)))

            for output_file in output_files:
                with open(f"{test_dir}/{output_file}", "r") as f:
                    self.assertEqual(2, len(f.readlines()), "Expected 2 lines in worker output file, one for each attempt")

    def test_e2e_supervisor_error(self):
        # Fails to create process group: handled in supervisor launcher.
        with tempfile.TemporaryDirectory() as test_dir:
            os.environ["MAST_HPC_MAX_TASK_FAILURES"] = "2"
            os.environ["SUPERVISOR_HEALTH_CHECKS"] = "--testing"
            os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"] = f"/{test_dir}/reply"
            os.environ["PYTHONPATH"] = "."
            rest = "python -m src.startup.supervise --nnodes 4 --failover 2 --max-attempts 2 --nproc-per-node 2"
            args = [*rest.split(" "), "DOES_NOT_EXIST"]

            emulate_mast_launch(args)

            self.assertTrue(os.path.exists(f"{test_dir}/reply"))


    def test_e2e_no_failures(self):
        with tempfile.TemporaryDirectory() as test_dir:
            os.environ["MAST_HPC_MAX_TASK_FAILURES"] = "2"
            os.environ["SUPERVISOR_HEALTH_CHECKS"] = "--testing"
            os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"] = f"/{test_dir}/reply"
            os.environ["PYTHONPATH"] = "."
            rest = "python -m src.startup.supervise --nnodes 4 --failover 2 --max-attempts 2 --nproc-per-node 2"
            args = [*rest.split(" "), "python", f"{os.path.dirname(os.path.abspath(__file__))}/worker.py",
                    "--path", test_dir, "--happy"]

            emulate_mast_launch(args)

            self.assertFalse(os.path.exists(f"{test_dir}/reply"))
