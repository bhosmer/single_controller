import sys
import subprocess
import signal
from contextlib import contextmanager
import time
import os
import socket

# run as `python tests/supervisor/emulate_xlformers_mast_launch.py`
# from the xlformers directory.

# this script will emulate a mast launcher running src.startup.supervise
# but using a fake trainer, and injecting some fake health test errors.

N = 4
C = 7

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

    hosts = [create_host(i) for i in range(C)]
    if multi_group:
        hosts.append(create_host(None))
    print("PIDS", [h.pid for h in hosts])
    while hosts:
        finished = []
        status = [h.poll() for h in hosts]
        for i in range(len(hosts)):
            if status[i] is not None and status[i] != 0 and i == 0:
                print(f"Host {i} manager exited with {status[i]}, exiting...")
                for c in hosts:
                    c.send_signal(signal.SIGINT)
                return
        hosts = [h for h, s in zip(hosts, status) if s is None]
        time.sleep(0.1)


if len(sys.argv) == 2 and sys.argv[1] == "--fake":
    from src.startup.bootstrap import bootstrap, ProgressReport
    from supervisor import get_message_queue
    bootstrap()

    import logging
    logger = logging.getLogger()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    message_queue = get_message_queue()
    for i in range(10):
        logger.info(f"{rank} Training {i}")
        if i == 3 and rank == 3 and os.environ["TORCH_SUPERVISOR_ATTEMPT"] != "2":
            raise Exception("test error" + os.environ["TORCH_SUPERVISOR_ATTEMPT"])
        if rank == 0:
            message_queue.send(ProgressReport(i))
        time.sleep(1)

else:
    os.environ["MAST_HPC_MAX_TASK_FAILURES"] = "2"
    os.environ["SUPERVISOR_EMULATED"] = "1"
    os.environ["TORCH_SUPERVISOR_HEARTBEAT_LIVENESS"] = "3.0"
    os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"] = "/tmp/reply"
    os.environ["PYTHONPATH"] = "."
    os.environ["DUMP_DIR"] = '/tmp'
    rest = "python -m src.startup.supervise --nnodes 4 --failover 2 --max-attempts 2 --nproc-per-node 2"
    args = [*rest.split(" "), "python", __file__, "--fake"]

    emulate_mast_launch(args)
