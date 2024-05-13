import io
import json
import logging
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import Callable
import signal

from .host import main
from .logging import gethostname, initialize_logging
from . import Context, Host

PORT = 55555

logger = logging.getLogger(__name__)


NON_RETRYABLE_FAILURE = 100
JOB_RESTART_SCOPE_ESCALATION = 101


def _write_reply_file(msg, reply_file=None):
    if reply_file is None:
        reply_file = os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"]
    job_attempt = int(os.environ["MAST_HPC_JOB_ATTEMPT_INDEX"])
    logger.info(
        f"Supervisor writing a reply file with JOB_RESTART_SCOPE_ESCALATION to {reply_file} (attempt {job_attempt})."
    )
    with open(reply_file, "w") as reply_file:
        timestamp_ns = time.time_ns()
        error_data = {
            "message": msg,
            "errorCode": JOB_RESTART_SCOPE_ESCALATION,
            "timestamp": int(timestamp_ns // 1e9),
            "timestamp_us": int(timestamp_ns // 1e3),
        }
        json.dump(error_data, reply_file)


def mast(supervise: Callable[[int, int], None]):
    """
    This function is the entrypoint for starting the supervisor when
    running on MAST. Each host should call `mast(supervise)` where
    `supervise` is the supervisor policy function for the job.
    Supervisor will be called only on the supervisor machine with
    `supervisor(n_hosts_in_task, port)` where `n_hosts_in_task` is
    the number of hosts reserved in the task group, and `port` is the
    port that supervisor should listen on.

    The supervise function can then create a supervisor Context object,
    request up to n_hosts_in_tasks hosts, and then
    """

    N = int(os.environ["MAST_HPC_TASK_GROUP_SIZE"])
    my_host_name = os.environ.get("HOSTNAME", socket.gethostname())
    # Get first host in the task group
    hostname_0 = min(os.environ["MAST_HPC_TASK_GROUP_HOSTNAMES"].split(","))
    hostname_0 = socket.getfqdn(hostname_0)
    is_supervisor = hostname_0 == my_host_name
    initialize_logging(
        "supervisor" if is_supervisor else f"{gethostname()} host-manager"
    )

    supervisor_addr = f"tcp://{hostname_0}:{PORT}"
    logger.info(
        "hostname %s, supervisor host is %s, supervisor=%s",
        my_host_name,
        hostname_0,
        is_supervisor,
    )
    if is_supervisor:
        _write_reply_file(
            "Supervisor deadman's switch. "
            "This reply file is written when the supervisor starts and deleted right before a successful exit. "
            "It is used to cause the whole job to restart if for some reason the "
            "supervisor host is unscheduled without it throwing an exception itself."
        )
        # local host manager on supervisor machine
        host_process = subprocess.Popen(
            [sys.executable, "-m", "supervisor.host", supervisor_addr]
        )
        try:
            ctx = Context(port=PORT)
            hosts: List[Host] = ctx.request_hosts(n=N)
            supervise(ctx, hosts)
            ctx.shutdown()
            logger.info("Supervisor shutdown complete.")
        except:
            ty, e, st = sys.exc_info()
            s = io.StringIO()
            traceback.print_tb(st, file=s)
            _write_reply_file(f"{ty.__name__}: {str(e)}\n{s.getvalue()}")
            host_process.send_signal(signal.SIGINT)
            raise
        return_code = host_process.wait(timeout=10)
        if return_code != 0:
            # Host manager may have been instructed to write a reply file, so
            # we do not write a reply file here which would clobber it.
            logger.warning(
                f"Host manager on supervisor returned non-zero code: {return_code}."
            )
            sys.exit(return_code)
        else:
            # successful exit, so we remove the deadman's switch reply file we wrote earlier.
            reply_file = os.environ["MAST_HPC_TASK_FAILURE_REPLY_FILE"]
            os.unlink(reply_file)
    else:
        # host manager on non-supervisor machine
        main(supervisor_addr)
