import logging


from supervisor import Host, Process, ProcessExited, HostDisconnected, TTL
from typing import List, Any, Set
import sys
import signal
from supervisor import get_message_queue
from supervisor.launchers import mast
import os
import itertools
import time
from typing import NamedTuple

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def start_training(
    ctx, N: int, hosts: Set[Host], npp: int, run_fraction, rank_fraction, attempt
):
    # we will use `run_fraction` of machines as training machines.
    # the rest will be used as failover machines.
    desired_run_size: int = int(run_fraction * N)

    # The supervisor can now create processes on hosts.
    # We will start by running a health check on all of our machines
    # to find the top `rank_fraction` of machines and exclude the remaining stragglers.

    logger.info(f"starting health checks host {len(hosts)} hosts")
    env = {"ATTEMPT": str(attempt)}
    _pg: List[Process] = ctx.create_process_group(
        hosts,
        args=[sys.executable, __file__, "--health", sys.argv[2]],
        env=env,
        processes_per_host=1,
        name="health_check",
    )

    ttl = TTL(60 * 5)
    to_sort = int(rank_fraction * N)
    responses = ctx.messagefilter(HealthResponse)
    to_rank = [responses.recv(timeout=ttl()) for _ in range(to_sort)]
    logger.info(f"Found {len(to_rank)} hosts that passed health checks, ranking...")
    to_rank = sorted(to_rank, key=lambda x: x[1])

    if len(to_rank) < desired_run_size:
        raise Exception("Not enough healthy hosts")

    good_hosts = [p.host for p, _ in to_rank[:desired_run_size]]
    logger.info(f"Chose hosts: {good_hosts}")

    # Let's get training started.
    logger.info(f"Launching {npp*desired_run_size} processes")

    return ctx.create_process_group(
        good_hosts,
        args=[sys.executable, __file__, "--train", sys.argv[2]],
        env=env,
        processes_per_host=npp,
        name="train",
    )


def wait_for_exit(ctx, process_group, hosts):
    process_group_set = set(process_group)
    exited = 0
    while exited < len(process_group):
        sender, msg = ctx.recv()
        if isinstance(msg, ProcessExited) and sender in process_group_set:
            if msg.result != 0:
                for p in process_group:
                    p.signal(
                        signal.SIGTERM
                    )  # TODO: maybe have a broadcasting signal function.
                return False
            else:
                exited += 1
        elif isinstance(msg, HostDisconnected):
            hosts.remove(sender)
            hosts.update(ctx.replace_hosts([sender]))

    return True


def supervise(ctx, host_list):
    N = config["N"]
    hosts: Set[Host] = set(host_list[:N])
    npp = 2
    for i in itertools.count():
        if i >= len(config["train"]):
            ctx.shutdown()
            raise Exception("Too many attempts")
        logger.info(
            f"Starting training attemp {i} with {N} hosts, {npp} processes per host."
        )
        process_group = start_training(
            ctx, N, hosts, npp, config["run_fraction"], config["rank_fraction"], i
        )
        if wait_for_exit(ctx, process_group, hosts):
            break
        logger.info("Training has failed, attempting restart...")
    logger.info(f"Training exited successfully.")


def train():
    rank = int(os.environ["RANK"])
    attempt = int(os.environ["ATTEMPT"])
    action = config["train"][attempt][rank]
    if action == "F":
        raise Exception("Failed!")
    elif action == ".":
        pass
    elif action == "E":
        os.kill(os.getppid(), signal.SIGKILL)

class HealthResponse(NamedTuple):
    health: Any

def health():
    rank = int(os.environ["RANK"])
    attempt = int(os.environ["ATTEMPT"])
    q = get_message_queue()
    health = config["health"][attempt][rank]
    if health == "hang":
        while True:
            time.sleep(1)
    q.send(HealthResponse(health))


if __name__ == "__main__":
    config = eval(sys.argv[2])
    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--health":
        health()
    else:
        assert sys.argv[1] == "--supervise"
        mast(supervise)
