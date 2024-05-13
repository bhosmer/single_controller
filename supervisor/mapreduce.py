# only in itertools after 3.12
from supervisor import Context, Host, Process, ProcessStarted, ProcessExited, get_message_queue
from typing import Iterator, List, TypeVar, Sequence, Dict
import sys
import zmq
from itertools import repeat
import pickle
import io

T = TypeVar("T")


def batched(xs: Sequence[T], n: int) -> Iterator[List[T]]:
    b = []
    for x in xs:
        b.append(x)
        if len(b) == n:
            yield b
            b = []
    if b:
        yield b


def mapreduce(
    hosts: Sequence[Host],
    mapper,
    inputs=None,
    branch=4,
    timeout=None,
):
    """
    Run finish(reduce([map(input) for input in inputs)]))
    Reduce must be associative and commutative.
    Finish will be run once on the final value.

    map, reduce, finish, inputs, and the outputs of each must be picklable.

    Not really map-reduce in the Hadoop sense with sort/shuffle/keys, but easy to implement for
    simple debug info gathering.
    """
    if len(hosts) == 0:
        raise ValueError("Must have at least one host")
    ctx = hosts[0]._context
    processes = ctx.create_process_group(
        hosts, [sys.executable, "-m", "supervisor.mapreduce"]
    )
    process_set = set(processes)
    broken = set()
    active = set(process_set)

    if inputs is None:
        for proc in processes:
            proc.send((mapper, None))
    else:
        inputs_per_proc = (len(inputs) - 1) // len(hosts) + 1
        for i, proc in enumerate(processes):
            start = inputs_per_proc * i
            proc.send((mapper, inputs[start:start + inputs_per_proc]))

    responses = ctx.messagefilter(lambda r: r.sender in process_set)
    batch = []
    while True:
        sender, message = responses.recv(timeout=timeout)
        if isinstance(message, ProcessExited) and message.result != 0:
            broken.add(sender)
        elif not isinstance(message, ProcessStarted):
            batch.append((sender, message))
        if len(batch) == min(branch, len(active)):
            (proc, port), *tail = batch
            batch.clear()
            if not tail:
                proc.send(("finish", None))
                break
            dst = f"tcp://{proc.host.hostname}:{port}"
            for src, port in tail:
                src.send(("send", dst))
                active.remove(src)
            proc.send(("reduce", len(tail)))
    return responses.recv(timeout=timeout).message


def _socket(context):
    s = context.socket(zmq.DEALER)
    s.setsockopt(zmq.IPV6, True)
    return s


def main():
    q = get_message_queue()
    _, (mapper, inputs) = q.recv()
    reduce = getattr(mapper, "reduce", sum)
    map = getattr(mapper, "map", lambda x: x)
    finish = getattr(mapper, "finish", lambda x: x)
    name = "tcp://*:*"
    router = _socket(q._ctx)
    router.bind(name)
    port = router.getsockopt(zmq.LAST_ENDPOINT).decode().split(":")[-1]
    r = map(inputs)
    # print(f"{port}: r = reduce(map({inputs})")
    q.send(port)
    while True:
        _, (action, value) = q.recv()
        if action != "reduce":
            break
        inputs = [r, *(router.recv_pyobj() for i in range(value))]
        # print(f"{port}: r = reduce({inputs})")
        r = reduce(inputs)
        q.send(port)

    if action == "send":
        # print(f"{port}: send {r} -> {value}")
        s = _socket(q._ctx)
        s.connect(value)
        s.send_pyobj(r)
    else:
        assert action == "finish"
        # print(f"{port}: finish {r}")
        q.send(finish(r))


if __name__ == "__main__":
    main()
