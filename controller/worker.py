from supervisor import LocalMessageQueue, ProcessList, get_message_queue, Context, Host
from supervisor.logging import initialize_logging
from typing import Dict, NamedTuple, Any, Sequence, TypedDict, Union, Literal, Optional, List, Tuple, Callable
from typing_extensions import Unpack
from contextlib import contextmanager
import math
import os
import importlib
import logging
from .tree import flatten, tree_map
import torch

logger = logging.getLogger(__name__)

class Ref:
    def __init__(self, id: int):
        self.id = id

    def __repr__(self):
        return f'r{self.id}'

    def __reduce__(self):
        return Ref, (self.id,)

class CreateDeviceMesh(NamedTuple):
    result: int
    dims: Dict[str, int]
    ranks: List[int]

class CallFunction(NamedTuple):
    results: Tuple[int,...]
    function: Union[str, Callable]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class Exit(NamedTuple):
    pass

class CommandGroup(NamedTuple):
    commands: List[NamedTuple]

class DeleteRefs(NamedTuple):
    refs: List[int]

class Restarted(NamedTuple):
    result: int

def log(*args):
    logger.info(*args)


class Dim(NamedTuple):
    rank: int
    size: int
    process_group: Any

class DeviceMesh:
    def __init__(self, dims: Dict[str, int], ranks: List[int], index: int):
        self.dims = {}
        stride = 1
        initial_index = index
        objects = []
        for size in reversed(dims.values()):
            rank = index % size
            index //= size
            group_start = initial_index - rank*stride
            members = ranks[group_start:group_start+stride*size:stride]
            assert members[rank] == ranks[initial_index]
            process_group = torch.distributed.new_group(members, use_local_synchronization=True)
            objects.append(Dim(rank, size, process_group))
            stride *= size
        self.dims = dict(zip(dims.keys(), reversed(objects)))


class Worker:
    def __init__(self, q: LocalMessageQueue, rank: int, world: int, local_rank: int):
        # remote ref id to local value
        self.env: Dict[int, Any] = {}
        self.q = q
        self.rank = rank
        self.world = world
        self.local_rank = local_rank

    def handle_message(self, event: NamedTuple):
        cmd = event.__class__.__name__
        fn = getattr(self, cmd, None)
        if fn is not None:
            return fn(*event)
        raise RuntimeError(f"unhandled event: {event}")

    def CreateDeviceMesh(self, result: int, dims: Dict[str, int], ranks: List[int]):
        index = self.ranks.index(self.rank)
        self.define(result, DeviceMesh(dims, ranks, index))

    def lookup(self, a: Any):
        if isinstance(a, Ref):
            return self.env[a.id]
        return a

    def CallFunction(self, results: Tuple[int], function: Union[str, Callable], args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        args, kwargs = tree_map(self.lookup, (args, kwargs))
        if isinstance(function, str):
            first, *parts = function.split('.')
            if first == 'torch':
                function = globals()[first]
                for p in parts:
                    function = getattr(function, p)
                assert isinstance(function, Callable)
            else:
                modulename, funcname = function.rsplit('.', 1)
                module = importlib.import_module(modulename)
                function = getattr(module, funcname)

        result = function(*args, **kwargs)
        tensors, _ = flatten(result, lambda x: isinstance(x, torch.Tensor))
        assert len(results) == len(tensors)
        for r, t in zip(results, tensors):
            self.define(r, t)

    def Exit(self):
        raise StopIteration()

    def CommandGroup(self, commands: List[NamedTuple], deletes: List[int]):
        for cmd in commands:
            self.handle_message(cmd)

    def DeleteRefs(self, refs: List[int]):
        for id in refs:
            del self.env[id]

    def define(self, r: int, value: Any):
        self.env[r] = value

    def event_loop(self):
        while True:
            _, msg = self.q.recv()
            try:
                logger.info(f"event: {msg}")
                self.handle_message(msg)
            except StopIteration:
                return


def worker_main(_restartable):
    rank = int(os.environ['RANK'])
    initialize_logging(process_name=f'worker_{rank}')
    logger.info("starting, restartable=%s", _restartable)
    q = get_message_queue()
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    # CUDA_VISIBLE_DEVICES should be set on launch to LOCAL_RANK
    worker = Worker(q, rank, world, local_rank)
    worker.event_loop()
    while _restartable:
        q.send(Restarted(0))
        logger.info("restarting")
        worker.event_loop()
