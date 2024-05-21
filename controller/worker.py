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
    result: Ref
    dims: Dict[str, int]
    ranks: List[int]

class CallFunction(NamedTuple):
    function: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class CallBuiltin(NamedTuple):
    results: Tuple[Ref]
    function: Union[str, Callable]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class Exit(NamedTuple):
    pass

class CommandGroup(NamedTuple):
    commands: List[NamedTuple]

class DeleteRefs(NamedTuple):
    refs: List[int]

def log(*args):
    logger.info(*args)

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

    def CreateDeviceMesh(self, result: Ref, dims: Dict[str, int], ranks: List[int]):
        if self.rank in ranks:
            self.define(result, ('devicemesh', dims, ranks))

    def CallFunction(self, function: str, args: List[Any], kwargs: Dict[str, Any]):
        modulename, funcname = function.rsplit('.', 1)
        module = importlib.import_module(modulename)
        func = getattr(module, funcname)
        args, kwargs = tree_map(self.lookup, (args, kwargs))
        func(*args, **kwargs)

    def lookup(self, a: Any):
        if isinstance(a, Ref):
            return self.env[a.id]
        return a

    def CallBuiltin(self, results: Tuple[Ref], function: Union[str, Callable], args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        args, kwargs = tree_map(self.lookup, (args, kwargs))
        if isinstance(function, str):
            first, *parts = function.split('.')
            function = globals()[first]
            for p in parts:
                function = getattr(function, p)
            assert isinstance(function, Callable)
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

    def define(self, r: Ref, value: Any):
        self.env[r.id] = value

    def event_loop(self):
        while True:
            _, msg = self.q.recv()
            try:
                logger.info(f"event: {msg}")
                self.handle_message(msg)
            except StopIteration:
                return


def worker_main():
    rank = int(os.environ['RANK'])
    initialize_logging(process_name=f'worker_{rank}')
    logger.info(f"starting")
    q = get_message_queue()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    # CUDA_VISIBLE_DEVICES should be set on launch to LOCAL_RANK
    worker = Worker(q, rank, world, local_rank)
    worker.event_loop()
