from supervisor import LocalMessageQueue, ProcessList, get_message_queue, Context, Host
from typing import Dict, NamedTuple, Any, Sequence, TypedDict, Union, Literal, Optional, List, Tuple
from torch import dtype, layout, device, memory_format
from typing_extensions import Unpack
import torch
from contextlib import contextmanager
from .base_tensor import BaseTensor
import math
import os
import importlib

class Ref(NamedTuple):
    id: int

class CreateDeviceMesh(NamedTuple):
    result: Ref
    dims: Dict[str, int]
    ranks: List[int]

class CallFunction(NamedTuple):
    function: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class Exit(NamedTuple):
    pass

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
        print(f"Create DeviceMesh: {dims} {ranks}")
        self.define(result, 4)

    def CallFunction(self, function: str, args: List[Any], kwargs: Dict[str, Any]):
        modulename, funcname = function.rsplit('.', 1)
        module = importlib.import_module(modulename)
        func = getattr(module, funcname)
        r = func(*args, **kwargs)
        print(f"PRODUCED RESULT: {r}")

    def Exit(self):
        raise StopIteration()

    def define(self, r: Ref, value: Any):
        self.env[r.id] = value

    def event_loop(self):
        while True:
            _, msg = self.q.recv()
            try:
                print(msg)
                self.handle_message(msg)
            except StopIteration:
                return


def worker_main():
    q = get_message_queue()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    # CUDA_VISIBLE_DEVICES should be set on launch to LOCAL_RANK
    worker = Worker(q, rank, world, local_rank)
    worker.event_loop()
