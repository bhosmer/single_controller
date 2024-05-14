from supervisor import ProcessList
from typing import Dict, NamedTuple, Any, Sequence, TypedDict, Union
from torch import dtype, layout, device, memory_format
from typing_extensions import Unpack
import torch

from enum import Enum

class Reduction(Enum):
    SUM = 1
    AVG = 2
    PRODUCT = 3
    MIN = 4
    MAX = 5
    BAND = 6
    BOR = 7
    BXOR = 8
    PREMUL_SUM = 8
    UNUSED = 9

class DeviceMesh:
    def __init__(self, processes: ProcessList):
        self.dims: Dict[str, int] = {}  # ordered dict, name -> size
        # will start as host, device
        self.procs = processes

    def stack(self, **kwargs):
        raise NotImplementedError()

    def __getattr__(self, name):
        if name in self.dims:
            return DeviceMeshDim(self, name)


class DeviceMeshDim(NamedTuple):
    _mesh: DeviceMesh
    _dim: str

    def __getitem__(self, index) -> Any:
        raise NotImplementedError()

    def split(self, **kwargs):
        raise NotImplementedError()


class Stream:
    name: str

    def wait_for(self, other: 'Stream'):
        """
        Blocks execution of this stream until the other stream completes the work that has been scheduled.
        Any tensors which have been borrowed from this stream to other, and then freed, will be returned
        to this stream, reclaiming the memory if there are no other references to them.
        """
        raise NotImplementedError()

class TensorOptions(TypedDict, total=False):
    dtype: dtype
    layer: layout
    device: Union[str, device]
    requires_grad: bool
    pin_memory: bool
    memory_format: memory_format


class Pipe:
    def push(self, tensor: 'Tensor'):
        raise NotImplementedError()

    def pop(self, sizes: Sequence[int], **kwargs: Unpack[TensorOptions]) -> 'Tensor':
        raise NotImplementedError()

class TensorMeshDim(NamedTuple):
    _tensor: Tensor
    _dim: str

    def all_reduce(self, op: Reduction) -> 'Tensor':
        """
        Perform op across all members of this dimension, the resulting tensor
        is the same across the dimension.
        """
        raise NotImplementedError()


class Tensor:
    stream: Stream
    mesh: DeviceMesh

    def borrow(self, to: Stream, mutable: bool = False) -> 'Tensor':
        """
        Returns a tensor that can on stream `to` that temporarily borrows this tensor for use on that stream.
        The memory of this tensor will stay alive until the borrowed tensor is freed AND then self.stream has waited
        on `to`, either because of another borrow or an call to `wait_for`.
        """
        raise NotImplementedError()

    def broadcast(self, dim: str) -> 'Tensor':
        """
        Transfer shard at `index` to all members dim.
        """
        raise NotImplementedError()
