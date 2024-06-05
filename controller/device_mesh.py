from supervisor import ProcessList
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple
from torch.utils._python_dispatch import TorchDispatchMode

import torch
from contextlib import contextmanager
import math
from . import messages
from .reference import Referenceable
from . import stream
from .tensor import dtensor_dispatch
import itertools

if TYPE_CHECKING:
    from .controller import Controller

def remote_function(path: str):
    return lambda func: lambda *args, **kwargs: dtensor_dispatch(path, args, kwargs, _active, stream._active, func)

class DeviceMesh(Referenceable):
    def __init__(self, ctrl: 'Controller', processes: ProcessList, dims):
        self.ctrl = ctrl
        assert len(processes) == math.prod(dims.values())
        self.dims: Dict[str, int] = dims
        self.processes = processes
        self.ref = None

    def __repr__(self):
        return f'<DeviceMesh({tuple(self.dims.keys())}, {tuple(self.dims.values())}) at {hex(id(self))}>'

    def define_ref(self):
        # Optimize: we do not have to send device meshes to all workers if we can
        # Create process groups as subsets
        msg = messages.CreateDeviceMesh(self.ctrl.ref(), self.dims, [p.rank for p in self.processes])
        self.processes.send(msg)

        return msg.result

    def delete_ref(self, ref: int):
        if not self.ctrl._shutdown:
            self._send(messages.DeleteRefs([ref]))

    def _send(self, cmd: NamedTuple):
        to_delete = self.ctrl._flush_deletes(self)
        if to_delete:
            self.processes.send(messages.CommandGroup([to_delete, cmd]))
        else:
            self.processes.send(cmd)

    def stack(self, **kwargs):
        raise NotImplementedError()

    def __call__(self, **kwargs) -> 'DeviceMesh':
        """
        m.index(batch=3) or m.index(batch=slice(3, None))
        """
        ranges = []
        stride = 1
        offset = 0
        dims = {}
        sizes = list(self.dims.values())
        for i, (k, v) in enumerate(self.dims.items()):
            stride = math.prod(sizes[i+1:])
            if k in kwargs:
                e = kwargs.pop(k)
                if isinstance(e, slice):
                    the_range = range(*e.indices(v))
                    dims[k] = len(the_range)
                    ranges.append(stride*x for x in the_range)
                else:
                    if e >= v or e < 0:
                        raise IndexError('index out of range')
                    offset += e*stride
            else:
                dims[k] = v
                ranges.append(range(0, v*stride, stride))
        if kwargs:
            raise TypeError(f'{self} does not have dimension(s) named {tuple(kwargs.keys())}')

        indices = [offset + sum(x) for x in itertools.product(*ranges)]
        processes = ProcessList(self.processes[x] for x in indices)
        return DeviceMesh(self.ctrl, processes, dims)

    def split(self, **kwargs: Dict[str, Tuple[str, ...]]):
        raise NotImplementedError()

    def rotate(self, **kwargs: Dict[str, int]):
        raise NotImplementedError()

    @remote_function('controller.worker._rank')
    def rank(self, dim: str):
        if dim not in self.dims:
            raise KeyError(f'{self} does not have dimension {repr(dim)}')
        return torch.full((), 0, dtype=torch.long)

    def _process(self, coordinates: Optional[Dict[str, int]]):
        if coordinates is None:
            return self.processes[0]
        stride = 1
        offset = 0
        extra = set(coordinates.keys()) - set(self.dims.keys())
        if extra:
            raise KeyError(f'{list(extra)}')
        for dim, size in reversed(self.dims.items()):
            idx = coordinates[dim]
            if idx < 0:
                idx += size
            if idx < 0 or idx >= size:
                raise IndexError(f'{dim} of size {size} has index {idx} out of range')
            offset += stride*idx
            stride *= size
        return self.processes[offset]

    def _use(self, tensor):
        borrows = tensor._borrows
        if tensor.stream is not borrows.origin_stream:
            borrow = borrows.active[tensor.stream]
            if not borrow.used:
                self._send(messages.BorrowFirstUse(borrow.id))
                borrows.active[tensor.stream] = borrow._replace(used=True)


_active: Optional[DeviceMesh] = None
_dispatch_enabled = False


class _ActiveMesh(TorchDispatchMode):
    ignore = ['profiler._record_function_exit._RecordFunction']

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if _active is None:
            return func(*args, **kwargs)
        return dtensor_dispatch(func, args, kwargs, _active, stream._active, func)

@contextmanager
def _dispatch():
    global _dispatch_enabled
    if _dispatch_enabled:
        yield
    else:
        _dispatch_enabled = True
        try:
            with _ActiveMesh():
                yield
        finally:
            _dispatch_enabled = False

@contextmanager
def active_mesh(mesh: Optional[DeviceMesh]):
    global _active
    _active, old = mesh, _active
    try:
        with _dispatch():
            yield
    finally:
        _active = old
