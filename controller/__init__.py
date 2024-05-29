from torch import device
from torch.types import Device
from supervisor import ProcessList, Context, Host, FunctionCall, TTL
from typing import Dict, NamedTuple, Any, Sequence, TypedDict, Union, Literal, Optional, List, Tuple, Callable
from torch.utils._python_dispatch import TorchDispatchMode

from supervisor.logging import gethostname
from .tree import flatten, tree_map
from torch._subclasses.fake_tensor import FakeTensorMode
from .base_tensor import BaseTensor

from torch import dtype, layout, device, memory_format
from typing_extensions import Unpack
import torch
from contextlib import contextmanager
# from .base_tensor import BaseTensor
import math
import os
from . import worker
from .worker import RemoteException, Ref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import itertools
import socket
import logging
import traceback

logger = logging.getLogger(__name__)

check_correctness_per_operator = False
if check_correctness_per_operator:
    class RealMode:
        def from_tensor(self, t):
            return t

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    fake_mode = RealMode()
else:
    fake_mode = FakeTensorMode()

_CONTROLLER_STATUS_INTERVAL = 2

class _Controller:
    def __init__(self, ctx: Context, hosts: List[Host], gpu_per_host: int, _processes=None, _store=None):
        self.ctx = ctx
        self.hosts = hosts
        self.store = self._create_store() if _store is None else _store
        self.all_processes = self._create_pg(ctx, hosts, gpu_per_host, self.store) if _processes is None else _processes
        self.last_completed_idents = [-1 for _ in self.all_processes]
        self.min_last_completed_ident = -1
        self.next_ref = itertools.count()
        self.next_ident = itertools.count()
        self.exited = {}
        self.failures: Dict[int, Dict[int, RemoteException]] = defaultdict(dict)
        self.tracebacks: deque[List[traceback.FrameSummary]] = deque()
        self.pending_del: Dict[DeviceMesh, List[int]] = defaultdict(list)
        self._shutdown = False
        self.incomplete_futures = {}
        self.controller_status_ttl = TTL(_CONTROLLER_STATUS_INTERVAL)
        global _active_stream
        _active_stream = Stream("main")

    @staticmethod
    def _create_store():
        hostname = socket.gethostname()
        with socket.socket() as sock:
            sock.bind(('', 0))
            port = sock.getsockname()[1]
        return torch.distributed.TCPStore(hostname, port, is_master=True)

    @staticmethod
    def _create_pg(ctx: Context, hosts: List[Host], gpu_per_host: int, store,
                   _restartable=False):
        return ctx.create_process_group(hosts,
                                        FunctionCall('controller.worker.worker_main',
                                                     _restartable=_restartable),
                                        processes_per_host=gpu_per_host,
                                        env={'CUDA_VISIBLE_DEVICES': '$LOCAL_RANK',
                                             'STORE_HOSTNAME': store.host,
                                             'STORE_PORT': str(store.port),})

    def shutdown(self):
        self._shutdown = True
        self.all_processes.send(worker.Exit())
        while len(self.exited) < len(self.all_processes):
            self._read_messages(None)

    def ref(self) -> int:
        return next(self.next_ref)

    def ident(self) -> int:
        r = next(self.next_ident)
        self.tracebacks.append(traceback.extract_stack())
        return r

    def _flush_deletes(self, device_mesh: 'DeviceMesh') -> Optional[worker.DeleteRefs]:
        to_delete = None
        if device_mesh in self.pending_del:
            to_delete = worker.DeleteRefs(self.pending_del.pop(device_mesh))
        # we also have to make sure if we have deletes to other device meshes,
        # they get processed before we do an op that will try to allocate memory
        for k, v in self.pending_del.items():
            k._send(worker.DeleteRefs(v))
        self.pending_del.clear()
        return to_delete

    def _read_messages(self, timeout: Optional[float]):
        if self.controller_status_ttl() == 0:
            self.all_processes.send(worker.ControllerStatus(self.ident()))
            self.controller_status_ttl = TTL(_CONTROLLER_STATUS_INTERVAL)
        for msg in self.ctx.recvready(timeout):
            self.handle_message(msg)

    def future(self):
        ident = self.ident()
        f = self.incomplete_futures[ident] = Future(self)
        return f, ident

    def handle_message(self, msg):
        sender, value = msg
        getattr(self, value.__class__.__name__)(sender, *value)

    def ProcessExited(self, proc, result):
        self.exited[proc] = result

    def Restarted(self, proc, result):
        self.exited[proc] = result

    def ValueResult(self, proc, ident, value):
        self.incomplete_futures.pop(ident)._set_result(value)

    def ValueException(self, proc, ident, failing_ident):
        self.incomplete_futures[ident]._set_result(self.failures[failing_ident][proc.rank])

    def Status(self, proc, last_completed_ident, failed_idents):
        # record failure messages in status report for later reporting
        for ident, (exception, worker_frames) in failed_idents.items():
            frames = self.tracebacks[ident - (self.min_last_completed_ident + 1)]
            self.failures[ident][proc.rank] = worker.RemoteException(exception, frames + worker_frames)

        # advance what our last completed action was, and
        # trim the list of tracebacks if we no longer need them.
        prev = self.last_completed_idents[proc.rank]
        self.last_completed_idents[proc.rank] = last_completed_ident
        if prev == self.min_last_completed_ident:
            self.min_last_completed_ident = min(self.last_completed_idents)
            for _ in range(self.min_last_completed_ident - prev):
                self.tracebacks.popleft()

class Referenceable:
    def delete_ref(self, ref):
        raise NotImplementedError("no delete_ref method")

    def define_ref(self):
        raise NotImplementedError("undefined ref with no define_ref method")

    def __reduce_ex__(self, protocol):
        if self.ref is None:
            self.ref = self.define_ref()
        return Ref, (self.ref,)

    def __del__(self):
        if self.ref is not None:
            self.delete_ref(self.ref)

def remote_function(path: str):
    return lambda func: lambda *args, **kwargs: dtensor_dispatch(path, args, kwargs, _active_mesh, _active_stream, func)

def _ident(*args):
    return args

def remote_value(path: str):
    def wrapper(**mesh_coordinates: int):
        return lambda *args, **kwargs: _dispatch_remote_value(path, mesh_coordinates, args, kwargs)

def _dispatch_remote_value(path: str, mesh_coordinates: Dict[str, int], args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    if _active_mesh is None:
        raise ValueError('No device mesh is active')
    fut, ident = _active_mesh.ctrl.future()
    process = _active_mesh._process(mesh_coordinates)
    process.send(worker.RemoteValue(ident, path, args, kwargs))
    return fut

PyTree = Union[Dict[str, 'PyTree'], List['PyTree'], Tuple['PyTree',...], 'Tensor']

class DeviceMesh(Referenceable):
    def __init__(self, ctrl: _Controller, processes: ProcessList, dims):
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
        msg = worker.CreateDeviceMesh(self.ctrl.ref(), self.dims, [p.rank for p in self.processes])
        self.processes.send(msg)

        return msg.result

    def delete_ref(self, ref: int):
        if not self.ctrl._shutdown:
            self._send(worker.DeleteRefs([ref]))

    def _send(self, cmd: NamedTuple):
        to_delete = self.ctrl._flush_deletes(self)
        if to_delete:
            self.processes.send(worker.CommandGroup([to_delete, cmd]))
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

    def _process(self, mesh_coordinates: Dict[str, int]):
        stride = 1
        offset = 0
        extra = set(mesh_coordinates.keys()) - set(self.dims.keys())
        if extra:
            raise KeyError(f'{list(extra)}')
        for dim, size in reversed(self.dims.items()):
            idx = mesh_coordinates.get(dim, 0)
            if idx < 0:
                idx += size
            if idx < 0 or idx >= size:
                raise IndexError(f'{dim} of size {size} has index {idx} out of range')
            offset += stride*idx
            stride *= size
        return self.processes[offset]



@remote_function('controller.worker._reduce')
def _reduce(tensor, source_mesh, dim, reduction, scatter, destination_mesh, inplace):
    if scatter:
        N = source_mesh.dims[dim]
        if tensor.ndim == 0 or tensor.size(0) != N:
            raise TypeError(f'When scattering results the outer most dimension of tensor ({list(tensor.size())} must match the size ({N}) of the dimension "{dim}" being reduced')
        if reduction == 'stack':
            # scatter removes a dimension of mesh size
            # but gather adds the dimension back
            return tensor
        return tensor.sum(dim=0)
    else:
        if reduction == 'stack':
            return torch.empty([source_mesh.dims[dim], *tensor.shape],
                               dtype=tensor.dtype, device=tensor.device, layout=tensor.layout)
        return tensor.add(tensor)

def world_mesh(ctx: Context, hosts: List[Host], gpu_per_host: int, _processes=None):
    ctrl = _Controller(ctx, hosts, gpu_per_host, _processes=_processes)
    return DeviceMesh(ctrl, ctrl.all_processes, {'host': len(ctrl.all_processes) // gpu_per_host, 'gpu': gpu_per_host})



class Stream:
    name: str

    def __init__(self, name: str):
        self.name = name
        self.ctrl = None

    def __repr__(self):
        return f'<Stream({repr(self.name)}) at {hex(id(self))}>'

    def _use_controller(self, ctrl: '_Controller'):
        if self.ctrl is None:
            self.ctrl = ctrl
        elif self.ctrl is not ctrl:
            raise TypeError('DeviceMesh and stream controller are different.')

    def wait_for(self, other: 'Stream'):
        """
        Blocks execution of this stream until the other stream completes the work that has been scheduled.
        Any tensors which have been borrowed from this stream to other, and then freed, will be returned
        to this stream, reclaiming the memory if there are no other references to them.
        """
        if other.ctrl is None:
            # nothing has happened yet on other stream, so we
            # can return
            return
        self._use_controller(other.ctrl)

        raise NotImplementedError()

    @contextmanager
    def coalesce(self):
        """
        Delay issuing operators to this stream, grouping them into one big operation that will run once this context manager exits.
        For data movement, this allows us to group the operators together. However coalescing too many ops together will expose
        more scheduling overhead that is normally pipelined with work. So avoid globally coalescing huge parts of a network.
        """
        raise NotImplementedError()

    def borrow(self, t: 'Tensor', mutable: bool = False) -> 'Tensor':
        """
        Borrows tensor 't' for use on this stream.
        The memory of t will stay alive until the borrowed tensor is freed AND then self has waited
        on t.stream, either because of another borrow or an call to `wait_for`.

        If `mutable` then self can write to the storage of `t`, but t.stream cannot read or write `t` until,
        the borrow is returned (becomes free and a wait_for has been issued).

        If not `mutable` both `self` and `t.stream` can read from t's storage but neither can write to it.
        """
        self._use_controller(t.mesh.ctrl)
        raise NotImplementedError()


class TensorOptions(TypedDict, total=False):
    dtype: 'torch.dtype'
    layer: 'torch.layout'
    device: Union[str, 'torch.device']
    requires_grad: bool
    pin_memory: bool
    memory_format: 'torch.memory_format'


class Pipe:
    def push(self, tensor: 'Tensor'):
        raise NotImplementedError()

    def pop(self, sizes: Sequence[int], **kwargs: Unpack[TensorOptions]) -> 'Tensor':
        raise NotImplementedError()

_valid_reduce = ['stack', 'sum', 'avg', 'product', 'min', 'max', 'band', 'bor', 'bxor']

class Tensor(Referenceable, BaseTensor):
    stream: Stream
    mesh: DeviceMesh


    def __new__(
        cls,
        fake: torch.Tensor, mesh: DeviceMesh, stream: Stream
    ):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            fake.size(),
            strides=fake.stride(),
            storage_offset=fake.storage_offset(),
            device=fake.device,  # This is the device of of either input tensor or first tensor of a list
            dtype=fake.dtype,
            layout=fake.layout,
            requires_grad=fake.requires_grad,
        )
        r._fake = fake
        r.ref = mesh.ctrl.ref()
        r.mesh = mesh
        r.stream = stream
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if isinstance(func, torch._ops.OpOverload):
            function = "torch.ops."  + str(func)
        else:
            function = func

        return dtensor_dispatch(function, args, kwargs, _active_mesh, _active_stream, func)

    def __init__(self, fake: torch.Tensor, mesh: DeviceMesh, stream: Stream):
        pass

    def __repr__(self):
       return f"DTensor(mesh={self.mesh}, stream={self.stream}, fake={self._fake})"


    def to_mesh(self, mesh: DeviceMesh):
        """
        Move data between one device mesh and another. Sizes of named dimensions must match.
        If mesh has dimensions that self.mesh does not, it will broadcast to those dimensions.


        broadcast:
            t.slice_mesh(batch=0).to_mesh(t.mesh)

        """
        raise NotImplementedError()

    def reduce_(self, dim: str, reduction: Literal["gather", "sum", "max"], scatter=False, mesh=None):
        # TODO: checks that this can actually happen in place e.g. if scatter is True, operation must be gather.
        inplace_valid = (reduction == 'gather' and scatter) or not scatter
        if not inplace_valid:
            raise ValueError(f'reduction {reduction} is not valid for in-place operation because the output size will not match the input size')
        return self.reduce(dim, reduction, scatter, mesh, _inplace=True)

    def reduce(self, dim: str, reduction: Literal["gather", "sum", "max"], scatter=False, mesh=None, _inplace=False):
        """
        Perform a reduction operation along dim, and move the data to mesh. If mesh=None, then mesh=self.mesh
        'gather' will concat the values along dim, and produce a local result tensor with an addition outer dimension of len(dim).
        If scatter=True, the local result tensor will be evenly split across dim.

        allreduce:
            t.reduce('batch', 'sum')

            First reduces dim 'batch' creating a local tensor with the 'batch' dimension, then because output_mesh=input_mesh, and it still has dim 'batch',
            we broadcast the result reduced tensor to all members of batch.

        reducescatter:
            t.reduce('batch', 'sum', scatter=True)

            Same as above except that scatter=True introduces a new 'batch' dimension that is the result of splitting the local tensor across 'batch'

        allgather:
            t.reduce('batch', 'gather')

            First reduces dim 'batch' creating a bigger local tensor, then because output_mesh=input_mesh, and it still has dim 'batch',
            broadcasts the result concatenated tensor to all members of batch.

        alltoall:
            t.reduce('batch', 'gather', scatter=True)


            First reduces dim 'batch' creating a bigger local tensor, then introduces a new 'batch' dimension that is the result of splitting this
            (bigger) tensor across 'batch'. The result is the same dimension as the original tensor, but with each rank sending to all other ranks.


        gather (to dim 0):
            t.reduce('batch', 'gather', mesh=t.mesh.index(batch=0))

            First gathers dim 'batch' and then places it on the first rank. t.mesh.batch[0] doesn't have a 'batch' dimension, but this is
            ok because we eliminated the 'batch' dim via reduction.

        reduce:
            t.reduce('batch', 'sum', mesh=t.mesh.index(batch=0))

            First reduces dim 'batch' and then places it on the first rank. t.mesh.batch[0] doesn't have a 'batch' dimension, but this is
            ok because we eliminated the 'batch' dim via reduction.
        """
        if mesh is not None:
            raise NotImplementedError()
        if dim not in self.mesh.dims:
            raise KeyError(f'dim {dim} not found in {self.mesh}')
        if reduction not in _valid_reduce:
            raise ValueError(f'reduction {reduction} not supported, reductions are {_valid_reduce}')
        if mesh is None:
            mesh = self.mesh
        return _reduce(self, self.mesh, dim, reduction, scatter, mesh, _inplace)

    def slice_mesh(self, **kwargs: Dict[str, Union[int, slice]]) -> 'MeshSliceTensor':
        pass

    def delete_ref(self, ref: int):
        mesh = self.mesh
        mesh.ctrl.pending_del[mesh].append(ref)


class MeshSliceTensor:
    def __init__(self, tensor: 'Tensor', slicing):
        self.tensor = tensor
        self.slicing = slicing

    def to_mesh(self, mesh: 'DeviceMesh') -> 'Tensor':
        pass


_explain = """\
LOCAL_TENSOR
A local (non-distributed) tensor is being passed while a device_mesh is active.
If you want to do local tensor compute use `with active_mesh(None):`

WRONG_MESH
A tensor being passed is on a device mesh that is not the current device_mesh.
Use `with active_mesh(m):` to switch the active mesh, or move the tensor to the correct device mesh with `to_mesh`/`on_mesh`.

WRONG_STREAM
A tensor being passed is on a stream that is not the current active stream. Use with `active_stream(s)` to switch streams, or
move the tensor to the correct stream with `.borrow`.
"""

explain = dict(entry.split('\n', 1) for entry in _explain.split('\n\n'))


def dtensor_check(func, args, kwargs, device_mesh, stream) -> Tuple[List['Tensor'], Any]:
    def stringify(t):
        if isinstance(t, Tensor):
            if t.mesh is not device_mesh:
                return 'WRONG_MESH'
            elif t.stream is not stream:
                return 'WRONG_STREAM'
            else:
                return '.'
        elif isinstance(t, torch.Tensor):
            return 'LOCAL_TENSOR'
        else:
            return t

    def tensor_check(x):
        if isinstance(x, torch.Tensor):
            if isinstance(x, Tensor) and x.mesh is device_mesh and x.stream is stream:
                return True
            fargs, fkwargs = tree_map(stringify, (args, kwargs))
            actuals = ', '.join(str(a) for a in fargs)
            if fkwargs:
                actuals = f'{actuals}, ' + ', '.join(f'{k}={v}' for k, v in kwargs.items())

            call = f"{func}({actuals})"
            dmesh = f'active_mesh = {device_mesh}\nactive_stream = {stream}'
            help = '\n'.join(f'{k}: {v}' for k, v in explain.items() if k in call)
            raise TypeError(f'Mismatched arguments to distributed tensor operation:\n\n  {call}\n\n{dmesh}\n{help}')

    dtensors, unflatten = flatten((args, kwargs), tensor_check)
    return dtensors, unflatten

def dtensor_dispatch(func, args, kwargs, device_mesh: Optional[DeviceMesh], stream: Stream, result_type):
    dtensors, unflatten = dtensor_check(func, args, kwargs, device_mesh, stream)
    assert device_mesh is not None
    ctrl = device_mesh.ctrl
    stream._use_controller(ctrl)

    if callable(result_type):
        fake_input_tensors = [d._fake for d in dtensors]
        fake_args, fake_kwargs = unflatten(fake_input_tensors)
        with fake_mode, active_mesh(None):
            result = result_type(*fake_args, **fake_kwargs)
        fake_map = {id(f): i for i, f in enumerate(fake_input_tensors)}
    else:
        result = result_type
        fake_map = {}

    fake_result_dtensors, unflatten_result = flatten(result, lambda x: isinstance(x, torch.Tensor))

    # sometimes operators return references to inputs, in which case the result should be the same DTensor object
    # otherwise we create a new DTensor with a new RemoteRef for the result
    result_dtensors = tuple(dtensors[fake_map[id(fake)]] if id(fake) in fake_map else Tensor(fake, device_mesh, stream) for fake in fake_result_dtensors)
    device_mesh._send(worker.CallFunction(device_mesh.ctrl.ident(), tuple(r.ref for r in result_dtensors), func, args, kwargs))
    results = unflatten_result(result_dtensors)
    return results


_active_stream: Stream = Stream('main')
_active_mesh: Optional[DeviceMesh] = None
_dispatch_enabled = False


class _ActiveMesh(TorchDispatchMode):
    ignore = ['profiler._record_function_exit._RecordFunction']

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if _active_mesh is None:
            return func(*args, **kwargs)
        return Tensor.__torch_dispatch__(func, types, args, kwargs)


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
def active_stream(stream: Stream):
    global _active_stream
    _active_stream, old = stream, _active_stream
    try:
        yield
    finally:
        _active_stream = old

@contextmanager
def active_mesh(mesh: Optional[DeviceMesh]):
    global _active_mesh
    _active_mesh, old = mesh, _active_mesh
    try:
        with _dispatch():
            yield
    finally:
        _active_mesh = old

class Future:
    def __init__(self, ctrl: '_Controller'):
        self._ctrl = ctrl
        self._status = 'incomplete'
        self._callbacks = None
        self._result = None

    def _set_result(self, r):
        assert self._status == 'incomplete'
        self._result = r
        self._status = 'exception' if isinstance(r, RemoteException) else 'complete'
        if self._callbacks:
            for cb in self._callbacks:
                try:
                    cb(self)
                except Exception:
                    logger.exception("exception in controller's Future callback")
        self._callbacks = None
        self._ctrl = None

    def _wait(self, timeout: Optional[float]):
        if self._status != 'incomplete':
            return True
        if timeout is None:
            while self._status == 'incomplete':
                self._ctrl._read_messages(timeout=None)
        else:
            ttl = TTL(timeout)
            while self._status == 'incomplete':
                remaining = ttl()
                self._ctrl._read_messages(timeout=remaining)
                if remaining == 0:
                    return self._status != 'incomplete'
        return True

    def result(self, timeout: Optional[float]=None):
        if not self._wait(timeout):
            raise TimeoutError()
        assert self._result is not None
        if self._status == 'exception':
            raise self._result
        return self._result

    def done(self) -> bool:
        return self._wait(0)

    def exception(self, timeout: Optional[float]=None):
        if not self._wait(timeout):
            raise TimeoutError()
        return self._result if self._status == 'exception' else None

    def add_callback(self, callback):
        if not self._callbacks:
            self._callbacks = [callback]
        else:
            self._callbacks.append(callback)
