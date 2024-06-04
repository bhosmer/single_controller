from supervisor import ProcessList, Context, Host, FunctionCall, TTL
from typing import Dict, NamedTuple, Any, Sequence, TypedDict, Union, Literal, Optional, List, Tuple
from torch.utils._python_dispatch import TorchDispatchMode
from concurrent.futures import ThreadPoolExecutor
from .tree import flatten, tree_map
from torch._subclasses.fake_tensor import FakeTensorMode
from .base_tensor import BaseTensor

from typing_extensions import Unpack
import torch
from contextlib import contextmanager
# from .base_tensor import BaseTensor
import math
from . import messages
from .reference import Referenceable
from collections import defaultdict, deque
import itertools
import socket
import logging
import traceback
import warnings
from weakref import WeakKeyDictionary, WeakSet

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

class _Borrow(NamedTuple):
    id: int
    used: bool
    frames: List[traceback.FrameSummary]

class _Borrows:
    def __init__(self, origin_stream: 'Stream'):
        self.origin_stream = origin_stream
        # who can write to this storage?
        self.writing_stream: Optional[Stream] = origin_stream
        # what Tensor aliases exist for this storage
        self.aliases = WeakSet()
        # what active borrows of this exist?
        self.active: Dict['Stream', _Borrow] = {}

class RemoteException(Exception):
    def __init__(self, ident: int, exception: Exception, controller_frames: List[traceback.FrameSummary], worker_frames: List[traceback.FrameSummary]):
        self.ident = ident
        self.exception = exception
        self.controller_frames = controller_frames
        self.worker_frames = worker_frames

    def __str__(self):
        try:
            exe = str(self.exception)
            worker_tb = ''.join(traceback.format_list(self.worker_frames))
            controller_tb = ''.join(traceback.format_list(self.controller_frames)) if self.controller_frames is not None else '<unset>'
            return f'A remote function has failed asynchronously.\n' \
                f'Traceback of where the remote function was issued on controller (most recent call last):\n{controller_tb}' \
                f'Traceback of where the remote function failed on worker (most recent call last):\n{worker_tb}{type(self.exception).__name__}: {exe}'
        except Exception as e:
            print(e)
            return "oops"

class _Invocation:
    def __init__(self, traceback: List[traceback.FrameSummary]):
        self.traceback = traceback
        self.users = set['_Invocation']()
        self.failure: Optional[RemoteException] = None
        self.fut: Optional[Future] = None
        self.fut_value: Any = None
 
    def fail(self, remote_exception: RemoteException):
        if self.failure is None or self.failure.ident > remote_exception.ident:
            self.failure = remote_exception
            return True
        return False
        

class _History:
    def __init__(self, N):
        self.next_ident = itertools.count()
        self.first_uncompleted_ident = [0 for _ in range(N)]
        self.min_first_uncompleted_ident = 0
        self.invocations = deque[_Invocation]()
        self.last_assigned_ident = -1
    
    def invocation(self, defs: Sequence['Tensor'], uses: Sequence['Tensor']):
        r = _Invocation(traceback.extract_stack()[:-2])
        for t in uses:
            u = t._invocation
            u.users.add(r)
            if u.failure is not None:
                r.fail(u.failure)
        for t in defs:
            t._invocation = r
        return r

    def ident(self, defs: Sequence['Tensor'], uses: Sequence['Tensor'], future: Optional['Future'] = None) -> int:
        r = self.last_assigned_ident = next(self.next_ident)
        invocation = self.invocation(defs, uses)
        invocation.fut = future
        self.invocations.append(invocation)
        return r

    def propagate_failure(self, ident, exception, worker_frames):
        invocation = self.invocations[ident - self.min_first_uncompleted_ident]
        remote_exception = RemoteException(ident, exception, invocation.traceback, worker_frames)
        worklist = deque((invocation,))
        while worklist:
            invocation = worklist.popleft()
            if invocation.fail(remote_exception):
                worklist.extend(invocation.users)
    
    def rank_completed(self, rank, first_uncompleted_ident):
        # advance what our last completed action was, and
        # trim the list of tracebacks if we no longer need them.
        prev = self.first_uncompleted_ident[rank]
        self.first_uncompleted_ident[rank] = first_uncompleted_ident
        if prev == self.min_first_uncompleted_ident:
            self.min_first_uncompleted_ident = min(self.first_uncompleted_ident)            
            for _ in range(self.min_first_uncompleted_ident - prev):
                invocation = self.invocations.popleft()
                if invocation.fut is not None:
                    invocation.fut._set_result(invocation.fut_value if invocation.failure is None else invocation.failure)

    def future_completed(self, ident, value):
        invocation = self.invocations[ident - self.min_first_uncompleted_ident]
        invocation.fut_value = value

class _Controller:
    def __init__(self, ctx: Context, hosts: List[Host], gpu_per_host: int, _processes=None, _store=None):
        self.ctx = ctx
        self.hosts = hosts
        self.store = self._create_store() if _store is None else _store
        self.all_processes = self._create_pg(ctx, hosts, gpu_per_host, self.store) if _processes is None else _processes
        self.next_ref = itertools.count()
        self.exited = {}
        self.failures: Dict[int, Dict[int, RemoteException]] = defaultdict(dict)
        self.pending_del: Dict[DeviceMesh, List[int]] = defaultdict(list)
        self._shutdown = False
        self.incomplete_futures = {}
        self.controller_status_ttl = TTL(_CONTROLLER_STATUS_INTERVAL)
        self.allocation_borrows: WeakKeyDictionary[torch.UntypedStorage, _Borrows] = WeakKeyDictionary()
        self.fake_mode_worker = ThreadPoolExecutor(max_workers=1)
        self.history = _History(len(self.all_processes))
        global _active_stream
        _active_stream = Stream("main", _default=True)

    def _run_fake(self, func, args, kwargs):
        def work():
            with fake_mode:
                return func(*args, **kwargs)
        return self.fake_mode_worker.submit(work).result()

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
                                        env={'CUDA_VISIBLE_DEVICES':  '$LOCAL_RANK',
                                             # supervisor_pipe is a unique ID per Host object,
                                             # so it lets us put multiple processes on the same GPU.
                                             'NCCL_HOSTID': '$SUPERVISOR_PIPE',
                                             'STORE_HOSTNAME': store.host,
                                             'STORE_PORT': str(store.port), })

    def shutdown(self):
        self._shutdown = True
        self.all_processes.send(messages.Exit())
        while len(self.exited) < len(self.all_processes):
            self.handle_message(self.ctx.recv())

    def ref(self) -> int:
        return next(self.next_ref)

    def _flush_deletes(self, device_mesh: 'DeviceMesh') -> Optional[messages.DeleteRefs]:
        to_delete = None
        if device_mesh in self.pending_del:
            to_delete = messages.DeleteRefs(self.pending_del.pop(device_mesh))
        # we also have to make sure if we have deletes to other device meshes,
        # they get processed before we do an op that will try to allocate memory
        for k, v in self.pending_del.items():
            k._send(messages.DeleteRefs(v))
        self.pending_del.clear()
        return to_delete

    def _request_status(self):
            self.all_processes.send(messages.RequestStatus(self.history.last_assigned_ident))

    def _read_messages(self, timeout: Optional[float]):
        # XXX - how can we avoid always requesting status when waiting on futures?
        #       we need to figure out what submesh we need to hear from before a future
        #       is considered 'good'. This means not just waiting for the future value
        #       but also for signal that any failures that could invalidate the future have
        #       not happened. We could do better if tensors/collectives had an invalid bit
        #       that we propagate. In real uses fetches might lag behind anyway so we would not
        #       have to send out so many requests for current status.
        for msg in self.ctx.recvready(timeout):
            self.handle_message(msg)

    def handle_message(self, msg):
        sender, value = msg
        getattr(self, value.__class__.__name__)(sender, *value)

    def ProcessExited(self, proc, result):
        self.exited[proc] = result

    def ProcessStarted(self, proc, pid):
        pass

    def Restarted(self, proc, result):
        self.exited[proc] = result

    def FetchResult(self, proc, ident, value):
        self.history.future_completed(ident, value)

    def RemoteFunctionFailed(self, proc, failing_ident, exception: Exception, worker_frames: List[traceback.FrameSummary]):
        self.history.propagate_failure(failing_ident, exception, worker_frames)
        self.history.rank_completed(proc.rank, failing_ident)

    def Status(self, proc, first_uncompleted_ident):
        self.history.rank_completed(proc.rank, first_uncompleted_ident)
 
def remote_function(path: str):
    return lambda func: lambda *args, **kwargs: dtensor_dispatch(path, args, kwargs, _active_mesh, _active_stream, func)

def fetch_shard(obj, coordinates: Optional[Dict[str, int]] = None, preprocess: Optional[str] = None):
    """
    Retrieve the shard at `coordinates` of the current device mesh of each tensor in obj.
        obj - a pytree containing the tensors the fetch
        coordinates - a dictionary from mesh dimension name to coordinate of the shard
                      If None, this will fetch from coordinate 0 for all dimensions (useful after all_reduce/all_gather)
        preprocess - an optional specifier for a remote function that is applyied to obj before returning the value.
                     This can be used to turn the tensor into some non-tensor values before copying it back
    """
    tensors, _ = dtensor_check('fetch_shard', (obj,), {}, _active_mesh, _active_stream)
    fut = Future(_active_mesh.ctrl)
    ident = _active_mesh.ctrl.history.ident((), tensors, fut)
    process = _active_mesh._process(coordinates)
    process.send(messages.FetchValue(ident, preprocess, obj, _active_stream))
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


def _fake_reduce(tensor, source_mesh, dim, reduction, scatter):
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

class Stream(Referenceable):
    name: str

    def __init__(self, name: str, _default=False):
        self.name = name
        self.default = _default
        self.ctrl = None
        self.ref = None

    def __repr__(self):
        return f'<Stream({repr(self.name)}) at {hex(id(self))}>'

    def __str__(self):
        return f'stream {repr(self.name)}'

    def _use_controller(self, ctrl: '_Controller'):
        if self.ctrl is None:
            self.ctrl = ctrl
        elif self.ctrl is not ctrl:
            raise TypeError('DeviceMesh and stream controller are different.')

    def delete_ref(self, ref):
        # streams are not frequently created/destroyed so
        # no need to buffer the delets
        assert self.ctrl is not None
        if not self.ctrl._shutdown:
            self.ctrl.all_processes.send(messages.DeleteRefs([ref]))

    def define_ref(self):
        assert self.ctrl is not None
        r = self.ctrl.ref()
        self.ctrl.all_processes.send(messages.CreateStream(r, self.default))
        return r

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
        The memory of t will stay alive until t.drop() is called, which will free t and
        and any of its alises on stream `self` and will cause t.stream to wait on self at that point so
        that the memory of t can be reused.

        If `mutable` then self can write to the storage of `t`, but t.stream cannot read or write `t` until,
        the borrow is returned (becomes free and a wait_for has been issued).

        If not `mutable` both `self` and `t.stream` can read from t's storage but neither can write to it.
        """
        self._use_controller(t.mesh.ctrl)
        assert self.ctrl is not None
        borrows = t._borrows
        if mutable and borrows.active:
            raise RuntimeError(f"Cannot borrow this tensor mutably because it (or a view) is already being borrowed non-mutably.")

        already_borrowed = self in borrows.active
        r = Tensor(t._fake, t.mesh, self, True)
        self.ctrl.history.invocation((r,), (t,))
        assert r.ref is not None
        t.mesh._send(messages.BorrowCreate(r.ref, t, t.stream, self, already_borrowed))
        if not already_borrowed:
            borrows.active[self] = _Borrow(r.ref, False, traceback.extract_stack())
            borrows.writing_stream = self if mutable else None
        return r


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
        fake: torch.Tensor, mesh: DeviceMesh, stream: Stream, borrowed: bool
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
        ctrl = mesh.ctrl
        r.ref = ctrl.ref()
        r.mesh = mesh
        r.stream = stream
        r._borrowed = borrowed
        r._borrows.aliases.add(r)
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if isinstance(func, torch._ops.OpOverload):
            function = "torch.ops."  + str(func)
        else:
            function = func

        return dtensor_dispatch(function, args, kwargs, _active_mesh, _active_stream, func)

    def __init__(self, fake: torch.Tensor, mesh: DeviceMesh, stream: Stream, borrowed: bool):
        pass

    def __repr__(self):
       return f"DTensor(mesh={self.mesh}, stream={self.stream}, fake={self._fake})"

    def drop(self):
        if self.ref is None:
            return
        borrows = self._borrows
        for alias in borrows.aliases:
            if alias.stream is not self.stream:
                continue
            alias._drop_ref()
        if borrows.origin_stream is not self.stream:
            borrow = borrows.active.pop(self.stream)
            self.mesh._send(messages.BorrowDrop(borrow.id))
            if not borrows.active:
                borrows.writing_stream = borrows.origin_stream

        # we should be in the tensors list as well
        assert self.ref is None

    @property
    def _borrows(self) -> _Borrows:
        storage = self._fake.untyped_storage()
        ctrl = self.mesh.ctrl
        if storage not in ctrl.allocation_borrows:
            ctrl.allocation_borrows[storage] = _Borrows(self.stream)
        return ctrl.allocation_borrows[storage]

    @property
    def dropped(self):
        return self.ref is None

    def _drop_ref(self):
        assert self.ref is not None
        # silence borrowing warning
        self._borrowed = False
        self.delete_ref(self.ref)
        self.ref = None


    def to_mesh(self, mesh: DeviceMesh):
        """
        Move data between one device mesh and another. Sizes of named dimensions must match.
        If mesh has dimensions that self.mesh does not, it will broadcast to those dimensions.


        broadcast:
            t.slice_mesh(batch=0).to_mesh(t.mesh)

        """
        return MeshSliceTensor(self, self.mesh).to_mesh(mesh)

    def reduce_(self, dim: str, reduction: Literal["gather", "sum", "max"] = "sum", scatter=False, mesh=None):
        # TODO: checks that this can actually happen in place e.g. if scatter is True, operation must be gather.
        inplace_valid = (reduction == 'gather' and scatter) or not scatter
        if not inplace_valid:
            raise ValueError(f'reduction {reduction} is not valid for in-place operation because the output size will not match the input size')
        return self.reduce(dim, reduction, scatter, mesh, _inplace=True)

    def reduce(self, dim: str, reduction: Literal["gather", "sum", "max"] = "sum", scatter=False, mesh=None, _inplace=False):
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

        if _inplace:
            fake_output = self._fake
        else:
            fake_output = self.mesh.ctrl._run_fake(_fake_reduce, (self._fake, self.mesh, dim, reduction, scatter), {})
        r = Tensor(fake_output, self.mesh, self.stream, borrowed=False)
        self.mesh._send(messages.Reduce(r.ref, self, self._factory(), self.mesh, self.stream, dim, reduction, scatter, _inplace))
        self.mesh.ctrl.history.invocation((r,), (self,))
        return r

    def slice_mesh(self, **kwargs: Dict[str, Union[int, slice]]) -> 'MeshSliceTensor':
        # technically a slice of a device mesh and a device mesh are not same thing
        # because a device mesh also has caches for doing collectives.
        # but this is an easy way to create a MeshSliceTensor until we optimize
        # how we represent mesh slices.
        slicing = self.mesh(**kwargs)
        return MeshSliceTensor(self, slicing)

    def delete_ref(self, ref: int):
        if self._borrowed:
            current = ''.join(traceback.format_stack())
            borrowtb = ''.join(traceback.format_list(self._borrows.active[self.stream].frames))
            warnings.warn('t.drop() must be called before a borrowed tensor is freed to specify when the borrowed tensor should return to its origin stream, but this tensor is being deleted before drop.'
                          't.drop() is being called automatically here to ensure correctness, but this will force a synchronization back to the original stream at this point which might not be intended.'
                          f'\nTraceback of __del__(most recent call last):\n{current}\nTraceback of original borrow (most recent call last):{borrowtb}')
            self.drop()
            return
        mesh = self.mesh
        mesh.ctrl.pending_del[mesh].append(ref)
    
    def _factory(self):
        return messages.TensorFactory.from_tensor(self._fake)


class MeshSliceTensor:
    def __init__(self, tensor: 'Tensor', slicing: 'DeviceMesh'):
        self.tensor = tensor
        self.slicing = slicing

    def to_mesh(self, mesh: 'DeviceMesh') -> 'Tensor':
        if self.slicing.dims != mesh.dims:
            raise ValueError(f'input of dimensions {self.slicing.dims} does not match destination mesh of dimensions {mesh.dims}')

        # XXX - this is a very ineffiecient algorithm for figuring out which groups need messages. With O(Workers)
        # individual sends. The message size is also O(Workers).
        # An O(Workers) algorithm can easily send O(1) individual messages
        # but this is suppose to be a stub for an optimized algorithm where:
        # 1. We can represent submeshes as NDSlice(offet, sizes, strides) on rank.
        # 2. A message can be efficiently broadcast to List[NDSlice] ranks by a smart tree based algorithm that can
        #    figure out which subtrees need the message.
        # 3. The message itself will use List[NDSlice] objects to express the send/recv set and so it is very small

        # so basically both the way the messsage is broadcast and its size will be compressed but the
        # send pattern and the meaning of the message will be the same as this ineffiecient form

        combined_processes = self.slicing.processes
        if self.slicing.processes is not mesh.processes:
            combined_processes = ProcessList(sorted(set(self.slicing.processes).union(mesh.processes), key=lambda p: p.rank))

        from_ranks = [p.rank for p in self.slicing.processes]
        to_ranks = [p.rank for p in mesh.processes]
        r = Tensor(self.tensor._fake, mesh, _active_stream, False)
        combined_processes.send(messages.SendTensor(r.ref, from_ranks, to_ranks, self.tensor, self.tensor._factory(), self.tensor.stream))
        self.tensor.mesh.ctrl.history.invocation((r,), (self.tensor,))
        return r


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

DROPPED
This tensor, or a view of it, was explicitly deleted with the t.drop() function and is no longer usable.
"""

explain = dict(entry.split('\n', 1) for entry in _explain.split('\n\n'))


def dtensor_check(func, args, kwargs, device_mesh, stream) -> Tuple[List['Tensor'], Any]:
    def stringify(t):
        if isinstance(t, Tensor):
            if t.mesh is not device_mesh:
                return 'WRONG_MESH'
            elif t.stream is not stream:
                return 'WRONG_STREAM'
            elif t.dropped:
                return 'DROPPED'
            else:
                return '.'
        elif isinstance(t, torch.Tensor):
            return 'LOCAL_TENSOR'
        else:
            return t

    def tensor_check(x):
        if isinstance(x, torch.Tensor):
            if isinstance(x, Tensor) and x.mesh is device_mesh and x.stream is stream and not x.dropped:
                device_mesh._use(x)
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

    mutates = []
    if callable(result_type):
        fake_input_tensors = [d._fake for d in dtensors]
        before_versions = [f._version for f in fake_input_tensors]
        fake_args, fake_kwargs = unflatten(fake_input_tensors)
        result = ctrl._run_fake(result_type, fake_args, fake_kwargs)
        for i in range(len(dtensors)):
            if before_versions[i] < fake_input_tensors[i]._version:
                borrows = dtensors[i]._borrows
                writing_stream = borrows.writing_stream
                if writing_stream is not stream:
                    reason = "it is read only because it is being borrowed" if writing_stream is None \
                             else f"it can only be mutated by f{writing_stream}"
                    tbs = ''.join(f"Traceback of borrow to {k} (most recent frame last):\n{''.join(traceback.format_list(b.frames))}" for k,b in borrows.active.items())
                    raise ValueError(f"\n{tbs}\nTensor input {i} would be mutated by this operator but {reason}")
                mutates.extend(dtensors[i]._borrows.aliases)
        fake_map = {id(f): i for i, f in enumerate(fake_input_tensors)}
    else:
        result = result_type
        fake_map = {}

    fake_result_dtensors, unflatten_result = flatten(result, lambda x: isinstance(x, torch.Tensor))

    # sometimes operators return references to inputs, in which case the result should be the same DTensor object
    # otherwise we create a new DTensor with a new RemoteRef for the result
    result_dtensors = tuple(dtensors[fake_map[id(fake)]] if id(fake) in fake_map else Tensor(fake, device_mesh, stream, False) for fake in fake_result_dtensors)
    ident = device_mesh.ctrl.history.ident(result_dtensors + tuple(mutates), dtensors)
    device_mesh._send(messages.CallFunction(ident, tuple(r.ref for r in result_dtensors), tuple(r.ref for r in mutates), func, args, kwargs, stream))    
    results = unflatten_result(result_dtensors)
    # XXX - realistically this would be done on a non-python thread, keeping our messages up to date
    # but we can approximate it by checking for all ready meassages whenever we schedule new work
    device_mesh.ctrl._read_messages(0)
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
        # see if the future is done already
        # and we just haven't processed the messages
        self._ctrl._read_messages(0)
        if self._status != 'incomplete':
            return True
        # we might need to ask for status updates
        # from workers to be sure they have finished
        # enough work to count this future as finished.
        self._ctrl._request_status()
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
