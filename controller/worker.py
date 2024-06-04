from contextlib import contextmanager
from supervisor import LocalMessageQueue, get_message_queue, TTL
from supervisor.logging import initialize_logging
from typing import Dict, NamedTuple, Any, Union, Optional, List, Tuple, Callable
import os
import importlib
import logging
from .tree import flatten, tree_map
import torch
import torch.distributed
from traceback import extract_tb, FrameSummary

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

class CreateStream(NamedTuple):
    result: int
    default: bool

class CallFunction(NamedTuple):
    ident: int
    results: Tuple[int, ...]
    mutates: Tuple[int, ...]
    function: Union[str, Callable]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    stream: Any

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

class FetchValue(NamedTuple):
    ident: int
    function: Optional[str]
    object: Any
    stream: Any

class FetchResult(NamedTuple):
    ident: int
    value: Any

class RemoteFunctionFailed(NamedTuple):
    failing_ident: int
    exception: Exception
    worker_frames: List[FrameSummary]

class Status(NamedTuple):
    first_uncompleted_ident: int

# When the controller is waiting on a status update,
# it will request one even if it is before the
# periodic one.
class RequestStatus(NamedTuple):
    ident: int

class BorrowCreate(NamedTuple):
    result: int
    tensor: Any
    from_stream: Any
    to_stream: Any
    alias: bool # is this an alias of an already existing borrow

class BorrowDrop(NamedTuple):
    borrow: int

class BorrowFirstUse(NamedTuple):
    borrow: int

class SendTensor(NamedTuple):
    result: int
    from_ranks: List[int]
    to_ranks: List[int]
    tensor: Any
    factory: 'TensorFactory'
    stream: Any

class Reduce(NamedTuple):
    result: int
    local_tensor_ref: Any
    factory: 'TensorFactory'
    source_mesh_ref: Any
    stream_ref: Any
    dim: str
    reduction: str
    scatter: bool
    inplace: bool

class TensorFactory(NamedTuple):
    size: Tuple[int,...]
    dtype: torch.dtype
    layout: torch.layout
    device: torch.device

    @staticmethod
    def from_tensor(t):
        return TensorFactory(t.size(), t.dtype, t.layout, t.device)

    def empty(self):
        return torch.empty(self.size, dtype=self.dtype, layout=self.layout, device=self.device)

    def zeros(self):
        return torch.full(self.size, 0, dtype=self.dtype, layout=self.layout, device=self.device)

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

def _rank(mesh: 'DeviceMesh', dim: str):
    return torch.full((), mesh.dims[dim].rank, dtype=torch.long)

class DependentOnError(Exception):
    def __init__(self, ident: int):
        self.ident = ident

class Stream:
    cuda_stream: Optional[torch.cuda.Stream]

    def __init__(self, default: bool):
        if default:
            self._cuda_stream = None
        else:
            self._cuda_stream = torch.cuda.Stream()

    @property
    def cuda_stream(self):
        if self._cuda_stream is None:
            return torch.cuda.current_stream()
        else:
            return self._cuda_stream

    @contextmanager
    def enable(self):
        if self._cuda_stream is None:
            yield
            return
        with torch.cuda.stream(self._cuda_stream):
            yield

    def event(self):
        e = torch.cuda.Event()
        self.cuda_stream.record_event(e)
        return e

    def wait_event(self, event):
        self.cuda_stream.wait_event(event)

    def wait_stream(self, stream):
        self.cuda_stream.wait_stream(stream.cuda_stream)

class Borrow:
    def __init__(self, from_stream: Stream, to_stream: Stream):
        self.from_stream = from_stream
        self.to_stream = to_stream
        self.event = from_stream.event()

    def use(self):
        self.to_stream.wait_event(self.event)
        self.event = None

    def drop(self):
        if self.event is not None:
            return
        self.from_stream.wait_stream(self.to_stream)


class Worker:
    def __init__(self, q: LocalMessageQueue, store, rank: int, world: int, local_rank: int):
        # remote ref id to local value
        self.env: Dict[int, Any] = {}
        self.q = q
        self.store = store
        self.rank = rank
        self.world = world
        self.local_rank = local_rank
        self.first_uncompleted_ident = 0
        self.last_send_status = 0
        self.borrows: Dict[int, Optional[Borrow]] = {}

    def handle_message(self, event: NamedTuple):
        cmd = event.__class__.__name__
        fn = getattr(self, cmd, None)
        if fn is not None:
            return fn(*event)
        raise RuntimeError(f"unhandled event: {event}")

    def CreateDeviceMesh(self, result: int, dims: Dict[str, int], ranks: List[int]):
        index = ranks.index(self.rank)
        self.define(result, DeviceMesh(dims, ranks, index))

    def lookup(self, a: Any):
        if isinstance(a, Ref):
            r = self.env[a.id]
            if isinstance(r, DependentOnError):
                raise r
            return r
        return a

    def _resolve_function(self, function_str: str) -> Callable:
        first, *parts = function_str.split('.')
        if first == 'torch':
            function = globals()[first]
            for p in parts:
                function = getattr(function, p)
            assert isinstance(function, Callable)
        else:
            modulename, funcname = function_str.rsplit('.', 1)
            module = importlib.import_module(modulename)
            function = getattr(module, funcname)
        return function

    def CallFunction(self, ident: int, results: Tuple[int], mutates: Tuple[int], function: Union[str, Callable], args: Tuple[Any, ...], kwargs: Dict[str, Any], streamref: Ref):
        with self.try_define(ident, results + mutates):
            stream: Stream = self.lookup(streamref)
            args, kwargs = tree_map(self.lookup, (args, kwargs))
            if isinstance(function, str):
                function = self._resolve_function(function)
            with stream.enable():
                result = function(*args, **kwargs)
            tensors, _ = flatten(result, lambda x: isinstance(x, torch.Tensor))
            assert len(results) == len(tensors)
            for r, t in zip(results, tensors):
                self.define(r, t)

    def CreateStream(self, result: int, default: bool):
        self.define(result, Stream(default))

    @contextmanager
    def try_define(self, ident: int, results: Tuple[int, ...]):
        try:
            yield
        except DependentOnError as e:
            for r in results:
                self.define(r, e)
            # note: there is no need to to send RemoteFunctionFailed
            # because the controller would have already gotten and propagated the
            # original created of DependentOnError.
        except Exception as e:
            exc = DependentOnError(ident)
            for r in results:
                self.define(r, exc)
            self.q.send(RemoteFunctionFailed(ident, e, extract_tb(e.__traceback__)))
        self.first_uncompleted_ident = ident + 1          

    def FetchValue(self, ident: int, function_str: Optional[str], obj: Any, streamref: Ref):
        with self.try_define(ident, ()):
            stream: Stream = self.lookup(streamref)
            obj = tree_map(self.lookup, obj)
            with stream.enable():
                if function_str is not None:
                    function = self._resolve_function(function_str)
                    obj = function(obj)
                self.q.send(FetchResult(ident, obj))

    def RequestStatus(self, ident: int):
        self.first_uncompleted_ident = ident + 1
        self._send_status()

    def Exit(self):
        raise StopIteration()

    def CommandGroup(self, commands: List[NamedTuple]):
        for cmd in commands:
            self.handle_message(cmd)

    def DeleteRefs(self, refs: List[int]):
        for id in refs:
            del self.env[id]

    def BorrowCreate(self, result: int, tensorref: Ref, from_streamref, to_streamref, already_borrowed: bool):
        try:
            from_stream = self.lookup(from_streamref)
            to_stream = self.lookup(to_streamref)
            tensor = self.lookup(tensorref)
            self.define(result, tensor)
            if not already_borrowed:
                self.borrows[result] = Borrow(from_stream, to_stream)
        except DependentOnError as e:
            self.define(result, e)
            if not already_borrowed:
                self.borrows[result] = None

    def BorrowFirstUse(self, borrow: int):
        b = self.borrows[borrow]
        # can be none if the originator of the borrow errored.
        if b is not None:
            b.use()

    def BorrowDrop(self, borrow: int):
        b = self.borrows.pop(borrow)
        if b is not None:
            b.drop()

    def _reduce(self, local_tensor: torch.Tensor, source_mesh: DeviceMesh, dim: str, reduction: str, scatter: bool, inplace: bool):
        group = source_mesh.dims[dim].process_group

        if reduction == 'stack':
            if scatter:
                output = local_tensor
                if not inplace:
                    output = local_tensor.clone()
                torch.distributed.all_to_all_single(output, local_tensor, group=group)
                return output

            assert not inplace
            output = torch.empty([source_mesh.dims[dim].size, *local_tensor.shape],
                                dtype=local_tensor.dtype, device=local_tensor.device, layout=local_tensor.layout)
            print(output.shape, local_tensor.shape)
            torch.distributed.all_gather_into_tensor(output, local_tensor, group=group)
            return output

        op = getattr(torch.distributed.ReduceOp, reduction.upper())

        if scatter:
            assert not inplace
            output = torch.empty(local_tensor.shape[1:], dtype=local_tensor.dtype,
                                device=local_tensor.device, layout=local_tensor.layout)
            torch.distributed.reduce_scatter_tensor(output, local_tensor, op=op, group=group)
            return output

        output = local_tensor
        if not inplace:
            output = local_tensor.clone()
        torch.distributed.all_reduce(output, op=op, group=group)
        return output

    def Reduce(self, result: int, local_tensor_ref: Ref, factory: TensorFactory, source_mesh_ref: Ref, stream_ref: Ref, dim: str, reduction: str, scatter: bool, inplace: bool):
        source_mesh = self.lookup(source_mesh_ref)
        stream = self.lookup(stream_ref)
        with stream.enable():
            try:
                local_tensor = self.lookup(local_tensor_ref)
            except DependentOnError as e:
                # even if we were broken before, we have to participate in the collective
                # because we cannot signal to other ranks that we were broken
                # the controller will see the error message we sent before and know
                # the downstream values are broken.
                local_tensor = factory.zeros()
            # XXX - we should be careful about starting the collective with a tensor that doesn't match the expected
            # factory size. It can error. however, before we can do something about it we need to assign a failure
            # identity to this reduce object.
            output = self._reduce(local_tensor, source_mesh, dim, reduction, scatter, inplace)
            self.define(result, output)

    def SendTensor(self, result: int, from_ranks: List[int], to_ranks: List[int], tensorref: Ref, factory: TensorFactory, streamref):
        try:
            stream = self.lookup(streamref)
        except DependentOnError as e:
            self.define(result, e)
            return
        with stream.enable():
            ops = []
            P2POp = torch.distributed.P2POp
            isend, irecv = torch.distributed.isend, torch.distributed.irecv
            try:
                index = from_ranks.index(self.rank)
                try:
                    tensor = self.lookup(tensorref)
                except DependentOnError:
                    # XXX - how do we propagate this error on the host correctly?
                    # the host will see on status, but it will not immediately know
                    # what dependended on this downstream that also has to be invalid now.
                    tensor = factory.zeros()
                to_rank = to_ranks[index]
                ops.append(P2POp(isend, tensor, to_rank))
            except ValueError:
                to_rank = None

            try:
                index = to_ranks.index(self.rank)
                from_rank = from_ranks[index]
                if from_rank == to_rank:
                    assert tensor is not None
                    self.define(result,  tensor)
                recv = factory.empty()
                ops.append(P2POp(irecv, recv, from_rank))
                self.define(result, recv)
            except ValueError:
                pass
            # invoke batched p2p ops
            for op in torch.distributed.batch_isend_irecv(ops):
                op.wait()

    def define(self, r: int, value: Any):
        self.env[r] = value

    def _send_status(self):
        if self.first_uncompleted_ident > self.last_send_status:
            self.q.send(Status(self.first_uncompleted_ident))
            self.last_send_status = self.first_uncompleted_ident
            logger.info("updating last send status %s", self.last_send_status)

    def event_loop(self):
        STATUS_INTERVAL = 1.0
        status_ttl = TTL(STATUS_INTERVAL)
        while True:
            try:
                _, msg = self.q.recv(timeout=status_ttl())
                logger.info(f"event: {msg}")
                self.handle_message(msg)
            except TimeoutError:
                pass
            except StopIteration:
                self.q.recvready(0)
                self.q.recvready(.01)
                return
            if status_ttl() == 0:
                status_ttl = TTL(STATUS_INTERVAL)
                self._send_status()


def worker_main(_restartable):
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    initialize_logging(process_name=f'worker_{rank}')
    logger.info("starting, restartable=%s", _restartable)
    store = torch.distributed.TCPStore(os.environ['STORE_HOSTNAME'], int(os.environ['STORE_PORT']))
    torch.distributed.init_process_group(backend='nccl', world_size=world, rank=rank, store=store)
    b = torch.zeros(1, device='cuda')
    torch.distributed.all_reduce(b)
    q = get_message_queue()
    # CUDA_VISIBLE_DEVICES should be set on launch to LOCAL_RANK
    while True:
        worker = Worker(q, store, rank, world, local_rank)
        worker.event_loop()
        if not _restartable:
            break
        q.send(Restarted(0))
        logger.info("restarting")


# 1. Test the mesh movement implementation for correctness
# 2. figure out how to move the 'success' criteria for fetch_shard to the controller
#    so it can be aware of any errors that happened on other workers. We may need to
#    broadcast a status/response call.
