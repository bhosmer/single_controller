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
from traceback import extract_tb, FrameSummary, format_list

logger = logging.getLogger(__name__)

class Ref:
    def __init__(self, id: int):
        self.id = id

    def __repr__(self):
        return f'r{self.id}'

    def __reduce__(self):
        return Ref, (self.id,)

class RemoteException(Exception):
    def __init__(self, exception: Exception, worker_frames: Optional[List[FrameSummary]] = None):
        self.exception = exception
        if worker_frames is None:
            worker_frames = extract_tb(exception.__traceback__)
        self.worker_frames = worker_frames
        self.controller_frames: Optional[List[FrameSummary]] = None

    def __str__(self):
        try:
            exe = str(self.exception)
            worker_tb = ''.join(format_list(self.worker_frames))
            controller_tb = ''.join(format_list(self.controller_frames)) if self.controller_frames is not None else '<unset>'
            return f'A remote function has failed asynchronously.\n' \
                f'Traceback of where the remote function was issued on controller (most recent call last):\n{controller_tb}' \
                f'Traceback of where the remote function failed on worker (most recent call last):\n{worker_tb}{type(self.exception).__name__}: {exe}'
        except Exception as e:
            print(e)
            return "oops"

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

class FetchException(NamedTuple):
    ident: int
    failing_ident: int

class Status(NamedTuple):
    last_completed_ident: int
    failed_idents: Dict[int, RemoteException]  # since last status

# used to force ident to update periodically even
# if this particular worker has received no new work
# eventually can be used to send more status info
# to workers
class ControllerStatus(NamedTuple):
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


def _reduce(local_tensor, source_mesh: 'DeviceMesh', dim: str, reduction: str, scatter: bool, destination_mesh: 'DeviceMesh', inplace: bool):
    if destination_mesh is None:
        destination_mesh = source_mesh
    group = source_mesh.dims[dim].process_group
    assert source_mesh is destination_mesh

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
        self.last_completed_ident = -1
        self.last_send_status = -1
        self.failed_idents: Dict[int, RemoteException] = {}
        self.borrows: Dict[int, Borrow] = {}

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

    def _dependent_error(self, ident: int, exception: Exception) -> DependentOnError:
        if isinstance(exception, DependentOnError):
            return exception
        self.failed_idents[ident] = RemoteException(exception)
        exception = DependentOnError(ident)
        return exception

    def CallFunction(self, ident: int, results: Tuple[int], function: Union[str, Callable], args: Tuple[Any, ...], kwargs: Dict[str, Any], streamref: Ref):
        try:
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
        except Exception as e:
            err = self._dependent_error(ident, e)
            for r in results:
                self.define(r, err)
        finally:
            self.last_completed_ident = ident

    def CreateStream(self, result: int, default: bool):
        self.define(result, Stream(default))

    def FetchValue(self, ident: int, function_str: Optional[str], obj: Any, streamref: Ref):
        self.last_completed_ident = ident
        try:
            stream: Stream = self.lookup(streamref)
            obj = tree_map(self.lookup, obj)
            with stream.enable():
                if function_str is not None:
                    function = self._resolve_function(function_str)
                    obj = function(obj)
                self.q.send(FetchResult(ident, obj))
        except Exception as e:
            err = self._dependent_error(ident, e)
            # make sure controller knows the error message
            # if we have not already reported it.
            if err.ident in self.failed_idents:
                self._send_status()
                assert err.ident not in self.failed_idents
            self.q.send(FetchException(ident, err.ident))

    def ControllerStatus(self, ident: int):
        self.last_completed_ident = ident

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
        except DependentOnError as e:
            self.define(result, e)
            return
        self.define(result, tensor)
        if not already_borrowed:
            self.borrows[result] = Borrow(from_stream, to_stream)

    def BorrowFirstUse(self, borrow: int):
        self.borrows[borrow].use()

    def BorrowDrop(self, borrow: int):
        self.borrows.pop(borrow).drop()

    def define(self, r: int, value: Any):
        self.env[r] = value

    def _send_status(self):
        if self.last_completed_ident > self.last_send_status:
            self.q.send(Status(self.last_completed_ident, self.failed_idents))
            self.failed_idents.clear()
            self.last_send_status = self.last_completed_ident
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
    q = get_message_queue()
    # CUDA_VISIBLE_DEVICES should be set on launch to LOCAL_RANK
    while True:
        worker = Worker(q, store, rank, world, local_rank)
        worker.event_loop()
        if not _restartable:
            break
        q.send(Restarted(0))
        logger.info("restarting")


# the_borrowed_tensor = stream.borrow(x)
# the_borrow.drop_borrow() # now any views of this can no longer be used on the stream
                           # have to track via weak references to storages we care about
# controller has a weakkeydictionary of storages for fake tensors to weakrefs to dtensors
# controller also keeps the restriction of who can read/write the storage
