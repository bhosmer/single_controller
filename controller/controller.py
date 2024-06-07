from supervisor import Context, Host, FunctionCall, TTL
from typing import Dict, Optional, List, Sequence, NamedTuple
from concurrent.futures import ThreadPoolExecutor
from torch._subclasses.fake_tensor import FakeTensorMode

import torch
import torch.distributed
from torch.distributed import TCPStore
from . import messages, stream
from .history import History, RemoteException
from .borrows import Borrows
from .stream import Stream
from .device_mesh import DeviceMesh

from collections import defaultdict
import itertools
import socket
import traceback
from weakref import WeakKeyDictionary

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

class Controller:
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
        self.allocation_borrows: WeakKeyDictionary[torch.UntypedStorage, Borrows] = WeakKeyDictionary()
        self.fake_mode_worker = ThreadPoolExecutor(max_workers=1)
        self.history = History(len(self.all_processes))
        stream._active = Stream("main", _default=True)

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
        return TCPStore(hostname, port, is_master=True)

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

    def send(self, ranks: Sequence[int], msg: NamedTuple):
        for rank in ranks:
            self.all_processes[rank].send(msg)

    @property
    def all_ranks(self):
        return range(len(self.all_processes))

    def shutdown(self):
        self._shutdown = True
        self.send(self.all_ranks, messages.Exit())
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
        self.send(self.all_ranks, messages.RequestStatus(self.history.last_assigned_ident))

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
