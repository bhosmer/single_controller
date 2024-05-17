from supervisor import ProcessExited, ProcessList, Context, Host, FunctionCall
from typing import Dict, NamedTuple, Any, Sequence, TypedDict, Union, Literal, Optional, List, Tuple
from torch import dtype, layout, device, memory_format
from typing_extensions import Unpack
import torch
from contextlib import contextmanager
from .base_tensor import BaseTensor
import math
import os
from . import worker
from .worker import Ref
from abc import ABC, abstractmethod

class _Controller:
    def __init__(self, ctx: Context, hosts: List[Host], gpu_per_host: int):
        self.ctx = ctx
        self.hosts = hosts
        self.all_processes = ctx.create_process_group(hosts, FunctionCall('controller.worker.worker_main'),
                                                      processes_per_host=gpu_per_host,
                                                      env={'CUDA_VISIBLE_DEVICES': '$LOCAL_RANK'})
        self.next_ref = 0
        self.exited = {}
        self.pending_del = []

    def shutdown(self):
        self.all_processes.send(worker.Exit())
        while len(self.exited) < len(self.all_processes):
            self._process_event()

    def _process_event(self):
        sender, event = self.ctx.recv()
        if isinstance(event, ProcessExited):
            self.exited[sender] = event.result

    def ref(self) -> 'Ref':
        r = Ref(self.next_ref)
        self.next_ref += 1
        return r


class Referenceable:
    ctrl: _Controller

    def define_ref(self):
        raise NotImplementedError("undefined ref with no define_ref method")

    def __reduce__(self):
        if self.ref is None:
            self.ref = self.define_ref()
        return Ref, (self.ref.id,)

    def __del__(self):
        if self.ref is not None:
            self.ctrl.pending_del.append(self.ref)

PyTree = Union[Dict[str, 'PyTree'], List['PyTree'], Tuple['PyTree',...], 'Tensor']

class DeviceMesh(Referenceable):
    def __init__(self, ctrl: _Controller, processes: ProcessList, dims):
        self.ctrl = ctrl
        assert len(processes) == math.prod(dims.values())
        self.dims: Dict[str, int] = dims
        self.processes = processes
        self.ref = None

    def define_ref(self):
        # Optimize: we do not have to send device meshes to all workers if we can
        # Create process groups as subsets
        msg = worker.CreateDeviceMesh(self.ctrl.ref(), self.dims, [p.rank for p in self.processes])
        self.ctrl.all_processes.send(msg)
        return msg.result

    def stack(self, **kwargs):
        raise NotImplementedError()

    def __getattr__(self, name):
        if name in self.dims:
            return DeviceMeshDim(self, name)

    def call(self, func: str, *args, **kwargs):
        msg = worker.CallFunction(func, args, kwargs)
        self.processes.send(msg)

    def index(self, **kwargs) -> 'DeviceMesh':
        """
        m.index(batch=3) or m.index(batch=slice(3, None))
        """
        pass

    def split(self, **kwargs: Dict[str, Tuple[str, ...]]):
        raise NotImplementedError()

    def rotate(self, **kwargs: Dict[str, int]):
        raise NotImplementedError()

def world_mesh(ctx: Context, hosts: List[Host], gpu_per_host: int):
    ctrl = _Controller(ctx, hosts, gpu_per_host)
    return DeviceMesh(ctrl, ctrl.all_processes, {'host': len(ctrl.all_processes) // gpu_per_host, 'gpu': gpu_per_host})



class Stream:
    name: str

    def wait_for(self, other: 'Stream'):
        """
        Blocks execution of this stream until the other stream completes the work that has been scheduled.
        Any tensors which have been borrowed from this stream to other, and then freed, will be returned
        to this stream, reclaiming the memory if there are no other references to them.
        """
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


class Tensor:
    stream: Stream
    mesh: DeviceMesh

    def to_mesh(self, mesh: DeviceMesh, stream: Optional[Stream]=None):
        """
        Move data between one device mesh and another. Sizes of named dimensions must match.
        If mesh has dimensions that self.mesh does not, it will broadcast to those dimensions.


        broadcast:
            t.index_mesh(batch=0).to(t.mesh)

        """
        raise NotImplementedError()

    def reduce(self, dim: str, reduction: Literal["gather", "sum", "max"], scatter=False, mesh=None):
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

    def index_mesh(self, **kwargs: Dict[str, Union[int, slice]]):
        pass

    def on_mesh(mesh: DeviceMesh) -> 'Tensor':
        """
        Create a new tensor where self.mesh is replaced with mesh.  Each device in self.mesh must be present exactly once in mesh.
        The result ofo this operation is switches the device meshes, while leaving the local data on each device unchanged.

        rotate:
            t.to(t.mesh.batch.rotate(1)).on_mesh(t.mesh)

        """
        pass






def dtensor_dispatch(func, args=(), kwargs=None, sharding=None):
    if func is torch.ops.aten.detach.default:
        if not args[0].requires_grad:
            return args[0]

    worker = sharding.mesh.flat_workers[0] if sharding else None

    def stringify(t):
        if isinstance(t, DTensor):
            return 'DTensor'
        elif isinstance(t, torch.Tensor):
            return 'Tensor'
        else:
            return t

    def is_dtensor_no_tensors(x):
        if isinstance(x, DTensor):
            return True
        elif isinstance(x, torch.Tensor):
            raise NotImplementedError(f"mixed DTensor/local tensor {func}(args, kwargs)={tree_map(stringify, (args, kwargs))}")

    dtensors, unflatten = flatten((args, kwargs), is_dtensor_no_tensors)

    fake_input_tensors = [d._fake for d in dtensors]
    if _trace is None and do_fake_mode_caching:
        fake_input_tensors_key =  [fake_key(d._fake) for d in dtensors]
        fake_cache_key = str((id(func), unflatten(fake_input_tensors_key)))
        result = fake_cache.get(fake_cache_key, None)
        if result is None:
            with fake_mode:
                fake_args, fake_kwargs = unflatten(fake_input_tensors)
                result = func(*fake_args, **fake_kwargs)
            fake_cache[fake_cache_key] = result
    else:
        with fake_mode:
            fake_args, fake_kwargs = unflatten(fake_input_tensors)
            result = func(*fake_args, **fake_kwargs)


    fake_result_dtensors, unflatten_result = flatten(result, is_tensor)
    fake_map = {id(f): i for i, f in enumerate(fake_input_tensors)}

    # sometimes operators return references to inputs, in which case the result should be the same DTensor object
    # otherwise we create a new DTensor with a new RemoteRef for the result
    result_dtensors = [dtensors[fake_map[id(fake)]] if id(fake) in fake_map else DTensor(fake, RemoteRef(), None) for fake in fake_result_dtensors]
    mesh, modified_args, modified_kwargs = propagate_sharding(func, args, kwargs, dtensors, result_dtensors, sharding)

    for i, r in enumerate(result_dtensors):
        assert r._sharding is not None, f"sharding unset for output {i} of {str(func)} fake outputs: {fake_result_dtensors}"

    def get_ref(x):
        return x if not isinstance(x, DTensor) else x._ref

    ref_args, ref_kwargs = tree_map(get_ref, modified_args), tree_map(get_ref, modified_kwargs)
    ref_results = [r._ref for r in result_dtensors]
    if _trace is not None:
        m, cmds = _trace
        assert m is mesh, "multiple compiled mesh NYI"
        cmds.append((func, ref_args, ref_kwargs, ref_results, result))
    else:
        for worker in mesh.flat_workers:
            worker.send_command(func, ref_args, ref_kwargs, ref_results)

    results = unflatten_result(result_dtensors)

    if check_correctness_per_operator:
        key = str(func)
        print(key, args, kwargs, result_dtensors)
        if "_." in key:
            for i, dtensor in enumerate(dtensors):
                rem = dtensor.to_local().wait()
                # print(dtensor._fake.sum(), rem.sum())
                if torch.all(torch.isfinite(dtensor._fake)) and torch.all(torch.isfinite(rem)):
                    torch.testing.assert_close(dtensor._fake, rem, atol=1e-03, rtol=1e-03)
                else:
                    pass #print("nonfinite present...")
                # don't let small differences accumulate over time when correctness testing
                try:
                    dtensor._fake.copy_(rem)
                except RuntimeError as e:
                    assert "unsupported operation" in str(e), "Weird tensor shapes cannot be moved around"
        for i, dtensor in enumerate(result_dtensors):
            rem = dtensor.to_local().wait()
            if 'aten._scaled_dot_product_efficient_attention.default' == key and i > 1:
                break
            # print(dtensor._fake.sum(), rem.sum())
            if torch.all(torch.isfinite(dtensor._fake)) and torch.all(torch.isfinite(rem)):
                torch.testing.assert_close(dtensor._fake, rem, atol=1e-03, rtol=1e-03)
            else:
                pass #print("nonfinite present...")
            # don't let small differences accumulate over time when correctness testing
            try:
                dtensor._fake.copy_(rem)
            except RuntimeError as e:
                assert "unsupported operation" in str(e), "Weird tensor shapes cannot be moved around"

    return results


class DTensor(BaseTensor):
    @staticmethod
    def __new__(
        cls,
        fake: torch.Tensor,
        ref: 'Ref',
        sharding: 'Optional[Sharding]',
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
        r._ref = ref
        r._fake = fake
        assert sharding is None or isinstance(sharding, Sharding)
        r._sharding = sharding
        return r

    def __init__(self, fake, ref, worker):
        pass


    def __tensor_flatten__(self):
        return ['_fake'], (self._ref, self._sharding)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        return inner_tensors['_fake']

    @classmethod
    def to_remote(cls, t: torch.Tensor, sharding: 'Sharding'):
        sharding = Sharding.lift(sharding)
        f = fake_mode.from_tensor(t)
        r = RemoteRef()

        shape = sharding.mesh.shape

        batch_dim = 0

        def split_dim(i, to_split):
            nonlocal t
            sizes = t.size()
            split_adjusted = to_split + i
            d = sizes[split_adjusted]
            assert d % shape.size(i) == 0, "NOT EVENLY SPLIT"
            chunk_size = d // shape.size(i)
            t = t.reshape(*sizes[:split_adjusted], shape.size(i), chunk_size, *sizes[split_adjusted+1:])
            t = t.movedim(split_adjusted, i)
            return chunk_size

        for i, ann in enumerate(sharding.sharding):
            if ann == "r":
                t = t.expand(shape.size(i), *t.size())
                t = t.movedim(0, i)
            elif isinstance(ann, int):
                split_dim(i, ann)
            elif ann == "b":
                chunk_size = split_dim(i, batch_dim)
                sizes = f.size()
                index = tuple(slice(0, chunk_size) if i == batch_dim else slice(None) for i in range(f.dim()))
                f = f[index]
                batch_dim += 1
            else:
                raise NotImplementedError(f"Annotation: {ann}")

        if shape.dim() == 0:
            worker = sharding.mesh._workers.workers[shape.item()]
            worker.send_value(r, t)
        else:
            t = t.flatten(0, shape.dim() - 1)
            shape_flat = shape.flatten()
            for idx, local in zip(shape_flat, t):
                worker = sharding.mesh._workers.workers[idx]
                worker.send_value(r, local)

        return DTensor(f, r, sharding)

    def __repr__(self):
       return f"DTensor(sharding={self.sharding}, shape={list(self.shape)})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dtensor_dispatch(func, args, kwargs)

    def to_local(self):
        return self._sharding.manager.schedule(reconstruct_tensor(self))

    def __del__(self):
        if sys is None or _trace is not None:
            return # pytho shutdown
        if self._sharding is None:
            return # an error happening during construction and this wasn't initialized
        for worker in self._sharding.mesh.flat_workers:
            worker.del_value(self._ref)

    @property
    def sharding(self):
        return self._sharding

    def to_sharding_(self, new_sharding):
        if not isinstance(new_sharding, Sharding):
            new_sharding = Sharding(self.sharding.mesh, new_sharding)
        new_sharding.apply_inplace(self)
        return self
