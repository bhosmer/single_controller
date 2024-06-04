from typing import Dict, NamedTuple, Any, Union, Optional, List, Tuple, Callable
from traceback import FrameSummary
from .tensor_factory import TensorFactory

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
    alias: bool  # is this an alias of an already existing borrow

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
