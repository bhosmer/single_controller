
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, List
import traceback
from weakref import WeakSet

if TYPE_CHECKING:
    from .stream import Stream

class Borrow(NamedTuple):
    id: int
    used: bool
    frames: List[traceback.FrameSummary]

# we have one Borrows entry for each storage of live controller Tensors
class Borrows:
    def __init__(self, origin_stream: 'Stream'):
        self.origin_stream = origin_stream
        # who can write to this storage?
        self.writing_stream: Optional['Stream'] = origin_stream
        # what Tensor aliases exist for this storage
        self.aliases = WeakSet()
        # what active borrows of this storage exist?
        self.active: Dict['Stream', Borrow] = {}
