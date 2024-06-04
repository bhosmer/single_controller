
from typing import Dict, NamedTuple, Optional, List
import traceback
from weakref import WeakSet

class _Borrow(NamedTuple):
    id: int
    used: bool
    frames: List[traceback.FrameSummary]

class _Borrows:
    def __init__(self, origin_stream: 'Stream'):
        self.origin_stream = origin_stream
        # who can write to this storage?
        self.writing_stream: Optional['Stream'] = origin_stream
        # what Tensor aliases exist for this storage
        self.aliases = WeakSet()
        # what active borrows of this exist?
        self.active: Dict['Stream', _Borrow] = {}
