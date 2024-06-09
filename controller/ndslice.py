from typing import List
from dataclasses import dataclass
from itertools import product
import math
import random

@dataclass(frozen=True)
class NDSlice:
    offset: int
    sizes: List[int]
    strides: List[int]

    def __post_init__(self):
        if len(self.sizes) != len(self.strides):
            raise ValueError("sizes and strides must have the same length")
        prev_stride = None
        total = 1
        for stride, size in sorted(zip(self.strides, self.sizes)):
            if prev_stride is not None:
                if stride % prev_stride != 0:
                    raise ValueError("NDSlice must be rectangularly shaped.")
                if stride == prev_stride:
                    raise ValueError("Strides must be unique.")
            if size <= 0:
                raise ValueError("Slice sizes must be positive.")
            if total > stride:
                raise ValueError("Stride must be positive, and larger than size of previous space.")
            total = stride * size
            prev_stride = stride

    def __iter__(self):
        for loc in product(*(range(s) for s in self.sizes)):
            yield self.offset + sum(i*s for i, s in zip(loc, self.strides))

    def union(self, other: 'NDSlice') -> List['NDSlice']:
        # return a merged slice for pairs that meet regularity conditions
        # and can be detected cheaply, otherwise just return the pair.
        def sort(s):
            strides, sizes = zip(*sorted(zip(s.strides, s.sizes), reverse=True))
            return NDSlice(s.offset, list(sizes), list(strides))
        s1, s2 = sort(self), sort(other)
        s1, s2 = sorted([s1, s2], key=lambda s: s.offset)
        extent = s1.sizes[0] * s1.strides[0]
        if s1.offset + extent == s2.offset:
            if s1.strides == s2.strides and s1.sizes[1:] == s2.sizes[1:]:
                # congruent inner shape and strides, concatenated on long axis
                sizes = [s1.sizes[0] + s2.sizes[0]] + s1.sizes[1:]
                return [NDSlice(s1.offset, sizes, s1.strides)]
        # fallback
        return [self, other]
    
    def contains_any(self, start: int, end: int) -> bool:
        # does this slice contain any of the elements in [start, end)
        # will be used to figure out who to broadcast to.
        base = self.offset
        for stride, size in sorted(zip(self.strides, self.sizes), reverse=True):
            if end <= base or start >= base + size * stride:
                return False
            shift = max(0, min(size - 1, (end - base - 1) // stride))
            base += shift * stride
        return base >= start and base < end
    
    def index(self, value: int) -> int:
        # return index  where self[index] == value,
        # or raise ValueError if not found
        pos = value - self.offset
        if pos < 0:
            raise ValueError(f"value {value} not in NDSlice {self} ()")
        result = 0

        stuff = sorted(zip(self.strides, enumerate(self.sizes)))
        for stride, (i, size) in reversed(list(stuff)):
            index, pos = divmod(pos, stride)
            if index >= size:
                raise ValueError(f"value {value} not in NDSlice {self}")
            result += index * math.prod(self.sizes[i+1:])
        if pos != 0:
            raise ValueError(f"value {value} not in NDSlice {self}")
        return result

    def __getitem__(self, index: int) -> int:
        # return the value as if we did tuple(self)[index],
        # but do so in O(1) time
        value = self.offset
        rest = index
        N = 1
        for size, stride in zip(reversed(self.sizes), reversed(self.strides)):
            N *= size
            value += (rest % size) * stride
            rest //= size
        if index < 0 or index >= N:
            raise IndexError("index out of range")
        return value