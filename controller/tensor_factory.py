import torch
from typing import NamedTuple, Tuple

class TensorFactory(NamedTuple):
    size: Tuple[int, ...]
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
