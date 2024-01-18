import neon as ne
import warp as wp
from typing import TypeVar, Generic
import dense as neon_dense


class Field(ne.Field):
    def __init__(self,
                 name: str,
                 grid: ne.Grid,
                 layout_dim: ne.Index,
                 dtype):
        """
        1. init the abstract Field
        2. create partition table
        """
        ne.Field.__init__(self, id=0, name=name, grid=grid)
        self.layout_dim: ne.Index = layout_dim
        self.dtype = dtype
        self.memory = wp.empty(shape=self.grid.get_dim().reduce_mul(), dtype=dtype, device="cpu")
        self.partition = neon_dense.make_partition(int, self.grid.get_dim(), self.memory)

    def operator_brackets(self, idx: neon_dense.Index) :
        return self.partition[idx]

    def get_partition(self,
                      deviceId: int,
                      dataType=None) :
        return self.partition
