import neon as ne
import dense as neon_dense
import warp as wp


@wp.struct
class Span:
    dim: ne.Index
    origin: ne.Index
    partitionIdx: int

    def set(self, dim: ne.Index,
            origin: ne.Index,
            device_idx: int):
        self.dim: ne.Index = dim
        self.origin: ne.Index = origin
        self.partitionIdx: int = device_idx

    def get_dim(self):
        return self.dim

    def get_origin(self):
        return self.origin

    def get_device_idx(self):
        return self.partitionIdx

    def get_idx(self, a, b, c) -> neon_dense.Index:
        return neon_dense.Index(a, b, c)
