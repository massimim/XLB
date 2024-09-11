import warp as wp
from cryptography.hazmat.backends.openssl.backend import backend

import wpne
from .grid import Grid
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal
from xlb import DefaultConfig


class NeonGrid(Grid):
    def __init__(self, shape):
        self.bk = None
        self.dim = None
        self.grid = None
        self.stencil = None
        super().__init__(shape, ComputeBackend.NEON)


    def _initialize_backend(self):
        import py_neon as ne

        # FIXME: for now we hardcode the number of devices to 0
        num_devs = 1
        dev_idx_list = list(range(num_devs))

        if len(self.shape) == 2:
            import py_neon
            self.dim = py_neon.Index_3d(self.shape[0],
                                        1,
                                        self.shape[1])
            nine_point_2d = [[-1, 0, -1],
                             [-1, 0, 0],
                             [-1, 0, 1],
                             [0, 0, -1],
                             [0, 0, 0],
                             [0, 0, 1],
                             [1, 0, -1],
                             [1, 0, 0],
                             [1, 0, 1],
                             ]

            self.stencil = nine_point_2d

        else:
            self.dim = ne.Index_3d(self.shape[0],
                                        self.shape[1],
                                        self.shape[2])
            # Raise exception as it this feature is not implemented yet.
            raise NotImplementedError("3D grid is not implemented yet.")

        self.bk = ne.Backend(
            runtime=ne.Backend.Runtime.stream,
            dev_idx_list=dev_idx_list)

        self.grid = ne.dense.dGrid(
            backend=self.bk,
            dim=self.dim,
            sparsity=None,
            stencil=self.stencil)
        pass

    def create_field(
            self,
            cardinality: int,
            dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None,
            fill_value=None,
    ):
        dtype = dtype.wp_dtype if dtype else DefaultConfig.default_precision_policy.store_precision.wp_dtype
        field = self.grid.new_field(cardinality=cardinality,
                                    dtype=dtype, )

        if fill_value is None:
            wpne.Container.zero(field).run(0)
        else:
            wpne.Container.fill(field, fill_value).run(0)
        return field
