"""
Single-resolution dense grid backed by the Carbon multi-GPU runtime.

This module wraps ``carbon.Dense`` and exposes it through the :class:`Grid`
interface so that XLB operators can allocate and operate on fields
transparently. The public backend name stays ``ComputeBackend.NEON``; Carbon
is the implementation underneath.
"""

import carbon
from .grid import Grid
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Literal
from xlb import DefaultConfig


class NeonGrid(Grid):
    """Dense single-resolution grid on the Carbon backend (public name: NEON).

    Wraps a ``carbon.Dense``. The grid is sized to the bounding box and given a
    halo thick enough for the LBM stencil derived from *velocity_set* so Carbon
    sets up the correct z-halo exchanges for neighbour communication.

    Parameters
    ----------
    shape : tuple of int
        Bounding-box dimensions of the domain ``(nx, ny, nz)`` (or
        ``(nx, ny)`` for 2-D).
    velocity_set : VelocitySet
        Lattice velocity set whose stencil defines neighbour connectivity.
    backend_config : dict, optional
        Backend configuration.  Must contain ``"device_list"`` (list of GPU
        device indices).  Defaults to ``{"device_list": [0]}``.
    """

    def __init__(
        self,
        shape,
        velocity_set,
        backend_config=None,
    ):
        from .warp_grid import WarpGrid

        if backend_config is None:
            backend_config = {
                "device_list": [0],
                "skeleton_config": carbon.SkeletonConfig.OCC.none(),
            }

        # check that the config dictionary has the required keys
        required_keys = ["device_list"]
        for key in required_keys:
            if key not in backend_config:
                raise ValueError(f"backend_config must contain a '{key}' key")

        # check that the device list is a list of integers
        if not isinstance(backend_config["device_list"], list):
            raise ValueError("backend_config['device_list'] must be a list of integers")
        for device in backend_config["device_list"]:
            if not isinstance(device, int):
                raise ValueError("backend_config['device_list'] must be a list of integers")

        self.config = backend_config
        self.bk = None
        self.dim = None
        self.grid = None
        self.velocity_set = velocity_set

        super().__init__(shape, ComputeBackend.NEON)

    def _get_velocity_set(self):
        return self.velocity_set

    def _initialize_backend(self):
        # NOTE: device subset selection (dev_idx_list) is not yet plumbed through the
        # Carbon C-ABI; Carbon uses every visible CUDA device (honour CUDA_VISIBLE_DEVICES
        # to restrict). Tracked as a follow-up. We still record the requested list.
        dev_idx_list = self.config["device_list"]

        if len(self.shape) == 2:
            self.dim = carbon.Index_3d(self.shape[0], 1, self.shape[1])
            self.neon_stencil = [[int(self.velocity_set._c[0, q]), 0, int(self.velocity_set._c[1, q])] for q in range(self.velocity_set.q)]
        else:
            self.dim = carbon.Index_3d(self.shape[0], self.shape[1], self.shape[2])
            self.neon_stencil = [[int(self.velocity_set._c[0, q]), int(self.velocity_set._c[1, q]), int(self.velocity_set._c[2, q])] for q in range(self.velocity_set.q)]

        # Halo thickness = Chebyshev radius of the stencil (1 for standard LBM lattices).
        halo_radius = 1
        for offset in self.neon_stencil:
            halo_radius = max(halo_radius, abs(offset[0]), abs(offset[1]), abs(offset[2]))

        gate = DefaultConfig.carbon_gate
        if gate is None:
            raise RuntimeError("Carbon runtime not initialized; call xlb.init(..., default_backend=ComputeBackend.NEON, ...) first")

        self.bk = carbon.System(gate)
        self.grid = carbon.Dense(self.bk, self.dim.x, self.dim.y, self.dim.z, halo_radius=halo_radius)

    def create_field(
        self,
        cardinality: int,
        dtype: Literal[Precision.FP32, Precision.FP64, Precision.FP16] = None,
        fill_value=None,
    ):
        """Allocate a new Neon field on this grid.

        Parameters
        ----------
        cardinality : int
            Number of components per voxel (e.g. ``q`` for populations).
        dtype : Precision, optional
            Element precision.  Defaults to the store precision from the
            global config.
        fill_value : float, optional
            If provided every element is set to this value; otherwise the
            field is zero-initialized.

        Returns
        -------
        carbon.DenseField
            The newly allocated field.
        """
        dtype = dtype.wp_dtype if dtype else DefaultConfig.default_precision_policy.store_precision.wp_dtype
        field = self.grid.new_field(
            cardinality=cardinality,
            dtype=dtype,
        )

        if fill_value is None:
            field.zero_run(stream_idx=0)
        else:
            field.fill_run(value=fill_value, stream_idx=0)
        return field

    def get_neon_backend(self):
        """Return the underlying ``carbon.System`` instance (public name: neon backend)."""
        return self.bk
