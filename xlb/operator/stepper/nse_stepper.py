# Base class for all stepper operators

from functools import partial
from jax import jit
import warp as wp
from typing import Any

import py_neon
import wpne
from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep


class IncompressibleNavierStokesStepper(Stepper):
    def __init__(self, omega, boundary_conditions=[], collision_type="BGK"):
        velocity_set = DefaultConfig.velocity_set
        precision_policy = DefaultConfig.default_precision_policy
        compute_backend = DefaultConfig.default_backend

        # Construct the collision operator
        if collision_type == "BGK":
            self.collision = BGK(omega, velocity_set, precision_policy, compute_backend)
        elif collision_type == "KBC":
            self.collision = KBC(omega, velocity_set, precision_policy, compute_backend)

        # Construct the operators
        self.stream = Stream(velocity_set, precision_policy, compute_backend)
        self.equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)
        self.macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        operators = [self.macroscopic, self.equilibrium, self.collision, self.stream]

        super().__init__(operators, boundary_conditions)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_0, f_1, boundary_mask, missing_mask, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """
        # Cast to compute precisioni
        f_0 = self.precision_policy.cast_to_compute_jax(f_0)
        f_1 = self.precision_policy.cast_to_compute_jax(f_1)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_0)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(f_0, feq, rho, u)

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            f_post_collision = bc.prepare_bc_auxilary_data(f_0, f_post_collision, boundary_mask, missing_mask)
            if bc.implementation_step == ImplementationStep.COLLISION:
                f_post_collision = bc(
                    f_0,
                    f_post_collision,
                    boundary_mask,
                    missing_mask,
                )

        # Apply streaming
        f_1 = self.stream(f_post_collision)

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                f_1 = bc(
                    f_post_collision,
                    f_1,
                    boundary_mask,
                    missing_mask,
                )

        # Copy back to store precision
        f_1 = self.precision_policy.cast_to_store_jax(f_1)

        return f_1

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool
        _c = self.velocity_set.wp_c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.wp_opp_indices
        sound_speed = 1.0 / wp.sqrt(3.0)

        @wp.struct
        class BoundaryConditionIDStruct:
            # Note the names are hardcoded here based on various BC operator names with "id_" at the beginning
            # One needs to manually add the names of additional BC's as they are added.
            # TODO: Any way to improve this?
            id_EquilibriumBC: wp.uint8
            id_DoNothingBC: wp.uint8
            id_HalfwayBounceBackBC: wp.uint8
            id_FullwayBounceBackBC: wp.uint8
            id_ZouHeBC_velocity: wp.uint8
            id_ZouHeBC_pressure: wp.uint8
            id_RegularizedBC_velocity: wp.uint8
            id_RegularizedBC_pressure: wp.uint8
            id_ExtrapolationOutflowBC: wp.uint8

        @wp.func
        def apply_post_streaming_bc(
            f_pre: Any,
            f_post: Any,
            f_aux: Any,
            missing_mask: Any,
            _boundary_id: Any,
            bc_struct: Any,
        ):
            # Apply post-streaming type boundary conditions
            if _boundary_id == bc_struct.id_EquilibriumBC:
                # Equilibrium boundary condition
                f_post = self.EquilibriumBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_DoNothingBC:
                # Do nothing boundary condition
                f_post = self.DoNothingBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_HalfwayBounceBackBC:
                # Half way boundary condition
                f_post = self.HalfwayBounceBackBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ZouHeBC_velocity:
                # Zouhe boundary condition (bc type = velocity)
                f_post = self.ZouHeBC_velocity.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ZouHeBC_pressure:
                # Zouhe boundary condition (bc type = pressure)
                f_post = self.ZouHeBC_pressure.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_RegularizedBC_velocity:
                # Regularized boundary condition (bc type = velocity)
                f_post = self.RegularizedBC_velocity.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_RegularizedBC_pressure:
                # Regularized boundary condition (bc type = velocity)
                f_post = self.RegularizedBC_pressure.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
                # Regularized boundary condition (bc type = velocity)
                f_post = self.ExtrapolationOutflowBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            return f_post

        @wp.func
        def apply_post_collision_bc(
            f_pre: Any,
            f_post: Any,
            f_aux: Any,
            missing_mask: Any,
            _boundary_id: Any,
            bc_struct: Any,
        ):
            if _boundary_id == bc_struct.id_FullwayBounceBackBC:
                # Full way boundary condition
                f_post = self.FullwayBounceBackBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
                # f_aux is the neighbour's post-streaming values
                # Storing post-streaming data in directions that leave the domain
                f_post = self.ExtrapolationOutflowBC.prepare_bc_auxilary_data(f_pre, f_post, f_aux, missing_mask)

            return f_post

        @wp.func
        def get_normal_vectors_2d(
            missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                    return -wp.vec2i(_c[0, l], _c[1, l])

        @wp.func
        def get_normal_vectors_3d(
            missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                    return -wp.vec3i(_c[0, l], _c[1, l], _c[2, l])

        @wp.func
        def get_thread_data_2d(
            f_0: wp.array3d(dtype=Any),
            missing_mask: wp.array3d(dtype=Any),
            index: Any,
        ):
            # Get the boundary id and missing mask
            f_post_collision = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                f_post_collision[l] = f_0[l, index[0], index[1]]

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            return f_post_collision, _missing_mask

        @wp.func
        def get_thread_data_3d(
            f_0: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            index: Any,
        ):
            # Get the boundary id and missing mask
            f_post_collision = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                f_post_collision[l] = f_0[l, index[0], index[1], index[2]]

                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)
            return f_post_collision, _missing_mask

        @wp.func
        def get_bc_auxilary_data_2d(
            f_0: wp.array3d(dtype=Any),
            index: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            bc_struct: Any,
        ):
            # special preparation of auxiliary data
            f_auxiliary = _f_vec()
            if _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
                nv = get_normal_vectors_2d(_missing_mask)
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1):
                        # f_0 is the post-collision values of the current time-step
                        # Get pull index associated with the "neighbours" pull_index
                        pull_index = type(index)()
                        for d in range(self.velocity_set.d):
                            pull_index[d] = index[d] - (_c[d, l] + nv[d])
                        # The following is the post-streaming values of the neighbor cell
                        f_auxiliary[l] = f_0[l, pull_index[0], pull_index[1]]
            return f_auxiliary

        @wp.func
        def get_bc_auxilary_data_3d(
            f_0: wp.array4d(dtype=Any),
            index: Any,
            _boundary_id: Any,
            _missing_mask: Any,
            bc_struct: Any,
        ):
            # special preparation of auxiliary data
            f_auxiliary = _f_vec()
            if _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
                nv = get_normal_vectors_3d(_missing_mask)
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1):
                        # f_0 is the post-collision values of the current time-step
                        # Get pull index associated with the "neighbours" pull_index
                        pull_index = type(index)()
                        for d in range(self.velocity_set.d):
                            pull_index[d] = index[d] - (_c[d, l] + nv[d])
                        # The following is the post-streaming values of the neighbor cell
                        f_auxiliary[l] = f_0[l, pull_index[0], pull_index[1], pull_index[2]]
            return f_auxiliary

        @wp.kernel
        def kernel2d(
            f_0: wp.array3d(dtype=Any),
            f_1: wp.array3d(dtype=Any),
            boundary_mask: wp.array3d(dtype=Any),
            missing_mask: wp.array3d(dtype=Any),
            bc_struct: Any,
            timestep: int,
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)  # TODO warp should fix this

            # Read thread data for populations and missing mask
            f_post_collision, _missing_mask = get_thread_data_2d(f_0, missing_mask, index)

            # Apply streaming (pull method)
            f_post_stream = self.stream.warp_functional(f_0, index)

            # Prepare auxilary data for BC (if applicable)
            _boundary_id = boundary_mask[0, index[0], index[1]]
            f_auxiliary = get_bc_auxilary_data_2d(f_0, index, _boundary_id, _missing_mask, bc_struct)

            # Apply post-streaming type boundary conditions
            f_post_stream = apply_post_streaming_bc(f_post_collision, f_post_stream, f_auxiliary, _missing_mask, _boundary_id, bc_struct)

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(f_post_stream)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(
                f_post_stream,
                feq,
                rho,
                u,
            )

            # Apply post-collision type boundary conditions
            f_post_collision = apply_post_collision_bc(f_post_stream, f_post_collision, f_auxiliary, _missing_mask, _boundary_id, bc_struct)

            # Set the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1]] = f_post_collision[l]

        # Construct the kernel
        @wp.kernel
        def kernel3d(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            boundary_mask: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            bc_struct: Any,
            timestep: int,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO warp should fix this

            # Read thread data for populations and missing mask
            f_post_collision, _missing_mask = get_thread_data_3d(f_0, missing_mask, index)

            # Apply streaming (pull method)
            f_post_stream = self.stream.warp_functional(f_0, index)

            # Prepare auxilary data for BC (if applicable)
            _boundary_id = boundary_mask[0, index[0], index[1], index[2]]
            f_auxiliary = get_bc_auxilary_data_3d(f_0, index, _boundary_id, _missing_mask, bc_struct)

            # Apply post-streaming type boundary conditions
            f_post_stream = apply_post_streaming_bc(f_post_collision, f_post_stream, f_auxiliary, _missing_mask, _boundary_id, bc_struct)

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(f_post_stream)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(f_post_stream, feq, rho, u)

            # Apply post-collision type boundary conditions
            f_post_collision = apply_post_collision_bc(f_post_stream, f_post_collision, f_auxiliary, _missing_mask, _boundary_id, bc_struct)

            # Set the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] = f_post_collision[l]

        # Return the correct kernel
        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return BoundaryConditionIDStruct, kernel

    def _construct_neon(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool
        _c = self.velocity_set.wp_c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.wp_opp_indices
        sound_speed = 1.0 / wp.sqrt(3.0)

        @wp.struct
        class BoundaryConditionIDStruct:
            # Note the names are hardcoded here based on various BC operator names with "id_" at the beginning
            # One needs to manually add the names of additional BC's as they are added.
            # TODO: Any way to improve this?
            id_EquilibriumBC: wp.uint8
            id_DoNothingBC: wp.uint8
            id_HalfwayBounceBackBC: wp.uint8
            id_FullwayBounceBackBC: wp.uint8
            id_ZouHeBC_velocity: wp.uint8
            id_ZouHeBC_pressure: wp.uint8
            id_RegularizedBC_velocity: wp.uint8
            id_RegularizedBC_pressure: wp.uint8
            id_ExtrapolationOutflowBC: wp.uint8

        @wp.func
        def apply_post_streaming_bc(
                f_pre: Any,
                f_post: Any,
                f_aux: Any,
                missing_mask: Any,
                _boundary_id: Any,
                bc_struct: Any,
        ):
            # Apply post-streaming type boundary conditions
            if _boundary_id == bc_struct.id_EquilibriumBC:
                # Equilibrium boundary condition
                f_post = self.EquilibriumBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_DoNothingBC:
                # Do nothing boundary condition
                f_post = self.DoNothingBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_HalfwayBounceBackBC:
                # Half way boundary condition
                f_post = self.HalfwayBounceBackBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ZouHeBC_velocity:
                # Zouhe boundary condition (bc type = velocity)
                f_post = self.ZouHeBC_velocity.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ZouHeBC_pressure:
                # Zouhe boundary condition (bc type = pressure)
                f_post = self.ZouHeBC_pressure.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_RegularizedBC_velocity:
                # Regularized boundary condition (bc type = velocity)
                f_post = self.RegularizedBC_velocity.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_RegularizedBC_pressure:
                # Regularized boundary condition (bc type = velocity)
                f_post = self.RegularizedBC_pressure.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
                # Regularized boundary condition (bc type = velocity)
                f_post = self.ExtrapolationOutflowBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            return f_post

        @wp.func
        def apply_post_collision_bc(
                f_pre: Any,
                f_post: Any,
                f_aux: Any,
                missing_mask: Any,
                _boundary_id: Any,
                bc_struct: Any,
        ):
            if _boundary_id == bc_struct.id_FullwayBounceBackBC:
                # Full way boundary condition
                f_post = self.FullwayBounceBackBC.warp_functional(f_pre, f_post, f_aux, missing_mask)
            elif _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
                # f_aux is the neighbour's post-streaming values
                # Storing post-streaming data in directions that leave the domain
                f_post = self.ExtrapolationOutflowBC.prepare_bc_auxilary_data(f_pre, f_post, f_aux, missing_mask)

            return f_post

        @wp.func
        def get_normal_vectors_2d(
                missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                    return -wp.vec2i(_c[0, l], _c[1, l])

        @wp.func
        def get_normal_vectors_3d(
                missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                    return -wp.vec3i(_c[0, l], _c[1, l], _c[2, l])

        @wp.func
        def get_thread_data(
                f_0: Any,  # this is a dParition type with dtype=Any and cardinality q
                missing_mask: Any,  # this is a dParition type with dtype=uint, and cardinality q
                index: Any,  # this is a Neon dGrid index
        ):
            # Get the boundary id and missing mask
            f_post_collision = _f_vec()
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # q-sized vector of pre-streaming populations
                f_post_collision[l] = wp.neon_read(f_0, index, l)
                _missing_mask[l] = wp.neon_read(missing_mask, index,0)
            return f_post_collision, _missing_mask

        @wp.func
        def get_bc_auxilary_data_2d(
                f_0: Any,
                index: Any,
                _boundary_id: Any,
                _missing_mask: Any,
                bc_struct: Any,
        ):
            # special preparation of auxiliary data
            f_auxiliary = _f_vec()
            # if _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
            #     nv = get_normal_vectors_2d(_missing_mask)
            #     for l in range(self.velocity_set.q):
            #         if _missing_mask[l] == wp.uint8(1):
            #             # f_0 is the post-collision values of the current time-step
            #             # Get pull index associated with the "neighbours" pull_index
            #             # pull_index = type(index)()
            #             # for d in range(self.velocity_set.d):
            #             #     pull_index[d] = index[d] - (_c[d, l] + nv[d])
            #             # The following is the post-streaming values of the neighbor cell
            #             ngh = wp.neon_ngh_idx(wp.int8(-(_c[0, l] + nv[0])),
            #                                   wp.int8(0),
            #                                   wp.int8(-(_c[1, l] + nv[1])))
            #             unused_is_valid = wp.bool(False)
            #             # How do we get the compute type at this stage?
            #             f_auxiliary[l] = wp.neon_ngh(f_0, index, l, ngh, wp.float(0), unused_is_valid)
            return f_auxiliary

        @wp.func
        def get_bc_auxilary_data_3d(
                f_0: wp.array4d(dtype=Any),
                index: Any,
                _boundary_id: Any,
                _missing_mask: Any,
                bc_struct: Any,
        ):
            # special preparation of auxiliary data
            f_auxiliary = _f_vec()
            if _boundary_id == bc_struct.id_ExtrapolationOutflowBC:
                nv = get_normal_vectors_3d(_missing_mask)
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1):
                        # f_0 is the post-collision values of the current time-step
                        # Get pull index associated with the "neighbours" pull_index
                        # pull_index = type(index)()
                        # for d in range(self.velocity_set.d):
                        #     pull_index[d] = index[d] - (_c[d, l] + nv[d])
                        # # The following is the post-streaming values of the neighbor cell
                        # f_auxiliary[l] = f_0[l, pull_index[0], pull_index[1], pull_index[2]]
                        ngh = wp.neon_ngh_idx(wp.int8(-(_c[0, l] + nv[0])),
                                              wp.int8(-(_c[1, l] + nv[1])),
                                              wp.int8(-(_c[2, l] + nv[2])))
                        unused_is_valid = wp.bool(False)
                        # How do we get the compute type at this stage?
                        f_auxiliary[l] = wp.neon_ngh(f_0, index, l, ngh, wp.float(0), unused_is_valid)
            return f_auxiliary

        @wpne.Container.factory
        def container2d(
                f_0_field: wp.array3d(dtype=Any),
                f_1_field: wp.array3d(dtype=Any),
                boundary_mask_field: wp.array3d(dtype=Any),
                missing_mask_field: wp.array3d(dtype=Any),
                bc_struct: Any,
                timestep: int,
        ):
            def container_generator(loader: wpne.Loader):
                loader.declare_execution_scope(f_0_field.get_grid())

                f_0 = loader.get_read_handel(f_0_field)
                f_1 = loader.get_read_handel(f_1_field)
                boundary_mask = loader.get_read_handel(boundary_mask_field)
                missing_mask = loader.get_read_handel(missing_mask_field)

                @wp.func
                def steper_2d(index: Any):
                    # Read thread data for populations and missing mask
                    f_post_collision, _missing_mask = get_thread_data(f_0, missing_mask, index)

                    # Apply streaming (pull method)
                    f_post_stream = self.stream.neon_functional(f_0, index)

                    # Prepare auxilary data for BC (if applicable)
                    _boundary_id = wp.neon_read(boundary_mask, index, 0)
                    f_auxiliary = get_bc_auxilary_data_2d(f_0, index, _boundary_id, _missing_mask, bc_struct)

                    # Apply post-streaming type boundary conditions
                    f_post_stream = apply_post_streaming_bc(f_post_collision, f_post_stream, f_auxiliary, _missing_mask,
                                                            _boundary_id, bc_struct)

                    # Compute rho and u
                    rho, u = self.macroscopic.warp_functional(f_post_stream)

                    # Compute equilibrium
                    feq = self.equilibrium.warp_functional(rho, u)

                    # Apply collision
                    f_post_collision = self.collision.warp_functional(
                        f_post_stream,
                        feq,
                        rho,
                        u,
                    )

                    # Apply post-collision type boundary conditions
                    f_post_collision = apply_post_collision_bc(f_post_stream, f_post_collision, f_auxiliary,
                                                               _missing_mask,
                                                               _boundary_id, bc_struct)

                    # Set the output
                    for l in range(self.velocity_set.q):
                        f_1[l, index[0], index[1]] = f_post_collision[l]

                loader.declare_kernel(steper_2d)

            return container_generator

        # Construct the kernel
        @wp.kernel
        def kernel3d(
                f_0: wp.array4d(dtype=Any),
                f_1: wp.array4d(dtype=Any),
                boundary_mask: wp.array4d(dtype=Any),
                missing_mask: wp.array4d(dtype=Any),
                bc_struct: Any,
                timestep: int,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO warp should fix this

            # Read thread data for populations and missing mask
            f_post_collision, _missing_mask = get_thread_data_3d(f_0, missing_mask, index)

            # Apply streaming (pull method)
            f_post_stream = self.stream.warp_functional(f_0, index)

            # Prepare auxilary data for BC (if applicable)
            _boundary_id = boundary_mask[0, index[0], index[1], index[2]]
            f_auxiliary = get_bc_auxilary_data_3d(f_0, index, _boundary_id, _missing_mask, bc_struct)

            # Apply post-streaming type boundary conditions
            f_post_stream = apply_post_streaming_bc(f_post_collision, f_post_stream, f_auxiliary, _missing_mask,
                                                    _boundary_id, bc_struct)

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(f_post_stream)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(f_post_stream, feq, rho, u)

            # Apply post-collision type boundary conditions
            f_post_collision = apply_post_collision_bc(f_post_stream, f_post_collision, f_auxiliary, _missing_mask,
                                                       _boundary_id, bc_struct)

            # Set the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] = f_post_collision[l]

        # Return the correct kernel
        container = None if self.velocity_set.d == 3 else container2d

        if self.velocity_set.d == 3:
            raise NotImplementedError("Neon backend is not yet supported for 3D simulations")

        return BoundaryConditionIDStruct, container

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, boundary_mask, missing_mask, timestep):
        # Get the boundary condition ids
        from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id
        id_to_bc = boundary_condition_registry.id_to_bc
        bc_struct = self.warp_functional()
        active_bc_list = []
        for bc in self.boundary_conditions:
            # Setting the Struct attributes and active BC classes based on the BC class names
            bc_name = id_to_bc[bc.id]
            setattr(self, bc_name, bc)
            setattr(bc_struct, "id_" + bc_name, bc_to_id[bc_name])
            active_bc_list.append("id_" + bc_name)

        # Setting the Struct attributes and active BC classes based on the BC class names
        bc_fallback = self.boundary_conditions[0]
        # TODO: what if self.boundary_conditions is an empty list e.g. when we have periodic BC all around!
        for var in vars(bc_struct):
            if var not in active_bc_list and not var.startswith("_"):
                # set unassigned boundaries to the maximum integer in uint8
                setattr(bc_struct, var, 255)

                # Assing a fall-back BC for inactive BCs. This is just to ensure Warp codegen does not
                # produce error when a particular BC is not used in an example.
                setattr(self, var.replace("id_", ""), bc_fallback)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
                boundary_mask,
                missing_mask,
                bc_struct,
                timestep,
            ],
            dim=f_0.shape[1:],
        )
        return f_1

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_0, f_1, boundary_mask, missing_mask, timestep):
        # Get the boundary condition ids
        from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry

        # Read the list of bc_to_id created upon instantiation
        bc_to_id = boundary_condition_registry.bc_to_id
        id_to_bc = boundary_condition_registry.id_to_bc
        bc_struct = self.neon_functional()
        active_bc_list = []
        for bc in self.boundary_conditions:
            # Setting the Struct attributes and active BC classes based on the BC class names
            bc_name = id_to_bc[bc.id]
            setattr(self, bc_name, bc)
            setattr(bc_struct, "id_" + bc_name, bc_to_id[bc_name])
            active_bc_list.append("id_" + bc_name)

        # Setting the Struct attributes and active BC classes based on the BC class names
        bc_fallback = self.boundary_conditions[0]
        # TODO: what if self.boundary_conditions is an empty list e.g. when we have periodic BC all around!
        for var in vars(bc_struct):
            if var not in active_bc_list and not var.startswith("_"):
                # set unassigned boundaries to the maximum integer in uint8
                setattr(bc_struct, var, 255)

                # Assing a fall-back BC for inactive BCs. This is just to ensure Warp codegen does not
                # produce error when a particular BC is not used in an example.
                setattr(self, var.replace("id_", ""), bc_fallback)

        # Launch the warp kernel
        container = self.neon_kernel(f_0,
                                     f_1,
                                     boundary_mask,
                                     missing_mask,
                                     bc_struct,
                                     timestep, )
        container.run(0)
        return f_1
