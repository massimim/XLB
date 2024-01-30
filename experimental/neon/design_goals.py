"""
This file has two major functions:

- A 'target_interface_example' that showcases the type of interface we would like to target.
  The function does not run, but it includes comments to explain the neon mechanism according to the following paper: https://escholarship.org/uc/item/9fz7k633

- A 'main' function that runs a limited version of the Neon runtime.
  This version's goal is to showcase some of the challenges we have encountered while trying to implement our target interface.

From this preliminary work, we collected the following questions:

- The neon grid is a low-level abstraction, as is the wp.array. Can we design it as a native warp type? This would allow us to manage the field types nicely.
- Is it possible to have an object-oriented interface for wp.structs?

- Are the following features in the warp roadmap:
    - User-defined cuda blocks.
    - User access to CUDA shared memory mechanism.

- In terms of automatic differentiation, how would it be possible to extend warp capabilities to a multi-GPU setup?

"""

import warp as wp
from typing import Any


def target_interface_example(
):
    """
    This function does not run.
    Its goal is to show the interface we would like to achieve.
    """
    import warp as wp
    import neon as ne

    wp.init()

    devices = [0, 1]
    dim = wp.vec3i(10, 10, 10)

    def active_vox(idx):
        return True

    bk = ne.Backend('cuda', devices)

    # the framework should support different grid types
    grid_type = ['dense', 'block', 'multi-resolution']

    grid = ne.Grid(dim=dim,
                   backend=bk,
                   ghost_radius=1,
                   gridType=grid_type[0],
                   gridOptions={'blockSize': (4, 4, 4),
                                'spaceCurve': 'hilbert',
                                'layout': 'SoA'},
                   active_mask=active_vox)

    velocity_field = grid.new(dtyepe=float, card=3)

    population_in_field = grid.new(dtyepe=float, card=19)
    population_out_field = grid.new(dtyepe=float, card=19)

    density_field = grid.new(dtyepe=float, card=1)

    # My_LBM_solver_flag_type is a structure define by the user, a sort of bit field
    flag_field = grid.new(dtyepe=My_LBM_solver_flag_type, card=1)

    def increase_pressure(idx, f, increase_quantity):
        # a mechanism to transition from field to partition
        # and acquire from  the user some information in the computation semantic
        p_f = ne.use(f, 'map')

        @wp.func
        def operation(idx, p_f: f.partitionType, increase_quantity):
            # we would need a mechanism to get the partition type as it is not directly accessible by the user
            # Alternatively we could use ANY, but it does not help preventing errors
            print(f"I am cell {f.get_my_location()} and I am increasing my pressure")
            p_f[idx, 0] += increase_quantity

        return operation

    def laplace_filter(fin, fout):
        # a mechanism to transition from field to partition
        # and acquire from  the user some information in the computation semantic
        p_fin = ne.use(fin, 'stencil:lattice')
        p_fout = ne.use(fout, 'map')

        @wp.func
        def laplace_filter_wp(idx, p_in: fin.partitionType, p_out: fin.partitionType):
            # idx is an opaque type, that identify the current cell
            # We would need a mechanism (fin.partitionType) to get the partition type as it is not directly accessible by the user
            # Alternatively we could use ANY, but it does not help preventing errors

            # read_ngh would be a method of the partition to extract the neighbor values
            # the operator[] is use to write values into the
            p_out[idx, 0] = p_in.read_ngh(idx, location={0, 0, 1}, card=0)
            + p_in.read_ngh(idx, location={0, 0, -1}, card=0)
            + p_in.read_ngh(idx, location={0, 1, 0}, card=0)
            + p_in.read_ngh(idx, location={0, -1, 0}, card=0)
            + p_in.read_ngh(idx, location={1, 0, 0}, card=0)
            + p_in.read_ngh(idx, location={-1, 0, 0}, card=0)
            - 5 * p_in.read(idx, 0)

    # Containers are a closure of functions and data. Containers generates the kernel code that will call the user warp function
    # Containers can be launch at any time. The launch of a container is asynchronous launches the same kernel on all the devices, but with different inputs.
    # The different inputs are the partitions of the fields.
    increase_pressure_op = grid.new_container(increase_pressure, {'f': density_field, 'increase_quantity': 1.0})
    laplace_filter_op = grid.new_container(laplace_filter, {'f_in': population_in_field, 'f_out': population_out_field,
                                                            'increase_quantity': 1.0})

    # The graph is the mechanism that creates the user's application graph and optimizes it.
    # For example by injecting requires halo updates and synchronizations.
    # It would be nice to be also able to have operation fusion as optimization.
    application_graph = ne.ApplicationGraph()
    application_graph.add_next_operation(increase_pressure_op)
    application_graph.add_next_operation(laplace_filter_op)

    # Running the graph will execute teh user application on a multi-GPU system
    application_graph.run(optimization="overlapping_computation_communication", syncStream=0)



class Backend:
    """
    Backend class to handle the different devices.
    """
    type: str
    ids: list[int]
    device_codes: list[str]

    def __init__(self, type: str = 'cpu', ids: list[int] = [0]):
        self.type = type
        self.ids = ids
        self.device_codes = []
        if type not in ['cpu', 'cuda']:
            raise Exception(f"Backend type {type} not supported")
        for i, s in enumerate(self.ids):
            if self.type == 'cpu':
                self.device_codes.append(f"{self.type}")
            if self.type == 'cuda':
                self.device_codes.append(f"{self.type}:{s}")
        pass

    def for_each_device(self, foo):
        """
        Execute the function foo on each device.
        The device context for warp is automatically set
        """
        for idx, device_code in enumerate(self.device_codes):
            wp.set_device(device_code)
            foo(idx, device_code)

    def get_multigpu_memory(self, len: list[int], dtype) -> list[wp.array]:
        mem: list[wp.array] = []

        def allocate(idx, device_code):
            # WARNING - allocation size restricted to 32bit
            # QUESTION - how to allocate passing a 64bit size?
            # QUESTION - is there a lower level allocator API than wp.array?
            mem.append(wp.empty(shape=(len[idx], 0, 0), dtype=dtype, device=device_code))

        self.for_each_device(allocate)
        return mem

    def get_device_memory(self, device_code, len: int, dtype) -> wp.array:
        return wp.empty(shape=(len, 0, 0), dtype=dtype, device=device_code)

    def get_num_devices(self):
        return len(self.ids)

    def get_host_backend(self):
        host = Backend("cpu", self.ids)
        for i, s in enumerate(self.device_codes):
            s = 'cpu'
        return host


@wp.struct
class IdxDenseGrid:
    i: wp.vec3i

    def set(self, x: int, y: int, z: int):
        self.i[0] = x
        self.i[1] = y
        self.i[2] = z

    def get_x(self):
        return self.i[0]

    def get_y(self):
        return self.i[1]

    def get_z(self):
        return self.i[2]


@wp.func
def IdxDenseGrid_method_set(self: IdxDenseGrid, x: int, y: int, z: int):
    self.i[0] = x
    self.i[1] = y
    self.i[2] = z


@wp.func
def IdxDenseGrid_method_get_x(self: IdxDenseGrid):
    return self.i[0]


@wp.func
def IdxDenseGrid_method_get_y(self: IdxDenseGrid):
    return self.i[1]


@wp.func
def IdxDenseGrid_method_get_z(self: IdxDenseGrid):
    return self.i[2]


@wp.struct
class SpanDenseGrid:
    dim: wp.vec3i
    ghost_radius: int

    def set(self, x: int, y: int, z: int, ghost_radius: int):
        self.dim[0] = x
        self.dim[1] = y
        self.dim[2] = z
        self.ghost_radius = ghost_radius

    def get_x(self):
        return self.dim[0]

    def get_y(self):
        return self.dim[1]

    def get_z(self):
        return self.dim[2]

    def get_ghost_radius(self):
        return self.ghost_radius

    def get_dim(self):
        return self.dim

    def set_idx(self, x: int, y: int, z: int, idx: IdxDenseGrid):
        idx.set(x, y, z + self.ghost_radius)


@wp.func
def set_span(span: SpanDenseGrid, x: int, y: int, z: int):
    span.dim[0] = x
    span.dim[1] = y
    span.dim[2] = z
    span.ghost_radius = 1


@wp.func
def SpanDenseGrid_method_set_idx(span: SpanDenseGrid, x: int, y: int, z: int, idx: IdxDenseGrid):
    IdxDenseGrid_method_set(idx, x, y, z + span.ghost_radius)


@wp.struct
class PartitionDenseGrid_int:
    # -> QUESTION: Can we transform the following into constant types (i.e. compile time constants)?
    # pitch: wp.constant(wp.vec3i)
    # card: wp.constant(int)
    # ghost_radius: wp.constant(int)

    def set(self, bk: Backend, span: SpanDenseGrid, card: int, ghost_radius: int, device_code: str):
        self.card = card
        self.pitch[0] = span.dim[0]
        self.pitch[1] = span.dim[1] * span.dim[0]
        self.pitch[2] = (span.dim[2] + 2 * ghost_radius) * span.dim[1] * span.dim[0]
        self.memory = bk.get_device_memory(device_code, self.pitch[2] * card, wp.int32)


@wp.func
def partition_int_operator_get(p: PartitionDenseGrid_int, grid_idx: IdxDenseGrid, card: int):
    pitch = (
            IdxDenseGrid_method_get_x(grid_idx) +
            p.pitch[0] * IdxDenseGrid_method_get_y(grid_idx) +
            p.pitch[1] * IdxDenseGrid_method_get_z(grid_idx) +
            p.pitch[2] * card)
    return p.memory[pitch]


class FieldDenseGrid_int:
    def __init__(self,
                 bk: Backend,
                 span_table: dict[str, list[SpanDenseGrid]],
                 card: int,
                 ghost_radius: int):

        self.bk: Backend = bk
        self.ghost_radius: int = ghost_radius
        self.host_partition_table: dict[str, list[PartitionDenseGrid_int]] = {}
        self.device_partition_table: dict[str, list[PartitionDenseGrid_int]] = {}

        print(self.host_partition_table)
        for data_view in ['standard']:  # , 'internal', 'boundary]:
            self.host_partition_table[data_view] = []
            self.device_partition_table[data_view] = []

            def new_partition(idx, device_code):
                self.host_partition_table[data_view].append(PartitionDenseGrid_int())
                self.device_partition_table[data_view].append(PartitionDenseGrid_int())

            bk.for_each_device(new_partition)

        def set_partition(target_partition_table, target_bk: Backend):
            def set_partition(idx, device_code):
                for data_view in ['standard']:  # , 'internal', 'boundary]:
                    # partition_dim: wp.vec3i
                    # partition_dim[0] = self.dim[0]
                    # partition_dim[1] = self.dim[1]
                    # partition_dim[2] = int(self.dim[2] / len(self.bk.get_num_devices()))
                    target_span = span_table[data_view][idx]
                    target_partition_table[data_view][idx].set(bk=target_bk,
                                                               span=target_span,
                                                               card=card,
                                                               ghost_radius=self.ghost_radius,
                                                               device_code=device_code)

            return set_partition

        bk_host = self.bk.get_host_backend()
        bk_host.for_each_device(set_partition(self.host_partition_table, bk_host))
        self.bk.for_each_device(set_partition(self.host_partition_table, self.bk))

    def get_partition(self, data_view: str, device_idx: int) -> PartitionDenseGrid_int:
        return self.device_partition_table[data_view][device_idx]

    def updateHost(self, stream: int):
        """
        Move data from the host to the device buffer asynchronously.
        """
        print("TODO")
        pass

    def updateDevice(self, stream: int):
        """
        Move data from the device to the host buffer asynchronously.
        """
        print("TODO")
        pass


class GridDense:

    def __init__(self, dim: wp.vec3i, backend: Backend, ghost_radius: int = 1):
        self.dim: wp.vec3i = dim
        self.span_table: dict[str, list[SpanDenseGrid]] = {}
        self.ghost_radius: int = ghost_radius
        self.bk: Backend = backend

        for data_view in ['standard']:  # , 'internal', 'boundary]:
            self.span_table[data_view] = []

            def new_span(idx, device_code):
                self.span_table[data_view].append(SpanDenseGrid())

            self.bk.for_each_device(new_span)

        def set_span(idx, device_code):
            print('set_span')
            for data_view in ['standard']:  # , 'internal', 'boundary]:
                span_dim = wp.vec3i()
                span_dim[0] = self.dim[0]
                span_dim[1] = self.dim[1]
                span_dim[2] = int(self.dim[2] / self.bk.get_num_devices())
                self.span_table[data_view][idx].set(span_dim[0], span_dim[1], span_dim[2], ghost_radius)
                pass

        self.bk.for_each_device(set_span)
        pass

    def get_span(self, data_view: str, device_idx: int) -> SpanDenseGrid:
        return self.span_table[data_view][device_idx]

    def new_int_field(self, card: int):
        fied = FieldDenseGrid_int(self.bk, self.span_table, card, self.ghost_radius)
        return fied

    def get_backend(self):
        return self.bk


class Container:
    def __init__(self,
                 wp_kernel,
                 grid: GridDense,
                 input_set,
                 host_or_device):

        self.kernel = wp_kernel
        self.grid = grid
        self.input_set = input_set
        self.bk = grid.get_backend()

        if host_or_device == 'host':
            raise Exception(f"Work in progress")

    def run(self, data_view: str, stream: int):
        if self.bk.type == 'cpu':
            # error, through exception
            raise Exception(f"Work in progress")
            pass
        if self.bk.type == 'cuda':
            def run_kernel(idx, device_code):
                span = self.grid.get_span(data_view, idx)
                input = []
                for i in self.input_set:
                    input.append(i.get_partition('standard', idx))
                print("wp.launch")
                wp.launch(kernel=self.kernel,
                          dim=span.get_dim(),
                          inputs=[span] + input,
                          device=device_code)
                wp.synchronize()

            self.bk.for_each_device(run_kernel)


def get_kernel_add_one(grid: GridDense,
                       fieldA: FieldDenseGrid_int):
    @wp.func
    def my_foo_that_adds_one(idx: Any,
                             A: Any):
        """
        This is the function we would like the user to write.
        """
        val = partition_int_operator_get(A, idx, 0)
        # val = A(idx,0)
        print("kernel_add_one")

        # instead of this we would like the user to write:
        # val = A(idx,0)
        val = val + 1
        pass

    @wp.kernel
    def kernel_add_one(span: SpanDenseGrid,
                       partition: PartitionDenseGrid_int):
        """
        This is the function that should be generated by the runtime
        """
        x, y, z = wp.tid()
        g_idx = IdxDenseGrid()
        SpanDenseGrid_method_set_idx(span, x, y, z, g_idx)
        my_foo_that_adds_one(g_idx, partition)
        pass

    c = Container(wp_kernel=kernel_add_one,
                  grid=grid,
                  input_set=[fieldA],
                  host_or_device='device')

    # NOTE: what we would like to write is:
    # => grid.new_container(kernel_add_one, [fieldA])
    # the rest would be handled by the runtime
    # The execution would than be:
    # c.run(data_view='standard', stream=0, 'device')
    #
    # or better:
    #
    # s = Skeleton(a,b,c)
    # s.run('device', opt='overlapping_computation_communication')
    return c


def main():
    wp.init()

    devices = [0, 1]
    dim = wp.vec3i()
    dim[0] = 10
    dim[1] = 10
    # enforcing the number of z-slices to be multiple of the number of devices
    # this is a shortcut to simplify the example
    dim[2] = int(4 * len(devices))

    print(f"dim {dim}")

    bk = Backend('cuda', devices)
    grid = GridDense(dim=dim, backend=bk, ghost_radius=1)
    # we would acutally want to write something like:
    # grid = Grid(gtype='dense', dim=dim, backend=bk, ghost_radius=1, dtype=int)
    # grid = Grid(gtype='block_4_4_4', dim=dim, backend=bk, ghost_radius=1, dtype=int)
    fieldA = grid.new_int_field(card=1)
    # we would like to write something like:
    # fieldA = grid.new_field(card=1, dtype=int)
    # fieldB = grid.new_field(card=1, dtype=double)

    print(f"int_field {fieldA}")

    my_computation = get_kernel_add_one(grid, fieldA)
    my_computation.run(data_view='standard', stream=0)


if __name__ == '__main__':
    main()
