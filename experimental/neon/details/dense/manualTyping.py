import warp as wp


class Backend:
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
    pitch: wp.vec3i
    card: int
    ghost_radius: int
    memory: wp.array(dtype=int)

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
    dim: wp.vec3i
    ghost_radius: int
    bk: Backend
    host_partition_table: dict[str, list[PartitionDenseGrid_int]]
    device_partition_table: dict[str, list[PartitionDenseGrid_int]]

    def __init__(self,
                 bk: Backend,
                 span_table: dict[str, list[SpanDenseGrid]],
                 card: int,
                 ghost_radius: int):

        self.bk = bk
        self.ghost_radius = ghost_radius
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
    dim: wp.vec3i
    span_table: dict[str, list[SpanDenseGrid]]
    bk: Backend
    ghost_radius: int

    def __init__(self, dim: wp.vec3i, backend: Backend, ghost_radius: int = 1):
        self.dim = dim
        self.span_table = {}
        self.ghost_radius = ghost_radius
        self.bk = backend

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

    def run(self, stream: int):
        if self.bk.type == 'cpu':
            # error, through exception
            raise Exception(f"Work in progress")
            pass
        if self.bk.type == 'cuda':
            def run_kernel(idx, device_code):
                span = self.grid.get_span('standard', idx)
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
    @wp.kernel
    def kernel_add_one(span: SpanDenseGrid,
                       partition: PartitionDenseGrid_int):
        x, y, z = wp.tid()
        print("kernel_add_one")
        g_idx = IdxDenseGrid()
        SpanDenseGrid_method_set_idx(span, x, y, z, g_idx)
        val = partition_int_operator_get(partition, g_idx, 0)
        val = val + 1
        pass

    c = Container(wp_kernel=kernel_add_one,
                  grid=grid,
                  input_set=[fieldA],
                  host_or_device='device')

    return c


def main():
    wp.init()

    devices = [0, 1]
    dim = wp.vec3i()
    dim[0] = 10
    dim[1] = 10
    dim[2] = int(4 * len(devices))

    print(f"dim {dim}")

    bk = Backend('cuda', devices)
    grid = GridDense(dim=dim, backend=bk, ghost_radius=1)
    fieldA = grid.new_int_field(card=1)
    print(f"int_field {fieldA}")

    c = get_kernel_add_one(grid, fieldA)
    c.run(0)


if __name__ == '__main__':
    main()
