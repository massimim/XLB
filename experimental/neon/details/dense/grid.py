from typing import Any

import neon as ne
import dense as neon_dense


class Grid(ne.Grid):
    def __init__(self, name, backend, dim: ne.Index):
        ne.Grid.__init__(self, name=name, dim=dim, backend=backend)

    def new_field(self,dtype, name):
        return neon_dense.Field(name=name, grid=self, layout_dim=self.dim, dtype=dtype)

    def new_container(self, name, op, parameter_list) -> ne.Container:
        c: ne.Container = ne.Container(name=name, grid=self, function=op, parameter_list=parameter_list)
        return c


def main():
    dim = ne.Index()
    dim.set(10,10,10)
    backend = ne.Backend()
    grid = neon_dense.Grid(name="grid-test", backend=backend, dim=dim)
    fieldA = grid.new_field(dtype=int, name="field-test")

    def myOp(idx: Any, a: Any) -> None:
        a[idx] = 33
        return

    opContainer = grid.new_container(name="container-test", op=myOp, parameter_list=[fieldA])
    print(str(type(fieldA)))
    pass


if __name__ == '__main__':
    main()
