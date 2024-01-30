import neon as ne


class Field(ne.MultiGpuData):
    def __init__(self, id, name, grid: ne.Grid):
        ne.MultiGpuData.__init__(self, id=id, name=name, backend=grid.get_backend())
        self.grid = grid

        def get_backend(self):
            return self.grid.get_backend()

        def get_grid(self):
            return self.grid
