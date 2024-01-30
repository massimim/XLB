class MultiGpuData(object):
    def __init__(self, id, name, backend):
        self.id = id
        self.backend = backend
        self.name = name

    def getId(self):
        return self.id
