import neon as ne
import types
import numpy as np


class Grid(object):
    def __init__(self,
                 name: str,
                 dim: ne.Index,
                 backend: ne.Backend):
        self.name = name
        self.dim: ne.Index = dim
        self.backend = backend

    def get_backend(self):
        return self.backend

    def get_dim(self):
        return self.dim

    def get_name(self):
        return self.name
