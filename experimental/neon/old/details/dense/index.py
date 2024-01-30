import warp as wp
import numpy as np

class Index:
    def __init__(self, a: int = 0, b: int = 0, c: int = 0):
        self.x: int = a
        self.y: int = b
        self.z: int = c

    def get_array(self):
        return np.array([self.x, self.y, self.z])
