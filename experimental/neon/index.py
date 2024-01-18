import warp as wp
import numpy as np


@wp.struct
class Index:
    x: int
    y: int
    z: int

    def __init__(self, a=0, b=0, c=0):
        self.x = a
        self.y = b
        self.z = c

    def set(self, a=0, b=0, c=0):
        self.x = a
        self.y = b
        self.z = c

    def getArry(self):
        return np.array([self.x, self.y, self.z])

    def reduce_mul(self):
        return self.x * self.y * self.z
