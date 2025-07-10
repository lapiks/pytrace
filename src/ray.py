import taichi as ti
from vec3 import *

@ti.dataclass
class Ray:
    origin: Vec3
    direction: Vec3

    @ti.func
    def at(self, t: float):
        return self.origin + t * self.direction
