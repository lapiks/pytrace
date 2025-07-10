import taichi.math as tm
from typing import TypeAlias

Vec3 = tm.vec3
Point3 = tm.vec3

ZERO = Vec3(0.0, 0.0, 0.0)
X = Vec3(1.0, 0.0, 0.0)
X_NEG = Vec3(-1.0, 0.0, 0.0)
Y = Vec3(0.0, 1.0, 0.0)
Y_NEG = Vec3(0.0, -1.0, 0.0)
Z = Vec3(0.0, 0.0, 1.0)
Z_NEG = Vec3(0.0, 0.0, -1.0)