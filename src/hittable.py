import taichi as ti
from ray import Ray
from vec3 import *

@ti.dataclass
class Sphere:
    center: Point3
    radius: float

    @ti.func
    def hit(self, ray: Ray, tmin: float, tmax: float):
        oc = ray.origin - self.center
        a = ray.direction.norm_sqr()
        half_b = oc.dot(ray.direction)
        c = oc.norm_sqr() - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        hit = discriminant >= 0.0
        root = -1.0
        if hit:
            sqrtd = ti.sqrt(discriminant)
            root = (-half_b - sqrtd) / a

            if root <= tmin or root >= tmax:
                root = (-half_b + sqrtd) / a
                if root <= tmin or root >= tmax:
                    hit = False

        return hit, root
    
    @ti.func
    def normal_at(self, point: Point3):
        return 0.5 * ((point - self.center).normalized() + 1.0)