from time import time
from vec3 import *
from color import *
from ray import *
from hittable import Sphere
import math
import taichi as ti

ti.init(arch=ti.gpu)

aspect_ratio = 16.0 / 9.0
image_width = 1920
image_height = int(image_width / aspect_ratio)
image_height = 1 if image_height < 1 else image_height

pixels = ti.Vector.field(n=3, dtype=float, shape=(image_width, image_height))

focal_length = 1.0
viewport_height = 2.0
viewport_width = viewport_height * (image_width / image_height)
camera_center = ZERO

viewport_u = Vec3(viewport_width, 0.0, 0.0)
viewport_v = Vec3(0.0, viewport_height, 0.0)

pixel_delta_u = viewport_u / image_width
pixel_delta_v = viewport_v / image_height

viewport_bottom_left = camera_center - Vec3(0.0, 0.0, focal_length) - viewport_u/2.0 - viewport_v/2.0
pixel00_loc = viewport_bottom_left + (pixel_delta_u + pixel_delta_v) * 0.5

@ti.func
def background(r: Ray):
    unit_direction = r.direction.normalized()
    # [-1, 1] -> [0, 1]
    t = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - t) * WHITE + t * Color(0.5, 0.7, 1.0)

@ti.func
def ray_color(r: Ray):
    color = BLACK
    sphere = Sphere(center=Vec3(0.0, 0.0, -1.0), radius=0.5)
    hit, t = sphere.hit(r, 0.0, 100.0)
    if hit:
        color = sphere.normal_at(r.at(t))
    else:
        color = background(r)
    return color

@ti.kernel
def set_pixels():
    for i, j in pixels:
        pixel_center = pixel00_loc + pixel_delta_u * i + pixel_delta_v * j
        ray_direction = pixel_center - camera_center

        r = Ray(origin=camera_center, direction=ray_direction)

        pixel_color = ray_color(r)
        pixels[i, j] = pixel_color

t = time()

print(f"Start rendering")

set_pixels()
filename = f"image.png"
ti.tools.imwrite(pixels.to_numpy(), filename)

print(f"Rendering finished in {time() - t:.2f} seconds")
print(f"The image has been saved to {filename}")