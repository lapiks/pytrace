import taichi as ti

ti.init(arch=ti.gpu)

image_width, image_height = 256, 256
pixels = ti.Vector.field(n=3, dtype=float, shape=(image_width, image_height))

@ti.kernel
def set_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([
            ti.random(),
            ti.random(),
            ti.random()
        ])

set_pixels()
filename = f'image.png'
ti.tools.imwrite(pixels.to_numpy(), filename)
print(f'The image has been saved to {filename}')