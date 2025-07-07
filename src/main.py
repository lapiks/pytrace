import taichi as ti

ti.init()

pixels = ti.field(ti.u8, shape=(512, 512, 3))

@ti.kernel
def set_pixels():
    for i, j, k in pixels:
        pixels[i, j, k] = ti.random() * 255

set_pixels()
filename = f'image.png'
ti.tools.imwrite(pixels.to_numpy(), filename)
print(f'The image has been saved to {filename}')