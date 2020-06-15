#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    """Load the specified image as uint8 Numpy array."""
    return np.uint8(plt.imread(path) * 255)


def image2pixels(image):
    """Create a list of (R, G, B, X, Y) tuples from a given 3d image array."""
    return [(p[0],p[1],p[2], idx,idy) for idy,y in enumerate(image) for idx,p in enumerate(y)]
    pass # TODO


def pixels2image(pixels):
    """Create a 3d image array from a list of (R, G, B, X, Y) pixels."""
    sizey = max(pixels, key=lambda x: x[4])[4]
    sizex = max(pixels, key=lambda x: x[3])[3]
    img = np.zeros((sizey+1,sizex+1,3), dtype=np.uint8)
    for x in pixels:
        img[x[4],x[3]] = x[:3]
    return img


def bounding_box(pixels):
    """Return a tuple ((Rmin, Rmax), (Gmin, Gmax), (Bmin, Bmax)) with the
    respective minimum and maximum values of each color."""
    Rmin, Rmax = 255,0
    Gmin, Gmax = 255,0
    Bmin, Bmax = 255,0
    for x in pixels:
        Rmin = min(Rmin, x[0])
        Gmin = min(Gmin, x[1])
        Bmin = min(Bmin, x[2])
        Rmax = max(Rmax, x[0])
        Gmax = max(Gmax, x[1])
        Bmax = max(Bmax, x[2])
    return ((Rmin, Rmax), (Gmin, Gmax), (Bmin, Bmax))


def color_average(pixels):
    """Return list of tuples (Ravg, Gavg, Bavg) with averaged color values."""
    sums = [round(sum(x[j] for x in pixels)/len(pixels)) for j in range(3)] 
    return tuple(map(round, sums))

def main():
    shibuya = load_image('shibuya.png')
    pixels = image2pixels(shibuya)
    print(bounding_box(pixels))
    print(color_average(pixels))
    print(color_average([(3, 1, 253, 0, 2),
  (0, 2, 254, 1, 2),
  (0, 1, 255, 2, 2),
  (253, 1, 0, 0, 0),
  (254, 1, 0, 1, 0),
  (255, 1, 0, 2, 0)]))
    plt.imshow(pixels2image(pixels)/255)
    plt.show()


if __name__ == "__main__": main()