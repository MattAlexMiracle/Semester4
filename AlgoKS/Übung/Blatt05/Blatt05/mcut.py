#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
# Now import the utilities from task 1.  Make sure the file imgutils.py is
# in the same folder as this file.
import imgutils


def cut_dimension(bbox):
    """Determine the longest side of the bounding box, i.e. return 0, 1 or 2. In case of a draw, prefer
    the lower dimensions."""
    return np.argmax([x[1]-x[0] for x in bbox])


def recursive_median_cut(pixels, N, bbox=False):
    if len(pixels) < 2:
        return pixels
    if N == 0:
        (ravg, gavg, bavg) = imgutils.color_average(pixels)
        return [(ravg, gavg, bavg, pixel[3], pixel[4]) for pixel in pixels]
    if not bbox:
        bbox = imgutils.bounding_box(pixels)
    dim = cut_dimension(bbox)
    pixels.sort(key=lambda x: x[dim])
    return recursive_median_cut(pixels[:len(pixels)//2], N-1)+recursive_median_cut(pixels[len(pixels)//2:], N-1)

def median_cut(image, ncuts=8):
    """Perform the median cut algorithm on a given image."""
    pixels = imgutils.image2pixels(image)
    pixels = recursive_median_cut(pixels, ncuts)
    return imgutils.pixels2image(pixels)


def image_difference(imageA, imageB):
    """Compute the difference of two images in the maximum-norm."""
    if imageA.shape != imageB.shape:
        raise ValueError('Dimensions of both images must be equal!')
    diff = np.array(imageA,dtype='int16') - np.array(imageB,dtype='int16')
    return np.amax(abs(diff))


def main():
    shibuya = imgutils.load_image('shibuya.png')
    lena    = imgutils.load_image('lena.png')
    # median cut quantisation with 2^N colors
    N = 5
    shibuya2 = median_cut(shibuya, N)
    lena2 = median_cut(lena, N)
    print("shibuya 2^{} colors: {}".format(N, image_difference(shibuya, shibuya2)))
    print("lena    2^{} colors: {}".format(N, image_difference(lena, lena2)))
    plt.imshow(lena2/255)
    plt.show()
    plt.imshow(shibuya2/255)
    plt.show()
    lena3 = median_cut(lena, 20)
    print("lena3    2^{} colors: {}".format(N, image_difference(lena, lena3)))
    plt.imshow(lena3/255)
    plt.show()


if __name__ == "__main__": main()
