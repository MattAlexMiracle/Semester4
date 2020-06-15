#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
# Now import the utilities from task 1.  Make sure the file imgutils.py is
# in the same folder as this file.
import imgutils


def compute_means(clusters):
    return [imgutils.color_average(x) for x in clusters]


def compute_clusters(pixels, means):
    clusters = [[] for x in range(len(means))]
    for p in pixels:
        idx = np.argmin([((np.array(m)-np.array(p[:3]))**2).sum() for m in means])
        clusters[idx].append(p)
    return clusters


def averaged_pixels(clusters, means):
    assert(len(clusters) == len(means))
    pixels = []
    for idx, x in enumerate(clusters):
        pixels.extend([(*means[idx],p[3],p[4]) for p in x])
    return pixels



def kmeans(image, k):
    pixels = imgutils.image2pixels(image)
    pixels = pixel_kmeans(pixels, k)
    return imgutils.pixels2image(pixels)


def pixel_kmeans(pixels, k):
    # Ensure that k is never bigger than the number of pixels.
    k = min(k, len(pixels))
    # Choose some initial clusters.  A better strategy would improve the
    # algorithm quite a bit, but this simple technique is usually fine, too.
    clusters = [[pixels[i]] for i in range(0, len(pixels), len(pixels) // k)][:k]
    means = compute_means(clusters)
    while True:
        assert(len(means) == k)
        assert(len(clusters) == k)
        clusters = compute_clusters(pixels, means)
        new_means = compute_means(clusters)
        if means == new_means:
            return averaged_pixels(clusters, means)
        means = new_means


def image_difference(imageA, imageB):
    """Compute the difference of two images in the maximum-norm."""
    if imageA.shape != imageB.shape:
        raise ValueError('Dimensions of both images must be equal!')
    diff = np.array(imageA,dtype='int16') - np.array(imageB,dtype='int16')
    return np.amax(abs(diff))


def main():
    shibuya = imgutils.load_image('shibuya_small.png')
    k = 64
    shibuya2 = kmeans(shibuya, k)
    plt.imshow(shibuya2/255)
    plt.show()


if __name__ == "__main__": main()