import kmeans
import imgutils
import matplotlib.pyplot as plt
import time

# for image in ['rgb.png', 'shibuya.png', 'lena.png', 'shibuya_small.png']:

startt = time.time()
for image in ['shibuya_small.png']:
    print(image)
    image = imgutils.load_image(image)
    for k in range(6):
        if k <= 0: continue
        k = 2 ** k
        print(k)
        shibuya2 = kmeans.kmeans(image, k)
        plt.imshow(shibuya2/255)
        plt.show()
print(time.time() - startt)
