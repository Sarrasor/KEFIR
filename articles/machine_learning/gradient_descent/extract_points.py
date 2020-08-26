import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("images/test.jpg", cv2.IMREAD_GRAYSCALE)

res = np.zeros((img.shape[1], 1))

for c in range(img.shape[1]):
    for r in range(img.shape[0]):
        if img[r, c] < 127:
            print(r, c)
            res[c, 0] = r
            break
# print(res)
plt.plot(res)
plt.show()

for x, y in enumerate(res):
    print(f"{int(y[0])},")
    # print("{", end='')
    # print(f"x:{x}, y:{int(y[0])}", end='')
    # print("},")
