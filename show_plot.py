import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

print('sys.argv         : ', sys.argv)
filename = sys.argv[1]
print(filename[6:-4])
# filename="h80hz_x500_y500.npy"

arr = np.load(filename)
print(arr.shape)
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 400/my_dpi), dpi=my_dpi)
plt.title(filename[6:-4])
width_x = 1000
plt.plot(arr[:width_x, 0], color="red", label="R")
plt.plot(arr[:width_x, 1], color="green", label="G")
plt.plot(arr[:width_x, 2], color="blue", label="B")
plt.legend()
plt.ylim(10, 70)
plt.savefig("./plot/" + filename[6:-4]+".png") # -----(2)

plt.show()
