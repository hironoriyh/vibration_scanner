import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

print('sys.argv         : ', sys.argv)
filename = sys.argv[1]
print("npy name", filename[6:-4])
# filename="h80hz_x500_y500.npy"
# print("video name", sys.argv[2])


arr = np.load(filename)
# print(arr[:, bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
print(arr.shape)
plt.plot(arr)
# plt.ylim(0, 0.5)
# my_dpi=96
# plt.figure(figsize=(1000/my_dpi, 400/my_dpi), dpi=my_dpi)
plt.title(filename[6:-4])
# width_x = 1000
# plt.plot(arr[:width_x, 0], color="red", label="R")
# plt.plot(arr[:width_x, 1], color="green", label="G")
# plt.plot(arr[:width_x, 2], color="blue", label="B")
# plt.legend()
plt.savefig("./npy/" + filename[6:-4]+"_plot.png") # -----(2)
plt.show()
print("saved")
