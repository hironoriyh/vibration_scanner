import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

print('sys.argv         : ', sys.argv)
filename = sys.argv[1]
# filename = "0hz.avi"

print(filename[8:-4])
cap = cv2.VideoCapture(filename)
all_frame_nums  =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("num of frames", all_frame_nums)
frame_nums = np.arange(all_frame_nums)
pixel_time = []


#### clicked pos
mutable_object = {}
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig = plt.figure()
plt.imshow(frame)

def onclick(event):
    print('you pressed', event.key, event.xdata, event.ydata)
    X_coordinate = event.xdata
    Y_coordinate = event.ydata
    mutable_object['click'] =[ X_coordinate, Y_coordinate]


cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.savefig("./plot/" + filename[8:-4]+"_image.png")
plt.show()

clicked_pos = [int(mutable_object['click'][0]), int(mutable_object['click'][1])]
# fig = plt.figure()
# plt.imshow(frame)
# plt.plot(clicked_pos, color="red", markersize="5")
# plt.savefig("./plot/" + filename[8:-4]+"_x"+ str(clicked_pos[0])+"_y"+str(clicked_pos[1])+"_image.png")
# plt.show()

print(clicked_pos)
clicked_pos = [280, 551]

### get pixel color over time
for frame_num in frame_nums:
    print(frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # snapshot = np.zeros(frame.shape, dtype=np.uint8)
    pixel_time.append(frame[clicked_pos[0]][clicked_pos[1]])

    if not ret:
        break

    # cv2.imshow("Test", frame)
    # cv2.waitKey(1)
npy_filename = filename[8:-4] + "_x" + str(clicked_pos[0]) + "_y" + str(clicked_pos[1])
np.save("./npy/" + npy_filename, pixel_time)
cap.release()
cv2.destroyAllWindows()

my_dpi=96
plt.figure(figsize=(1000/my_dpi, 400/my_dpi), dpi=my_dpi)
plt.title(npy_filename)
pixel_time = np.array(pixel_time)
print(pixel_time.shape)
plt.plot(pixel_time[:300, 0], color="red", label="R")
plt.plot(pixel_time[:300, 1], color="green", label="G")
plt.plot(pixel_time[:300, 2], color="blue", label="B")
plt.legend()
plt.savefig("./plot/images/" + npy_filename+".png") # -----(2)
# plt.ylim(30, 80)
plt.show()



# print(pixel_time)
