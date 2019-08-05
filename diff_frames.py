import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("h80hz.avi")

all_frame_nums  =  cap.get(cv2.CAP_PROP_FRAME_COUNT)

print("num of frames", all_frame_nums)


count = 0
frame_nums = np.arange(1000)
# while True:
pixel_time = []


mutable_object = {}

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
fig= plt.imshow(frame)
fig = plt.figure()
def onclick(event):
    print('you pressed', event.key, event.xdata, event.ydata)
    X_coordinate = event.xdata
    Y_coordinate = event.ydata
    mutable_object['click'] = X_coordinate

cid = fig.canvas.mpl_connect('button_press_event', onclick)
lines, = plt.plot([1,2,3])
plt.show()
X_coordinate = mutable_object['click']
print(X_coordinate)

for frame_num in frame_nums:
    # cap.grab()
    print(frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


    # ret, frame = cap.retrieve()
    ret, frame = cap.read()
    snapshot = np.zeros(frame.shape, dtype=np.uint8)
    # colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
    # print(frame[500][500])
    pixel_time.append(frame[500][500])

    if not ret:
        break


    # cv2.imshow("Test", frame)
    # cv2.waitKey(1)

print(pixel_time)

# while(cap.isOpened()):
#     if(count%100==0):
#
#         ret, frame = cap.read()
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         cv2.imshow('frame',gray)
#
#
#         print(count)
#
#     else:
#         continue
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     count += 1

cap.release()
cv2.destroyAllWindows()
