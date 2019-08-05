import cv2

vid = cv2.VideoCapture("h80hz.avi")

count = 0
# while True:
#     vid.grab()
#
#     retval, image = vid.retrieve()
#
#     if not retval:
#         break
#
#
#     cv2.imshow("Test", image)
#     cv2.waitKey(1)


while(vid.isOpened()):
    if(count%100==0):

        ret, frame = vid.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)


        print(count)

    else:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
    count += 1

vid.release()
cv2.destroyAllWindows()
