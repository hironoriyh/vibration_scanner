import argparse
import cv2
import numpy as np
import sys


def lukaskanade(cap):

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)


    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

def harneback(cap):
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next

def nofilter(cap, filename, framecount=500, spatial_filter=False, init_diff=True, diff_num=5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_frame = cap.read()[1]
    if(spatial_filter):
         first_frame = cv2.medianBlur(first_frame, 5)
         # first_frame = cv2.fastNlMeansDenoisingColored(first_frame,None,10,10,7,21)
    # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    cap.set(cv2.CAP_PROP_POS_FRAMES, framecount)

    while True:
        if(init_diff==0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, framecount)
            first_frame = cap.read()[1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, framecount+diff_num)

        frame = cap.read()[1]
        if(spatial_filter):
             frame = cv2.medianBlur(frame, 5)
             #frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21) #realy slow!

        frame = cv2.absdiff(frame, first_frame)*5
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "FrameCount : " + str(framecount), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        elif(k == ord("s") or framecount == 900):
            cv2.imwrite("diff_images/subtract_%s_%i.png" %(filename, framecount), frame)
            print("image saved", "subtract_%s_%i.png" %(filename, framecount))
        framecount += 1

def temporalfilter(cap, filename):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_frames = [cap.read()[1] for i in range(5)]
    # first_frames = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.uint8) for i in first_frames]
    first_frame = np.median(first_frames, axis=0).astype(np.uint8)
    framecount = 500
    cap.set(cv2.CAP_PROP_POS_FRAMES, framecount)

    while True:
        imgs = [cap.read()[1] for i in range(3)]
        # imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.uint8) for i in imgs]
        frame = np.median(imgs, axis=0).astype(np.uint8) #or frame = np.average(imgs, axis=0)
        frame = cv2.subtract(frame, first_frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "FrameCount : " + str(framecount), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        elif (k == ord("s") or framecount == 900):
            cv2.imwrite("diff_images/temporal_filter_%s_%i.png" %(filename, framecount), frame)
            print("image saved")
        framecount += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--spatial_filter', type=int, default=0)
    parser.add_argument('--init_diff', type=int, default=1)
    args = parser.parse_args()

    print('sys.argv         : ', args.file)
    filename = args.file
    print("filename", filename[8:-4])
    cap = cv2.VideoCapture(filename)
    filename =filename[8:-4]

    nofilter(cap, filename, 500, args.spatial_filter, args.init_diff, 5)
    # temporalfilter(cap, filename)
    # lukaskanade(cap)
    # harneback(cap)

    cv2.destroyAllWindows()
    cap.release()
