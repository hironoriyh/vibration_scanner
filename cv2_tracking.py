import cv2
import sys
import numpy as np

def frame_resize(frame, n=2):
    """
    スクリーンショットを撮りたい関係で1/4サイズに縮小
    """
    return cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

# def bounding_box():


if __name__ == '__main__':
    """
    Tracking手法を選ぶ。適当にコメントアウトして実行する。
    """

    print('sys.argv         : ', sys.argv)
    filename = sys.argv[1]
    print("filename", filename[8:-4])
    # Boosting
    # tracker = cv2.TrackerBoosting_create()

    # MIL
    # tracker = cv2.TrackerMIL_create()

    # KCF
    tracker = cv2.TrackerKCF_create()

    # TLD #GPUコンパイラのエラーが出ているっぽい
    # tracker = cv2.TrackerTLD_create()

    # MedianFlow
    # tracker = cv2.TrackerMedianFlow_create()

    # GOTURN # モデルが無いよって怒られた
    # https://github.com/opencv/opencv_contrib/issues/941#issuecomment-343384500
    # https://github.com/Auron-X/GOTURN-Example
    # http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
    # tracker = cv2.TrackerGOTURN_create()

    cap = cv2.VideoCapture(filename)
    all_frame_nums  =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = (0,0,10,10)
        bbox = cv2.selectROI(frame, False)
        ok = tracker.init(frame, bbox)
        cv2.destroyAllWindows()
        break

    framecount = 0
    positions = []
    frames = []
    prev_frame = first_frame

    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        frames.append(frame)
        if len(frames) > 10:
            frames.pop(0)

        if not ret:
            k = cv2.waitKey(1)
            if k == 27 :
                break
            continue

        # Start timer
        timer = cv2.getTickCount()

        # トラッカーをアップデートする
        track, bbox = tracker.update(frame)
        position = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2]
        positions.append(position)

        # FPSを計算する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # frame = np.median(frames, axis=0).astype(np.uint8)
        diff = cv2.subtract(frame, frames[0])*10
        # diff = cv2.absdiff(frame, prev_frame)*5

        # diff = cv2.subtract(frame, first_frame)*10

        # 検出した場所に四角を書く
        if track:
            # Tracking success
            # print(bbox)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(diff, p1, p2, (0,255,0), 1, 1)
        else :
            # トラッキングが外れたら警告を表示する
            cv2.putText(diff, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # FPSを表示する
        cv2.putText(diff, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(diff, "BBOX : " + str(int(bbox[0])), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(diff, "FrameCount : " + str(framecount) + " , " + str(all_frame_nums), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        for pos in positions:
            cv2.circle(diff, (int(pos[0]), int(pos[1])), 2, (0,255,255), -1)
        # 加工済の画像を表示する
        cv2.imshow("Tracking", diff)
        # writer.write(diff)
        framecount += 1

        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27 :
            break
        elif k == ord('s') or framecount is 900:
            np.save("./npy/tracking_" + filename[8:-4], positions)

        prev_frame = frame

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()
