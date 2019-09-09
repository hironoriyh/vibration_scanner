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
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # cap = cv2.VideoCapture(0)

    ## writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # writer = cv2.VideoWriter('outpy.avi', -1, 20.0, (frame_width,frame_height))


    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # frame = frame_resize(frame)
        bbox = (0,0,10,10)
        bbox = cv2.selectROI(frame, False)
        ok = tracker.init(frame, bbox)
        cv2.destroyAllWindows()
        break

    framecount = 0
    positions = []
    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()

        # frame = frame_resize(frame)
        if not ret:
            k = cv2.waitKey(1)
            if k == 27 :
                break
            continue

        # Start timer
        timer = cv2.getTickCount()

        # トラッカーをアップデートする
        track, bbox = tracker.update(frame)
        positions.append(bbox)

        # FPSを計算する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame - first_frame
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 検出した場所に四角を書く
        if track:
            # Tracking success
            # print(bbox)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 1, 1)
        else :
            # トラッキングが外れたら警告を表示する
            cv2.putText(frame, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # FPSを表示する
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(frame, "BBOX : " + str(int(bbox[0])), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(frame, "FrameCount : " + str(framecount) + " , " + str(all_frame_nums), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        for pos in positions:
            center = [pos[0]+pos[2]/2, pos[1]+pos[3]/2]
            cv2.circle(frame, (int(center[0]), int(center[1])), 2, (0,255,255), -1)
        # 加工済の画像を表示する
        cv2.imshow("Tracking", frame)
        # writer.write(frame)
        framecount += 1

        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27 :
            break

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
