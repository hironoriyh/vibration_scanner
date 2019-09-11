import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def showplot(filenamem, saveimg=False):
    arr = np.load(filename)
    print("shape of input arr", arr.shape)

    print(arr.shape)
    start_x= 8
    end_x = 15

    print("max", np.max(arr[:, 0, start_x:end_x], axis=0)) # show max brightness for each x (y is fixed with 0)

    #brightness
    N = arr.shape[0] #samples
    dt = 1/2000
    time = np.arange(0, N*dt, dt)
    fn = 1/dt/2 # Nyquist frequency

    # plot
    plt.subplot(311)
    plt.title("50hz input vibration, 2000hz sampling", fontsize=20)
    plt.plot(time, arr[:, 0, start_x:end_x], label="brightness")
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Signal", fontsize=10)


    #frequency
    freq = np.linspace(0, 1/dt, arr.shape[0])
    # freqList = np.fft.fftfreq(N, d=1.0/4000)
    # print("freqList", freqList)
    # print("freq", freq.shape)
    # print(arr[:, 0, 2:5])
    F_xrow = []
    print(arr[:, 0].shape)
    for ind in np.arange(start_x, end_x):
        # print(x_arr.shape)
        F = np.fft.fft(arr[:, 0, ind])
        F[(freq >=fn)] = 0
        # F = np.abs(F)
        F_xrow.append(F)

    # plot
    plt.subplot(312)
    for f_y in F_xrow:
        plt.plot(freq[1:], np.abs(f_y)[1:], label="frequency")
    plt.xlabel('Frequency', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.xticks(np.arange(0, 2000, 100))


    # F1 = F_xrow[0].copy()
    # print(F1.shape)
    # y1 = np.fft.ifft(F1)*N
    #
    # print("ifft shape", y1.shape)
    # plt.subplot(313)
    # plt.plot(time, np.real(y1))

    if saveimg:
        plt.savefig("./npy/" + filename[6:-4]+"_plot.png") # -----(2)

    plt.grid()
    plt.show()
    print("saved")


def drawimgs(arr):
    for img in arr:
        # img = cv2.
        cv2.imshow("test", img)
        k = cv2.waitKey(1)
        if k == 27 :
            break
    cv2.destroyAllWindows()



if __name__ == '__main__':

    print('sys.argv         : ', sys.argv)
    filename = sys.argv[1]
    print("npy name", filename[6:-4])
    # filename="h80hz_x500_y500.npy"
    # print("video name", sys.argv[2])
    showplot(filename, False)
