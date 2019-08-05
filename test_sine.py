
import pyaudio
import numpy as np
import sys

print('sys.argv         : ', sys.argv)

p = pyaudio.PyAudio()

volume = 0.8     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 5.0   # in seconds, may be float
# f = 100.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
f_arr = np.arange(10) + 45

# print(f_arr)

f = float(sys.argv[1])
print(f , "Hz")
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
stream.write(volume*samples)

# for f in f_arr:
#     print(f , "Hz")
#     samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
#
#     # for paFloat32 sample values must be in range [-1.0, 1.0]
#     stream = p.open(format=pyaudio.paFloat32,
#                     channels=1,
#                     rate=fs,
#                     output=True)
#
#     # play. May repeat with different volume values (if done interactively)
#     stream.write(volume*samples)

# stream.stop_stream()
stream.close()

p.terminate()
