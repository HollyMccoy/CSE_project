import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import wavio


def record(seconds=20, out="out.wav", sample_rate=16000, channels=1):  #
    #
    myrecording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)  #
    sd.wait()  # Wait until recording is finished#
    # write(out, sample_rate, myrecording)  # Save as WAV file#


    T = 3  # sample duration (seconds)
    f = 441.0  # sound frequency (Hz)#
    t = np.linspace(0, T, T * sample_rate, endpoint=False)#
    x = np.sin(2 * np.pi * f * t)#
#
    wavio.write(out,x, sample_rate, sampwidth=2)#

    # print(out)
### import subprocess
### import os
###
###
### def record(seconds=0, out="out.wav", sample_rate=16000, channels=2):
###     proc_args = ["arecord", "-D", "hw:1,0", "-f", "S16_LE", "-r", str(sample_rate), "-c", str(channels), out]
###
###     if (seconds > 0):
###         proc_args.append("-d")
###         proc_args.append(str(seconds))
###
###     proc = subprocess.Popen(proc_args, shell=False, preexec_fn=os.setsid)
###
###     print("Starting recording... pid =" + str(proc.pid))
###
###     if (seconds > 0):
###         proc.wait()
###         print("Recording done")
