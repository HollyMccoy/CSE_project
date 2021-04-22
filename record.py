import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import wavio


def record(seconds=20, out="out.wav", sample_rate=16000, channels=2):  #

    myrecording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=2)  #
    sd.wait()  # Wait until recording is finished#

    wavio.write(out,myrecording, sample_rate, sampwidth=2)#


