import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import wavio
import time

for i in range(60):

    t = time.localtime()
    timestamp = time.strftime("%H%M%S", t)
    out = "wavfiles/quiet_room/"+timestamp+".wav"
    seconds = 20
    sample_rate=16000
    print(out)
    myrecording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1)  #
    sd.wait()  # Wait until recording is finished#
    #write(out, sample_rate, myrecording)  # Save as WAV file#

    wavio.write(out,myrecording, sample_rate, sampwidth=2)#

    print(out)
