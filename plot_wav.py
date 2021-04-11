import matplotlib.pyplot as plt
import numpy as np
import argparse
import wave
import sys

# example: python3 plot_wav.py --file='wavfiles/Bird Noises/4221.wav'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='wav file plotter')
    parser.add_argument('--file', type=str, default='wavfiles/Bird Noises/4221.wav',
                        help='wav file to plot')
    args, _ = parser.parse_known_args()

    spf = wave.open(args.file, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    fs = spf.getframerate()

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)


    Time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(Time, signal)
    plt.show()
