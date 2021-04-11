from scipy.io.wavfile import read
from scipy.io.wavfile import write
import numpy as np
import librosa
import argparse
import os

def add_noise(data, noise_factor=100):
    if (noise_factor == 0):
        return data
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def add_time_shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif self.shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def add_pitch_mod(data, sampling_rate, pitch_factor=0.9):
    if (pitch_factor == 0):
        return data
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def add_speed_mod(data, speed_factor=0.9):
    if (speed_factor == 0):
        return data
    return librosa.effects.time_stretch(data, speed_factor)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Audio data augmenter')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory of audio to augment')
    parser.add_argument('--trgt_dir', type=str, default='synthdata',
                        help='directory to add augmented output data to')
    parser.add_argument('--en_noise', type=bool, default=False,
                        help='enable noise injection')
    parser.add_argument('--en_time_shift', type=bool, default=False,
                        help='enable time shifting')
    parser.add_argument('--en_pitch_mod', type=bool, default=False,
                        help='enable pitch modulation')
    parser.add_argument('--en_speed_mod', type=bool, default=True,
                        help='enable speed modulation')
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.trgt_dir):
        os.makedirs(args.trgt_dir)

    print("Adding: ")

    if (args.en_noise):
        print("-noise")

    if (args.en_time_shift):
        print("-time shift")

    if (args.en_pitch_mod):
        print("-pitch mod")

    if (args.en_speed_mod):
        print("-speed mod")

    # apply enabled filters
    for directory in os.listdir(args.src_dir):
        if not os.path.exists(args.trgt_dir + "/" + directory):
            os.makedirs(args.trgt_dir + "/" + directory)
        for filename in os.listdir(args.src_dir + "/" + directory):
            # load file into np array
            rate, a = read(args.src_dir + "/" + directory + "/" + filename)        
            data = np.array(a, dtype=float)

            if (args.en_noise):
                data = add_noise(data)

            if (args.en_time_shift):
                data = add_time_shift(data, sampling_rate=rate, shift_max=5, shift_direction='right')

            if (args.en_pitch_mod):
                data = add_pitch_mod(data, rate)

            if (args.en_speed_mod):
                data = add_speed_mod(data)

            # write file
            write(args.trgt_dir + "/" + directory + "/" + "aug-" + filename, rate, data.astype(np.int16))

    print("Wrote augmented data to " + args.trgt_dir)
