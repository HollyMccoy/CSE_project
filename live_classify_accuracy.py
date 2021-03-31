# use for running live_classify for specific # of tests
# starts checking for accuracy after everything is loaded in the beginning
# command line: python live_classify.py {"class name"} {number of iterations}
# make sure class name is in quotation marks and matches exactly with the folder names

# from classify import make_classification
from record import record
import time
import os
import shutil
import argparse
import sys

import tensorflow as tf
import logging
# from original classify.py
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm

actual = sys.argv[1]
num = int(sys.argv[2])
correct = []
test_num = 0

# prevents retracing warnings from printing to console
logging.getLogger('tensorflow').disabled = True

def make_classification(args, src_dir, timestamp):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)

    rate, wav = downsample_mono(src_dir, args.sr)#
    mask, env = envelope(wav, rate, threshold=args.threshold)#
    clean_wav = wav[mask]#
    step = int(args.sr * args.dt)#
    batch = []#
#
    for i in range(0, clean_wav.shape[0], step):#
        sample = clean_wav[i:i + step]#
        sample = sample.reshape(-1, 1)#
        if sample.shape[0] < step:#
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)#
            tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)#
            sample = tmp#
        batch.append(sample)#
#    print(batch)#
    X_batch = np.array(batch, dtype=np.float32)#
    y_pred = model.predict(X_batch)#
    y_mean = np.mean(y_pred, axis=0)#
    y_pred = np.argmax(y_mean)#
    time_stamp = timestamp#
    
    if test_num > 0:
        print('Timestamp: {}, Predicted class: {}'.format(time_stamp, classes[y_pred]))#
        if classes[y_pred] == actual:
            correct.append(1)
        else:
            correct.append(0)

    # make post request
    # for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
    #     rate, wav = downsample_mono(wav_fn, args.sr)
    #     mask, env = envelope(wav, rate, threshold=args.threshold)
    #     clean_wav = wav[mask]
    #     step = int(args.sr*args.dt)
    #     batch = []

    #     for i in range(0, clean_wav.shape[0], step):
    #         sample = clean_wav[i:i+step]
    #         sample = sample.reshape(-1, 1)
    #         if sample.shape[0] < step:
    #             tmp = np.zeros(shape=(step, 1), dtype=np.float32)
    #             tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
    #             sample = tmp
    #         batch.append(sample)
    #     print(batch)
    #     X_batch = np.array(batch, dtype=np.float32)
    #     y_pred = model.predict(X_batch)
    #     y_mean = np.mean(y_pred, axis=0)
    #     y_pred = np.argmax(y_mean)
    #     time_stamp = timestamp
    #     print('\nTimestamp: {}, Predicted class: {}'.format(time_stamp, classes[y_pred]))
    #     # make post request


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/conv2d.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    # while (True):
    while(test_num < num):
        # 1. record 5 secs and store in a folder

        t = time.localtime()
        timestamp = time.strftime("%H%M%S", t)
        dir = timestamp
        if not os.path.exists(timestamp):
            os.makedirs(timestamp)  # make a directory
        output = dir + "/out.wav"
        
        if test_num > 0:
            print(f"\nTest #{test_num}\nRecording starting ({output})")

        record(seconds=5, out=output)
        print("Done.")
        
        # 2. call make_classification on this folder

        make_classification(args, output, timestamp)

        # 3. delete directory

        shutil.rmtree(dir)
        
        test_num+=1

print('n', 50*'-')
print(f"Total accuracy from test #1-{test_num - 1}: {100 * sum(correct)/len(correct)}%")
