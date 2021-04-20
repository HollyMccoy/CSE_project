from record import record
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os
import shutil
import argparse
import threading

def make_classification(args, src_dir, timestamp, dir, model):
    
    wav_paths = glob('{}/**'.format(src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    
    rate, wav = downsample_mono(src_dir, args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    clean_wav = wav[mask]
    step = int(args.sr * args.dt)
    batch = []
    
    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i + step]
        sample = sample.reshape(-1, 1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)
            sample = tmp
        batch.append(sample)
        
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    time_stamp = timestamp
    print('Timestamp: {}, Predicted class: {}'.format(time_stamp, classes[y_pred]))
    # make post request

    shutil.rmtree(dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Live classifier for mic audio')
    parser.add_argument('--rec_seconds', type=int, default=20,
                        help='number of seconds to record for classification')
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

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})

    # force pre-load tf dynamic libraries

    dummy_data = []
    dummy_data.append(np.empty((16000,1)))
    dummy_batch = np.array(dummy_data, dtype=np.float32)
    model.predict(dummy_batch)

    threads = []

    print("***********Starting***********")

    while (True):
        # record x secs and store in folder
        
        t = time.localtime()
        timestamp = time.strftime("%H%M%S", t)
        dir = timestamp
        if not os.path.exists(timestamp):
            os.makedirs(timestamp)
        output = dir + "/out.wav"
        print("Timestamp: {}, Recording starting".format(timestamp))

        record(seconds=args.rec_seconds, out=output)
        UTC = str(datetime.datetime.now().strftime("%H:%M:%S"))
        # call make_classification on folder

        #for thread in threads:
            #thread.join()

        t = threading.Thread(target=make_classification, args=(args, output, timestamp, dir, model))
        threads.append(t)
        t.start()
