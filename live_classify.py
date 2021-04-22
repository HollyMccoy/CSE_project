from record import record
import datetime
import os
import shutil
import argparse
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
from db_api import post_db
import logging

# prevents tensorflow retracing errors from displaying on console
logging.getLogger('tensorflow').disabled = True

def make_classification(args, src_dir, start_time, db_time):

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

    for i in range(0, clean_wav.shape[0], step):#
        sample = clean_wav[i:i + step]#
        sample = sample.reshape(-1, 1)#
        if sample.shape[0] < step:#
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)#
            tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)#
            sample = tmp#
        batch.append(sample)#

    # print(batch)#
    X_batch = np.array(batch, dtype=np.float32)#
    y_pred = model.predict(X_batch)#
    y_mean = np.mean(y_pred, axis=0)#
    y_pred = np.argmax(y_mean)#
    time_stamp = timestamp#
    #print('-Prediction: {}'.format(classes[y_pred]))#
    end_time = datetime.datetime.now()
    diff = end_time - start_time
    total_time = diff.total_seconds()
    print(f"> Prediction: {classes[y_pred]}")
    # print(f"> Prediction: {classes[y_pred]} ({round(total_time,2)} seconds)")
    
    # make post request
    start_post = str(datetime.datetime.now().strftime("%H:%M:%S"))
    #print(f"> Post start time: {start_post}")
    post_db(classes[y_pred], "DEVICE ID", db_time)   # replace "DEVICE ID" with desired name


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

    while (True):
        
        # 1. record 5 secs and store in a folder
        
        timestamp = str(datetime.datetime.now().strftime("%H:%M:%S"))
        dir = timestamp
        
        if not os.path.exists(timestamp):
            os.makedirs(timestamp)      # make a directory
        output = dir + "/out.wav"
        print(f"\nRecording starting ({output})")
        record(seconds=5, out=output)
        print("Done.")

        # 2. call make_classification on this folder
        start_time = datetime.datetime.now()
        db_time = str(datetime.datetime.now().strftime("%H:%M:%S"))
        make_classification(args, output, start_time, db_time)

        
        # 3. delete directory
        
        shutil.rmtree(dir)
