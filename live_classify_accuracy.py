# for testing accuracy of model and prints out % of all predicted sounds
# default number of tests = 30
# custom number of tests:
# ex: python live_classify_accuracy.py --test_length=50 <-- for 50 tests

from record import record
import time, os, shutil, argparse, sys, logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
import datetime

# prevents retracing errors from printing
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
    y_pred = model.predict(X_batch)
#    y_pred = predict(model, X_batch) # retracing warning fix attempt
    y_mean = np.mean(y_pred, axis=0)#
    y_pred = np.argmax(y_mean)#
    time_stamp = timestamp#
    
    if test_num > 0:
        print(f'-Predicted class: {classes[y_pred]}')#
        predicts.append(classes[y_pred])

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
    parser.add_argument('--test_length', type=int, default=31,
                        help='total number of tests per run')
    args, _ = parser.parse_known_args()
    
    # test_length = 31 by default to test 30 times, first one is ignored for loading tensorflow
    num = args.test_length 

    sound_classes = sorted(os.listdir(args.src_dir))
    predicts = []
    test_num = 0
    classify_times = []

    while(test_num < num):
        # 1. record 5 secs and store in a folder

        t = time.localtime()
        timestamp = time.strftime("%H%M%S", t)
        dir = timestamp
        if not os.path.exists(timestamp):
            os.makedirs(timestamp)  # make a directory
        output = dir + "/out.wav"
        if test_num > 0:
            print(f"\nTest #{test_num}\nRecording starting {(output)}")

        record(seconds=5, out=output)
        print("Done.")
        # 2. call make_classification on this folder
        
        start_classify = datetime.datetime.now()
        make_classification(args, output, timestamp)
        end_classify = datetime.datetime.now()
        diff = end_classify - start_classify
        time_classify = diff.total_seconds()

        if test_num > 0:
            print(f"-classification time: {round(time_classify,2)} seconds")
            classify_times.append(time_classify)

        # 3. delete directory

        shutil.rmtree(dir)

        test_num+=1


    # accuracy info
    print(50*'=')
    print("Prediction statistics: ")
    correct_stats = []
    for x in sound_classes:
        for y in predicts:
            if x == y:
                correct_stats.append(1)
            elif x != y:
                correct_stats.append(0)
        stats = 100 * sum(correct_stats)/len(predicts)
        if stats > 0:
            print(f"{x}: {round(stats,2)}%")
        correct_stats.clear()
    time_avg = sum(classify_times)/len(classify_times)
    print(50*'=')
    print(f"Classify time average: {round(time_avg,2)} seconds, min: {round(min(classify_times),2)}s, max: {round(max(classify_times),2)}s")
