from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm


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
    print(batch)#
    X_batch = np.array(batch, dtype=np.float32)#
    y_pred = model.predict(X_batch)#
    y_mean = np.mean(y_pred, axis=0)#
    y_pred = np.argmax(y_mean)#
    time_stamp = timestamp#
    print('\nTimestamp: {}, Predicted class: {}'.format(time_stamp, classes[y_pred]))#
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
