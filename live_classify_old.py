from classify import make_classification
from record import record
import time
import os
import shutil

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    while (True)
        # 1. record 5 secs and store in a folder
        
        t = time.localtime()
        timestamp = time.strftime("%H%M%S", t)
        dir = timestamp
        if not os.path.exists(timestamp):
            os.makedirs(timestamp)      # make a directory
        print("Recording starting ({})".format(output))
        output = dir + "/out.wav"

        record(seconds=12, out=output)  # for some reason "12" is needed to get 5 seconds
    
        # 2. call make_classification on this folder

        make_classification(args, dir, timestamp)

        # 3. delete directory
        
        shutil.rmtree(dir)
