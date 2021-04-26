# Nielsen Ambient Noise Sound Classifier for Attentiveness
CSE senior capstone project with Nielsen


To install all necessary requirements, refer to the commandline folder and follow these steps:

1. Install librosa (librosa_install in commandline folder)

2. Install requirements (pip3 install -r requirements.txt)

3. Install tensorflow (tensorflow-installation in commandline folder)

If you have issues installing numba, try the following commands:

git clone https://github.com/wjakob/tbb.git

cd tbb/build

cmake ..

make -j

sudo make install

cd ~ 

pip3 install numba

4. Install mic libraries (mic_install.txt in commandline folder)


List of sound categories:
- Bird noises
- Cat meowing
- Cutlery
- Dog barking
- Doorbell
- Keyboard typing
- Laughter
- Speech
- Vacuuming


If you want to collect quiet room samples specific to your environment, collect_quiet_room.py can be used. This is a script that will collect quiet audio of your specific environment so that the model classifies your room as quiet when no sounds are being made. Once the samples are created, add the audio clips to their own folder (ex: quiet) and place the new folder in your wavfiles folder for them to be cleaned with the other data sets.

python3 collect_quiet_room.py

The clean.py file is used to clean the .WAV files stored in the wavfiles folder and prepare them to be used to train the model.
After running this file, a clean folder will be created containing the newly cleaned files.

python3 clean.py

![image](https://user-images.githubusercontent.com/57106938/115637362-fc578b80-a2dd-11eb-8342-b0e03bb309ef.png)

Once the wavfiles are cleaned, train.py can be used to train a model with 3 options: lstm, 1D convolutional and 2D convolutional.

python3 train.py

![train_model](https://user-images.githubusercontent.com/57106938/115637424-1f823b00-a2de-11eb-91cb-8b8946dc0920.png)

After the model is trained, live audio can be captured by live_classify.py and classify sounds in real time. live_classify_accuracy.py has the same functionality but provides additional metrics (average time to make classification and accuracy of each class) for data purposes and only runs through 30 tests before terminating. record.py provides the functionality for the capturing of live audio and is included in the live classify programs. db_api.py is also included in live_classify.py and sends the sound classifications to the DynamoDB database.

python3 live_classify.py

![classes](https://user-images.githubusercontent.com/57106938/115637444-28730c80-a2de-11eb-8426-2e9809e959b0.png)

NOTE: Everytime the terminal is restarted, the TLS block will need to be set. This command can be found in the commandline folder.

If more datasets are needed, the augment_data.py script can be used to modify existing wav files to increase the noise levels, shift time, increase or decrease the pitch, and/or speed up or slow down the audio.

Current accuracy by class of 2D convolutional model:

![Screen Shot 2021-04-21 at 5 31 05 PM](https://user-images.githubusercontent.com/57106938/115623596-64e73e00-a2c7-11eb-860c-f99ac6913e6c.png)

Overall accuracy: 86%

This specific model can be found in the models folder.


The lambda code to set up both databases (classification and correlation) can be found in the awsLambda folder.
