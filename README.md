Author: Sonesh Kumar Swain

This project's aim is to make accurate machine learning models for embedded devices (raspberry pi in our case) that help with
voice activation of the smart rollator that we're working on.

First and foremost, Python must be installed in the system you will be working on, preferably v3.11.9. Then below is the list of
packages you must install for the making of the machine learning models:-

numpy<br>
pandas<br>
scikit-learn<br>
tensorflow (v2.15.0 only)<br>
keras (v2.15.0 only)<br>
sounddevice<br>
librosa<br>
matplotlib<br>
scikeras (**)<br>

** This particular package has version issues with keras, as it works with keras >= v3.2.0 only. Please install this after you've
made the tensorflow-lite models, as anything greater than keras v.2.15.0 will give errors while converting to tensorflow lite.

Then, you have to go and download the google commands dataset for audio processing, the link for which is given below:-
https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Once you've finished with the above instructions, you can now view the python files in here along with the models and datasets
present here.

=> The files with the '.csv' extensions represent the audio files processed into numbers to be fed as inputs into the machine 
learning models.

=> The files with the '.keras' extensions represent the ML models saved as files. These models can be run on machines with proper
CPUs like laptops, PCs, etc.

=> Then the files with the '.tflite' extensions represent the tensorflow-lite versions of the ML models we made. These are made
to be run on embedded devices like Raspberry Pi 5.

=> Then finally come the cpp header files ('.h'), which represent the ML models in tinyML format. These are made to be run on even
smaller devices like Arduino Nano ESP32.

Now comes the main view - the python files. These files are the backbone of this project, as they will essentially just process,
make, train and test the ML models for you.

=> preparing_data.py - This file is used to generate audio files for processing when the standard dataset does not have the particular
word for training or does not have enough of the particular word for training. Also, the best use of this file is actually for background
noise generation.

=> preprocessing.py - This file is used to process the audio stored in the two folders - wakeword_sounds & background_sounds - which according
to their names store exactly what they mention. The audio is labeled as 0 & 1 - 0 for background and 1 for wakeword. The file extracts a
particular feature of the audio, called Mel Frequency Cepstrom Coefficients or MFCC, which is basically a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. In simpler words,
they are coefficients of the discrete cosine transform of the Fourier transform of the sound wave. It is calculated as follow:-
    ==> Take the Fourier transform of (a windowed excerpt of) a signal
    ==> Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows or alternatively, cosine overlapping windows.
    ==> Take the logs of the powers at each of the mel frequencies.
    ==> Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
    ==> The MFCCs are the amplitudes of the resulting spectrum.
The internet can give better ways to calculate the MFCCs but again, the file does it for us. Now, we get the numbers from the audio, and we
pack it into a binary-encoded csv file.

=> training.py - The main file, and it basically trains the model we want. First we extract the data from the dataset file and store it as 'x'
for the MFCC values, and 'y' for the labels. Then we make a Sequential model with an input layer of 40 neurons (40 MFCC coefficients - this can differ from audio to audio), an optional hidden layer of 27 neurons (thumb rule of hidden layers - at max can contain (2/3 * previous layer neurons) + (neurons in output layer)) and an output layer with a single neuron. We use the Dense layer from keras here and it is recommended we use this one. Since the dataset is quite small, we have practically no hidden layers, otherwise we'll be overfitting the model. But if you encounter a dataset whose length goes more than 15000, then one hidden layer is recommended and the greater the dataset, the more the number of hidden layers. We use the 'relu' activation function for the input and hidden layers, and it stands for Rectified Linear Unit, and it is actually a non-linear function used to activate the layers in deep neural networks. Then, we use the sigmoid function as activation function for the output layer, and it basically maps all output in the range [0, 1). Then we use binary cross entropy as the loss function for the model. These settings are all recommended for binary classification problems. Then we get the score and the .keras model. The commented parts after the model generation are essentially used for accurately scoring the model. It is highly recommended to do that on a separate system as it requires keras >= v.3.2.0, which will hinder the tensorflow-lite model creation.

=> prediction.py - This file is used to test the model and it checks the accuracy by measuring the output against a threshold, which can be set according to the size of the dataset and the score obtained during training.

=> embedded_prediction.py - This file is the blueprint of the file to be used in the Raspberry Pi for prediction and it is an ensemble of all the ML models, where the wake word is compared with all the models at once to distinguish between the correct wake word.

=> embedded_prediction_final.py - This file is the one that will be running on the Raspberry Pi on its startup, using the Tensorflow-Lite models to classify the wake word when it's detected using the USB Microphone that will be attached to it. It will send output signals through the GPIO header pins depending on which wake word was detected.
** If this file immediately crashes on startup of the Raspberry Pi, it's most likely due to the instantaneous changing of the index of the USB port which is outside my area of expertise. As of now, there is a question reagrding this on teh stackoverflow community, but it hasn't been answered yet. Please update the index in this file if you get the answer.

=> rename.py - This files renames the audio files in the two folders to numbers from 0-(total no of audio files - 1)

These are all the files in here and if you feel that these are too easy and basic for you, please make changes to them to suit your needs and make the models more accurate.

P.S., this is the diagram of the Raspberry Pi's GPIO headers and which ones you should be using for which model:-

    -------------------------------------------
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                           GPIO 17 - ··  |
    |                           GPIO 27 - ··  |
    |                           GPIO 22 - ··  | - GPIO 23
    |                                     ··  | - GPIO 24
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  |
    |                                     ··  | - GPIO 16
    |                                     ··  |
    |                                     ··  |
    |                                         |
    -------------------------------------------

Pin 16 - Start<br>
Pin 17 - Stop<br>
Pin 22 - Left<br>
Pin 23 - Right<br>
Pin 24 - Forward<br>
Pin 27 - Backward<br>
