import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from keras.models import load_model

fs = 44100
seconds = 2
filename = "./prediction.wav"
class_names = ["Wake word NOT detected", "Wake word detected"]

model_name = "left_model"
model = load_model("./" + model_name + ".keras")
print("Prediction started:\n")
i = 0
while True:
    print("Say now:")
    recording = sd.rec(int(fs * seconds), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, recording)
    audio, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    print(prediction)
    if prediction[0][0] > 0.8:
        print(f"Wake word HAS been detected! ({i})")
        i += 1
    else:
        print(f"Wake word NOT detected! ({i})")