import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from keras.models import load_model

fs = 44100
seconds = 2
filename = "./prediction.wav"
class_names = ["Wake word NOT detected", "Wake word detected"]

model_start = load_model("./start_model.keras")
model_left = load_model("./left_model.keras")
model_right = load_model("./right_model.keras")
model_forward = load_model("./forward_model.keras")
model_backward = load_model("./backward_model.keras")
model_stop = load_model("./stop_model.keras")
print("Prediction started:\n")

while True:
    print("Say now:")
    recording = sd.rec(int(fs * seconds), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, recording)
    audio, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    data = np.expand_dims(mfcc_processed, axis=0)
    prediction_start = model_start.predict(data)
    prediction_stop = model_stop.predict(data)
    prediction_left = model_left.predict(data)
    prediction_right = model_right.predict(data)
    prediction_forward = model_forward.predict(data)
    prediction_backward = model_backward.predict(data)
    prediction_array = [prediction_start[0][0], prediction_stop[0][0], prediction_right[0][0], prediction_left[0][0], prediction_forward[0][0], prediction_backward[0][0]]
    best_prediction = max(prediction_array)
    # print(prediction_array)
    if best_prediction <= 0.5:
        print("Wake word NOT detected!")
    elif best_prediction == prediction_start:
        print("Start the Rollator!")
    elif best_prediction == prediction_stop:
        print("Stop the Rollator!")
    elif best_prediction == prediction_left:
        print("Rollator, go left!")
    elif best_prediction == prediction_right:
        print("Rollator, go right!")
    elif best_prediction == prediction_forward:
        print("Rollator, go forward!")
    elif best_prediction == prediction_backward:
        print("Rollator, go backward!")