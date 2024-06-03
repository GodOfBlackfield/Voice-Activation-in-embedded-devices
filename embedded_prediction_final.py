from scipy.io.wavfile import write
import os
import librosa
import numpy as np
import tensorflow as tf
import pyaudio
import wave
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.output(16, False)
GPIO.output(17, False)
GPIO.output(22, False)
GPIO.output(23, False)
GPIO.output(24, False)
GPIO.output(27, False)

def record_audio(): 
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = 2 # seconds to record
    wav_output_filename = './prediction.wav' # name of .wav file
    audio = pyaudio.PyAudio() # create pyaudio instantiation
    dev_index = 1 # device index found by p.get_device_info_by_index(ii)
    
    for device_index in range(audio.get_device_count()):
      dev = audio.get_device_info_by_index(device_index)
      if dev["name"] == "USB PnP Sound Device: Audio (hw:1,0)":
        dev_index = dev["index"]
        break
     
    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, input_device_index = dev_index,input = True, frames_per_buffer=chunk)
    print("recording")
    frames = []
     
    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
	    data = stream.read(chunk, exception_on_overflow=False)
	    frames.append(data)
     
    print("finished recording")
     
    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()
     
    # save the audio frames as .wav file
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()
    
    return wav_output_filename

interpreter_start = tf.lite.Interpreter("./start_model_lite.tflite")
interpreter_stop = tf.lite.Interpreter("./stop_model_lite.tflite")
interpreter_left = tf.lite.Interpreter("./left_model_lite.tflite")
interpreter_right = tf.lite.Interpreter("./right_model_lite.tflite")
interpreter_forward = tf.lite.Interpreter("./forward_model_lite.tflite")
interpreter_backward = tf.lite.Interpreter("./backward_model_lite.tflite")
interpreter_start.allocate_tensors()
interpreter_stop.allocate_tensors()
interpreter_left.allocate_tensors()
interpreter_right.allocate_tensors()
interpreter_forward.allocate_tensors()
interpreter_backward.allocate_tensors()

print("success")
print("Prediction started:\n")

while True:
    print("Say now:")
    filename = record_audio()
    audio, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    input_data = np.expand_dims(mfcc_processed, axis=0)
    
    # Get Input and Output tensors
    input_details_start = interpreter_start.get_input_details()
    output_details_start = interpreter_start.get_output_details()
    input_details_stop = interpreter_stop.get_input_details()
    output_details_stop = interpreter_stop.get_output_details()
    input_details_left = interpreter_left.get_input_details()
    output_details_left = interpreter_left.get_output_details()
    input_details_right = interpreter_right.get_input_details()
    output_details_right = interpreter_right.get_output_details()
    input_details_forward = interpreter_forward.get_input_details()
    output_details_forward = interpreter_forward.get_output_details()
    input_details_backward = interpreter_backward.get_input_details()
    output_details_backward = interpreter_backward.get_output_details()
    
    # Test the model on input data
    interpreter_start.set_tensor(input_details_start[0]['index'], input_data)
    interpreter_start.invoke()
    interpreter_stop.set_tensor(input_details_stop[0]['index'], input_data)
    interpreter_stop.invoke()
    interpreter_left.set_tensor(input_details_left[0]['index'], input_data)
    interpreter_left.invoke()
    interpreter_right.set_tensor(input_details_right[0]['index'], input_data)
    interpreter_right.invoke()
    interpreter_forward.set_tensor(input_details_forward[0]['index'], input_data)
    interpreter_forward.invoke()
    interpreter_backward.set_tensor(input_details_backward[0]['index'], input_data)
    interpreter_backward.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    prediction_start = interpreter_start.get_tensor(output_details_start[0]['index'])
    prediction_stop = interpreter_stop.get_tensor(output_details_stop[0]['index'])
    prediction_left = interpreter_left.get_tensor(output_details_left[0]['index'])
    prediction_right = interpreter_right.get_tensor(output_details_right[0]['index'])
    prediction_forward = interpreter_forward.get_tensor(output_details_forward[0]['index'])
    prediction_backward = interpreter_backward.get_tensor(output_details_backward[0]['index'])
    prediction_array = [prediction_start[0][0], prediction_stop[0][0], prediction_left[0][0], prediction_right[0][0], prediction_forward[0][0], prediction_backward[0][0]]
    best_prediction = max(prediction_array)
    print(prediction_array)
    
    if best_prediction <= 0.5:
      print("Wake word NOT detected!")
    elif best_prediction == prediction_start[0][0]:
      GPIO.output(16, True)
      GPIO.output(17, False)
      GPIO.output(22, False)
      GPIO.output(23, False)
      GPIO.output(24, False)
      GPIO.output(27, False)
      print("Rollator, Start!")
    elif best_prediction == prediction_stop[0][0]:
      GPIO.output(16, False)
      GPIO.output(17, True)
      GPIO.output(22, False)
      GPIO.output(23, False)
      GPIO.output(24, False)
      GPIO.output(27, False)
      print("Rollator, Stop!")
    elif best_prediction == prediction_left[0][0]:
      GPIO.output(16, False)
      GPIO.output(17, False)
      GPIO.output(22, True)
      GPIO.output(23, False)
      GPIO.output(24, False)
      GPIO.output(27, False)
      print("Rollator, go left!")
    elif best_prediction == prediction_right[0][0]:
      GPIO.output(16, False)
      GPIO.output(17, False)
      GPIO.output(22, False)
      GPIO.output(23, True)
      GPIO.output(24, False)
      GPIO.output(27, False)
      print("Rollator, go right!")
    elif best_prediction == prediction_forward[0][0]:
      GPIO.output(16, False)
      GPIO.output(17, False)
      GPIO.output(22, False)
      GPIO.output(23, False)
      GPIO.output(24, True)
      GPIO.output(27, False)
      print("Rollator, go forward!")
    elif best_prediction == prediction_backward[0][0]:
      GPIO.output(16, False)
      GPIO.output(17, False)
      GPIO.output(22, False)
      GPIO.output(23, False)
      GPIO.output(24, False)
      GPIO.output(27, True)
      print("Rollator, go backward!")
