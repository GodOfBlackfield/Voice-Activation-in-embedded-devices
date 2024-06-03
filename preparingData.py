import sounddevice as sd
from scipy.io.wavfile import write

def record_audio_and_save(save_path, n_times=100):
    input("To record audio press Enter:")
    for i in range(n_times):
        fs = 44100
        seconds = 2
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, recording)
        input(f"Press to record next or to stop press Ctrl + C ({i + 1}/{n_times})")

def record_background_audio(save_path, n_times=100):
    input("To record background sounds press Enter:")
    for i in range(n_times):
        fs = 44100
        seconds = 2
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, recording)
        print(f"Currently on ({i + 1}/{n_times})")

print("Recording the wake word: ")
record_audio_and_save("./audio_sounds/")

print("Recording the background sounds: ")
record_background_audio("./background_sounds/")