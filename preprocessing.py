import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sample = "./background_sounds/72.wav"
# data,sample_rate = librosa.load(sample)

# plt.title("Wave Form")
# librosa.display.waveshow(data, sr=sample_rate, color="blue")
# plt.show()

# mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
# print("Shape of mfcc: ", mfccs.shape)

# plt.title("MFCC")
# librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
# plt.show()

all_data = []

csv_file_name = "forward_model_data"

data_path_dict = {
    0: ["./background_sounds/" + file_path for file_path in os.listdir("./background_sounds/")],
    1: ["./audio_sounds/" + file_path for file_path in os.listdir("./audio_sounds/")],
}

# headers = ["MFCC_" + str(i) for i in range(1, 41)]
# headers.append("Label")
# # print(headers)
# df = pd.DataFrame(columns=headers)
# length = 0

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        data, sample_rate = librosa.load(single_file)
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfcc_processed = np.mean(mfccs.T, axis=0)
        all_data.append([mfcc_processed, class_label])
        # mfcc_processed = np.append(mfcc_processed, class_label)
        # df.loc[length] = mfcc_processed
        # length += 1
    print(f"INFO: Successfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])
df.to_pickle("./" + csv_file_name + ".csv")
# df.to_csv("./" + csv_file_name + ".csv")