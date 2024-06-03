import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

model_name = "forward_model"
csv_file_name = model_name + "_data"
df = pd.read_pickle("./" + csv_file_name + ".csv")
x = df["feature"].values
x = np.concatenate(x, axis=0).reshape(len(x), 40)
y = np.array(df["class_label"].tolist())
# df = pd.read_csv("./" + csv_file_name +".csv")
# dataset = df.values
# x = dataset[:,1:41].astype(float)
# y = dataset[:,41].astype(int)
# print(x[0])
# print(len(y))

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train)

# def create_model():
#     model = Sequential([
#         Dense(40, input_shape=(40,), activation='relu'),
#         Dense(1, activation="sigmoid")
#     ])
#     print(model.summary())
#     model.compile(
#         loss="binary_crossentropy",
#         optimizer="adam",
#         metrics=["accuracy"]
#     )
#     return model

# model = create_model()
model = Sequential([
    Dense(40, input_shape=x[0].shape, activation='relu'),
    # Dense(27, activation='relu'),
    Dense(1, activation="sigmoid")
])
print(model.summary())
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
print("Model Score:\n")
history = model.fit(x, y, epochs=100)
model.save("./" + model_name + ".keras")
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(model=create_model, epochs=100, batch_size=10, verbose=1)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(pipeline, x, y, cv=kfold)
# print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# print("Model Classification Report:\n")
# y_pred = np.argmax(model.predict(x_test), axis=1)
# cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
# print(classification_report(np.argmax(y_test, axis=1), y_pred))
# plotted_cm = ConfusionMatrixDisplay(cm, display_labels=["Does not contain Wake word", "Contains wake word"])
# plotted_cm.plot()
# plt.show()
