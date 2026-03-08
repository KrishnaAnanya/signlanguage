import os
import numpy as np

DATA_PATH = "processed_data"

X = []
y = []

for label in os.listdir(DATA_PATH):

    label_path = os.path.join(DATA_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):

        if file.endswith(".npy"):

            data = np.load(os.path.join(label_path, file))

            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# SAVE DATASET
np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset saved successfully")