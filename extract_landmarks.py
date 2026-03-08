import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.2
)

DATASET_PATH = "datsets/alphabetimages/asl_alphabet_train"
OUTPUT_PATH = "processed_data"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# letters we want to generate
TARGET_LETTERS = ["A","B","C","D","E","F","G","H","del"]


def prepare_image(image):
    # enlarge small images
    image = cv2.resize(image, (400, 400))

    # add padding to improve palm detection
    image = cv2.copyMakeBorder(
        image,
        100,100,100,100,
        cv2.BORDER_CONSTANT,
        value=[0,0,0]
    )

    return image


for label in os.listdir(DATASET_PATH):

    # process only A–H
    if label not in TARGET_LETTERS:
        print(f"Skipping {label}")
        continue

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\nProcessing class: {label}")

    save_path = os.path.join(OUTPUT_PATH, label)
    os.makedirs(save_path, exist_ok=True)

    # check if npy already exist
    existing = [f for f in os.listdir(save_path) if f.endswith(".npy")]

    if len(existing) > 0:
        print(f"{label} already processed. Skipping.")
        continue

    saved = 0

    for img_name in tqdm(os.listdir(label_path)):

        if not img_name.lower().endswith((".jpg",".png",".jpeg")):
            continue

        img_path = os.path.join(label_path, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = prepare_image(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        coords = []

        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])

        coords = np.array(coords)

        # normalization
        wrist = coords[0]
        coords = coords - wrist

        hand_size = np.linalg.norm(coords[12])

        if hand_size > 0:
            coords = coords / hand_size

        coords = coords.flatten()

        file_name = img_name.split(".")[0] + ".npy"

        np.save(os.path.join(save_path,file_name),coords)

        saved += 1

    print(f"{label} saved landmarks:", saved)

print("\nExtraction complete!")