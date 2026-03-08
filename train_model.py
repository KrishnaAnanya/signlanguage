import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
X = np.load("X.npy")
y = np.load("y.npy")

print("Dataset shape:", X.shape)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Create RandomForest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save trained model
with open("sign_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")