from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import os

def train_model():
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    encodings_path = config["encodings_path"]
    recognizer_path = config["recognizer_path"]
    le_path = config["le_path"]
    training_size = config["training_size"]
    
    # Check if encodings exist
    if not os.path.exists(encodings_path):
        print("[ERROR] Encodings file not found. Run encode_faces.py first.")
        return
    
    # Load the face encodings
    print("[INFO] Loading encodings...")
    with open(encodings_path, "rb") as f:
        data = pickle.loads(f.read())
    
    if len(data["encodings"]) == 0:
        print("[ERROR] No encodings found in file.")
        return
    
    # Encode the labels
    print("[INFO] Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    
    # Split the data into training and testing sets
    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["encodings"], labels, test_size=(1-training_size), random_state=42, stratify=labels
    )
    
    # Train the SVM model
    print("[INFO] Training SVM classifier...")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[SUCCESS] Model accuracy: {accuracy * 100:.2f}%")
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save the trained model and label encoder
    print("[INFO] Saving model...")
    model_data = {
        "model": model,
        "le": le,
        "classes": le.classes_.tolist(),
        "accuracy": accuracy
    }
    
    with open(recognizer_path, "wb") as f:
        f.write(pickle.dumps(model_data))
    
    # Save label encoder separately
    with open(le_path, "wb") as f:
        f.write(pickle.dumps(le))
    
    print(f"[SUCCESS] Model training completed!")
    print(f"[INFO] Model saved to: {recognizer_path}")
    print(f"[INFO] Label encoder saved to: {le_path}")

if __name__ == "__main__":
    train_model()