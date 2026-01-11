import face_recognition
import pickle
import cv2
import json
import os
from datetime import datetime
import numpy as np

class FaceRecognizer:
    def __init__(self):
        # Load configuration
        with open('config/config.json', 'r') as f:
            self.config = json.load(f)
        
        self.detection_method = self.config["detection_method"]
        self.confidence_threshold = self.config["confidence_threshold"]
        
        # Load the trained model
        try:
            with open(self.config["recognizer_path"], "rb") as f:
                self.model_data = pickle.loads(f.read())
            
            self.model = self.model_data["model"]
            self.le = self.model_data["le"]
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
            self.le = None
        
        # Initialize attendance records
        self.attendance_records = self.load_attendance()
        self.recognized_names = set()
        
    def load_attendance(self):
        attendance_path = self.config["attendance_path"]
        if os.path.exists(attendance_path):
            with open(attendance_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_attendance(self):
        attendance_path = self.config["attendance_path"]
        os.makedirs(os.path.dirname(attendance_path), exist_ok=True)
        with open(attendance_path, 'w') as f:
            json.dump(self.attendance_records, f, indent=4)
    
    def mark_attendance(self, name):
        if name == "Unknown":
            return False
            
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if today not in self.attendance_records:
            self.attendance_records[today] = {}
        
        if name not in self.attendance_records[today]:
            self.attendance_records[today][name] = {
                "first_seen": current_time,
                "last_seen": current_time,
                "count": 1
            }
            self.save_attendance()
            print(f"[ATTENDANCE] Marked attendance for {name} at {current_time}")
            return True
        else:
            self.attendance_records[today][name]["last_seen"] = current_time
            self.attendance_records[today][name]["count"] += 1
            self.save_attendance()
            return False
    
    def recognize_faces(self, frame):
        if self.model is None:
            return [], [], []
            
        # Convert the image from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using HOG method
        boxes = face_recognition.face_locations(rgb, model=self.detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        names = []
        confidences = []
        
        # Loop over the face encodings
        for encoding in encodings:
            # Predict the face using SVM probabilities
            preds = self.model.predict_proba([encoding])[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = self.le.classes_[j]
            
            # Filter weak detections
            if proba >= self.confidence_threshold:
                names.append(name)
                confidences.append(proba)
                
                # Mark attendance (only for new recognitions in this session)
                if name not in self.recognized_names:
                    self.mark_attendance(name)
                    self.recognized_names.add(name)
            else:
                names.append("Unknown")
                confidences.append(proba)
        
        return boxes, names, confidences
    
    def draw_recognitions(self, frame, boxes, names, confidences):
        # Loop over the recognized faces
        for ((top, right, bottom, left), name, confidence) in zip(boxes, names, confidences):
            # Draw the bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            thickness = 2
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Draw the label
            label = f"{name} ({confidence * 100:.1f}%)"
            label_bg_color = color
            label_text_color = (255, 255, 255)
            
            # Calculate label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            label_left = left
            label_top = bottom - label_size[1] - 10
            
            # Ensure label stays within frame
            if label_top < top:
                label_top = top + label_size[1] + 10
            
            cv2.rectangle(frame, 
                         (label_left, label_top - label_size[1] - 10),
                         (label_left + label_size[0], label_top + 10),
                         label_bg_color, cv2.FILLED)
            
            cv2.putText(frame, label, (label_left + 6, label_top - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, label_text_color, 1)
        
        return frame

    def reset_recognized_names(self):
        self.recognized_names.clear()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    print("Face Recognizer initialized successfully")