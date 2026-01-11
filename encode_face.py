import face_recognition
import pickle
import cv2
import os
import json
from imutils import paths

def encode_faces():
    # Load configuration
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    dataset_path = config["dataset_path"]
    encodings_path = config["encodings_path"]
    detection_method = config["detection_method"]
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
    
    # Get the paths to the images in dataset
    image_paths = list(paths.list_images(dataset_path))
    
    if len(image_paths) == 0:
        print("[ERROR] No images found in dataset directory")
        return
    
    known_encodings = []
    known_names = []
    
    # Loop over the image paths
    for (i, image_path) in enumerate(image_paths):
        print(f"[INFO] Processing image {i+1}/{len(image_paths)}")
        
        # Extract the person name from the image path
        name = image_path.split(os.path.sep)[-2].split('_')[0]
        
        # Load the input image and convert it from BGR to RGB
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not load image: {image_path}")
            continue
            
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image using HOG
        boxes = face_recognition.face_locations(rgb, model=detection_method)
        
        if len(boxes) == 0:
            print(f"[WARNING] No faces detected in {image_path}")
            continue
            
        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        # Loop over the encodings
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)
    
    # Save the encodings and names
    print("[INFO] Saving encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    
    with open(encodings_path, "wb") as f:
        f.write(pickle.dumps(data))
    
    print(f"[SUCCESS] Encoding completed! Total faces encoded: {len(known_names)}")
    print(f"[INFO] Encodings saved to: {encodings_path}")

if __name__ == "__main__":
    encode_faces()