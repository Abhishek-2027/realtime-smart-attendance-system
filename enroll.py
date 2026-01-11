import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import time
import threading
import json
import face_recognition
from PIL import Image, ImageTk

class FaceEnrollment:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Enrollment")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        # Load configuration
        self.config_path = "config/config.json"
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.stop_event = threading.Event()
        self.cap = None
        self.face_count = 0
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Enrollment", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Person ID
        id_frame = ttk.Frame(main_frame)
        id_frame.pack(fill='x', pady=8)
        ttk.Label(id_frame, text="Person ID:", width=12).pack(side=tk.LEFT)
        self.id_entry = ttk.Entry(id_frame, width=25)
        self.id_entry.pack(side=tk.LEFT, padx=5)
        
        # Person Name  
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill='x', pady=8)
        ttk.Label(name_frame, text="Person Name:", width=12).pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(name_frame, width=25)
        self.name_entry.pack(side=tk.LEFT, padx=5)
        
        # Config Path
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill='x', pady=8)
        ttk.Label(config_frame, text="Config Path:", width=12).pack(side=tk.LEFT)
        ttk.Label(config_frame, text="config/config.json", foreground="blue").pack(side=tk.LEFT, padx=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate', length=360)
        self.progress.pack(fill='x', pady=15)
        self.progress['value'] = 0
        
        # Progress percentage
        self.progress_var = tk.StringVar(value="0%")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.pack()
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=15)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.enroll_btn = ttk.Button(button_frame, text="Enroll", 
                                    command=self.start_enrollment, width=15)
        self.enroll_btn.pack(side=tk.LEFT, padx=8)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Enrollment", 
                                  command=self.stop_enrollment, state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        
        self.reset_btn = ttk.Button(button_frame, text="Reset", 
                                   command=self.reset_enrollment, width=15)
        self.reset_btn.pack(side=tk.LEFT, padx=8)
        
        # Status
        self.status_var = tk.StringVar(value="Ready for enrollment")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="green")
        status_label.pack(pady=5)
        
    def start_enrollment(self):
        person_id = self.id_entry.get().strip()
        person_name = self.name_entry.get().strip()
        
        if not person_id or not person_name:
            messagebox.showerror("Error", "Please enter both Person ID and Person Name")
            return
            
        # Create dataset directory
        if not os.path.exists(self.config["dataset_path"]):
            os.makedirs(self.config["dataset_path"])
            
        # Create user directory
        user_dir = os.path.join(self.config["dataset_path"], f"{person_name}_{person_id}")
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        self.stop_event.clear()
        self.face_count = 0
        self.enroll_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.DISABLED)
        
        # Start enrollment thread
        thread = threading.Thread(target=self.capture_faces, 
                                 args=(user_dir, person_name, person_id))
        thread.daemon = True
        thread.start()
        
    def stop_enrollment(self):
        self.stop_event.set()
        self.status_var.set("Enrollment stopped")
        self.enroll_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)
        
    def reset_enrollment(self):
        self.id_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)
        self.progress['value'] = 0
        self.progress_var.set("0%")
        self.status_var.set("Ready for enrollment")
        
    def capture_faces(self, user_dir, person_name, person_id):
        try:
            self.cap = cv2.VideoCapture(self.config["camera_index"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])
            
            total_faces = self.config["face_count"]
            detection_model = self.config["detection_method"]
            
            self.status_var.set("Starting face detection...")
            
            while self.face_count < total_faces and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if ret:
                    # Convert to RGB for face detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces using HOG method
                    face_locations = face_recognition.face_locations(rgb_frame, model=detection_model)
                    
                    if len(face_locations) > 0:
                        # Save image with face
                        img_path = os.path.join(user_dir, f"{person_name}_{self.face_count:02d}.jpg")
                        cv2.imwrite(img_path, frame)
                        self.face_count += 1
                        
                        # Update progress
                        progress = (self.face_count / total_faces) * 100
                        self.progress['value'] = progress
                        self.progress_var.set(f"{int(progress)}%")
                        self.status_var.set(f"Captured {self.face_count}/{total_faces} faces")
                    
                    time.sleep(self.config["capture_delay"])
                    
                else:
                    break
                    
            self.cap.release()
            
            if not self.stop_event.is_set():
                # Update enrollment database
                self.update_enrollment_db(person_id, person_name, user_dir)
                self.status_var.set(f"Enrollment completed for {person_name}")
                messagebox.showinfo("Success", f"Successfully enrolled {person_name} with {self.face_count} faces!")
                
            self.enroll_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Enrollment failed: {str(e)}")
            if self.cap:
                self.cap.release()
                
    def update_enrollment_db(self, person_id, person_name, user_dir):
        # Create database directory if not exists
        db_dir = os.path.dirname(self.config["db_path"])
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        # Load existing database or create new
        if os.path.exists(self.config["db_path"]):
            with open(self.config["db_path"], 'r') as f:
                enroll_db = json.load(f)
        else:
            enroll_db = []
            
        # Add new enrollment
        enrollment = {
            "id": person_id,
            "name": person_name,
            "folder": os.path.basename(user_dir),
            "class": self.config["class"],
            "enrollment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "face_count": self.face_count,
            "dataset_path": user_dir
        }
        
        # Check if ID already exists
        for i, record in enumerate(enroll_db):
            if record["id"] == person_id:
                enroll_db[i] = enrollment  # Update existing
                break
        else:
            enroll_db.append(enrollment)  # Add new
            
        # Save database
        with open(self.config["db_path"], 'w') as f:
            json.dump(enroll_db, f, indent=4)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEnrollment(root)
    root.mainloop()