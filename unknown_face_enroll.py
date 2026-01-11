import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import time
import threading
import json
import face_recognition
from datetime import datetime
from PIL import Image, ImageTk

class UnknownFaceEnroll:
    def __init__(self, root):
        self.root = root
        self.root.title("Unknown Face Enrollment")
        self.root.geometry("500x400")
        
        # Load configuration
        with open('config/config.json', 'r') as f:
            self.config = json.load(f)
        
        self.stop_event = threading.Event()
        self.cap = None
        self.face_count = 0
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Unknown Face Enrollment", 
                               font=('Arial', 14, 'bold'), foreground="red")
        title_label.pack(pady=10)
        
        # Info label
        info_label = ttk.Label(main_frame, 
                              text="This module enrolls unknown faces detected during recognition",
                              foreground="blue", wraplength=400)
        info_label.pack(pady=5)
        
        # Unknown folder path
        path_frame = ttk.Frame(main_frame)
        path_frame.pack(fill='x', pady=10)
        ttk.Label(path_frame, text="Unknown Faces Folder:").pack(side=tk.LEFT)
        unknown_path = os.path.join(self.config["dataset_path"], "unknown")
        ttk.Label(path_frame, text=unknown_path, foreground="green").pack(side=tk.LEFT, padx=5)
        
        # Auto ID generation
        id_frame = ttk.Frame(main_frame)
        id_frame.pack(fill='x', pady=8)
        ttk.Label(id_frame, text="Auto-generated ID:").pack(side=tk.LEFT)
        self.id_var = tk.StringVar(value=f"UNK_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ttk.Label(id_frame, textvariable=self.id_var, foreground="purple").pack(side=tk.LEFT, padx=5)
        
        # Name entry
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill='x', pady=8)
        ttk.Label(name_frame, text="Enter Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(name_frame, width=25)
        self.name_entry.pack(side=tk.LEFT, padx=5)
        self.name_entry.insert(0, "Unknown_Person")
        
        # Progress
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.pack(fill='x', pady=15)
        self.progress['value'] = 0
        
        self.progress_var = tk.StringVar(value="0%")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.pack()
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=15)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Capture", 
                                   command=self.start_capture, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=8)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Capture", 
                                  command=self.stop_capture, state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        
        self.enroll_btn = ttk.Button(button_frame, text="Enroll Unknown", 
                                    command=self.enroll_unknown, width=15)
        self.enroll_btn.pack(side=tk.LEFT, padx=8)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to capture unknown faces")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="green")
        status_label.pack(pady=5)
        
        # Captured faces counter
        self.counter_var = tk.StringVar(value="Faces captured: 0")
        counter_label = ttk.Label(main_frame, textvariable=self.counter_var)
        counter_label.pack(pady=5)
        
        self.captured_faces = []
        
    def start_capture(self):
        person_name = self.name_entry.get().strip()
        if not person_name:
            messagebox.showerror("Error", "Please enter a name for the unknown person")
            return
            
        self.stop_event.clear()
        self.captured_faces = []
        self.face_count = 0
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.enroll_btn.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.capture_unknown_faces)
        thread.daemon = True
        thread.start()
        
    def stop_capture(self):
        self.stop_event.set()
        self.status_var.set(f"Capture stopped. Captured {len(self.captured_faces)} faces")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if len(self.captured_faces) > 0:
            self.enroll_btn.config(state=tk.NORMAL)
        
    def capture_unknown_faces(self):
        try:
            self.cap = cv2.VideoCapture(self.config["camera_index"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])
            
            detection_model = self.config["detection_method"]
            max_faces = 20  # Maximum unknown faces to capture
            
            self.status_var.set("Capturing unknown faces...")
            
            while len(self.captured_faces) < max_faces and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_frame, model=detection_model)
                    
                    if len(face_locations) > 0:
                        # Save the first face found
                        face_location = face_locations[0]
                        self.captured_faces.append(frame.copy())
                        self.face_count = len(self.captured_faces)
                        
                        # Update progress
                        progress = (self.face_count / max_faces) * 100
                        self.progress['value'] = progress
                        self.progress_var.set(f"{int(progress)}%")
                        self.counter_var.set(f"Faces captured: {self.face_count}")
                        self.status_var.set(f"Captured {self.face_count}/{max_faces} unknown faces")
                    
                    time.sleep(0.2)  # Slower capture for unknown faces
                    
                else:
                    break
                    
            self.cap.release()
            
            if not self.stop_event.is_set():
                self.status_var.set(f"Capture completed: {len(self.captured_faces)} faces")
                if len(self.captured_faces) > 0:
                    self.enroll_btn.config(state=tk.NORMAL)
                    
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Capture failed: {str(e)}")
            if self.cap:
                self.cap.release()
                
    def enroll_unknown(self):
        if len(self.captured_faces) == 0:
            messagebox.showwarning("Warning", "No faces captured to enroll")
            return
            
        person_name = self.name_entry.get().strip()
        person_id = self.id_var.get()
        
        if not person_name:
            messagebox.showerror("Error", "Please enter a valid name")
            return
            
        # Create user directory
        user_dir = os.path.join(self.config["dataset_path"], f"{person_name}_{person_id}")
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        # Save captured faces
        saved_count = 0
        for i, face_img in enumerate(self.captured_faces):
            try:
                img_path = os.path.join(user_dir, f"{person_name}_{i:02d}.jpg")
                cv2.imwrite(img_path, face_img)
                saved_count += 1
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                
        # Update enrollment database
        self.update_enrollment_db(person_id, person_name, user_dir, saved_count)
        
        messagebox.showinfo("Success", 
                          f"Successfully enrolled {person_name}!\n"
                          f"ID: {person_id}\n"
                          f"Faces saved: {saved_count}")
        
        self.reset_ui()
        
    def update_enrollment_db(self, person_id, person_name, user_dir, face_count):
        db_dir = os.path.dirname(self.config["db_path"])
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        if os.path.exists(self.config["db_path"]):
            with open(self.config["db_path"], 'r') as f:
                enroll_db = json.load(f)
        else:
            enroll_db = []
            
        enrollment = {
            "id": person_id,
            "name": person_name,
            "folder": os.path.basename(user_dir),
            "class": "UNKNOWN_AUTO",
            "enrollment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "face_count": face_count,
            "dataset_path": user_dir,
            "auto_enrolled": True
        }
        
        enroll_db.append(enrollment)
            
        with open(self.config["db_path"], 'w') as f:
            json.dump(enroll_db, f, indent=4)
            
    def reset_ui(self):
        self.progress['value'] = 0
        self.progress_var.set("0%")
        self.counter_var.set("Faces captured: 0")
        self.status_var.set("Ready to capture unknown faces")
        self.id_var.set(f"UNK_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, "Unknown_Person")
        self.captured_faces = []
        self.enroll_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = UnknownFaceEnroll(root)
    root.mainloop()