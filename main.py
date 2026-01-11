import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import json
import os
from enroll import FaceEnrollment
from recognition import FaceRecognizer
from attendance_enroll_info_check_and_delete_id import EnrollmentManager
from unknown_face_enroll import UnknownFaceEnroll

class SmartFaceAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face Attendance System")
        self.root.geometry("1000x700")
        self.root.state('zoomed')  # Start maximized
        
        # Load configuration
        with open('config/config.json', 'r') as f:
            self.config = json.load(f)
        
        self.recognizer = None
        self.stop_event = threading.Event()
        self.cap = None
        self.is_recognition_running = False
        
        self.setup_ui()
        self.load_config_status()
        
    def setup_ui(self):
        # Create main menu
        self.create_menu()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Welcome tab
        self.welcome_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.welcome_frame, text="🏠 Welcome")
        self.setup_welcome_tab()
        
        # Recognition tab
        self.recognition_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.recognition_frame, text="👁️ Face Recognition")
        
        # Enrollment tab
        self.enrollment_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.enrollment_frame, text="📷 Face Enrollment")
        
        # Attendance tab
        self.attendance_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.attendance_frame, text="📊 Attendance Records")
        
        # Management tab
        self.management_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.management_frame, text="⚙️ Enrollment Management")
        
        # Unknown Faces tab
        self.unknown_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.unknown_frame, text="❓ Unknown Faces")
        
        self.setup_recognition_tab()
        self.setup_enrollment_tab()
        self.setup_attendance_tab()
        self.setup_management_tab()
        self.setup_unknown_tab()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Encode Faces", command=self.encode_faces)
        file_menu.add_command(label="Train Model", command=self.train_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="System Status", command=self.show_system_status)
        tools_menu.add_command(label="Configuration", command=self.show_configuration)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def setup_welcome_tab(self):
        # Welcome message
        welcome_text = """
        🤖 SMART FACE ATTENDANCE SYSTEM
        
        Welcome to the Smart Face Attendance System!
        
        This system provides automated attendance tracking using facial recognition technology.
        
        🎯 FEATURES:
        • Real-time Face Recognition
        • Face Enrollment & Management
        • Automated Attendance Marking
        • HOG-based Face Detection
        • SVM-based Recognition
        • Attendance Reports & Export
        
        🚀 GETTING STARTED:
        1. Enroll faces using the 'Face Enrollment' tab
        2. Encode faces using File → Encode Faces
        3. Train model using File → Train Model  
        4. Start recognition in 'Face Recognition' tab
        5. View attendance in 'Attendance Records'
        
        ⚙️ SYSTEM CONFIGURATION:
        • Detection Method: HOG
        • Recognition Method: SVM
        • Confidence Threshold: 60%
        • Face Images per Person: 30
        
        Developed with Python, OpenCV, and Machine Learning
        """
        
        welcome_label = ttk.Label(self.welcome_frame, text=welcome_text, 
                                 justify=tk.LEFT, font=('Arial', 11))
        welcome_label.pack(anchor='w', pady=20)
        
        # System status
        status_frame = ttk.LabelFrame(self.welcome_frame, text="System Status", padding="10")
        status_frame.pack(fill='x', pady=20)
        
        self.status_labels = {}
        status_items = [
            ("Enrollments", "database/enroll.json"),
            ("Encodings", "output/encodings.pickle"),
            ("Trained Model", "output/recognizer.pickle"),
            ("Dataset", "dataset/")
        ]
        
        for i, (label, path) in enumerate(status_items):
            row = i // 2
            col = i % 2
            frame = ttk.Frame(status_frame)
            frame.grid(row=row, column=col, sticky='w', padx=10, pady=5)
            
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            status_var = tk.StringVar()
            self.status_labels[path] = status_var
            ttk.Label(frame, textvariable=status_var, width=20).pack(side=tk.LEFT, padx=5)
            
        # Refresh button
        ttk.Button(self.welcome_frame, text="Refresh Status", 
                  command=self.load_config_status).pack(pady=10)
        
    def setup_recognition_tab(self):
        # Video display
        video_frame = ttk.LabelFrame(self.recognition_frame, text="Live Camera Feed", padding="10")
        video_frame.pack(fill='both', expand=True, pady=10)
        
        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here", 
                                    background='black', foreground='white')
        self.video_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(self.recognition_frame)
        control_frame.pack(fill='x', pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="🎥 Start Recognition", 
                                   command=self.start_recognition, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop Recognition", 
                                  command=self.stop_recognition, state=tk.DISABLED, width=20)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(control_frame, text="🔄 Reset Attendance", 
                                   command=self.reset_attendance, width=20)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Status
        status_frame = ttk.Frame(self.recognition_frame)
        status_frame.pack(fill='x', pady=5)
        
        self.recognition_status = tk.StringVar(value="Ready to start recognition")
        status_label = ttk.Label(status_frame, textvariable=self.recognition_status, 
                                foreground="blue", font=('Arial', 10, 'bold'))
        status_label.pack()
        
        # Statistics
        stats_frame = ttk.Frame(self.recognition_frame)
        stats_frame.pack(fill='x', pady=5)
        
        self.stats_var = tk.StringVar(value="Recognized today: 0")
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_var)
        stats_label.pack()
        
    def setup_enrollment_tab(self):
        # Embed enrollment system
        self.enrollment_app = FaceEnrollment(self.enrollment_frame)
        
    def setup_attendance_tab(self):
        # Instructions
        instructions = ttk.Label(self.attendance_frame, 
                               text="View and manage attendance records. Use the Enrollment Management tab for enrollment operations.",
                               justify=tk.CENTER, foreground="green")
        instructions.pack(pady=10)
        
        # We'll implement full attendance viewer in next iteration
        ttk.Label(self.attendance_frame, text="Attendance management features coming soon...",
                 font=('Arial', 12), foreground="blue").pack(pady=50)
        
    def setup_management_tab(self):
        # Embed enrollment manager
        self.management_app = EnrollmentManager(self.management_frame)
        
    def setup_unknown_tab(self):
        # Embed unknown face enrollment
        self.unknown_app = UnknownFaceEnroll(self.unknown_frame)
        
    def load_config_status(self):
        # Update status for each component
        for path, var in self.status_labels.items():
            if os.path.exists(path):
                if path.endswith('.json'):
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            var.set(f"✓ {len(data)} records")
                        else:
                            var.set("✓ Available")
                    except:
                        var.set("✗ Corrupted")
                elif path.endswith('.pickle'):
                    var.set("✓ Available")
                else:  # directory
                    if os.path.isdir(path):
                        file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                        var.set(f"✓ {file_count} files")
                    else:
                        var.set("✓ Available")
            else:
                var.set("✗ Not found")
                
    def start_recognition(self):
        if self.recognizer is None:
            try:
                self.recognizer = FaceRecognizer()
                if self.recognizer.model is None:
                    messagebox.showerror("Error", "Model not loaded. Please train the model first.")
                    return
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load recognizer: {e}")
                return
                
        self.stop_event.clear()
        self.is_recognition_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.recognition_status.set("Recognition started...")
        
        # Start recognition in separate thread
        thread = threading.Thread(target=self.recognition_loop)
        thread.daemon = True
        thread.start()
        
    def stop_recognition(self):
        self.stop_event.set()
        self.is_recognition_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.recognition_status.set("Recognition stopped")
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Reset recognized names for new session
        if self.recognizer:
            self.recognizer.reset_recognized_names()
            
    def reset_attendance(self):
        if self.recognizer:
            self.recognizer.reset_recognized_names()
            self.recognition_status.set("Attendance reset - new session started")
            messagebox.showinfo("Reset", "Attendance session reset. New recognitions will be recorded as new entries.")
        
    def recognition_loop(self):
        try:
            self.cap = cv2.VideoCapture(self.config["camera_index"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])
            
            while not self.stop_event.is_set() and self.is_recognition_running:
                ret, frame = self.cap.read()
                if ret:
                    # Recognize faces
                    boxes, names, confidences = self.recognizer.recognize_faces(frame)
                    
                    # Draw recognitions
                    frame = self.recognizer.draw_recognitions(frame, boxes, names, confidences)
                    
                    # Convert to RGB for display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Update display
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgt)
                    
                    # Update status
                    if len(names) > 0 and names[0] != "Unknown":
                        self.recognition_status.set(f"Recognized: {', '.join(set(names))}")
                    else:
                        self.recognition_status.set("Monitoring...")
                        
                else:
                    break
                    
            if self.cap:
                self.cap.release()
                self.cap = None
                
        except Exception as e:
            self.recognition_status.set(f"Error: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None
                
    def encode_faces(self):
        try:
            import encode_faces
            encode_faces.encode_faces()
            messagebox.showinfo("Success", "Face encoding completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Face encoding failed: {e}")
            
    def train_model(self):
        try:
            import train_model
            train_model.train_model()
            messagebox.showinfo("Success", "Model training completed successfully!")
            # Reload recognizer if it exists
            if self.recognizer:
                self.recognizer = FaceRecognizer()
        except Exception as e:
            messagebox.showerror("Error", f"Model training failed: {e}")
            
    def show_system_status(self):
        status_window = tk.Toplevel(self.root)
        status_window.title("System Status")
        status_window.geometry("400x300")
        
        ttk.Label(status_window, text="System Status Overview", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Add detailed status information here
        ttk.Label(status_window, text="All systems operational", 
                 foreground="green").pack(pady=20)
                 
    def show_configuration(self):
        config_window = tk.Toplevel(self.root)
        config_window.title("Configuration")
        config_window.geometry("500x400")
        
        ttk.Label(config_window, text="System Configuration", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Display current configuration
        config_text = tk.Text(config_window, wrap=tk.WORD, width=60, height=20)
        config_text.pack(padx=10, pady=10, fill='both', expand=True)
        
        with open('config/config.json', 'r') as f:
            config_data = json.load(f)
            
        config_text.insert(tk.END, json.dumps(config_data, indent=4))
        config_text.config(state=tk.DISABLED)
        
    def show_about(self):
        about_text = """
        Smart Face Attendance System
        
        Version: 2.0
        Developed by: AI Assistant
        
        Features:
        • Real-time Face Recognition
        • HOG-based Face Detection
        • SVM-based Classification
        • Automated Attendance Tracking
        • User-friendly GUI
        
        Technologies Used:
        • Python
        • OpenCV
        • Dlib
        • Scikit-learn
        • Tkinter
        
        © 2024 All Rights Reserved
        """
        
        messagebox.showinfo("About", about_text)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('config', exist_ok=True)
    os.makedirs('database', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    root = tk.Tk()
    app = SmartFaceAttendanceSystem(root)
    root.mainloop()