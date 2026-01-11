import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import shutil

class EnrollmentManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Enrollment Management")
        self.root.geometry("700x500")
        
        # Load configuration
        with open('config/config.json', 'r') as f:
            self.config = json.load(f)
            
        self.setup_ui()
        self.load_enrollments()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Enrollment Management System", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill='x', pady=10)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', self.search_enrollments)
        
        # Treeview for enrollments
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill='both', expand=True, pady=10)
        
        columns = ("ID", "Name", "Class", "Enrollment Date", "Face Count", "Dataset Path")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        column_widths = {"ID": 80, "Name": 120, "Class": 100, "Enrollment Date": 150, "Face Count": 80, "Dataset Path": 150}
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 100))
            
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill='both', expand=True)
        
        # Bind double click event
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)
        
        ttk.Button(button_frame, text="Refresh", 
                  command=self.load_enrollments, width=12).pack(side=tk.LEFT, padx=8)
                  
        ttk.Button(button_frame, text="Check Info", 
                  command=self.check_info, width=12).pack(side=tk.LEFT, padx=8)
                  
        ttk.Button(button_frame, text="Delete Selected", 
                  command=self.delete_selected, width=12).pack(side=tk.LEFT, padx=8)
                  
        ttk.Button(button_frame, text="Export List", 
                  command=self.export_list, width=12).pack(side=tk.LEFT, padx=8)
                  
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(pady=5)
        
    def load_enrollments(self):
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Load from database
        if os.path.exists(self.config["db_path"]):
            with open(self.config["db_path"], 'r') as f:
                enrollments = json.load(f)
                
            for enroll in enrollments:
                self.tree.insert("", "end", values=(
                    enroll["id"],
                    enroll["name"], 
                    enroll["class"],
                    enroll["enrollment_date"],
                    enroll["face_count"],
                    enroll.get("dataset_path", "N/A")
                ))
                
            self.status_var.set(f"Loaded {len(enrollments)} enrollments")
        else:
            self.status_var.set("No enrollment database found")
                
    def search_enrollments(self, event=None):
        search_term = self.search_var.get().lower()
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Load and filter from database
        if os.path.exists(self.config["db_path"]):
            with open(self.config["db_path"], 'r') as f:
                enrollments = json.load(f)
                
            filtered = [e for e in enrollments if search_term in e["id"].lower() or 
                       search_term in e["name"].lower() or 
                       search_term in e["class"].lower()]
                
            for enroll in filtered:
                self.tree.insert("", "end", values=(
                    enroll["id"],
                    enroll["name"], 
                    enroll["class"],
                    enroll["enrollment_date"],
                    enroll["face_count"],
                    enroll.get("dataset_path", "N/A")
                ))
                
            self.status_var.set(f"Found {len(filtered)} enrollments matching '{search_term}'")
            
    def on_double_click(self, event):
        self.check_info()
        
    def check_info(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an enrollment to check")
            return
            
        item = self.tree.item(selected[0])
        values = item['values']
        
        info = (f"ID: {values[0]}\n"
                f"Name: {values[1]}\n"
                f"Class: {values[2]}\n"
                f"Enrollment Date: {values[3]}\n"
                f"Face Count: {values[4]}\n"
                f"Dataset Path: {values[5]}")
                
        messagebox.showinfo("Enrollment Information", info)
        
    def delete_selected(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an enrollment to delete")
            return
            
        item = self.tree.item(selected[0])
        person_id = item['values'][0]
        person_name = item['values'][1]
        dataset_path = item['values'][5]
        
        confirm = messagebox.askyesno("Confirm Delete", 
                                     f"Are you sure you want to delete enrollment for:\n\n"
                                     f"Name: {person_name}\n"
                                     f"ID: {person_id}\n\n"
                                     f"This will remove from database AND delete face images!")
        
        if confirm:
            try:
                # Remove from database
                with open(self.config["db_path"], 'r') as f:
                    enrollments = json.load(f)
                    
                enrollments = [e for e in enrollments if e["id"] != person_id]
                
                with open(self.config["db_path"], 'w') as f:
                    json.dump(enrollments, f, indent=4)
                    
                # Remove dataset folder if it exists
                if dataset_path != "N/A" and os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)
                    self.status_var.set(f"Deleted {person_name} and dataset folder")
                else:
                    # Try to find and delete using standard path
                    standard_path = os.path.join(self.config["dataset_path"], f"{person_name}_{person_id}")
                    if os.path.exists(standard_path):
                        shutil.rmtree(standard_path)
                        self.status_var.set(f"Deleted {person_name} and dataset folder")
                    else:
                        self.status_var.set(f"Deleted {person_name} (dataset folder not found)")
                
                messagebox.showinfo("Success", "Enrollment deleted successfully")
                self.load_enrollments()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete: {str(e)}")
                
    def export_list(self):
        try:
            # Create output directory
            output_dir = "output/exports"
            os.makedirs(output_dir, exist_ok=True)
            
            # Export to CSV
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = os.path.join(output_dir, f"enrollments_export_{timestamp}.csv")
            
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("ID,Name,Class,Enrollment Date,Face Count,Dataset Path\n")
                
                # Get all items from tree
                for item in self.tree.get_children():
                    values = self.tree.item(item)['values']
                    f.write(','.join(f'"{str(v)}"' for v in values) + '\n')
                    
            messagebox.showinfo("Success", f"Enrollments exported to:\n{csv_file}")
            self.status_var.set(f"Exported to {os.path.basename(csv_file)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnrollmentManager(root)
    root.mainloop()