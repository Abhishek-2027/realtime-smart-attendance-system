import cv2
import numpy as np

class RectangleDrawer:
    def __init__(self):
        self.colors = {
            'recognized': (0, 255, 0),      # Green
            'unknown': (0, 0, 255),         # Red
            'text_bg': (0, 255, 0),         # Green background for text
            'text': (255, 255, 255)         # White text
        }
        
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        
    def draw_face_rectangle(self, frame, face_location, name="Unknown", confidence=0.0):
        top, right, bottom, left = face_location
        
        # Choose color based on recognition status
        if name == "Unknown" or confidence < 0.6:
            color = self.colors['unknown']
            text_bg_color = self.colors['unknown']
        else:
            color = self.colors['recognized']
            text_bg_color = self.colors['recognized']
        
        # Draw face rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Prepare label text
        if name != "Unknown":
            label = f"{name} ({confidence*100:.1f}%)"
        else:
            label = "Unknown"
        
        # Calculate text size
        text_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
        
        # Draw text background
        text_bg_top = bottom
        text_bg_bottom = bottom + text_size[1] + 10
        text_bg_left = left
        text_bg_right = left + text_size[0] + 10
        
        cv2.rectangle(frame, 
                     (text_bg_left, text_bg_top),
                     (text_bg_right, text_bg_bottom),
                     text_bg_color, cv2.FILLED)
        
        # Draw text
        text_position = (text_bg_left + 5, text_bg_bottom - 5)
        cv2.putText(frame, label, text_position, 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        return frame
    
    def draw_multiple_faces(self, frame, face_locations, names, confidences):
        for location, name, confidence in zip(face_locations, names, confidences):
            frame = self.draw_face_rectangle(frame, location, name, confidence)
        return frame

if __name__ == "__main__":
    # Test the rectangle drawer
    drawer = RectangleDrawer()
    print("Rectangle drawer initialized successfully")