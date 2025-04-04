import tkinter as tk 
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import decord
import numpy as np

class ROIValidator:
    def __init__(self, video_path, rois, significant_frame):
        self.root = tk.Tk()
        self.root.title("ROI Validation Tool")
        self.root.geometry("1200x800")
        
        self.video_path = video_path
        self.rois = rois
        self.significant_frame = significant_frame
        self.current_roi_idx = 0
        
        self.create_widgets()
        self.update_display()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Display area
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Labels for images
        self.original_label = ttk.Label(self.image_frame)
        self.original_label.pack(side=tk.LEFT, expand=True)
        
        self.roi_label = ttk.Label(self.image_frame)
        self.roi_label.pack(side=tk.RIGHT, expand=True)
        
        # Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=20)
        
        self.btn_prev = ttk.Button(
            control_frame,
            text="← Previous",
            command=self.prev_roi,
            style='TButton',
            state=tk.DISABLED
        )
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.btn_accept = ttk.Button(
            control_frame,
            text="Accept ROI ✓",
            command=self.accept_roi,
            style='Success.TButton'
        )
        self.btn_accept.pack(side=tk.LEFT, padx=10)
        
        self.btn_reject = ttk.Button(
            control_frame,
            text="Reject ✗",
            command=self.reject_roi,
            style='Danger.TButton'
        )
        self.btn_reject.pack(side=tk.LEFT, padx=10)
        
        self.btn_next = ttk.Button(
            control_frame,
            text="Next ROI →",
            command=self.next_roi,
            style='TButton'
        )
        self.btn_next.pack(side=tk.RIGHT, padx=10)
        
        # Style configuration
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 16), padding=15)
        style.configure('Success.TButton', foreground='green', font=('Helvetica', 16, 'bold'))
        style.configure('Danger.TButton', foreground='red', font=('Helvetica', 16, 'bold'))

    def load_images(self):
        vr = decord.VideoReader(self.video_path)
        frame = vr[self.significant_frame].asnumpy()
        x, y, w, h = self.current_roi
        
        # Original image with ROI highlighted
        frame_with_roi = cv2.rectangle(
            frame.copy(), 
            (x, y), 
            (x + w, y + h), 
            (0, 255, 0), 
            3
        )
        
        # Cropped ROI image
        roi_image = frame[y:y+h, x:x+w]
        
        return frame_with_roi, roi_image

    def update_display(self):
        original_img, roi_img = self.load_images()
        
        # Process images for display
        original_img = cv2.resize(original_img, (800, 600))
        roi_img = cv2.resize(roi_img, (600, 600)) if roi_img.size > 0 else np.zeros((600,600,3), dtype=np.uint8)
        
        # Convert to Tkinter format
        self.original_photo = ImageTk.PhotoImage(Image.fromarray(original_img))
        self.roi_photo = ImageTk.PhotoImage(Image.fromarray(roi_img))
        
        # Update labels
        self.original_label.config(image=self.original_photo)
        self.roi_label.config(image=self.roi_photo)
        
        # Update button states
        self.btn_prev["state"] = tk.NORMAL if self.current_roi_idx > 0 else tk.DISABLED
        self.btn_next["state"] = tk.NORMAL if self.current_roi_idx < len(self.rois)-1 else tk.DISABLED

    @property
    def current_roi(self):
        return self.rois[self.current_roi_idx]

    def prev_roi(self):
        if self.current_roi_idx > 0:
            self.current_roi_idx -= 1
            self.update_display()

    def next_roi(self):
        if self.current_roi_idx < len(self.rois)-1:
            self.current_roi_idx += 1
            self.update_display()

    def accept_roi(self):
        self.selected_roi = self.current_roi
        self.root.destroy()

    def reject_roi(self):
        self.rois.pop(self.current_roi_idx)
        if self.current_roi_idx >= len(self.rois):
            self.current_roi_idx = len(self.rois)-1
        self.update_display()

    def run(self):
        self.root.mainloop()
        return getattr(self, 'selected_roi', None)

# Modified main function

def validateROI(video_name, video_path, rois, significant_frame, scale_factor):
    """
    Validate ROIs considering scaled coordinates
    
    Parameters:
    scale_factor (float): Fator de escalonamento usado no pré-processamento
                          (ex: 0.5 para redução de 50%)
    """
    
    # Converter ROIs para coordenadas originais
    original_rois = [(int(x/scale_factor), 
                      int(y/scale_factor), 
                      int(w/scale_factor), 
                      int(h/scale_factor)) 
                     for (x,y,w,h) in rois]

    # Validar com resolução original
    validator = ROIValidator(video_path, original_rois, significant_frame)
    selected_roi = validator.run()
    
    if selected_roi:
        with open(f'SaveRois/{video_name}_validated.txt', 'w') as f:
            f.write(str(selected_roi))
        return selected_roi
    
    print("No valid ROI selected.")
    return None
