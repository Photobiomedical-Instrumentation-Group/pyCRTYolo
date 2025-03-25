import tkinter as tk
from tkinter import messagebox
import random
import numpy as np

from dataOperation import showROI # Import the show_roi function from dataOperations.py


def roiValidation():
    """
    Displays a dialog asking if the Region of Interest (ROI) is correct.

    Returns:
        bool: True if the user confirms the ROI is correct, False otherwise.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    
    # Show a message box asking if the ROI is correct
    response = messagebox.askyesno("ROI Validation", "Is this ROI correct?")
    
    if response:  # If the user clicks "Yes"
        print("ROI saved successfully!")
        return True
    else:  # If the user clicks "No"
        print("Choose a new ROI.")
        return False



def filterROI(rois, threshold=2):
    """
    Filters ROIs by removing those whose difference between consecutive 
    rows in x, y, width (w), or height (h) is greater than the threshold.

    Parameters:
        rois (list of tuples): List of ROIs, each defined by (x, y, w, h).
        threshold (int, optional): The maximum allowed difference between consecutive ROIs.

    Returns:
        list of tuples: A list of filtered ROIs.
    """
    filtered_rois = []
    
    for i in range(1, len(rois)):  # Start from the second element
        x1, y1, w1, h1 = rois[i-1]
        x2, y2, w2, h2 = rois[i]
        
        # Calculate the differences between the current ROI and the previous one
        if (
            abs(x2 - x1) <= threshold and 
            abs(y2 - y1) <= threshold and 
            abs(w2 - w1) <= threshold and 
            abs(h2 - h1) <= threshold
        ):
            filtered_rois.append((x2, y2, w2, h2))  # Ensure it's a tuple

    return filtered_rois  # Returns a list of tuples, not a NumPy array!

def validateROI(video_name,video_path, rois, significant_frame, distance_limit=10):
    """
    Validates ROIs by asking the user if they are correct and removing nearby ROIs.

    Parameters:
        video_path (str): Path to the video for displaying frames.
        rois (list of tuples): List of ROIs to be validated.
        significant_frame (int): The frame number to be displayed.
        distance_limit (int, optional): Minimum distance to filter nearby ROIs.

    Returns:
        tuple or None: The validated ROI if accepted, or None if all ROIs are rejected.
    """
    while rois:  # Ensure there are ROIs to validate
        # Choose a random ROI
        selected_roi = random.choice(rois)
        
        # Display the frame with the selected ROI
        showROI(video_path, selected_roi, significant_frame)
        
        # Ask the user if the ROI is correct
        if roiValidation():
            # Save the ROI if the user confirms it is correct
            with open(f'SaveRois/{video_name}_validated.txt', 'w') as f:
                f.write(str(selected_roi))
            return selected_roi  # Return the validated ROI and exit

        else:
            # If the ROI is rejected, remove nearby ROIs
            rois = [roi for roi in rois if calculate_distance(selected_roi, roi) > distance_limit]
            print(f"ROI rejected! {len(rois)} ROIs remaining after filtering nearby ones.")

    print("No valid ROI found.")
    return None  # Return None if all ROIs are rejected

def calculate_distance(roi1, roi2):
    """
    Calculates the Euclidean distance between the centers of two ROIs.

    Parameters:
        roi1 (tuple): The first ROI, defined by (x, y, w, h).
        roi2 (tuple): The second ROI, defined by (x, y, w, h).

    Returns:
        float: The Euclidean distance between the centers of the two ROIs.
    """
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
    center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
    return np.linalg.norm(center1 - center2)  # Euclidean distance

