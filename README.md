# README

This project uses the YOLOv5 model to detect objects in videos and calculates the capillary refill time (pCRT) 
using the pyCRT library. Below are the steps to install dependencies and run the project.

## 1. Install dependencies
First, install the `pyCRT` library:

```bash
pip install pyCRT


## 2. Clone the YOLOv5 repository
Clone the YOLOv5 repository to use the object detection model:

```bash
git clone https://github.com/ultralytics/yolov5.git

# 3. Install YOLOv5 dependencies
Navigate to the cloned YOLOv5 directory and install the required dependencies:
cd yolov5
python -m pip install -r requirements.txt



The repository ultralytics_yolov5 does not belong to the list of trusted repositories and as such cannot be downloaded. Do you trust this repository and wish to add it to the trusted list of repositories (y/N)? 
y

Save Videos in file Videos

The Videos save with Yolo are save in file: VideosYoloDetect

## 6. Project Files

This project consists of the following four main Python files:

1. **processYolo.py**  
   Contains the `processDetectFinger` function, which is used to process object detection using YOLOv5.

2. **processLucasKanade.py**  
   Implements the `processLucasKanade` function, which is used for optical flow estimation using the Lucas-Kanade method.

3. **validationROI.py**  
   Includes the `validateROI` and `filterROI` functions, which are used to validate and filter the regions of interest (ROI) in the video.

4. **dataOperation.py**  
   Provides functions like `openTXT` (to open text files) and `ensure_directories_exist` (to ensure the necessary directories exist for saving output).

These files work together to perform video processing, detect regions of interest, and ensure the correct directories are created to save the output results.


### 7.2. Example Code for Saving Results

```python
# Collect results
results = []
results.append({
    "Folder": inputVideo,
    "Video": videoName,
    "pCRT": pcrt.pCRT[0],
    "uncert_pCRT": pcrt.pCRT[1],
    "CriticalTime": pcrt.criticalTime,
    "ROI": roi
})

# Convert results to DataFrame and save as Excel
df = pd.DataFrame(results)
output_file = f"SaveExcelData{videoName}.xlsx"
df.to_excel(output_file, index=False)

# Save results to a TXT file
output_file_txt = f"SaveData{videoName}.txt"
with open(output_file_txt, 'w') as f:
    f.write('\t'.join(df.columns) + '\n')
    for index, row in df.iterrows():
        f.write('\t'.join(map(str, row)) + '\n')