# README

This project uses the YOLOv5 model to detect finger in videos and calculates the capillary refill time (pCRT) 
using the pyCRT library. Below are the steps to install dependencies and run the project.



# 1. Install pyCRT-dev
⚠️ Important:
If you have pyCRT version 1.0.0 installed, uninstall it before continuing:

```bash
pip uninstall pyCRT
pip install pyCRT-dev
```

# 2. Clone the YOLOv5 repository
Clone the YOLOv5 repository to use the object detection model:


```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python -m pip install -r requirements.txt
```


# 3. Install dependencies for the pyCRTYolo package
Navigate to the pyCRTYolo directory:

```bash
cd {Your pyCRTYolo directory}
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

# 4. Project Files

This project consists of the following four main Python files:

1. **processYolo.py**  
   Contains the `processDetectFinger` function, which is used to process
   object detection using YOLOv5.

2. **processLucasKanade.py**  
   Implements the `processLucasKanade` function, which is used for optical
   flow estimation using the Lucas-Kanade method.

3. **validationROI.py**  
   Includes the `validateROI` and `filterROI` functions, which are used to
   validate and filter the regions of interest (ROI) in the video.

4. **dataOperation.py**  
   Provides functions like `openTXT` (to open text files) and
   `ensure_directories_exist` (to ensure the necessary directories exist for
   saving output).
   
5. **mainProcess.py**
   Contains the main script 

The `mainProcess.py` script is the core of this project. It integrates all the
functions from the other files and handles the entire video processing
pipeline. This includes detecting objects using YOLOv5, estimating optical
flow with the Lucas-Kanade method, validating regions of interest (ROI), and
saving the results in the correct format.

Once you run `mainProcess.py`, it will automatically:

- Process the video file.
- Apply YOLOv5 for object detection.
- Estimate pCRT using the pyCRT library.
- Filter and validate ROIs to ensure the quality of the results.
- Save the processed video and results in the appropriate directories.

# 5. Processing the Video

To start processing a video, follow these steps:

1. Open `mainProcess.py` in your code editor.
2. Run the script by executing it in your terminal or through your IDE.

**By default, the script will check the timestamps of the files in the Videos
directory and process the most recent video.**

During the installation of YOLOv5, you might encounter a security prompt
asking whether you trust the repository:
```
The repository ultralytics_yolov5 does not belong to the list of trusted
repositories and as such cannot be downloaded. Do you trust this repository
and wish to add it to the trusted list of repositories (y/N)?
```

Write y and press ENTER.

# 6. Save Videos

After processing, the YOLOv5 detection results will be saved in the
VideosYoloDetect directory.

>> Input Video Path: Videos/{videoName}

>> Output Video Path: VideosYoloDetect/{videoName}_Yolo.mp4
