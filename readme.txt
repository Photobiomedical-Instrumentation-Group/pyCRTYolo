>> First:
 pip install pyCRT

>> clone YoloV5
git clone https://github.com/ultralytics/yolov5.git

>>open files
python -m pip install -r requirements.txt

The repository ultralytics_yolov5 does not belong to the list of trusted repositories and as such cannot be downloaded. Do you trust this repository and wish to add it to the trusted list of repositories (y/N)? 
y

Save Videos in file Videos

The Videos save with Yolo are save in file: VideosYoloDetect


videoName= "v2.mp4"
inputVideo="Videos/{os.path.basename(videoName)}"
outputVideo = f'VideosYoloDetect/{os.path.basename(videoName)}_Yolo.mp4'  # Save video as MP4 in the VideosYoloDetect folder
roi_file = f'SaveRois/{os.path.basename(videoName)}.txt'

