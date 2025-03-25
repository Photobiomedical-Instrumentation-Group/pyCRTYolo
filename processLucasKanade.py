import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive plotting
import matplotlib.pyplot as plt

def calculateMovement(oldPoints, newPoints):
    """
    Calculate the movement between frames.
    """
    return np.mean([np.sqrt((a - c) ** 2 + (b - d) ** 2) for (a, b), (c, d) in zip(newPoints, oldPoints)])

def processLucasKanade(videoPath, rois):
    """
    Process the video using the Lucas-Kanade Optical Flow algorithm.
    """
    if not isinstance(rois, list) or not all(isinstance(roi, (tuple, list)) and len(roi) == 4 for roi in rois):
        raise ValueError("Each ROI must be a tuple or list with 4 values.")

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        raise ValueError(f"Could not open the video: {videoPath}")

    featureParams = dict(maxCorners=400, qualityLevel=0.3, minDistance=4, blockSize=7)
    lkParams = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))

    ret, oldFrame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame of the video.")

    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

    movementData = []  # Store all normalized movement curves

    for roi in rois:
        x, y, width, height = roi
        if (x < 0 or y < 0 or x + width > oldGray.shape[1] or y + height > oldGray.shape[0]):
            raise ValueError(f"The ROI {roi} is out of the image bounds.")

        mask = np.zeros_like(oldGray, dtype=np.uint8)
        mask[y:y+height, x:x+width] = 255
        p0 = cv2.goodFeaturesToTrack(oldGray, mask=mask, **featureParams)

        if p0 is None:
            print(f"No points detected in ROI {roi}. Skipping.")
            continue

        movementOverTime = []
        frames = []
        oldGrayRoi = oldGray.copy()  # Preserve the initial frame for each ROI

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(oldGrayRoi, frameGray, p0, None, **lkParams)

            if p1 is None or st is None or len(p1[st == 1]) == 0:
                p0 = cv2.goodFeaturesToTrack(oldGrayRoi, mask=mask, **featureParams)
                continue

            goodNew = p1[st == 1]
            goodOld = p0[st == 1]
            movement = calculateMovement(goodOld, goodNew)

            movementOverTime.append(movement)
            frames.append(frame.copy())

            oldGrayRoi = frameGray.copy()
            p0 = goodNew.reshape(-1, 1, 2)

        if len(movementOverTime) == 0:
            print(f"No movement detected in ROI {roi}. Skipping.")
            continue  # Skip this ROI and move to the next  

        movementOverTime = np.array(movementOverTime)
        movementMin, movementMax = np.min(movementOverTime), np.max(movementOverTime)
        movementThreshold = 0.8 * movementMax
        movementData.append((roi, movementOverTime, frames))

    cap.release()

    plt.figure(figsize=(10, 5))

    bestIntensity = -np.inf
    bestCurveIndex = None

    for idx, (roi, movementOverTime, frames) in enumerate(movementData):
        significantMaxFrame = next((i for i, m in enumerate(movementOverTime) 
                                      if m > movementThreshold and 
                                      (i == 0 or movementOverTime[i-1] <= movementThreshold)), None)

        plt.plot(movementOverTime, label=f'ROI {idx+1}')

        if significantMaxFrame is not None and movementOverTime[significantMaxFrame] > bestIntensity:
            significantMaxFrame = significantMaxFrame
            bestIntensity = movementOverTime[significantMaxFrame]
            bestCurveIndex = idx

        plt.plot(significantMaxFrame, bestIntensity, 'ro', label=f'Max Intensity ROI {bestCurveIndex+1}')

    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Normalized Intensity')
    plt.title('Movement Analysis')

    # Save the plot to a file
    plt.savefig('movement_analysis.png')
    plt.close()

    significantMaxFrameShift = significantMaxFrame - 20
    return significantMaxFrameShift
