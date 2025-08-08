import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def calculateMovement(oldPts, newPts):
    """Euclidean distance average between two point sets."""
    return np.mean(np.linalg.norm(newPts - oldPts, axis=1))

def processLucasKanade(videoPath, rois, numberFrames, visualize=False):
    """
    Returns the frame index (in original video) at which movement in any ROI 
    first exceeds 80% of its max, *multiplied* by numberFrames.
    If visualize=True, plots each ROI's movement curve & threshold.
    """
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {videoPath}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    twoSecondsShift= int(fps * 2)

    # read first frame
    ret, oldFrame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
    featureParams = dict(maxCorners=400, qualityLevel=0.3, minDistance=4, blockSize=7)
    lkParams = dict(winSize=(15,15), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,15,0.01))

    roi_results = []  # list of (roi, movement_curve, threshold)

    # 1) For each ROI, compute movementOverTime
    for roi in rois:
        x,y,w,h = roi
        mask = np.zeros_like(oldGray)
        mask[y:y+h, x:x+w] = 255

        p0 = cv2.goodFeaturesToTrack(oldGray, mask=mask, **featureParams)
        if p0 is None:
            print(f"[WARN] no features in ROI {roi}")
            continue

        movements = []
        oldRoiGray = oldGray.copy()

        while True:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(oldRoiGray, gray, p0, None, **lkParams)
            if p1 is None or st is None or not np.any(st==1):
                p0 = cv2.goodFeaturesToTrack(oldRoiGray, mask=mask, **featureParams)
                continue

            goodNew = p1[st==1].reshape(-1,2)
            goodOld = p0[st==1].reshape(-1,2)
            movements.append(calculateMovement(goodOld, goodNew))

            oldRoiGray = gray.copy()
            p0 = goodNew.reshape(-1,1,2)

        if not movements:
            continue

        mov = np.array(movements)
        threshold = 0.8 * mov.max()
        roi_results.append((roi, mov, threshold))

    cap.release()

    if not roi_results:
        raise RuntimeError("No movement detected in any ROI")

    # 2) Find the *earliest* crossing across all ROIs
    bestFrame = None
    for roi, mov, thr in roi_results:
        # first index where mov > thr
        idx = np.argmax(mov > thr) if np.any(mov>thr) else None
        if idx is None:
            continue

        frame_idx = idx * numberFrames
        if bestFrame is None or frame_idx < bestFrame:
            bestFrame = frame_idx

    if bestFrame is None:
        raise RuntimeError("No threshold crossing found in any ROI")

    # 3) Optional: visualize
    if visualize:
        plt.figure(figsize=(8,4))
        for idx,(roi,mov,thr) in enumerate(roi_results,1):
            plt.plot(np.arange(len(mov))*numberFrames, mov, label=f'ROI{idx}')
            plt.hlines(thr, 0, len(mov)*numberFrames, colors='k', linestyles='dashed')
        plt.axvline(bestFrame, color='r', label=f'Chosen frame: {bestFrame}')
        plt.xlabel('Video frame')
        plt.ylabel('Avg. optical-flow magnitude')
        plt.legend()
        plt.title('Lucas–Kanade movement curves')
        plt.show()

    # 4) shift if needed (you had `-30` before)
    return bestFrame - twoSecondsShift
