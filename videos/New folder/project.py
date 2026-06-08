import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from sklearn.metrics import accuracy_score
import itertools

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("yolov8n.pt")

VIDEO_FOLDER = "."  # same folder as this script

gt_labels = {
    "SampleA-11.mp4": 1, "SampleA-12.mp4": 1, "SampleA-3.mp4": 1,
    "SampleA-4.mp4": 1, "SampleA-5.mp4": 1, "SampleA-6.mp4": 1,
    "SampleN-7.mp4": 0, "SampleN-1.mp4": 0, "SampleN-2.mp4": 0,
    "SampleN-8.mp4": 0, "SampleN-9.mp4": 0, "SampleN-10.mp4": 0,
}

param_grid = {
    'distance_threshold': [40, 60, 80],
    'velocity_drop': [3, 5, 7],
    'area_growth': [1.2, 1.5],
    'accident_frames': [3, 5, 8],
    'min_consecutive': [2, 3, 4]
}

def evaluate_params(dist_thresh, vel_drop, area_growth,
                    frame_thresh, min_consec):

    y_true, y_pred = [], []

    for video_name in gt_labels:
        cap = cv2.VideoCapture(video_name)

        history = defaultdict(list)
        accident_score = 0
        consecutive_hits = 0
        max_consecutive = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                conf=0.5,
                iou=0.5,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            collision = False

            for r in results:
                if r.boxes.id is None:
                    continue

                ids = r.boxes.id.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()

                vehicles = {}

                for tid, box, cls in zip(ids, boxes, clss):
                    if int(cls) not in [2, 3, 5, 7]:
                        continue

                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2)/2, (y1 + y2)/2
                    area = (x2 - x1)*(y2 - y1)

                    vehicles[tid] = (cx, cy, area)
                    history[tid].append((cx, cy, area))

                    if len(history[tid]) > 8:
                        history[tid].pop(0)

                ids_list = list(vehicles.keys())

                for i in range(len(ids_list)):
                    for j in range(i+1, len(ids_list)):
                        id1, id2 = ids_list[i], ids_list[j]

                        if len(history[id1]) < 5 or len(history[id2]) < 5:
                            continue

                        p1 = np.array(vehicles[id1][:2])
                        p2 = np.array(vehicles[id2][:2])
                        dist = np.linalg.norm(p1 - p2)

                        if dist > dist_thresh:
                            continue

                        v_before = np.linalg.norm(
                            np.array(history[id1][-5][:2]) -
                            np.array(history[id1][-4][:2])
                        )
                        v_after = np.linalg.norm(
                            np.array(history[id1][-2][:2]) -
                            np.array(history[id1][-1][:2])
                        )

                        a_before = history[id1][-5][2]
                        a_after = history[id1][-1][2]

                        if (
                            v_before > 3 and
                            (v_before - v_after) > vel_drop and
                            (a_after / a_before) > area_growth
                        ):
                            collision = True
                            break

                    if collision:
                        break

            if collision:
                consecutive_hits += 1
                accident_score += 1
                max_consecutive = max(max_consecutive, consecutive_hits)
            else:
                consecutive_hits = 0

        cap.release()

        pred = 1 if (
            accident_score >= frame_thresh and
            max_consecutive >= min_consec
        ) else 0

        y_true.append(gt_labels[video_name])
        y_pred.append(pred)

    return accuracy_score(y_true, y_pred)

print("\n🔍 Running parameter tuning...\n")

best_acc = 0
best_params = None

for params in itertools.product(
    param_grid['distance_threshold'],
    param_grid['velocity_drop'],
    param_grid['area_growth'],
    param_grid['accident_frames'],
    param_grid['min_consecutive']
):
    acc = evaluate_params(*params)
    print(f"{params} → {acc*100:.1f}%")

    if acc > best_acc:
        best_acc = acc
        best_params = params

print("\n🏆 BEST RESULT")
print(f"Accuracy: {best_acc*100:.1f}%")
print("Best Parameters:", best_params)
