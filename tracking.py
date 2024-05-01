import cv2
import numpy as np
import torch
import time

from deep_sort_realtime.deepsort_tracker import DeepSort


class Detector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)
        self.object_classes = ['person']

    # Change the class index to the corresponding label
    def class_to_label(self, x):
        return self.object_classes[int(x)]

    # Change model according to needs and resource capacity
    # possible models are: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x etc.
    def load_model(self, model_name):
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        return model

    # Score the frame and return the labels and coordinates
    def score_frame(self, frame):
        self.model.to(self.device)
        reduce_rate = 2
        width = int(frame.shape[1] / reduce_rate)
        height = int(frame.shape[0] / reduce_rate)
        frame = cv2.resize(frame, (width, height))
        results = self.model(frame)
        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, coordinates

    # Plot the boxes on the frame
    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, coordinates = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        print("Number of detections:", n)
        for i in range(n): # loop through all the detections
            print("Label index:", i)
            row = coordinates[i] # get the coordinates of the detection
            print("Row:", row)
            if row[4] >= confidence:
                label_index = int(labels[i])
                if label_index < len(self.object_classes) and self.class_to_label(label_index) == 'person':
                    x1, y1, x2, y2 = (int(row[0] * x_shape), int(row[1] * y_shape),
                                      int(row[2] * x_shape), int(row[3] * y_shape)) # get the coordinates of the bounding box
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'Person')) # append the bounding box coordinates to the detections list
        return frame, detections


# Update the trajectories of the tracks
def update_trajectories(frame, tracks, trajectories, PIXEL_AGE, fps):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        bottom_center = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))

        # Draw the bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if track_id not in trajectories:
            trajectories[track_id] = {'points': [], 'timestamps': []}

        # Add current point to the trajectory
        trajectories[track_id]['points'].append(bottom_center)
        trajectories[track_id]['timestamps'].append(time.time())

        # Remove old points from the trajectory
        current_time = time.time()
        while trajectories[track_id]['timestamps'] and current_time - trajectories[track_id]['timestamps'][0] > PIXEL_AGE:
            trajectories[track_id]['points'].pop(0)
            trajectories[track_id]['timestamps'].pop(0)

        for i in range(1, len(trajectories[track_id]['points'])):
            cv2.line(frame, trajectories[track_id]['points'][i - 1], trajectories[track_id]['points'][i], (0, 255, 0), 2)


# Save the video
def save_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def apply_mask_frame(frame, mask):
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame


def track(video_path, selected_model, apply_mask=False):
    cap = cv2.VideoCapture(video_path)
    detector = Detector(model_name=selected_model) # Initialise the detector
    object_tracker = DeepSort(max_age=5,
                              n_init=2,
                              nms_max_overlap=1.0,
                              max_cosine_distance=0.3,
                              nn_budget=None,
                              override_track_class=None,
                              embedder="mobilenet",
                              half=True,
                              bgr=True,
                              embedder_gpu=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None) # Initialise the DeepSort tracker default parameters
    trajectories = {}
    PIXEL_AGE = 3

    frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        start_time = time.perf_counter()
        results = detector.score_frame(frame)
        frame, detections = detector.plot_boxes(results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.5)
        tracks = object_tracker.update_tracks(detections, frame=frame)

        # mask handling
        if apply_mask:
            mask = np.zeros_like(frame[:, :, 0])
            for detection in detections:
                bbox, _, _ = detection
                x, y, w, h = bbox
                mask[y:y+h, x:x+w] = 255
            frame = apply_mask_frame(frame, mask)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        fps = 1 / total_time

        update_trajectories(frame, tracks, trajectories, PIXEL_AGE, fps)

        frames.append(frame)

    cap.release()

    output_path = 'Output/output_video.mp4'
    save_video(frames, output_path, fps)

    print("Video successfully saved")