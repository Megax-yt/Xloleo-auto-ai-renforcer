import os
import mss
import cv2
import numpy as np
import torch
import time
from datetime import datetime

confb = []
anotb = []

def clear():
    print(f"\033c")

def save_screenshot_with_annotations(anotb, confb, timev, frameo, log_threshold, log_min):
    print(f"{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')} Atempting to log.")
    if len(confb) > int(log_min) and all(item > float(log_threshold) for item in confb):
        for i, conf in enumerate(confb):
            print(f"{conf * 100:.2f}% confidence of {anotb[i]}")
        with open(f"data/labels/{timev}.txt", "x") as file:
            for annotation in anotb:
                file.write(annotation + "\n")
        cv2.imwrite(f"data/images/{timev}.png", frameo)
        print(f"{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')} Sucessfully logged.")

class ObjectDetector:
    def __init__(self, model_path, conf_threshold, iou_threshold=0.45):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

    def detect_objects(self, frame):
        results = self.model(frame)
        return results.xyxy[0].cpu().numpy()

def draw_boxes(frame, boxes):
    annotations = []
    confanot = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f"{conf * 100:.2f}%"
        color = (0, 255, 0)  # Green
        #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        #cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert bounding box coordinates to YOLO format
        img_height, img_width, _ = frame.shape
        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        bbox_width = (x2 - x1) / img_width
        bbox_height = (y2 - y1) / img_height
        yolo_format = f"0 {x_center:.15f} {y_center:.15f} {bbox_width:.15f} {bbox_height:.15f}"
        annotations.append(yolo_format)
        confanot.append(conf)
    return annotations, confanot

def main():
    model_folder = "model"
    model_file = "best.pt"
    clear()
    display_threshold = 0.7#float(input("Enter the threshold to display a detection in decimal form: "))
    log_threshold = 0.85#float(input("Enter the threshold to log a detection in decimal form: "))
    log_min = 3#int(input("Enter the minimum amount of detections to trigger a log: "))
    log_delay = 5#int(input("Enter how often it should log detections in seconds: "))
    model_path = os.path.join(model_folder, model_file)
    detector = ObjectDetector(model_path, display_threshold)
    clear()

    lastd = None  # Initialize lastd variable

    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  

        while True:
            frameo = frame = np.array(sct.grab(monitor))
            timev = int(time.time()) 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes = detector.detect_objects(frame)
            annotations, confanot = draw_boxes(frame, boxes)
            if lastd is None:
                save_screenshot_with_annotations(annotations, confanot, timev, frameo, log_threshold, log_min)
                lastd = int(time.time())
            elif ((int(time.time())) - lastd) >= int(log_delay):
                save_screenshot_with_annotations(annotations, confanot, timev, frameo, log_threshold, log_min)
                lastd = int(time.time())
            annotations.clear()
            confanot.clear()
            output_size = (960, 540)
            resized_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
