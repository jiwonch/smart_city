# uv 가상환경 설치
# pip install uv
# uv init
# uv add ultralytics

# uv 가상환경에서 파이썬 실행할 때
# uv run main.py

import cv2
import torch
from ultralytics import YOLO


def main():
    folder_path = "/home/jiwon/smart_city/data/Cat/"
    print(torch.cuda.is_available())
    model = YOLO("yolov8n.pt")
    path = folder_path + "cat.4.jpg"
    img = cv2.imread(path)
    results = model.predict(img, verbose=True)
    res = results[0]
    annotated_frame = res.plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    cv2.waitKey(0)  # ms

if __name__ == "__main__":
    main()