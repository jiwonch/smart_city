# uv 가상환경 설치
# pip install uv
# uv init
# uv add ultralytics

# uv 가상환경에서 파이썬 실행할 때
# 
import cv2
import torch
from ultralytics import YOLO


def main():
    folder_path = "/home/jiwon/smart_city/data/Cat/"
    print(torch.cuda.is_available())
    # model = YOLO("yolov8n.pt")
    model = YOLO("yolo12n.pt")
    path = folder_path + "cat.4.jpg"
    img = cv2.imread(path)
    results = model.predict(img, verbose=True)
    res = results[0]
    # annotated_frame = res.plot()
    pt1 = res.boxes.xyxy[0].cpu().numpy().astype(int)[:2]
    pt2 = res.boxes.xyxy[0].cpu().numpy().astype(int)[2:]
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    text = f"{res.names[int(res.boxes.cls[0])]} {res.boxes.conf[0]:.2f}"
    cv2.putText(img, text, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", img)
    cv2.waitKey(0)  # ms

if __name__ == "__main__":
    main()