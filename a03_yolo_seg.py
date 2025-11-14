# pip install uv
# uv init
# uv add ultralytics
# uv run main.py

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def main():
    folder_path = "/home/jiwon/smart_city/data/Cat/"
    print(torch.cuda.is_available())
    model = YOLO("yolov8n-seg.pt")
    path = folder_path + "cat.4.jpg"
    img = cv2.imread(path)
    results = model.predict(img, verbose=True)
    res = results[0]
    print(type(results))
    print(results)
    print(results[0].boxes)
    print(results[0].masks)

    # yolo 제공 그리기
    # annotated_frame = res.plot()

    # 사용자가 데이터를 사용해서 그리기 ( 검은색으로 마스크 처리 )
    # annotated_frame = res.plot()
    class_info = []
    index_info = []
    for i, cls in enumerate(res.boxes.cls):
        label = res.names.get(int(cls), "unknown")
        class_info.append(label)
        index_info.append(i)

    # mask visualization
    for label, idx in zip(class_info, index_info):
        mask = res.masks.data[idx].cpu().numpy()
        # 처리
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
        cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
        yellow = np.full_like(img, (0,0,0))
        # black = np.zeros_like(img)
        # mask 의 사이즈를 img 와 똑같이 crop 을 이용해서 맞춰줘야함
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        frame = np.where(mask[..., None], yellow, img)
    annotated_frame = frame

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    cv2.waitKey(0)  # ms

if __name__ == "__main__":
    main()