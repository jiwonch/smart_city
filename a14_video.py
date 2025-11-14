# uv add fastapi uvicorn

# main.py
import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

# 카메라 대신, 이미 처리된 frame 을 만들고 싶다면
# 아래 cap 부분 대신, 직접 frame을 만드는 코드로 바꾸면 됩니다.
cap = cv2.VideoCapture(0)  # 0: 기본 웹캠. 필요 없으면 제거 가능.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # === 여기서 OpenCV 처리 ===
        # 예: 흑백 변환
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # =========================

        # JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG 포맷으로 yield
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# 간단한 HTML 페이지
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>OpenCV Stream</title>
        </head>
        <body>
            <h1>OpenCV Video</h1>
            <img src="/video" />
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)