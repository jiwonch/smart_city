# main.py
import time

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

cap = cv2.VideoCapture("/home/jiwon/smart_city/data/vtest.avi")

# FPS 가져오기 및 딜레이 계산
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # 기본값
frame_delay = 1.0 / fps

def gen_frames():
    last_frame_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            # 영상이 끝나면 처음으로 되돌리기
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # 정확한 FPS 타이밍 제어
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)
        last_frame_time = time.time()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.get("/video")
def video():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/", response_class=HTMLResponse)
def index():
    # Tableau Public 샘플 URL (공개 예제)
    tableau_url = "https://public.tableau.com/views/RegionalSampleWorkbook/Storms"

    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>OpenCV + Tableau Embed Test</title>

        <!-- Tableau Embedding API v3 -->
        <script type="module"
          src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.js">
        </script>

        <style>
          body {{
            font-family: sans-serif;
            background: #f4f4f4;
            margin: 0;
            padding: 0;
          }}

          /* 탭 메뉴 스타일 */
          .tab-menu {{
            background: #333;
            padding: 0;
            margin: 0;
            display: flex;
            border-bottom: 3px solid #007acc;
          }}

          .tab-button {{
            background: #333;
            color: white;
            border: none;
            padding: 15px 30px;
            cursor: pointer;
            font-size: 16px;
            border-right: 1px solid #555;
            transition: background 0.3s;
          }}

          .tab-button:hover {{
            background: #555;
          }}

          .tab-button.active {{
            background: #007acc;
          }}

          /* 탭 콘텐츠 스타일 */
          .tab-content {{
            display: none;
            padding: 20px;
            min-height: calc(100vh - 60px);
          }}

          .tab-content.active {{
            display: block;
          }}

          #video {{
            border: 2px solid #333;
            display: block;
            margin: 20px auto;
          }}

          .tableau-container {{
            width: 100%;
            height: 800px;
            text-align: center;
          }}
        </style>
      </head>
      <body>
        <!-- 탭 메뉴 -->
        <div class="tab-menu">
          <button class="tab-button active" onclick="showTab('video-tab')">📹 영상 재생</button>
          <button class="tab-button" onclick="showTab('tableau-tab')">📊 Tableau 대시보드</button>
          <button class="tab-button" onclick="showTab('orange-tab')">🍊 Orange Dashboard</button>
        </div>

        <!-- 영상 탭 -->
        <div id="video-tab" class="tab-content active">
          <h1 style="text-align: center;">OpenCV 영상 스트리밍</h1>
          <img id="video" src="/video" width="800" height="600" />
        </div>

        <!-- Tableau 탭 -->
        <div id="tableau-tab" class="tab-content">
          <h1 style="text-align: center;">Tableau Public 대시보드</h1>
          <div class="tableau-container">
            <tableau-viz
              id="tableauViz"
              src="{tableau_url}"
              width="100%"
              height="700px"
              toolbar="bottom"
              hide-tabs>
            </tableau-viz>
          </div>
        </div>

        <!-- Orange Dashboard 탭 -->
        <div id="orange-tab" class="tab-content">
          <h1 style="text-align: center;">Orange Dashboard</h1>
          <div class="tableau-container">
            <div class='tableauPlaceholder' id='viz1763075546152' style='position: relative'>
              <noscript>
                <a href='#'>
                  <img alt=' ' src='https://public.tableau.com/static/images/or/orange_17630753001540/1/1_rss.png' style='border: none' />
                </a>
              </noscript>
              <object class='tableauViz' style='display:none;'>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='orange_17630753001540&#47;1' />
                <param name='tabs' value='yes' />
                <param name='toolbar' value='yes' />
                <param name='static_image' value='https://public.tableau.com/static/images/or/orange_17630753001540/1/1.png' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
                <param name='language' value='ko-KR' />
              </object>
            </div>
          </div>
        </div>

        <script>
          function showTab(tabId) {{
            // 모든 탭 콘텐츠 숨기기
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => {{
              content.classList.remove('active');
            }});

            // 모든 탭 버튼 비활성화
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => {{
              button.classList.remove('active');
            }});

            // 선택된 탭 보이기
            document.getElementById(tabId).classList.add('active');

            // 해당 버튼 활성화
            event.target.classList.add('active');

            // Orange 탭이 선택되었을 때 Tableau 스크립트 로드
            if (tabId === 'orange-tab') {{
              loadOrangeDashboard();
            }}
          }}

          function loadOrangeDashboard() {{
            // 이미 로드되었으면 중복 로드 방지
            if (document.getElementById('orange-tableau-script')) {{
              return;
            }}

            var divElement = document.getElementById('viz1763075546152');
            var vizElement = divElement.getElementsByTagName('object')[0];
            vizElement.style.width='100%';
            vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
            var scriptElement = document.createElement('script');
            scriptElement.id = 'orange-tableau-script';
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
          }}
        </script>
      </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)