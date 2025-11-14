# uv add fastapi uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()


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
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)