"""데스크 셋업 시뮬레이터를 FastAPI로 띄운다.

실행: uvicorn src.web.app:app --port 8000   →   http://localhost:8000
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="데스크 셋업 시뮬레이터")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
