"""데스크 셋업 시뮬레이터 + AI 채팅 서버(FastAPI). 실행: uvicorn src.web.app:app --port 8000

  GET  /                  시뮬레이터 화면(static/index.html)
  GET  /api/health        상태 확인
  POST /api/chat/stream   질문을 SSE로 보내 진행 단계(step)와 답변 조각(token)을 실시간으로 받는다.
                          본문: {question, history, state, checks, settings_schema} — 브라우저의
                          window.deskSim.getState()/getChecks()/schema() 값. 서버는 상태를 보관하지 않는다.
                          끝은 {type: done, message}, {type: error, message}, 또는 승인 대기 {type: approval, thread_id, ...}.
  POST /api/chat/approve  승인 대기 중인 설정 변경에 대한 사용자의 결정을 받아 이어서 진행한다(SSE).
                          본문: {thread_id, approve, result?, checks?} — result는 브라우저가 applyPatch로 적용한 결과,
                          checks는 적용 뒤 점검 결과.
"""
import contextlib
import functools
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agent import graph as agent
from src.core import config

STATIC_DIR = Path(__file__).parent / "static"
MAX_HISTORY = 10  # 대화 맥락으로 LLM에 넘기는 최근 메시지 수


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=40)
    state: dict = Field(default_factory=dict)
    checks: list[dict] = Field(default_factory=list, max_length=40)
    settings_schema: list[dict] = Field(default_factory=list, max_length=80)


class ApproveRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    approve: bool
    result: dict = Field(default_factory=dict)
    checks: list[dict] = Field(default_factory=list, max_length=40)


@functools.lru_cache(maxsize=1)
def get_graph():
    return agent.build_default_graph()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config.load_settings()  # 필수 설정(.env)이 없으면 여기서 ConfigError를 내서 서버가 뜨지 않는다
    yield


app = FastAPI(title="데스크 셋업 시뮬레이터", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _relay(events):
    """그래프 이벤트를 SSE로 흘려보낸다. 오류는 내용을 숨기고 종류만 알린다."""
    try:
        for event in events:
            if event["type"] == "result":
                yield _sse({"type": "done", "message": event["answer"]})
            else:
                yield _sse(event)
    except Exception as e:
        # 자격 증명·모델 권한 오류 등. 오류 내용은 노출하지 않고 종류만 알린다
        yield _sse({"type": "error", "message": f"답변을 만들지 못했어요: {type(e).__name__}. .env의 Bedrock 설정을 확인해 주세요."})


def _sse_response(events) -> StreamingResponse:
    return StreamingResponse(
        _relay(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    history = [h.model_dump() for h in req.history][-MAX_HISTORY:]

    def events():
        yield from agent.ask_stream(get_graph(), req.question, history, req.state, req.checks, req.settings_schema)

    return _sse_response(events())


@app.post("/api/chat/approve")
def chat_approve(req: ApproveRequest) -> StreamingResponse:
    def events():
        yield from agent.resume_stream(get_graph(), req.thread_id, req.approve, req.result, req.checks)

    return _sse_response(events())
