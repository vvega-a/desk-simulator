"""데스크 셋업 시뮬레이터 + AI 채팅 서버(FastAPI). 실행: uvicorn src.app:app --port 8000

  GET  /                  시뮬레이터 화면(static/index.html)
  GET  /api/health        상태 확인
  POST /api/chat/stream   질문을 SSE로 보내 진행 단계(step)와 답변 조각(token)을 실시간으로 받는다.
                          본문: {question, history, state, prev_state, checks, settings_schema, collections, slots} — 브라우저의
                          window.deskSim.getState()/getChecks()/schema()/collections()/slots() 값. 서버는 상태를 보관하지 않는다.
                          끝은 {type: done, message}, {type: error, message}, 또는 승인 대기 {type: approval, thread_id, ...}.
  POST /api/chat/approve  승인 대기 중인 설정 변경에 대한 사용자의 결정을 받아 이어서 진행한다(SSE).
                          본문: {thread_id, approve, result?, checks?, state?} — result는 브라우저가 applyChanges로 적용한 결과,
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

from src import agent
from src import config

STATIC_DIR = Path(__file__).parent / "static"
MAX_HISTORY = 40  # 받는 대화 기록 수. 최근 10개는 원문, 그보다 오래된 것은 요약해서 LLM에 넘긴다(src/memory.py)


class ToolTrace(BaseModel):
    """어시스턴트 턴에서 실제로 실행된 도구 호출 하나. 다음 질문에서 호출·결과를 되살리는 데 쓴다."""

    name: str = Field(max_length=40)
    args: dict = Field(default_factory=dict)
    result: str = Field(default="", max_length=1000)


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
    tools: list[ToolTrace] = Field(default_factory=list, max_length=8)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=40)
    state: dict = Field(default_factory=dict)
    checks: list[dict] = Field(default_factory=list, max_length=40)
    settings_schema: list[dict] = Field(default_factory=list, max_length=80)
    collections: dict = Field(default_factory=dict)  # 추가 책상·서랍장 등 배열 물건의 값 정의(deskSim.collections())
    slots: dict = Field(default_factory=dict)  # 저장 슬롯 요약(deskSim.slots())
    prev_state: dict | None = None  # (첫 질문이면 null) 지난 답변을 마친 때의 상태. 지금 상태와 비교해 사용자가 화면에서 직접 바꾼 것을 알린다


class ApproveRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    approve: bool
    result: dict = Field(default_factory=dict)
    checks: list[dict] = Field(default_factory=list, max_length=40)
    state: dict = Field(default_factory=dict)  # 적용 뒤의 deskSim.getState(). 번호가 바뀐 목록을 에이전트가 읽는다


@functools.lru_cache(maxsize=1)
def get_graph():
    return agent.build_default_graph()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config.load_settings()  # 필수 설정(.env)이 없으면 여기서 ConfigError를 내서 서버가 뜨지 않는다
    get_graph()  # 그래프와 가이드 문서 검색기를 미리 만든다(문서 임베딩은 백그라운드에서 진행)
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
                yield _sse({"type": "done", "message": event["answer"], "tools": event.get("tools", [])})
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
        yield from agent.ask_stream(get_graph(), req.question, history, req.state, req.checks, req.settings_schema, req.collections, req.slots,
                                    prev_state=req.prev_state)

    return _sse_response(events())


@app.post("/api/chat/approve")
def chat_approve(req: ApproveRequest) -> StreamingResponse:
    def events():
        yield from agent.resume_stream(get_graph(), req.thread_id, req.approve, req.result, req.checks, req.state)

    return _sse_response(events())
