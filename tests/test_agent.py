"""Bedrock 없이 도는 테스트: 상태 설명, 그래프 스트리밍, SSE 엔드포인트."""
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from src.agent import graph as agent
from src.agent.context import describe
from src.web import app as web

STATE = {
    "deskW": 1400, "deskD": 700, "deskH": 720, "mon": "27", "curved": False, "mount": "stand",
    "monBack": 200, "monLift": 100, "monX": 0, "subCount": 0, "kb": "tkl", "mouse": True, "mat": "900x400",
    "laptop": "none", "tower": "none", "speakers": False, "chair": False, "userH": 175,
    "extraDesks": [], "drawers": [{"where": "underL"}], "sets": [], "pegs": [], "room": False,
}
CHECKS = [{"title": "눈 ↔ 화면 거리", "value": "약 620 mm", "status": "warn", "note": "권장 670~870mm"}]


class FakeLLM:
    """stream이 조각을 내보내는 가짜 LLM. 받은 메시지를 기록한다."""

    def __init__(self, pieces):
        self.pieces, self.seen = pieces, None

    def stream(self, messages):
        self.seen = messages
        for p in self.pieces:
            yield AIMessageChunk(content=p)


def test_describe_uses_only_state_values():
    text = describe(STATE, CHECKS)
    assert "1400mm × 세로 700mm" in text
    assert '27" 16:9' in text and "텐키리스" in text
    assert "서랍장 1개" in text
    assert "[주의] 눈 ↔ 화면 거리: 약 620 mm" in text


def test_describe_tolerates_empty_state():
    assert "기타 물건: 없음" in describe({}, [])


def test_graph_streams_steps_and_sentence_tokens():
    llm = FakeLLM(["거리가 조금 ", "가까워요. 모니터를 ", "뒤로 밀어 보세요."])
    events = list(agent.ask_stream(agent.build_graph(llm), "괜찮아?", [], STATE, CHECKS))
    kinds = [(e["type"], e.get("node"), e.get("phase")) for e in events]
    assert kinds[:2] == [("step", "context", "start"), ("step", "context", "done")]
    assert [e["text"] for e in events if e["type"] == "token"] == ["거리가 조금 가까워요. ", "모니터를 뒤로 밀어 보세요."]
    assert events[-1] == {"type": "result", "answer": "거리가 조금 가까워요. 모니터를 뒤로 밀어 보세요."}
    system = llm.seen[0].content
    assert "약 620 mm" in system  # 상태 글이 프롬프트에 들어갔다
    assert llm.seen[-1].content == "괜찮아?"


def test_history_is_passed_to_llm():
    llm = FakeLLM(["네."])
    history = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요"}]
    list(agent.ask_stream(agent.build_graph(llm), "다음 질문", history, STATE, CHECKS))
    assert [m.content for m in llm.seen[1:]] == ["안녕", "안녕하세요", "다음 질문"]


def _client(monkeypatch, llm):
    monkeypatch.setattr(web, "get_graph", lambda: agent.build_graph(llm))
    return TestClient(web.app)  # lifespan은 with 없이는 실행되지 않으므로 .env 없이도 된다


def _events(res):
    return [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]


def test_chat_stream_endpoint(monkeypatch):
    res = _client(monkeypatch, FakeLLM(["좋아요."])).post(
        "/api/chat/stream", json={"question": "어때?", "state": STATE, "checks": CHECKS}
    )
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = _events(res)
    assert events[-1] == {"type": "done", "message": "좋아요."}
    assert any(e["type"] == "token" for e in events)


def test_chat_stream_reports_error_without_details(monkeypatch):
    class Boom:
        def stream(self, messages):
            raise RuntimeError("secret-key-123")
            yield

    events = _events(_client(monkeypatch, Boom()).post("/api/chat/stream", json={"question": "어때?"}))
    assert events[-1]["type"] == "error"
    assert "RuntimeError" in events[-1]["message"] and "secret-key-123" not in events[-1]["message"]


def test_chat_stream_validates_request(monkeypatch):
    client = _client(monkeypatch, FakeLLM(["x"]))
    assert client.post("/api/chat/stream", json={"question": ""}).status_code == 422
    assert client.post("/api/chat/stream", json={"question": "a" * 1001}).status_code == 422
    assert client.post("/api/chat/stream", json={"question": "a", "history": [{"role": "system", "content": "x"}]}).status_code == 422


def test_index_and_health(monkeypatch):
    client = _client(monkeypatch, FakeLLM(["x"]))
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "window.deskSim" in client.get("/").text
