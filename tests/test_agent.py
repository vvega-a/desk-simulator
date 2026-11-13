"""Bedrock 없이 도는 테스트: 상태 설명, 설정 검증, ReAct 그래프(도구·승인), SSE 엔드포인트."""
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, ToolMessage

from src import agent
from src.context import describe
from src.settings import diff_items, format_settings, validate_patch
from src import app as web

STATE = {
    "deskW": 1400, "deskD": 700, "deskH": 720, "mon": "27", "curved": False, "mount": "stand",
    "monBack": 200, "monLift": 100, "monX": 0, "subCount": 0, "kb": "tkl", "mouse": True, "mat": "900x400",
    "laptop": "none", "tower": "none", "speakers": False, "chair": False, "userH": 175,
    "extraDesks": [], "drawers": [{"where": "underL"}], "sets": [], "pegs": [], "room": False,
}
CHECKS = [{"title": "눈 ↔ 화면 거리", "value": "약 620 mm", "status": "warn", "note": "권장 670~870mm"}]
SCHEMA = [
    {"key": "deskW", "label": "가로 (폭)", "section": "책상", "type": "range", "min": 600, "max": 2400, "step": 10, "unit": "mm"},
    {"key": "mon", "label": "모델", "section": "모니터 1", "type": "select",
     "options": [{"value": "27", "label": '27" 16:9'}, {"value": "32", "label": '32" 16:9'}]},
    {"key": "speakers", "label": "북쉘프 스피커", "section": "기타 물건", "type": "toggle"},
    {"key": "subCount", "label": "개수", "section": "모니터 2", "type": "seg",
     "options": [{"value": 0, "label": "없음"}, {"value": 1, "label": "1대"}, {"value": 2, "label": "2대 (양옆)"}]},
]


def text(s):
    return [AIMessageChunk(content=s)]


def call(name, args, cid="c1"):
    return [AIMessageChunk(content="", tool_call_chunks=[{"name": name, "args": json.dumps(args, ensure_ascii=False), "id": cid, "index": 0}])]


class FakeLLM:
    """턴마다 미리 정한 조각을 내보내는 가짜 LLM. 매 호출에 받은 메시지를 기록한다."""

    def __init__(self, *turns):
        self.turns, self.calls = list(turns), []

    def bind_tools(self, tools):
        return self

    def stream(self, messages):
        self.calls.append(list(messages))
        yield from self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]


def run(llm, question="어때?", history=None, collections=None, slots=None, state=None):
    g = agent.build_graph(llm)
    events = agent.ask_stream(g, question, history or [], state or STATE, CHECKS, SCHEMA, collections, slots, thread_id="t1")
    return g, list(events)


def tool_messages(llm, call_index):
    return [m.content for m in llm.calls[call_index] if isinstance(m, ToolMessage)]


# ---- 상태 설명 -------------------------------------------------------------------------------

def test_describe_uses_only_state_values():
    out = describe(STATE, CHECKS)
    assert "1400mm × 세로 700mm" in out and '27" 16:9' in out and "텐키리스" in out
    assert "서랍장 1개" in out and "[주의] 눈 ↔ 화면 거리: 약 620 mm" in out


def test_describe_tolerates_empty_state():
    assert "기타 물건: 없음" in describe({}, [])


# ---- 설정 검증 -------------------------------------------------------------------------------

def test_validate_snaps_range_and_reports_note():
    clean, rejected, notes = validate_patch({"deskW": 1643}, SCHEMA)
    assert clean == {"deskW": 1640} and not rejected and "deskW" in notes


def test_validate_clamps_to_range():
    assert validate_patch({"deskW": 99999}, SCHEMA)[0] == {"deskW": 2400}
    assert validate_patch({"deskW": "100"}, SCHEMA)[0] == {"deskW": 600}


def test_validate_rejects_bad_values():
    clean, rejected, _ = validate_patch(
        {"bogus": 1, "deskW": "abc", "speakers": "maybe", "mon": "99", "x": None}, SCHEMA
    )
    assert clean == {}
    assert set(rejected) == {"bogus", "deskW", "speakers", "mon", "x"}
    assert "27=" in rejected["mon"]  # 허용 선택지를 알려 준다


def test_validate_accepts_option_by_label_and_keeps_type():
    assert validate_patch({"mon": '32" 16:9'}, SCHEMA)[0] == {"mon": "32"}
    assert validate_patch({"subCount": "2"}, SCHEMA)[0] == {"subCount": 2}
    assert validate_patch({"speakers": "true"}, SCHEMA)[0] == {"speakers": True}
    assert validate_patch({"deskW": True}, SCHEMA)[1].keys() == {"deskW"}  # bool은 숫자로 안 친다


def test_diff_skips_unchanged_values():
    items, same = diff_items({"deskW": 1600, "mon": "27"}, STATE, SCHEMA)
    assert items == [{"key": "deskW", "label": "가로 (폭)", "from": "1400mm", "to": "1600mm"}]
    assert same == ["mon"]


def test_format_settings_lists_sections_and_ranges():
    assert "[책상] deskW" in format_settings(SCHEMA)
    detail = format_settings(SCHEMA, "책상")
    assert "600~2400mm, 10mm 단위" in detail
    assert "27=" in format_settings(SCHEMA, "모니터 1")
    assert "섹션이 없어요" in format_settings(SCHEMA, "없는섹션")


# ---- 그래프 ----------------------------------------------------------------------------------

def test_plain_answer_streams_sentences_without_tools():
    llm = FakeLLM([AIMessageChunk(content="거리가 조금 "), AIMessageChunk(content="가까워요. 뒤로 밀어 보세요.")])
    _, events = run(llm, "괜찮아?")
    assert [e["text"] for e in events if e["type"] == "token"] == ["거리가 조금 가까워요. ", "뒤로 밀어 보세요."]
    assert events[-1] == {"type": "result", "answer": "거리가 조금 가까워요. 뒤로 밀어 보세요.", "tools": []}
    assert "약 620 mm" in llm.calls[0][0].content and llm.calls[0][-1].content == "괜찮아?"


def test_history_is_passed_to_llm():
    llm = FakeLLM(text("네."))
    run(llm, "다음", [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "안녕하세요"}])
    assert [m.content for m in llm.calls[0][1:]] == ["안녕", "안녕하세요", "다음"]


def test_list_settings_tool_result_goes_back_to_llm():
    llm = FakeLLM(call("ListSettings", {"section": "책상"}), text("가로는 600~2400이에요."))
    _, events = run(llm)
    assert "600~2400mm" in tool_messages(llm, 1)[0]
    assert events[-1]["type"] == "result" and not any(e["type"] == "approval" for e in events)


def test_propose_pauses_for_approval_then_applies():
    llm = FakeLLM(
        call("ListSettings", {"section": "책상"}, "c1"),
        call("ProposeChanges", {"changes": {"deskW": 1643}, "reason": "넓게 쓰려고"}, "c2"),
        text("가로를 1640으로 바꿨어요."),
    )
    g, events = run(llm)
    approval = events[-1]
    assert approval["type"] == "approval" and approval["thread_id"] == "t1"
    assert approval["patch"] == {"changes": {"deskW": 1640}, "array_ops": []}
    assert approval["changes"] == [{"key": "deskW", "label": "가로 (폭)", "from": "1400mm", "to": "1640mm"}]
    assert approval["reason"] == "넓게 쓰려고" and "deskW" in approval["notes"]
    assert not any(e["type"] == "result" for e in events)  # 승인 전에는 끝나지 않는다

    new_checks = [{"title": "좌우 여유", "value": "왼쪽 400", "status": "ok", "note": "충분"}]
    resumed = list(agent.resume_stream(g, "t1", True, {"applied": {"deskW": 1640}, "rejected": {}}, new_checks))
    assert resumed[-1]["answer"] == "가로를 1640으로 바꿨어요."
    assert [t["name"] for t in resumed[-1]["tools"]] == ["ListSettings", "ProposeChanges"]  # 실제 실행된 도구 기록
    assert "적용했어요: deskW=1640" in resumed[-1]["tools"][1]["result"]
    seen = tool_messages(llm, 2)[-1]
    assert "적용했어요: deskW=1640" in seen and "좌우 여유" in seen  # 적용 결과와 새 점검 결과를 LLM이 본다


def test_user_rejection_applies_nothing():
    llm = FakeLLM(call("ProposeChanges", {"changes": {"mon": "32"}, "reason": "크게"}), text("알겠어요."))
    g, events = run(llm)
    assert events[-1]["type"] == "approval"
    resumed = list(agent.resume_stream(g, "t1", False))
    assert resumed[-1]["type"] == "result"
    assert "거절" in tool_messages(llm, 1)[-1] and "적용했어요" not in tool_messages(llm, 1)[-1]


def test_invalid_proposal_never_asks_for_approval():
    llm = FakeLLM(call("ProposeChanges", {"changes": {"bogus": 1}, "reason": "x"}), text("다시 확인할게요."))
    _, events = run(llm)
    assert not any(e["type"] == "approval" for e in events) and events[-1]["type"] == "result"
    assert "알 수 없는 항목" in tool_messages(llm, 1)[0]


def test_proposal_with_no_real_change_is_not_shown():
    llm = FakeLLM(call("ProposeChanges", {"changes": {"mon": "27"}, "reason": "x"}), text("이미 그래요."))
    _, events = run(llm)
    assert not any(e["type"] == "approval" for e in events)
    assert "이미 그 값이에요" in tool_messages(llm, 1)[0]


def test_loop_is_capped():
    llm = FakeLLM(call("ListSettings", {}))  # 항상 도구만 부른다
    _, events = run(llm)
    assert len(llm.calls) == agent.MAX_LOOPS
    assert events[-1]["type"] == "result" and "멈출게요" in events[-1]["answer"]


# ---- 엔드포인트 ------------------------------------------------------------------------------

def _client(monkeypatch, llm):
    graph = agent.build_graph(llm)
    monkeypatch.setattr(web, "get_graph", lambda: graph)
    return TestClient(web.app)  # lifespan은 with 없이는 실행되지 않으므로 .env 없이도 된다


def _events(res):
    return [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]


BODY = {"question": "가로 1600으로", "state": STATE, "checks": CHECKS, "settings_schema": SCHEMA}


def test_stream_endpoint_plain_answer(monkeypatch):
    res = _client(monkeypatch, FakeLLM(text("좋아요."))).post("/api/chat/stream", json=BODY)
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = _events(res)
    assert events[-1] == {"type": "done", "message": "좋아요.", "tools": []} and any(e["type"] == "token" for e in events)


def test_stream_then_approve_endpoints(monkeypatch):
    llm = FakeLLM(call("ProposeChanges", {"changes": {"deskW": 1600}, "reason": "넓게"}), text("바꿨어요."))
    client = _client(monkeypatch, llm)
    first = _events(client.post("/api/chat/stream", json=BODY))
    assert first[-1]["type"] == "approval" and first[-1]["patch"] == {"changes": {"deskW": 1600}, "array_ops": []}
    second = _events(client.post("/api/chat/approve", json={
        "thread_id": first[-1]["thread_id"], "approve": True, "result": {"applied": {"deskW": 1600}}, "checks": CHECKS}))
    assert second[-1]["type"] == "done" and second[-1]["message"] == "바꿨어요." and second[-1]["tools"][0]["name"] == "ProposeChanges"


def test_approve_unknown_thread_reports_error_without_details(monkeypatch):
    client = _client(monkeypatch, FakeLLM(text("x")))
    events = _events(client.post("/api/chat/approve", json={"thread_id": "nope", "approve": True}))
    assert events[-1]["type"] == "error"


def test_error_hides_details(monkeypatch):
    class Boom(FakeLLM):
        def stream(self, messages):
            raise RuntimeError("secret-key-123")
            yield

    events = _events(_client(monkeypatch, Boom([])).post("/api/chat/stream", json=BODY))
    assert events[-1]["type"] == "error"
    assert "RuntimeError" in events[-1]["message"] and "secret-key-123" not in events[-1]["message"]


def test_request_validation(monkeypatch):
    client = _client(monkeypatch, FakeLLM(text("x")))
    assert client.post("/api/chat/stream", json={"question": ""}).status_code == 422
    assert client.post("/api/chat/stream", json={"question": "a" * 1001}).status_code == 422
    assert client.post("/api/chat/stream", json={"question": "a", "history": [{"role": "system", "content": "x"}]}).status_code == 422
    assert client.post("/api/chat/approve", json={"thread_id": "t"}).status_code == 422


def test_index_and_health(monkeypatch):
    client = _client(monkeypatch, FakeLLM(text("x")))
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "window.deskSim" in client.get("/").text
