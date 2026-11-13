"""Bedrock 없이 도는 테스트: 사용자가 대화 사이에 화면에서 직접 바꾼 것(마우스로 옮기기·삭제·슬라이더)을 에이전트가 알게 하는지."""
import copy

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage

from src import agent
from src.context import describe, describe_changes
from src import app as web
from tests.test_agent import CHECKS, SCHEMA, STATE, FakeLLM, text

COLL = {"drawers": {"label": "서랍장", "max": 4, "fields": [{"key": "x", "label": "좌우 이동"}, {"key": "z", "label": "앞뒤 이동"}]},
        "monitors": {"label": "추가 모니터", "max": 3, "fields": [{"key": "x", "label": "좌우 위치"}]}}
MON2 = {"desk": 0, "mon": "27", "x": 650, "back": 200}


def states():
    prev = copy.deepcopy(STATE)
    prev.update(monitors=[MON2], drawers=[{"where": "underR", "x": 0, "z": 0}])
    cur = copy.deepcopy(prev)
    cur.update(monitors=[], drawers=[{"where": "underR", "x": -410, "z": 370}], deskW=1600)
    return prev, cur


def test_changes_list_removed_moved_and_slider_values():
    prev, cur = states()
    out = describe_changes(prev, cur, SCHEMA, COLL)
    assert any("추가 모니터: 1개 → 0개" in c and "삭제" in c for c in out)
    assert "서랍장 1: 좌우 이동(x) 0 → -410, 앞뒤 이동(z) 0 → 370" in out
    assert "책상 가로 (폭)(deskW): 1400 → 1600" in out


def test_no_changes_without_previous_state_or_when_equal():
    prev, _ = states()
    assert describe_changes(None, prev) == [] and describe_changes(prev, copy.deepcopy(prev)) == []


def test_many_changes_are_capped():
    prev = {f"k{i}": 0 for i in range(30)}
    out = describe_changes(prev, {k: 1 for k in prev})
    assert len(out) == 16 and out[-1] == "그 밖에 15건"


def test_describe_says_explicitly_when_a_collection_is_empty():
    _, cur = states()
    assert "추가 모니터: 없음 (모니터는 모니터 1 하나뿐)" in describe(cur, [])
    assert "없음" not in describe({}, []).replace("기타 물건: 없음", "")  # 상태에 키가 없으면 적지 않는다


def test_changes_are_put_right_before_the_question():
    prev, cur = states()
    llm = FakeLLM(text("모니터 1 하나예요."))
    g = agent.build_graph(llm)
    history = [{"role": "user", "content": "모니터 추가해줘"}, {"role": "assistant", "content": "모니터 2를 추가했어요."}]
    events = list(agent.ask_stream(g, "모니터 몇 대야?", history, cur, CHECKS, SCHEMA, COLL, thread_id="t1", prev_state=prev))
    last = llm.calls[0][-1]
    assert isinstance(last, HumanMessage) and last.content.startswith("[화면에서 직접 바꾼 것")
    assert "1개 → 0개" in last.content and last.content.endswith("[질문]\n모니터 몇 대야?")
    assert "'현재 셋업'은 이번 질문을 받은 때의 화면" in llm.calls[0][0].content and isinstance(llm.calls[0][0], SystemMessage)
    assert any("직접 바꾼 것 3건" in e.get("detail", "") for e in events if e["type"] == "step")


def test_api_passes_prev_state_through(monkeypatch):
    prev, cur = states()
    llm = FakeLLM(text("네."))
    monkeypatch.setattr(web, "get_graph", lambda: agent.build_graph(llm))
    r = TestClient(web.app).post("/api/chat/stream", json={"question": "어때?", "history": [], "state": cur, "prev_state": prev,
                                                           "checks": CHECKS, "settings_schema": SCHEMA, "collections": COLL})
    assert r.status_code == 200 and "직접 바꾼 것" in llm.calls[0][-1].content


def test_eval_synthesizes_the_state_after_approval():
    from src.evaluate import synth_state
    state = {"deskW": 1400, "drawers": [{"where": "underL"}, {"where": "underR"}]}
    after = synth_state(state, {"changes": {"deskW": 1600}, "array_ops": [{"op": "add", "collection": "drawers", "values": {"tiers": 4}},
                                                                         {"op": "remove", "collection": "drawers", "index": 0}]})
    assert after == {"deskW": 1600, "drawers": [{"where": "underR"}, {"tiers": 4}]} and state["deskW"] == 1400


def test_api_accepts_null_prev_state_on_the_first_question(monkeypatch):
    llm = FakeLLM(text("네."))
    monkeypatch.setattr(web, "get_graph", lambda: agent.build_graph(llm))
    r = TestClient(web.app).post("/api/chat/stream", json={"question": "어때?", "history": [], "state": STATE, "prev_state": None})
    assert r.status_code == 200 and "직접 바꾼 것" not in llm.calls[0][-1].content


def test_room_rotation_is_described_as_window_direction():
    room = {**STATE, "room": True, "roomW": 3000, "roomD": 3300, "roomH": 2300, "win": True}
    assert "앉았을 때 창문이 앞(모니터 뒤)" in describe({**room, "roomRot": 0}, [])
    assert "방 돌리기(roomRot) 90° = 앉았을 때 창문이 왼쪽" in describe({**room, "roomRot": 90}, [])
    assert "창문 없음" in describe({**room, "win": False}, [])
