"""물건 추가·수정·삭제, 장면 동작(슬롯·초기화), 화면 보기 도구 테스트(Bedrock 없이)."""
from langchain_core.messages import AIMessageChunk

from src.agent import graph as agent
from src.agent.context import describe
from src.agent.settings import format_settings, validate_ops, validate_scene_action
from tests.test_agent import CHECKS, SCHEMA, STATE, FakeLLM, call, run, text, tool_messages

COLLECTIONS = {
    "drawers": {"label": "서랍장", "max": 4, "first": 1, "fields": [
        {"key": "where", "label": "위치", "type": "select",
         "options": [{"value": "underL", "label": "책상 밑 왼쪽"}, {"value": "underR", "label": "책상 밑 오른쪽"}]},
        {"key": "w", "label": "가로", "type": "range", "min": 250, "max": 1200, "step": 10, "unit": "mm"},
        {"key": "tiers", "label": "서랍 칸 수", "type": "range", "min": 1, "max": 6, "step": 1, "unit": "단"},
        {"key": "casters", "label": "바퀴", "type": "toggle"}]},
    "extraDesks": {"label": "추가 책상", "max": 3, "first": 2, "fields": [
        {"key": "w", "label": "가로", "type": "range", "min": 400, "max": 2400, "step": 10, "unit": "mm"}]},
}
DRAWER = {"where": "underL", "w": 400, "tiers": 3, "casters": True}
WITH_DRAWER = {**STATE, "drawers": [DRAWER]}
SLOTS = {"A": '1400×700 · 27"', "B": None, "C": None}


# ---- 검증 ------------------------------------------------------------------------------------

def test_add_validates_and_clamps_values():
    ops, items, problems, notes = validate_ops(
        [{"op": "add", "collection": "drawers", "values": {"w": 99999, "tiers": 4, "bogus": 1}}], COLLECTIONS, WITH_DRAWER)
    assert ops == [{"op": "add", "collection": "drawers", "values": {"w": 1200, "tiers": 4}}]
    assert items[0]["label"] == "서랍장 추가" and "가로 1200mm" in items[0]["to"] and items[0]["from"] == ""
    assert any("bogus" in p for p in problems) and "drawers.w" in notes


def test_add_respects_max_including_earlier_ops():
    full = {**STATE, "drawers": [DRAWER] * 3}
    ops, _, problems, _ = validate_ops([{"op": "add", "collection": "drawers"}] * 2, COLLECTIONS, full)
    assert len(ops) == 1 and any("최대 4개" in p for p in problems)


def test_update_keeps_only_changed_values_and_shows_diff():
    ops, items, problems, _ = validate_ops(
        [{"op": "update", "collection": "drawers", "index": 0, "values": {"w": 500, "tiers": 3}}], COLLECTIONS, WITH_DRAWER)
    assert ops == [{"op": "update", "collection": "drawers", "index": 0, "values": {"w": 500}}]
    assert items == [{"key": "drawers.0.w", "label": "서랍장 1: 가로", "from": "400mm", "to": "500mm"}] and not problems


def test_update_with_no_change_is_a_problem():
    ops, items, problems, _ = validate_ops(
        [{"op": "update", "collection": "drawers", "index": 0, "values": {"w": 400}}], COLLECTIONS, WITH_DRAWER)
    assert not ops and not items and "이미 그 값" in problems[0]


def test_remove_and_bad_index():
    ops, items, problems, _ = validate_ops([
        {"op": "remove", "collection": "drawers", "index": 0},
        {"op": "remove", "collection": "drawers", "index": 0},  # 이미 하나 지웠으니 더 없다
        {"op": "update", "collection": "extraDesks", "index": 1, "values": {"w": 800}},
    ], COLLECTIONS, WITH_DRAWER)
    assert ops == [{"op": "remove", "collection": "drawers", "index": 0}]
    assert items[0]["label"] == "서랍장 1 삭제"
    assert len(problems) == 2 and all("번호" in p for p in problems)


def test_numbering_follows_first_label():
    state = {**STATE, "extraDesks": [{"w": 1000}]}
    _, items, _, _ = validate_ops([{"op": "update", "collection": "extraDesks", "index": 0, "values": {"w": 1200}}], COLLECTIONS, state)
    assert items[0]["label"] == "추가 책상 2: 가로"  # 화면에서는 2번부터 보인다


def test_unknown_collection_and_op():
    ops, _, problems, _ = validate_ops(
        [{"op": "add", "collection": "lamps"}, {"op": "explode", "collection": "drawers", "index": 0}, "x"], COLLECTIONS, WITH_DRAWER)
    assert not ops and len(problems) == 3


def test_scene_action_validation():
    assert validate_scene_action("reset", None, SLOTS)[0] == {"action": "reset"}
    action, item, _ = validate_scene_action("load_slot", "A", SLOTS)
    assert action == {"action": "load_slot", "slot": "A"} and "불러오기" in item["label"]
    assert "비어 있어요" in validate_scene_action("load_slot", "B", SLOTS)[2]
    assert "A, B, C" in validate_scene_action("save_slot", "Z", SLOTS)[2]
    assert "덮어씀" in validate_scene_action("save_slot", "A", SLOTS)[1]["to"]
    assert validate_scene_action("explode", "A", SLOTS)[0] is None


def test_format_settings_lists_collections():
    overview = format_settings(SCHEMA, None, COLLECTIONS)
    assert "collection=drawers" in overview and "최대 4개" in overview
    detail = format_settings(SCHEMA, "서랍장", COLLECTIONS)
    assert "번호(index)는 0부터" in detail and "tiers" in detail and "250~1200mm" in detail
    assert "추가 책상" in format_settings(SCHEMA, "없음", COLLECTIONS)


def test_describe_lists_items_with_index_and_slots():
    out = describe(WITH_DRAWER, [], SLOTS)
    assert "서랍장 1개 (번호는 0부터)" in out and "[0] where=underL, w=400" in out
    assert "저장 슬롯: A:" in out and "B: 비어 있음" in out


# ---- 그래프 ----------------------------------------------------------------------------------

def test_array_ops_go_through_approval_and_result_reaches_llm():
    llm = FakeLLM(
        call("ProposeChanges", {"array_ops": [{"op": "add", "collection": "drawers", "values": {"w": 700}}], "reason": "수납"}),
        text("서랍장을 놓았어요."),
    )
    g, events = run(llm, "서랍장 추가", collections=COLLECTIONS, slots=SLOTS, state=WITH_DRAWER)
    approval = events[-1]
    assert approval["type"] == "approval"
    assert approval["patch"] == {"changes": {}, "array_ops": [{"op": "add", "collection": "drawers", "values": {"w": 700}}]}
    assert approval["changes"][0]["label"] == "서랍장 추가"
    result = {"applied": {}, "rejected": {}, "ops": [{"ok": True, "text": "서랍장 2 추가"}, {"ok": False, "text": "타공판은(는) 최대 4개예요"}]}
    resumed = list(agent.resume_stream(g, "t1", True, result, CHECKS))
    assert resumed[-1]["type"] == "result"
    seen = tool_messages(llm, 1)[-1]
    assert "서랍장 2 추가" in seen and "적용하지 못한 것: 타공판은(는) 최대 4개예요" in seen


def test_scene_action_asks_approval_and_empty_slot_does_not():
    llm = FakeLLM(call("SceneAction", {"action": "load_slot", "slot": "A", "reason": "비교"}), text("불러왔어요."))
    g, events = run(llm, collections=COLLECTIONS, slots=SLOTS)
    assert events[-1]["type"] == "approval" and events[-1]["patch"] == {"scene_action": {"action": "load_slot", "slot": "A"}}
    resumed = list(agent.resume_stream(g, "t1", True, {"ops": [{"ok": True, "text": "슬롯 A를 불러왔어요"}]}, CHECKS))
    assert resumed[-1]["type"] == "result" and "슬롯 A를 불러왔어요" in tool_messages(llm, 1)[-1]

    empty = FakeLLM(call("SceneAction", {"action": "load_slot", "slot": "B", "reason": "x"}), text("비어 있네요."))
    _, events = run(empty, collections=COLLECTIONS, slots=SLOTS)
    assert not any(e["type"] == "approval" for e in events) and "비어 있어요" in tool_messages(empty, 1)[0]


def test_set_view_notifies_browser_without_approval():
    llm = FakeLLM(call("SetView", {"view": "front", "auto_rotate": False}), text("정면으로 바꿨어요."))
    _, events = run(llm)
    action = next(e for e in events if e["type"] == "client_action")
    assert action == {"type": "client_action", "action": "set_view", "view": "front", "auto_rotate": False}
    assert not any(e["type"] == "approval" for e in events) and events[-1]["type"] == "result"
    assert "시점=정면" in tool_messages(llm, 1)[0]


def test_set_view_rejects_unknown_view_and_empty_call():
    llm = FakeLLM(call("SetView", {}), text("뭘 바꿀까요?"))
    _, events = run(llm)
    assert not any(e["type"] == "client_action" for e in events) and "바꿀 것이 없어요" in tool_messages(llm, 1)[0]


def test_one_approval_per_tool_per_turn():
    two_calls = [AIMessageChunk(content="", tool_call_chunks=[
        {"name": "ProposeChanges", "args": '{"changes": {"deskW": 1600}, "reason": "a"}', "id": "c1", "index": 0},
        {"name": "ProposeChanges", "args": '{"changes": {"mon": "32"}, "reason": "b"}', "id": "c2", "index": 1},
    ])]
    llm = FakeLLM(two_calls, text("끝."))
    g, events = run(llm)
    assert sum(e["type"] == "approval" for e in events) == 1
    list(agent.resume_stream(g, "t1", True, {"applied": {"deskW": 1600}}, CHECKS))
    assert any("한 번에 하나만" in m for m in tool_messages(llm, 1))


# ---- 대화 기록의 도구 호출 복원 ---------------------------------------------------------------

def test_history_restores_tool_calls_so_model_does_not_just_claim_done():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    history = [
        {"role": "user", "content": "서랍장 추가해줘"},
        {"role": "assistant", "content": "추가했어요.", "tools": [
            {"name": "ProposeChanges", "args": {"reason": "수납"}, "result": "적용했어요: 서랍장 1 추가"}]},
    ]
    llm = FakeLLM(text("네."))
    run(llm, "또 해줘", history)
    msgs = [m for m in llm.calls[0][1:]]  # 시스템 메시지 제외
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]
    assert msgs[1].tool_calls[0]["name"] == "ProposeChanges" and msgs[1].tool_calls[0]["id"] == msgs[2].tool_call_id
    assert msgs[2].content == "적용했어요: 서랍장 1 추가" and msgs[3].content == "추가했어요."


def test_result_trace_lists_only_executed_tools_of_this_question():
    llm = FakeLLM(call("ListSettings", {"section": "책상"}, "c1"), text("가로는 600~2400이에요."))
    old = [{"role": "user", "content": "이전"}, {"role": "assistant", "content": "이전 답", "tools": [
        {"name": "SetView", "args": {"view": "top"}, "result": "화면을 바꿨어요"}]}]
    _, events = run(llm, "가로 범위가 뭐야?", old)
    tools = events[-1]["tools"]
    assert [t["name"] for t in tools] == ["ListSettings"]  # 이전 턴의 SetView는 섞이지 않는다
    assert tools[0]["args"] == {"section": "책상"} and "600~2400mm" in tools[0]["result"]


def test_history_tools_are_validated_by_the_api(monkeypatch):
    from fastapi.testclient import TestClient

    from src.web import app as web

    monkeypatch.setattr(web, "get_graph", lambda: agent.build_graph(FakeLLM(text("x"))))
    client = TestClient(web.app)
    ok = {"question": "a", "history": [{"role": "assistant", "content": "b", "tools": [{"name": "SetView", "args": {}, "result": "r"}]}]}
    assert client.post("/api/chat/stream", json=ok).status_code == 200
    too_many = {"question": "a", "history": [{"role": "assistant", "content": "b", "tools": [{"name": "t"}] * 9}]}
    assert client.post("/api/chat/stream", json=too_many).status_code == 422


# ---- 완료 주장 안전장치 -----------------------------------------------------------------------

def test_claiming_done_without_any_applied_tool_gets_a_truthful_notice():
    llm = FakeLLM(text("슬롯 A에 저장했어요."))  # 도구를 부르지 않고 했다고만 말한다
    _, events = run(llm, "슬롯 A에 저장해줘")
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert any("실제로 적용된 것이 없어요" in t for t in tokens)
    assert "실제로 적용된 것이 없어요" in events[-1]["answer"]  # 대화 기록에도 사실이 남는다


def test_no_notice_when_something_was_really_applied():
    llm = FakeLLM(call("SetView", {"view": "top"}), text("위에서 보여 드렸어요. 시점을 바꿨어요."))
    _, events = run(llm)
    assert not any("실제로 적용된" in e.get("text", "") for e in events if e["type"] == "token")


def test_read_only_tool_then_done_claim_still_gets_notice():
    llm = FakeLLM(call("ListSettings", {"section": "책상"}), text("서랍장을 추가했어요."))
    _, events = run(llm)
    assert "실제로 적용된 것이 없어요" in events[-1]["answer"]


def test_no_notice_for_plain_answers():
    llm = FakeLLM(text("눈과 화면 거리가 조금 가까워요."))
    _, events = run(llm)
    assert "실제로" not in events[-1]["answer"]
