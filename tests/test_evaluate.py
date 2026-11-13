"""평가 러너의 채점 로직과 평가 파일(test_queries.json·fixture.json)의 정합성 테스트(Bedrock 없이)."""
import json
from pathlib import Path

from src.agent.graph import NOT_APPLIED_NOTICE
from src.evaluation import evaluate as ev

EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluation"
FIXTURE = json.loads((EVAL_DIR / "fixture.json").read_text(encoding="utf-8"))
CASES = json.loads((EVAL_DIR / "test_queries.json").read_text(encoding="utf-8"))["cases"]


def phase(approval=None, actions=(), answer="네.", tools=(), error=None):
    """채점용 요약(summarize의 결과 모양)을 만든다."""
    return {"approval": approval, "client_actions": list(actions), "tokens": answer or "", "answer": answer,
            "tools": list(tools), "error": error}


def approval(**patch):
    return {"type": "approval", "thread_id": "t", "patch": patch}


def case(expect, **kw):
    return {"id": "x", "category": "in", "question": "q", "expect": expect, **kw}


# ---- 채점 함수 --------------------------------------------------------------------------------

def test_subset_matches_nested_and_ignores_extra_keys():
    assert ev.subset({"a": 1}, {"a": 1, "b": 2})
    assert ev.subset({"op": "add", "values": {"w": 700}}, {"op": "add", "collection": "x", "values": {"w": 700, "h": 1}})
    assert not ev.subset({"values": {"w": 700}}, {"values": {"w": 800}})
    assert not ev.subset({"a": 1}, {"b": 1})


def test_check_patch_flags_wrong_and_missing_values():
    assert ev.check_patch({"changes": {"deskW": 1600}}, {"changes": {"deskW": 1600}}, True) == []
    assert "deskW=1500" in ev.check_patch({"changes": {"deskW": 1600}}, {"changes": {"deskW": 1500}}, True)[0]
    assert "mon 변경이 없어요" in ev.check_patch({"changes": {"mon": "32"}}, {"changes": {}}, True)[0]


def test_strict_rejects_unrequested_changes_but_non_strict_allows():
    got = {"changes": {"deskW": 1600, "deskD": 800}}
    assert ev.check_patch({"changes": {"deskW": 1600}}, got, False) == []
    assert "deskD" in ev.check_patch({"changes": {"deskW": 1600}}, got, True)[0]


def test_check_patch_ops_and_scene_action():
    add = {"op": "add", "collection": "drawers", "values": {"tiers": 4}}
    assert ev.check_patch({"array_ops": [{"op": "add", "collection": "drawers"}]}, {"array_ops": [add]}, True) == []
    assert ev.check_patch({"array_ops": [{"op": "remove", "collection": "drawers", "index": 0}]}, {"array_ops": [add]}, True)
    assert "요청하지 않은 작업 1개" in ev.check_patch({"array_ops": [add]}, {"array_ops": [add, add]}, True)[0]
    assert ev.check_patch({"scene_action": {"action": "reset"}}, {"scene_action": {"action": "reset"}}, True) == []
    assert ev.check_patch({"scene_action": {"action": "load_slot", "slot": "A"}}, {"scene_action": {"action": "reset"}}, True)


def test_summarize_collects_events():
    s = ev.summarize([{"type": "token", "text": "가"}, {"type": "client_action", "view": "top"},
                      {"type": "result", "answer": "끝", "tools": [{"name": "SetView"}]}])
    assert s["tokens"] == "가" and s["client_actions"] == [{"type": "client_action", "view": "top"}]
    assert s["answer"] == "끝" and s["tools"] == [{"name": "SetView"}] and ev.shown_text(s) == "끝"
    assert ev.shown_text(ev.summarize([{"type": "token", "text": "대기 전"}])) == "대기 전"  # 승인 대기로 끝나면 흐른 글


# ---- 한 번의 실행 채점 ------------------------------------------------------------------------

def test_run_passes_when_proposal_matches():
    c = case({"approval": True, "patch": {"changes": {"deskW": 1600}}, "strict": True})
    assert ev.check_run(c, phase(approval(changes={"deskW": 1600}), answer=None), None) == []


def test_run_fails_when_card_missing_or_unexpected():
    c = case({"approval": True, "patch": {"changes": {"deskW": 1600}}})
    assert "안 떴어요" in ev.check_run(c, phase(None), None)[0]
    assert "떴어요" in ev.check_run(case({"approval": False}), phase(approval(changes={})), None)[0]


def test_view_expectation_and_unrequested_view_change():
    c = case({"approval": False, "client_action": {"view": "front"}})
    ok = phase(actions=[{"type": "client_action", "action": "set_view", "view": "front"}], answer="정면이에요")
    assert ev.check_run(c, ok, None) == []
    wrong = phase(actions=[{"type": "client_action", "action": "set_view", "view": "top"}], answer="위에서예요")
    assert "기대와 달라요" in ev.check_run(c, wrong, None)[0]
    assert "요청하지 않은 화면 보기" in ev.check_run(case({"approval": False}), ok, None)[0]


def test_false_completion_claim_is_a_failure_only_if_notice_missing():
    c = case({"approval": False})
    bare = phase(answer="슬롯 A에 저장했어요.")
    assert any("완료를 주장" in f for f in ev.check_run(c, bare, None))
    guarded = phase(answer="슬롯 A에 저장했어요." + NOT_APPLIED_NOTICE)
    assert not any("완료를 주장" in f for f in ev.check_run(c, guarded, None))
    # 승인 카드가 떴거나 화면 보기를 바꿨다면 완료 표현은 정상
    assert not any("완료를 주장" in f for f in ev.check_run(case({"approval": False, "client_action": {"view": "top"}}),
        phase(actions=[{"type": "client_action", "action": "set_view", "view": "top"}], answer="바꿨어요."), None))


def test_reply_any_none_and_judge():
    c = case({"approval": False, "reply_any": ["620"], "reply_none": ["규칙:"], "judge": "refuse_and_redirect"})
    ok = phase(answer="거리는 약 620mm예요.")
    assert ev.check_run(c, ok, None, {"ok": True, "reason": ""}) == []
    assert any("기대한 문구" in f for f in ev.check_run(c, phase(answer="괜찮아요"), None, {"ok": True, "reason": ""}))
    assert any("규칙:" in f for f in ev.check_run(c, phase(answer="620 규칙: ..."), None, {"ok": True, "reason": ""}))
    assert any("판정 실패" in f for f in ev.check_run(c, ok, None, {"ok": False, "reason": "정보를 알려 줬다"}))
    assert any("판정을 못 했어요" in f for f in ev.check_run(c, ok, None, None))


def test_server_error_short_circuits():
    fails = ev.check_run(case({"approval": True}), phase(answer=None, error="답변을 만들지 못했어요"), None)
    assert fails and fails[0].startswith("서버 오류")


def test_after_resume_approve_and_reject():
    approve_case = case({"approval": True}, resume="approve")
    good = phase(answer="적용됐어요.", tools=[{"name": "ProposeChanges", "result": "적용했어요: deskW=1600"}])
    assert ev.check_run(approve_case, phase(approval(changes={}), answer=None), good) == []
    wrong = phase(answer="안내", tools=[{"name": "ProposeChanges", "result": "사용자가 거절했어요"}])
    assert any("적용했어요" in f for f in ev.check_run(approve_case, phase(approval(changes={}), answer=None), wrong))

    reject_case = case({"approval": True, "after_resume": {"tool_result_contains": "거절", "answer_none": ["적용했어요"]}}, resume="reject")
    ok = phase(answer="알겠어요, 그대로 둘게요.", tools=[{"name": "ProposeChanges", "result": "사용자가 거절했어요. 아무것도 적용하지 않았어요."}])
    assert ev.check_run(reject_case, phase(approval(changes={}), answer=None), ok) == []
    bad = phase(answer="적용했어요!", tools=[{"name": "ProposeChanges", "result": "사용자가 거절했어요."}])
    assert any("적용했어요" in f for f in ev.check_run(reject_case, phase(approval(changes={}), answer=None), bad))


def test_synth_result_mirrors_the_patch():
    r = ev.synth_result({"changes": {"deskW": 1600}, "array_ops": [{"op": "add", "collection": "drawers"}], "scene_action": {"action": "reset"}})
    assert r["applied"] == {"deskW": 1600} and len(r["ops"]) == 2 and all(o["ok"] for o in r["ops"])


def test_summarize_runs_aggregates_by_case_and_category():
    cases = [{"id": "a", "category": "in", "question": "qa"}, {"id": "b", "category": "out", "question": "qb"}]
    runs = [{"id": "a", "category": "in", "passed": True, "failures": [], "seconds": 1.0},
            {"id": "a", "category": "in", "passed": False, "failures": ["x"], "seconds": 3.0},
            {"id": "b", "category": "out", "passed": True, "failures": [], "seconds": 2.0}]
    s = ev.summarize_runs(cases, runs)
    assert (s["passed_runs"], s["total_runs"], s["cases_all_pass"]) == (2, 3, 1)
    assert s["by_category"]["in"]["passed"] == 1 and s["per_case"][0]["failures"] == ["x"]
    assert "2 / 3" in ev.render_report({"when": "t", "model": "m", "judge_model": "j", "repeats": 1}, s)


# ---- 평가 파일 정합성 -------------------------------------------------------------------------

def test_query_set_has_the_planned_shape():
    ids = [c["id"] for c in CASES]
    assert len(CASES) == 35 and len(set(ids)) == 35
    counts = {k: sum(c["category"] == k for c in CASES) for k in ("in", "out", "guardrail")}
    assert counts == {"in": 25, "out": 5, "guardrail": 5}
    assert sum(c["id"].startswith("shop_") for c in CASES) == 2  # 상품 검색 케이스
    assert sum(c["id"].startswith("rag_") for c in CASES) == 3  # 가이드 문서 검색(RAG) 케이스
    assert all(c["question"].strip() and "expect" in c for c in CASES)
    assert all(c.get("resume") in (None, "approve", "reject") for c in CASES)


def test_expected_keys_exist_in_the_fixture():
    schema_keys = {f["key"] for f in FIXTURE["schema"]}
    collections = FIXTURE["collections"]
    for c in CASES:
        patch = (c["expect"].get("patch") or {})
        for key in patch.get("changes") or {}:
            assert key in schema_keys, (c["id"], key)
        for op in patch.get("array_ops") or []:
            assert op["collection"] in collections, (c["id"], op)
            field_keys = {f["key"] for f in collections[op["collection"]]["fields"]}
            assert set(op.get("values") or {}) <= field_keys, (c["id"], op)
        assert set(c["expect"]) <= {"approval", "patch", "strict", "client_action", "reply_any", "reply_none", "judge", "after_resume", "tools_any"}, c["id"]


def test_fixture_matches_the_default_simulator():
    assert len(FIXTURE["schema"]) == 39 and set(FIXTURE["collections"]) == {"monitors", "extraDesks", "drawers", "sets", "pegs"}
    assert FIXTURE["state"]["monitors"] == [] and FIXTURE["slots"] == {"A": None, "B": None, "C": None}
    assert any(c["title"] == "눈 ↔ 화면 거리" for c in FIXTURE["checks"])


# ---- 인프라 오류 분리, 취소 뒤 재제안 ------------------------------------------------------------

def test_infra_errors_are_recognised_but_agent_errors_are_not():
    assert ev.is_infra_error("ThrottlingException: Too many requests, please wait")
    assert ev.is_infra_error("EventStreamError: serviceUnavailableException ... Bedrock is unable to process your request")
    assert not ev.is_infra_error("KeyError: 'patch'")
    assert not ev.is_infra_error("답변이 이상해요")


def test_run_case_retries_infra_errors_then_gives_up_and_marks_them(monkeypatch):
    calls = []

    def flaky(graph, fixture, case, judge_llm):
        calls.append(1)
        return {"id": "x", "category": "in", "passed": False, "failures": ["인프라 오류: Throttling"], "infra": True, "seconds": 0, "answer": ""}

    monkeypatch.setattr(ev, "_run_once", flaky)
    r = ev.run_case(None, {}, {"id": "x"}, None, waits=(0, 0))
    assert len(calls) == 3 and r["infra"] is True  # 첫 시도 + 재시도 2번

    seq = iter([{"infra": True, "passed": False, "failures": ["인프라 오류"]}, {"infra": False, "passed": True, "failures": []}])
    monkeypatch.setattr(ev, "_run_once", lambda *a: next(seq))
    assert ev.run_case(None, {}, {"id": "x"}, None, waits=(0, 0))["passed"] is True  # 재시도로 회복


def test_summary_excludes_infra_runs_from_the_denominator_but_reports_them():
    cases = [{"id": "a", "category": "in", "question": "q"}, {"id": "b", "category": "in", "question": "q"}]
    runs = [{"id": "a", "category": "in", "passed": True, "failures": [], "seconds": 1, "infra": False},
            {"id": "a", "category": "in", "passed": False, "failures": ["인프라 오류"], "seconds": 1, "infra": True},
            {"id": "b", "category": "in", "passed": False, "failures": ["인프라 오류"], "seconds": 1, "infra": True}]
    s = ev.summarize_runs(cases, runs)
    assert (s["attempted_runs"], s["infra_runs"], s["total_runs"], s["passed_runs"]) == (3, 2, 1, 1)
    assert s["cases_all_pass"] == 1  # b는 측정 불가라 통과로 세지 않는다
    assert next(p for p in s["per_case"] if p["id"] == "b")["runs"] == 0
    assert "인프라 오류" in ev.render_report({"when": "t", "model": "m", "judge_model": "j", "repeats": 1}, s)


def test_second_approval_after_reject_is_a_failure():
    c = case({"approval": True}, resume="reject")
    again = phase(approval(changes={"monBack": 300}), answer=None)  # 취소했는데 또 카드가 뜸
    assert ev.check_run(c, phase(approval(changes={"mon": "32"}), answer=None), again) == ["취소했는데 같은 요청에서 승인 카드를 또 띄웠어요"]
