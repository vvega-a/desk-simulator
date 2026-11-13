"""Bedrock·네이버 API 없이 도는 상품 검색 테스트: 검색 대상 만들기, 상품명 치수 읽기·판정, API 호출(가짜), 키 없을 때 링크, 에이전트 도구."""
import pytest

from src import agent
from src import tools as shop
from tests.test_agent import CHECKS, SCHEMA, STATE, FakeLLM, call, text, tool_messages

STATE2 = {**STATE, "mon": "34uw", "curved": True, "monitors": [{"mon": "27", "curved": False}],
          "drawers": [{"where": "underL", "w": 400, "d": 500, "h": 600, "casters": True}]}


@pytest.mark.parametrize("name, query, dims", [
    ("책상 1", "컴퓨터 책상 1400x700", (1400, 700)),
    ("모니터 1", "34인치 울트라와이드 모니터 커브드", ()),
    ("모니터 2", "27인치 모니터", ()),
    ("서랍장 1", "이동식 서랍장 400x500x600", (400, 500, 600)),
    ("키보드", "텐키리스 키보드", ()),
])
def test_target_reads_sizes_from_the_current_setup(name, query, dims):
    t = shop.target_from_state(name, STATE2)
    assert t.query == query and t.dims == dims


def test_missing_item_is_reported_not_invented():
    assert "모니터 3가 없어요" in shop.target_from_state("모니터 3", STATE2)
    assert "찾을 수 있는 물건이 아니에요" in shop.target_from_state("소파", STATE2)


@pytest.mark.parametrize("title, dims", [
    ("컴퓨터책상 1400x700", [(1400, 700)]), ("일자 책상 140x70cm", [(1400, 700)]), ("스탠딩 1400*700*720", [(1400, 700, 720)]),
    ("<b>책상</b> 120×60", [(1200, 600)]), ("원목 책상", []),
])
def test_parse_dims(title, dims):
    assert shop.parse_dims(title) == dims


def test_match_handles_swapped_order_monitor_inches_and_missing_sizes():
    desk = shop.target_from_state("책상 1", STATE2)
    assert shop.match(desk, "책상 700x1400")[0] == "일치"
    assert shop.match(desk, "책상 1450x700")[0] == "근접"
    assert shop.match(desk, "책상 1200x600")[0] == "다름"
    assert shop.match(desk, "모던 책상")[0] == "표기 없음"
    mon = shop.target_from_state("모니터 1", STATE2)
    assert shop.match(mon, "LG 34인치 울트라와이드")[0] == "일치" and shop.match(mon, "27인치 모니터")[0] == "다름"


class FakeHTTP:
    def __init__(self, items=None, fail=False):
        self.items, self.fail, self.calls = items or [], fail, []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, params, headers))
        if self.fail:
            raise TimeoutError("slow")
        items = self.items

        class R:
            def raise_for_status(self): ...
            def json(self): return {"items": items}
        return R()


ITEMS = [
    {"title": "모던 <b>책상</b>", "link": "https://shopping.naver.com/a", "lprice": "50000", "mallName": "가"},
    {"title": "컴퓨터 책상 1200x600", "link": "https://shopping.naver.com/b", "lprice": "40000", "mallName": "나"},
    {"title": "컴퓨터 책상 1400x700 화이트", "link": "https://shopping.naver.com/c", "lprice": "90000", "mallName": "다"},
    {"title": "책상 1400x700 원목", "link": "https://shopping.naver.com/d", "lprice": "70000", "mallName": "라"},
    {"title": "피싱 책상 1400x700", "link": "javascript:alert(1)", "lprice": "1"},
]


def test_api_results_are_ranked_by_size_match_then_price():
    http = FakeHTTP(ITEMS)
    r = shop.search(shop.target_from_state("책상 1", STATE2), "화이트 010-1234-5678", "id", "secret", http)
    url, params, headers = http.calls[0]
    assert url == shop.SHOP_API and headers["X-Naver-Client-Id"] == "id" and params["display"] == shop.DISPLAY
    assert params["query"].startswith("컴퓨터 책상 1400x700 화이트") and "010-1234-5678" not in params["query"]  # 키워드도 마스킹
    assert [i["link"][-1] for i in r["items"]] == ["d", "c", "a", "b"]  # 일치(싼 순) → 표기 없음 → 다름, 위험한 링크는 버림
    assert r["items"][0]["match"] == "일치" and r["items"][2]["title"] == "모던 책상"


def test_without_keys_only_a_search_link_is_returned():
    http = FakeHTTP(ITEMS)
    r = shop.search(shop.target_from_state("책상 1", STATE2), "", "", "", http)
    assert http.calls == [] and r["mode"] == "link" and r["items"] == []
    assert r["url"].startswith(shop.SEARCH_URL) and "1400x700" in shop.urllib.parse.unquote(r["url"])
    assert "지어내지" in shop.format_result(r)


def test_api_failure_hides_details():
    r = shop.search(shop.target_from_state("책상 1", STATE2), "", "id", "secret", FakeHTTP(fail=True))
    assert r["error"] == "TimeoutError" and r["items"] == [] and "slow" not in shop.format_result(r)


def test_search_products_tool_emits_cards_and_feeds_results_to_llm(monkeypatch):
    real = shop.search
    monkeypatch.setattr(shop, "search", lambda t, kw, cid, sec: real(t, kw, cid, sec, FakeHTTP(ITEMS)))
    llm = FakeLLM(call("SearchProducts", {"target": "책상 1"}), text("1400x700 책상을 찾았어요."))
    g = agent.build_graph(llm, shop_keys=("id", "s"))
    events = list(agent.ask_stream(g, "지금 책상 맞는 모델 찾아줘", [], STATE2, CHECKS, SCHEMA, thread_id="t1"))
    cards = [e for e in events if e["type"] == "products"]
    assert len(cards) == 1 and cards[0]["items"][0]["match"] == "일치" and cards[0]["label"] == "책상 1"
    assert cards[0]["open"] is False  # 상품 목록이 있으면 새 탭을 열지 않는다
    assert "[일치] 책상 1400x700 원목" in tool_messages(llm, 1)[0]
    assert not any(e["type"] == "approval" for e in events)


def test_search_products_for_missing_item_returns_reason():
    llm = FakeLLM(call("SearchProducts", {"target": "타공판 1"}), text("타공판이 없어요."))
    g = agent.build_graph(llm)
    events = list(agent.ask_stream(g, "타공판 찾아줘", [], STATE2, CHECKS, SCHEMA, thread_id="t1"))
    assert tool_messages(llm, 1)[0] == "타공판이 없어요." and not any(e["type"] == "products" for e in events)


def test_without_keys_the_browser_is_told_to_open_the_search_page():
    llm = FakeLLM(call("SearchProducts", {"target": "책상 1"}), text("네이버 쇼핑을 열었어요."))
    g = agent.build_graph(llm)  # 키 없음
    events = list(agent.ask_stream(g, "책상 찾아줘", [], STATE2, CHECKS, SCHEMA, thread_id="t1"))
    card = next(e for e in events if e["type"] == "products")
    assert card["open"] is True and card["items"] == [] and card["url"].startswith(shop.SEARCH_URL)
    assert "새 탭으로 열었어요" in tool_messages(llm, 1)[0]
