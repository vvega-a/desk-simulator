"""상품 검색: 지금 셋업의 물건(책상·모니터·서랍장 …)과 치수가 맞는 상품을 네이버 쇼핑에서 찾는다.

  물건 이름("책상 1", "모니터 2", "서랍장 1") ─► 현재 상태에서 치수·규격을 읽어 검색어를 만든다(코드, LLM 없음)
    ─► 네이버 검색 API(쇼핑) ─► 상품명에서 치수(1400x700, 140x70cm, 27인치 …)를 읽어 일치·근접·다름으로 나눈다
    ─► 치수가 맞는 것부터 몇 개 + 네이버 쇼핑 검색 링크

- shopping.naver.com 화면을 긁어 오지 않는다. 자동 접근을 막는 사이트이고 이용약관에도 어긋난다.
  대신 네이버 개발자센터의 공식 검색 API(쇼핑)를 쓴다. 키(NAVER_CLIENT_ID·NAVER_CLIENT_SECRET)가 없으면 API 호출 없이
  같은 검색어의 네이버 쇼핑 검색 링크만 돌려준다.
- 상품명에 적힌 치수는 판매자가 쓴 글이라 틀릴 수 있다. '일치'는 "상품명에 그 치수가 적혀 있다"는 뜻이지 보증이 아니다.
- API로 보내는 것은 검색어(물건 종류·치수·LLM이 고른 키워드)뿐이다. 키워드도 개인정보 마스킹을 거친다.
"""
import html
import re
import urllib.parse
from dataclasses import dataclass, field

from src import guard

SHOP_API = "https://openapi.naver.com/v1/search/shop.json"
SEARCH_URL = "https://search.shopping.naver.com/ns/search?query="
DISPLAY = 40  # API에서 받아 올 상품 수(치수로 다시 고른다)
SHOW = 5  # 돌려줄 상품 수
TIMEOUT = 6.0

MONITOR_INCH = {"24": 24, "27": 27, "32": 32, "42": 42, "34uw": 34, "38uw": 38, "40uw": 40, "45uw": 45, "49suw": 49, "57suw": 57}
KEYBOARD_NAME = {"full": "풀배열 키보드", "tkl": "텐키리스 키보드", "75": "75% 키보드", "65": "65% 키보드"}
MATS = {"800x300": (800, 300), "900x400": (900, 400), "900x450": (900, 450), "1000x500": (1000, 500), "1200x600": (1200, 600)}


@dataclass
class Target:
    label: str  # 화면 이름(책상 1, 모니터 2 …)
    kind: str  # desk, monitor, drawer, peg, mat, keyboard, chair, arm
    query: str  # 기본 검색어
    dims: tuple[int, ...] = ()  # 비교할 치수(mm). 책상·매트는 (가로, 세로), 서랍장은 (가로, 깊이, 높이), 타공판은 (가로, 세로)
    inch: float | None = None  # 모니터 대각선(인치)
    notes: list[str] = field(default_factory=list)


def _num(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 1


def _monitor_target(label: str, model: str, curved: bool) -> Target:
    inch = MONITOR_INCH.get(str(model), 27)
    kind = " 슈퍼울트라와이드" if str(model).endswith("suw") else " 울트라와이드" if str(model).endswith("uw") else ""
    query = f"{inch}인치{kind} 모니터" + (" 커브드" if curved else "")
    return Target(label, "monitor", query, inch=inch)


def target_from_state(name: str, state: dict) -> Target | str:
    """'책상 1'·'모니터 2'·'서랍장 1'·'의자'처럼 화면 이름으로 물건을 찾아 검색 대상을 만든다. 없으면 이유 글을 돌려준다."""
    n = (name or "").replace(" ", "")
    s = state or {}
    if n.startswith("책상") or n in ("desk", "데스크"):
        i = _num(n)
        if i <= 1:
            if "deskW" not in s:
                return "현재 셋업에 책상 치수가 없어요."
            return Target("책상 1", "desk", f"컴퓨터 책상 {s['deskW']}x{s['deskD']}", dims=(s["deskW"], s["deskD"]),
                          notes=[f"높이 {s.get('deskH')}mm는 상품명에 잘 적히지 않아 비교하지 않았어요"])
        extra = s.get("extraDesks") or []
        if i - 2 >= len(extra):
            return f"책상 {i}이 없어요. 지금 책상은 {len(extra) + 1}개예요."
        e = extra[i - 2]
        return Target(f"책상 {i}", "desk", f"컴퓨터 책상 {e['w']}x{e['d']}", dims=(e["w"], e["d"]))
    if n.startswith("모니터암") or n.startswith("암"):
        return Target("모니터암", "arm", "모니터암" + (" 노트북 거치대" if s.get("mount") == "armLap" else ""))
    if n.startswith("모니터"):
        i = _num(n)
        if i <= 1:
            if "mon" not in s:
                return "현재 셋업에 모니터 정보가 없어요."
            return _monitor_target("모니터 1", s["mon"], bool(s.get("curved")))
        mons = s.get("monitors") or []
        if i - 2 >= len(mons):
            return f"모니터 {i}가 없어요. 지금 모니터는 {len(mons) + 1}대예요."
        e = mons[i - 2]
        return _monitor_target(f"모니터 {i}", e.get("mon"), bool(e.get("curved")))
    if n.startswith("서랍"):
        i, drawers = _num(n), s.get("drawers") or []
        if not 1 <= i <= len(drawers):
            return "서랍장이 없어요." if not drawers else f"서랍장 {i}가 없어요. 지금 서랍장은 {len(drawers)}개예요."
        e = drawers[i - 1]
        return Target(f"서랍장 {i}", "drawer", f"{'이동식 ' if e.get('casters') else ''}서랍장 {e['w']}x{e['d']}x{e['h']}",
                      dims=(e["w"], e["d"], e["h"]))
    if n.startswith("타공판"):
        i, pegs = _num(n), s.get("pegs") or []
        if not 1 <= i <= len(pegs):
            return "타공판이 없어요." if not pegs else f"타공판 {i}이 없어요. 지금 타공판은 {len(pegs)}개예요."
        e = pegs[i - 1]
        return Target(f"타공판 {i}", "peg", f"타공판 {e['w']}x{e['h']}", dims=(e["w"], e["h"]))
    if n.startswith("데스크매트") or n.startswith("매트"):
        mat = MATS.get(str(s.get("mat")))
        if not mat:
            return "데스크매트가 없어요."
        return Target("데스크매트", "mat", f"데스크매트 {mat[0]}x{mat[1]}", dims=mat)
    if n.startswith("키보드"):
        if s.get("kb", "none") == "none":
            return "키보드가 없어요."
        return Target("키보드", "keyboard", KEYBOARD_NAME.get(str(s.get("kb")), "키보드"))
    if n.startswith("의자"):
        return Target("의자", "chair", "사무용 의자", notes=[f"사용자 키 {s.get('userH')}cm"])
    return f"'{name}'은(는) 찾을 수 있는 물건이 아니에요. 책상 N, 모니터 N, 서랍장 N, 타공판 N, 데스크매트, 키보드, 의자, 모니터암 중에서 고르세요."


# ---- 상품명에서 치수 읽기 ------------------------------------------------------------------

_DIMS = re.compile(
    r"(?<![\d.])(\d{2,4}(?:\.\d)?)\s*(mm|cm)?\s*[xX×*]\s*(\d{2,4}(?:\.\d)?)\s*(mm|cm)?"
    r"(?:\s*[xX×*]\s*(\d{2,4}(?:\.\d)?)\s*(mm|cm)?)?(?![\d.])"
)
_INCH = re.compile(r"(?<![\d.])(\d{2}(?:\.\d)?)\s*(?:인치|형|\"|”|inch)", re.I)


def parse_dims(title: str) -> list[tuple[int, ...]]:
    """상품명 속 'AxB(xC)' 치수를 mm로 읽는다. 단위가 없으면 모두 300 미만일 때 cm로 본다(120x60 → 1200x600)."""
    out = []
    for m in _DIMS.finditer(title or ""):
        nums = [float(x) for x in (m.group(1), m.group(3), m.group(5)) if x]
        unit = next((u for u in (m.group(6), m.group(4), m.group(2)) if u), None)
        scale = 10 if unit == "cm" or (unit is None and max(nums) < 300) else 1
        out.append(tuple(round(v * scale) for v in nums))
    return out


def parse_inch(title: str) -> float | None:
    m = _INCH.search(title or "")
    return float(m.group(1)) if m else None


def match(target: Target, title: str) -> tuple[str, int | None, str]:
    """(판정, 차이, 읽은 치수 글). 판정: 일치(±20mm·±0.6인치) · 근접(±100mm) · 다름 · 표기 없음 · 비교 안 함."""
    if target.kind == "monitor":
        inch = parse_inch(title)
        if inch is None:
            return "표기 없음", None, ""
        diff = abs(inch - (target.inch or 0))
        return ("일치" if diff <= 0.6 else "다름"), round(diff * 25.4), f"{inch:g}인치"
    if not target.dims:
        return "비교 안 함", None, ""
    best = None
    for d in parse_dims(title):
        if len(d) < len(target.dims) and len(d) < 2:
            continue
        k = min(len(d), len(target.dims))
        want, got = target.dims[:k], d[:k]
        diffs = [abs(a - b) for a, b in zip(want, got)]
        if k >= 2:  # 가로·세로를 바꿔 쓴 상품명도 있다
            swapped = (got[1], got[0]) + tuple(got[2:])
            diffs = min(diffs, [abs(a - b) for a, b in zip(want, swapped)], key=max)
        if best is None or max(diffs) < max(best[0]):
            best = (diffs, d)
    if best is None:
        return "표기 없음", None, ""
    worst = max(best[0])
    label = "일치" if worst <= 20 else "근접" if worst <= 100 else "다름"
    return label, worst, "×".join(str(v) for v in best[1])


RANK = {"일치": 0, "근접": 1, "비교 안 함": 2, "표기 없음": 3, "다름": 4}


def _clean(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def search_url(query: str) -> str:
    return SEARCH_URL + urllib.parse.quote(query)


def search(target: Target, keywords: str = "", client_id: str = "", client_secret: str = "", http=None) -> dict:
    """상품을 찾아 {label, query, url, items, mode, error}를 돌려준다. 키가 없으면 mode='link'(검색 링크만)."""
    extra = guard.redact(re.sub(r"\s+", " ", keywords or "")).strip()[:40]
    query = (target.query + (" " + extra if extra else "")).strip()
    out = {"label": target.label, "query": query, "url": search_url(query), "items": [], "mode": "link", "error": None,
           "notes": list(target.notes)}
    if not (client_id and client_secret):
        return out
    try:
        if http is None:
            import httpx

            http = httpx.Client(timeout=TIMEOUT)
        res = http.get(SHOP_API, params={"query": query, "display": DISPLAY, "sort": "sim"},
                       headers={"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret})
        res.raise_for_status()
        raw = res.json().get("items") or []
    except Exception as e:  # 키 오류·한도 초과·네트워크. 내용은 숨기고 종류만 알린다
        out["error"] = type(e).__name__
        return out
    items = []
    for it in raw:
        title = _clean(it.get("title"))
        link = str(it.get("link") or "")
        if not title or not link.startswith("https://"):
            continue
        verdict, diff, size = match(target, title)
        try:
            price = int(it.get("lprice") or 0)
        except (TypeError, ValueError):
            price = 0
        items.append({"title": title, "link": link, "price": price, "mall": _clean(it.get("mallName")), "brand": _clean(it.get("brand")),
                      "match": verdict, "diff": diff, "size": size})
    items.sort(key=lambda x: (RANK[x["match"]], x["diff"] if x["diff"] is not None else 10**6, x["price"] or 10**9))
    out.update(items=items[:SHOW], mode="api", total=len(items))
    return out


def format_result(r: dict) -> str:
    """도구 결과 글(LLM이 읽는다). 상품 카드는 화면에 따로 뜨니 LLM은 몇 개만 짧게 짚으면 된다."""
    head = f"네이버 쇼핑 검색: '{r['query']}' ({r['label']} 기준)."
    if r.get("notes"):
        head += " 참고: " + "; ".join(r["notes"]) + "."
    if r["mode"] == "link" and not r.get("error"):
        return head + (" 쇼핑 API 키가 없어 상품 목록은 가져오지 못했어요. 대신 이 검색어로 네이버 쇼핑 검색 결과를 새 탭으로 열었어요"
                       "(브라우저가 팝업을 막으면 화면 카드의 링크로 열 수 있어요). 어떤 검색어로 열었는지와 새 탭에서 치수를 보고 고르라는 것만 말하고,"
                       " 상품 이름·가격은 모르니 지어내지 마세요.")
    if r.get("error"):
        return head + f" 상품 검색에 실패했어요({r['error']}). 대신 같은 검색어로 네이버 쇼핑 검색 결과를 새 탭으로 열었어요. 상품 이름·가격을 지어내지 마세요."
    if not r["items"]:
        return head + " 맞는 상품을 찾지 못했어요. 대신 같은 검색어로 네이버 쇼핑 검색 결과를 새 탭으로 열었어요."
    lines = [head, "치수 판정은 상품명에 적힌 치수 기준이라 판매 페이지에서 꼭 확인하라고 말하세요. 화면에 상품 카드가 떠 있어요:"]
    for n, it in enumerate(r["items"], 1):
        size = f", 상품명 치수 {it['size']}" if it["size"] else ""
        price = f"{it['price']:,}원" if it["price"] else "가격 정보 없음"
        lines.append(f"[{n}] [{it['match']}] {it['title']} · 최저가 {price} · {it['mall'] or '판매처 정보 없음'}{size}")
    return "\n".join(lines)
