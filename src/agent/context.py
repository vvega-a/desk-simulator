"""브라우저가 보낸 시뮬레이터 상태(S)와 점검 결과를 LLM이 읽기 좋은 한국어 글로 바꾼다.

LLM은 값을 지어내지 않고 여기서 만든 글에 적힌 값만 근거로 삼는다(코드가 읽고 LLM은 해석만 한다).
"""

MONITORS = {
    "24": '24" 16:9', "27": '27" 16:9', "32": '32" 16:9', "42": '42" 16:9 OLED',
    "34uw": '34" 21:9 울트라와이드', "38uw": '38" 24:10 울트라와이드', "40uw": '40" 5K2K 울트라와이드',
    "45uw": '45" 21:9 울트라와이드', "49suw": '49" 32:9 슈퍼울트라와이드', "57suw": '57" 32:9 슈퍼울트라와이드',
}
MOUNTS = {"stand": "기본 스탠드", "arm": "모니터암", "armLap": "모니터+노트북 암"}
KEYBOARDS = {"none": "없음", "full": "풀배열", "tkl": "텐키리스", "75": "75%", "65": "65%"}
TOWERS = {"none": "없음", "deskL": "책상 왼쪽", "deskR": "책상 오른쪽", "floor": "바닥"}
STATUS = {"ok": "좋음", "warn": "주의", "bad": "문제", "info": "참고"}
# 키: (목록 이름, 화면에 보이는 이름, 첫 번호). 화면·카드·채팅이 모두 '이름 번호'(서랍장 2, 책상 2)로 같은 번호를 쓴다
COLLECTIONS = {
    "monitors": ("추가 모니터", "모니터", 2),
    "extraDesks": ("추가 책상", "책상", 2),
    "drawers": ("서랍장", "서랍장", 1),
    "sets": ("모니터+노트북 세트", "모니터+노트북 세트", 1),
    "pegs": ("타공판", "타공판", 1),
}


def _name(table: dict, key) -> str:
    return table.get(str(key), str(key))


def describe(state: dict, checks: list[dict], slots: dict | None = None, collections: dict | None = None) -> str:
    """현재 셋업을 줄 단위 글로 설명한다. 상태에 없는 키는 건너뛴다."""
    s = state
    lines = []
    if "deskW" in s:
        lines.append(f"책상 1: 가로 {s['deskW']}mm × 세로 {s['deskD']}mm, 높이 {s['deskH']}mm")
    if "mon" in s:
        curved = " (커브드)" if s.get("curved") else ""
        lines.append(f"모니터 1: {_name(MONITORS, s['mon'])}{curved}, 거치 {_name(MOUNTS, s.get('mount'))}")
        lines.append(f"  화면 위치: 책상 뒤 끝에서 {s.get('monBack')}mm, 책상 위 {s.get('monLift')}mm, 좌우 {s.get('monX')}mm")
    if "kb" in s:
        mouse = "있음" if s.get("mouse") else "없음"
        lines.append(f"키보드: {_name(KEYBOARDS, s['kb'])}, 마우스 {mouse}, 데스크매트 {s.get('mat')}")
    extras = []
    if s.get("laptop", "none") != "none":
        extras.append(f"노트북 {s['laptop']}인치")
    if s.get("tower", "none") != "none":
        extras.append(f"PC 본체({_name(TOWERS, s['tower'])})")
    if s.get("speakers"):
        extras.append("북쉘프 스피커")
    if s.get("chair"):
        extras.append("의자")
    lines.append("기타 물건: " + (", ".join(extras) if extras else "없음"))
    if "userH" in s:
        lines.append(f"사용자 키: {s['userH']}cm")
    lines += describe_collections(s, collections)
    if s.get("room"):
        lines.append(f"방: {s.get('roomW')}mm × {s.get('roomD')}mm, 천장 {s.get('roomH')}mm")
    if slots:
        lines.append("저장 슬롯: " + " / ".join(f"{k}: {v or '비어 있음'}" for k, v in slots.items()))
    if checks:
        lines.append("\n" + describe_checks(checks))
    return "\n".join(lines)


def describe_collections(state: dict, collections: dict | None = None) -> list[str]:
    """추가 책상·서랍장·세트·타공판을 화면과 같은 '이름 번호'로 한 줄씩 설명한다. 지우면 뒤 번호가 당겨지므로 적용 뒤에도 다시 보여 준다."""
    lines = []
    for key, (label, item, first) in COLLECTIONS.items():
        items = state.get(key) or []
        if not items:
            continue
        spec = (collections or {}).get(key) or {}
        cap = ""
        if spec.get("max"):  # 한도를 알려 줘야 가득 찼을 때 되묻지 않고 바로 안내한다
            full = len(items) >= spec["max"]
            cap = f" ({spec.get('max_note') or '최대 ' + str(spec['max']) + '개'}{', 더 추가할 수 없음' if full else ''})"
        lines.append(f"{label} {len(items)}개{cap}:")
        for i, e in enumerate(items):
            lines.append(f"  {item} {i + first}: " + ", ".join(f"{k}={v}" for k, v in e.items()))
    return lines


def describe_checks(checks: list[dict]) -> str:
    lines = ["점검 결과(시뮬레이터가 계산한 값):"]
    for c in checks:
        lines.append(f"- [{STATUS.get(c.get('status'), c.get('status'))}] {c.get('title')}: {c.get('value')} — {c.get('note')}")
    return "\n".join(lines)
