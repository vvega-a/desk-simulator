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


def _name(table: dict, key) -> str:
    return table.get(str(key), str(key))


def describe(state: dict, checks: list[dict]) -> str:
    """현재 셋업을 줄 단위 글로 설명한다. 상태에 없는 키는 건너뛴다."""
    s = state
    lines = []
    if "deskW" in s:
        lines.append(f"책상: 가로 {s['deskW']}mm × 세로 {s['deskD']}mm, 높이 {s['deskH']}mm")
    if "mon" in s:
        curved = " (커브드)" if s.get("curved") else ""
        lines.append(f"메인 모니터: {_name(MONITORS, s['mon'])}{curved}, 거치 {_name(MOUNTS, s.get('mount'))}")
        lines.append(f"  화면 위치: 책상 뒤 끝에서 {s.get('monBack')}mm, 책상 위 {s.get('monLift')}mm, 좌우 {s.get('monX')}mm")
    if s.get("subCount"):
        portrait = ", 세로 피벗" if s.get("subPortrait") else ""
        lines.append(f"보조 모니터: {_name(MONITORS, s.get('sub'))} {s['subCount']}대{portrait}")
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
    counts = [
        (len(s.get("extraDesks") or []), "추가 책상"), (len(s.get("drawers") or []), "서랍장"),
        (len(s.get("sets") or []), "모니터+노트북 세트"), (len(s.get("pegs") or []), "타공판"),
    ]
    more = [f"{label} {n}개" for n, label in counts if n]
    if more:
        lines.append("추가 구성: " + ", ".join(more))
    if s.get("room"):
        lines.append(f"방: {s.get('roomW')}mm × {s.get('roomD')}mm, 천장 {s.get('roomH')}mm")
    if checks:
        lines.append("\n점검 결과(시뮬레이터가 계산한 값):")
        for c in checks:
            lines.append(f"- [{STATUS.get(c.get('status'), c.get('status'))}] {c.get('title')}: {c.get('value')} — {c.get('note')}")
    return "\n".join(lines)
