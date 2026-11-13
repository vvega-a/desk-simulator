"""설정 항목 스키마(브라우저의 deskSim.schema() 값) 다루기: 도구 응답 글, 값 검증, 변경 전후 비교.

LLM은 어떤 값이든 제안할 수 있지만, 실제로 적용되는 값은 여기서 스키마의 범위·선택지로 다시 맞춘다.
(브라우저의 applyPatch도 같은 규칙으로 한 번 더 검증한다.)
"""
import math


def _sections(schema: list[dict]) -> list[str]:
    seen: list[str] = []
    for f in schema:
        if f["section"] not in seen:
            seen.append(f["section"])
    return seen


def _options_text(f: dict) -> str:
    return ", ".join(f"{o['value']}={o['label']}" for o in f.get("options") or [])


def format_settings(schema: list[dict], section: str | None = None) -> str:
    """list_settings 도구가 돌려주는 글. 섹션을 주면 항목별 범위·선택지까지, 없으면 섹션별 키 목록만."""
    sections = _sections(schema)
    if not section:
        lines = ["섹션별 설정 키 (자세한 범위는 section을 지정해 다시 조회):"]
        for s in sections:
            lines.append(f"[{s}] " + ", ".join(f["key"] for f in schema if f["section"] == s))
        return "\n".join(lines)
    if section not in sections:
        return f"'{section}' 섹션이 없어요. 가능한 섹션: " + ", ".join(sections)
    lines = [f"[{section}]"]
    for f in schema:
        if f["section"] != section:
            continue
        unit = f.get("unit") or ""
        if f["type"] == "range":
            rule = f"{f['min']}~{f['max']}{unit}, {f['step']}{unit} 단위"
        elif f["type"] == "toggle":
            rule = "true/false"
        else:
            rule = "선택지 " + _options_text(f)
        lines.append(f"- {f['key']} ({f['label']}): {rule}")
    return "\n".join(lines)


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def validate_patch(patch: dict, schema: list[dict]) -> tuple[dict, dict, dict]:
    """제안된 변경을 스키마로 검증한다. (적용 가능한 값, 거절 사유, 맞춰 준 내용)을 돌려준다."""
    fields = {f["key"]: f for f in schema}
    clean: dict = {}
    rejected: dict = {}
    notes: dict = {}
    for key, raw in patch.items():
        f = fields.get(key)
        if f is None:
            rejected[key] = "알 수 없는 항목이에요"
            continue
        kind = f["type"]
        if kind == "range":
            try:
                num = float(raw)
            except (TypeError, ValueError):
                num = math.nan
            if isinstance(raw, bool) or not math.isfinite(num):
                rejected[key] = "숫자가 아니에요"
                continue
            lo, hi, step = f["min"], f["max"], f["step"]
            value = min(hi, max(lo, _round_half_up((num - lo) / step) * step + lo))
            if num != value:
                notes[key] = f"{raw}를 {value}{f.get('unit') or ''}로 맞췄어요(범위 {lo}~{hi}, {step} 단위)"
            clean[key] = value
        elif kind == "toggle":
            if isinstance(raw, bool):
                clean[key] = raw
            elif str(raw).lower() in ("true", "false"):
                clean[key] = str(raw).lower() == "true"
            else:
                rejected[key] = "true/false여야 해요"
        else:
            match = next((o for o in f["options"] if str(o["value"]) == str(raw) or o["label"] == str(raw)), None)
            if match is None:
                rejected[key] = "선택지: " + _options_text(f)
            else:
                clean[key] = match["value"]
    return clean, rejected, notes


def _show(f: dict, value) -> str:
    if value is None:
        return "?"
    if f["type"] == "toggle":
        return "켜짐" if value else "꺼짐"
    if f["type"] == "range":
        return f"{value}{f.get('unit') or ''}"
    match = next((o for o in f["options"] if str(o["value"]) == str(value)), None)
    return match["label"] if match else str(value)


def diff_items(clean: dict, current: dict, schema: list[dict]) -> tuple[list[dict], list[str]]:
    """검증된 값 중 실제로 달라지는 것만 화면에 보일 비교 항목으로 만든다. 이미 그 값인 키는 따로 돌려준다."""
    fields = {f["key"]: f for f in schema}
    items, same = [], []
    for key, value in clean.items():
        f = fields[key]
        if current.get(key) == value:
            same.append(key)
            continue
        items.append({"key": key, "label": f["label"], "from": _show(f, current.get(key)), "to": _show(f, value)})
    return items, same
