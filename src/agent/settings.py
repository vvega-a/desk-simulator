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


def _rule(f: dict) -> str:
    unit = f.get("unit") or ""
    if f["type"] == "range":
        return f"{f['min']}~{f['max']}{unit}, {f['step']}{unit} 단위"
    if f["type"] == "toggle":
        return "true/false"
    return "선택지 " + _options_text(f)


def format_settings(schema: list[dict], section: str | None = None, collections: dict | None = None) -> str:
    """list_settings 도구가 돌려주는 글. 섹션을 주면 항목별 범위·선택지까지, 없으면 섹션별 키 목록만.

    collections(추가 책상·서랍장·세트·타공판)도 섹션 이름처럼 조회할 수 있다.
    """
    collections = collections or {}
    by_label = {c["label"]: (name, c) for name, c in collections.items()}
    sections = _sections(schema)
    if not section:
        lines = ["단일 설정 (propose_changes의 changes에 {키: 값}). 자세한 범위는 section을 지정해 다시 조회:"]
        for s in sections:
            lines.append(f"[{s}] " + ", ".join(f["key"] for f in schema if f["section"] == s))
        if collections:
            lines.append("\n여러 개 놓는 물건 (propose_changes의 array_ops로 추가·수정·삭제):")
            for name, c in collections.items():
                lines.append(f"[{c['label']}] 최대 {c['max']}개, collection={name}, 값 키: " + ", ".join(f["key"] for f in c["fields"]))
        return "\n".join(lines)
    if section in by_label:
        name, c = by_label[section]
        lines = [f"[{section}] collection={name}, 최대 {c['max']}개. 번호(index)는 0부터이고 화면에는 {c['first']}번부터 보여요.",
                 "add/update의 values에 쓸 수 있는 키:"]
        lines += [f"- {f['key']} ({f['label']}): {_rule(f)}" for f in c["fields"]]
        return "\n".join(lines)
    if section not in sections:
        return f"'{section}' 섹션이 없어요. 가능한 섹션: " + ", ".join(sections + list(by_label))
    lines = [f"[{section}]"]
    lines += [f"- {f['key']} ({f['label']}): {_rule(f)}" for f in schema if f["section"] == section]
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


VIEW_NAMES = {"iso": "사선", "front": "정면", "top": "위에서", "side": "옆", "pov": "앉은 시점"}
SLOTS = ("A", "B", "C")


def validate_ops(ops: list, collections: dict, state: dict) -> tuple[list[dict], list[dict], list[str], dict]:
    """여러 개 놓는 물건의 추가·수정·삭제 제안을 검증한다. (적용할 작업, 비교 항목, 문제 목록, 맞춰 준 내용)을 돌려준다."""
    clean: list[dict] = []
    items: list[dict] = []
    problems: list[str] = []
    notes: dict = {}
    lens = {name: len(state.get(name) or []) for name in collections}
    for n, op in enumerate(ops, 1):
        if not isinstance(op, dict):
            problems.append(f"작업 {n}: 형식이 맞지 않아요")
            continue
        name, kind = op.get("collection"), op.get("op")
        spec = collections.get(name)
        if spec is None:
            problems.append(f"작업 {n}: 알 수 없는 종류 '{name}' (가능: {', '.join(collections)})")
            continue
        label, first = spec["label"], spec.get("first", 1)
        raw = op.get("values")
        values, rejected, vnotes = validate_patch(raw if isinstance(raw, dict) else {}, spec["fields"]) if kind in ("add", "update") else ({}, {}, {})
        for k, why in rejected.items():
            problems.append(f"{label} {k}: {why}")
        for k, why in vnotes.items():
            notes[f"{name}.{k}"] = why
        fields = {f["key"]: f for f in spec["fields"]}
        if kind == "add":
            if lens[name] >= spec["max"]:
                problems.append(f"{label}은(는) 최대 {spec['max']}개예요")
                continue
            summary = ", ".join(f"{fields[k]['label']} {_show(fields[k], v)}" for k, v in values.items()) or "기본값으로"
            items.append({"key": f"{name}+", "label": f"{label} 추가", "from": "", "to": summary})
            clean.append({"op": "add", "collection": name, "values": values})
            lens[name] += 1
            continue
        index = op.get("index")
        if kind not in ("update", "remove"):
            problems.append(f"작업 {n}: 알 수 없는 작업 '{kind}' (add, update, remove 중 하나)")
            continue
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < lens[name]:
            problems.append(f"{label} 번호 {index!r}가 맞지 않아요 (현재 {lens[name]}개, 0~{max(lens[name] - 1, 0)})")
            continue
        title = f"{label} {index + first}"
        if kind == "remove":
            items.append({"key": f"{name}-{index}", "label": f"{title} 삭제", "from": "", "to": "삭제"})
            clean.append({"op": "remove", "collection": name, "index": index})
            lens[name] -= 1
            continue
        current = (state.get(name) or [])[index] if index < len(state.get(name) or []) else {}
        changed = {k: v for k, v in values.items() if current.get(k) != v}
        if not changed:
            problems.append(f"{title}: 이미 그 값이거나 바꿀 값이 없어요")
            continue
        for k, v in changed.items():
            items.append({"key": f"{name}.{index}.{k}", "label": f"{title}: {fields[k]['label']}",
                          "from": _show(fields[k], current.get(k)), "to": _show(fields[k], v)})
        clean.append({"op": "update", "collection": name, "index": index, "values": changed})
    return clean, items, problems, notes


def validate_scene_action(action: str | None, slot: str | None, slots: dict) -> tuple[dict | None, dict | None, str]:
    """장면 전체 동작(초기화·슬롯 저장/불러오기) 검증. (동작, 비교 항목, 문제)를 돌려준다."""
    if action == "reset":
        return {"action": "reset"}, {"key": "scene", "label": "모든 설정", "from": "지금 설정", "to": "처음 상태로 초기화"}, ""
    if action not in ("load_slot", "save_slot"):
        return None, None, "action은 reset, load_slot, save_slot 중 하나여야 해요"
    if slot not in SLOTS:
        return None, None, "slot은 A, B, C 중 하나여야 해요"
    filled = slots.get(slot)
    if action == "load_slot":
        if not filled:
            return None, None, f"슬롯 {slot}는 비어 있어요"
        return ({"action": "load_slot", "slot": slot},
                {"key": "scene", "label": f"슬롯 {slot} 불러오기", "from": "지금 설정", "to": str(filled)}, "")
    to = "비어 있던 슬롯에 저장" if not filled else f"기존 내용을 덮어씀 ({filled})"
    return {"action": "save_slot", "slot": slot}, {"key": "scene", "label": f"슬롯 {slot}에 저장", "from": "", "to": to}, ""
