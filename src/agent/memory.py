"""대화 이력 미들웨어: 길어진 대화의 오래된 부분은 요약하고, 최근 대화는 원문(도구 호출 내역 포함)으로 넘긴다.
고양이 챗봇(pjt)의 src/rag/history.py를 옮겼다.

예전에는 최근 10개 메시지만 넘기고 나머지를 버렸다. 이제 브라우저가 최근 40개까지 보내고,
  - 최근 KEEP(10)개는 항상 원문으로 남긴다(도구 호출 내역도 되살린다 — graph._history_messages).
  - 그보다 오래된 부분을 LLM이 짧게 요약해 시스템 프롬프트 뒤에 붙인다.
  - 요약 구간의 끝(경계)은 BLOCK(4)개, 즉 두 턴 단위로만 올린다. 매 질문마다 다시 요약하지 않고 같은 구간은 캐시를 쓴다.
  - 요약에 넣는 글과 요약 결과 모두 개인정보 마스킹(guard.redact)을 거친다.
이 모듈은 어떤 구간을 요약하고 어떻게 줄로 만들지만 정한다. 요약 호출은 graph의 context 노드가 한다.
"""
import hashlib
import threading
from collections import OrderedDict

from src.core import guard

KEEP = 10
BLOCK = 4
CACHE_SIZE = 64
MAX_LINE = 300  # 요약에 넣을 때 메시지 하나를 자르는 길이

SUMMARY_PROMPT = """아래는 데스크 셋업 시뮬레이터 도우미와 사용자의 앞선 대화야. 이어지는 대화에 필요한 것만 한국어 3~5문장으로 요약해.
- 사용자가 원한 것(물건·치수·취향)과 거절한 것을 남겨.
- '[도구]' 줄에 '적용했어요'가 있는 변경만 적용된 것으로 적고, 제안만 하거나 취소된 것은 '적용 안 됨'으로 적어.
- 지금 셋업의 값은 따로 전달되니 숫자를 길게 옮기지 마.

대화:
{conversation}"""

_cache: "OrderedDict[str, str]" = OrderedDict()
_lock = threading.Lock()


def plan(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """(요약할 오래된 메시지, 원문으로 남길 최근 메시지)로 나눈다. 짧은 대화는 요약할 것이 없다."""
    n = len(messages)
    if n <= KEEP:
        return [], list(messages)
    boundary = (n - KEEP) // BLOCK * BLOCK
    return list(messages[:boundary]), list(messages[boundary:])


def format_lines(messages: list[dict]) -> str:
    """'사용자: …', '도우미: …', '[도구] 이름 → 결과' 줄로 만든다. 개인정보는 가린다."""
    lines = []
    for m in messages:
        who = "사용자" if m["role"] == "user" else "도우미"
        for t in m.get("tools") or []:
            lines.append(f"[도구] {t.get('name')} → {guard.redact(str(t.get('result') or ''))[:120]}")
        lines.append(f"{who}: {guard.redact(m.get('content') or '')[:MAX_LINE]}")
    return "\n".join(lines)


def cache_key(older: list[dict]) -> str:
    return hashlib.sha256(format_lines(older).encode("utf-8")).hexdigest()


def cached(key: str) -> str | None:
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def store(key: str, summary: str) -> None:
    with _lock:
        _cache[key] = summary
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _lock:
        _cache.clear()
