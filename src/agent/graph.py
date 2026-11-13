"""데스크 셋업 에이전트 그래프(LangGraph).

  context ─► answer
  context: 시뮬레이터 상태·점검 결과를 글로 정리(코드, LLM 없음)
  answer : 그 글만 근거로 답변을 문장 단위로 스트리밍

1단계는 읽기만 한다. 설정을 바꾸는 도구는 2단계에서 answer 앞에 붙인다.
스트리밍 이벤트(custom 스트림):
  {"type": "step", "phase": "start" | "done", "node": 노드, "label": 이름, "detail"?: 설명}
  {"type": "token", "text": 답변 조각}
"""
import re
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.agent.context import describe

SYSTEM_PROMPT = """너는 '데스크 셋업 시뮬레이터'에 붙어 있는 도우미야. 사용자가 3D로 꾸민 책상 셋업을 보고 한국어로 짧고 친근하게 답해.

규칙:
- 아래 '현재 셋업' 글에 적힌 값만 근거로 말해. 글에 없는 수치는 지어내지 말고 모른다고 해.
- 점검 결과의 '문제'·'주의' 항목을 먼저 짚고, 어떻게 바꾸면 좋을지 구체적인 값(mm 등)으로 제안해.
- 지금은 설정을 직접 바꿀 수 없어. 바꾸자고 하면 "아직 직접 바꾸는 기능은 준비 중이라 패널에서 바꿔 주세요"라고 안내해.
- 데스크 셋업과 관계없는 요청은 정중히 사양하고 셋업 얘기로 돌아와.
- 답은 5문장 이내의 평범한 글로 써. 화면이 마크다운을 그리지 못하니 **굵게**, # 제목, 표, 목록 기호를 쓰지 마.

현재 셋업:
{context}"""

STEP_LABELS = {"context": "셋업 읽기", "answer": "답변 작성"}


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    desk_state: dict
    checks: list[dict]
    context: str
    answer: str


def _emit(event: dict) -> None:
    get_stream_writer()(event)


def _traced(node: str, fn):
    def wrapper(state: AgentState) -> AgentState:
        _emit({"type": "step", "phase": "start", "node": node, "label": STEP_LABELS[node]})
        out = fn(state)
        detail = out.pop("_detail", "")
        _emit({"type": "step", "phase": "done", "node": node, "label": STEP_LABELS[node], "detail": detail})
        return out

    return wrapper


def _text_of(chunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


_BOUNDARY = re.compile(r"[.!?]\s|\n")


def _stream_answer(llm, messages: list) -> str:
    """답변을 문장이 끝날 때마다 token 이벤트로 내보내고 전체 텍스트를 돌려준다."""
    full, buf = "", ""
    stream = llm.stream(messages) if hasattr(llm, "stream") else [llm.invoke(messages)]
    for chunk in stream:
        piece = _text_of(chunk)
        full += piece
        buf += piece
        while (m := _BOUNDARY.search(buf)):
            segment, buf = buf[: m.end()], buf[m.end():]
            if segment.strip():
                _emit({"type": "token", "text": segment})
    if buf.strip():
        _emit({"type": "token", "text": buf})
    return full


def build_graph(llm):
    """llm: ChatBedrockConverse 호환(stream 또는 invoke를 가진 객체)."""

    def context(state: AgentState) -> AgentState:
        text = describe(state.get("desk_state") or {}, state.get("checks") or [])
        return {"context": text, "_detail": f"점검 항목 {len(state.get('checks') or [])}개를 읽었어요"}

    def answer(state: AgentState) -> AgentState:
        messages = [SystemMessage(SYSTEM_PROMPT.format(context=state["context"]))]
        for m in state.get("history") or []:
            messages.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
        messages.append(HumanMessage(state["question"]))
        return {"answer": _stream_answer(llm, messages), "_detail": "현재 셋업만 근거로 답했어요"}

    g = StateGraph(AgentState)
    g.add_node("context", _traced("context", context))
    g.add_node("answer", _traced("answer", answer))
    g.add_edge(START, "context")
    g.add_edge("context", "answer")
    g.add_edge("answer", END)
    return g.compile()


def build_default_graph():
    """.env 설정으로 Bedrock LLM을 만들어 그래프를 구성한다."""
    from langchain_aws import ChatBedrockConverse

    from src.core.config import bedrock_client_config, load_settings

    s = load_settings()
    retry = bedrock_client_config(s["max_retries"])
    llm = ChatBedrockConverse(model=s["llm_model_id"], region_name=s["region"], temperature=0.3, config=retry)
    return build_graph(llm)


def ask_stream(graph, question: str, history: list[dict], desk_state: dict, checks: list[dict]):
    """진행 이벤트(step·token)를 차례로 내보내고, 마지막에 {"type": "result", "answer": 전체 답변}을 낸다."""
    payload = {"question": question, "history": history, "desk_state": desk_state, "checks": checks}
    final = {}
    for mode, chunk in graph.stream(payload, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield chunk
        else:
            final = chunk
    yield {"type": "result", "answer": final.get("answer", "")}
