"""데스크 셋업 에이전트 그래프(LangGraph): ReAct 도구 호출 + 사람 승인.

  context ─► agent ─┬─ 도구 호출 없음 ─► END
                    └─ 도구 호출 있음 ─► act ─► agent ...   (최대 MAX_LOOPS번)

  context: 상태·점검 결과를 글로 정리하고 대화를 메시지로 만든다(코드, LLM 없음)
  agent  : LLM이 답하거나 도구를 고른다(문장 단위로 스트리밍)
  act    : 도구를 실행한다. list_settings는 바로, propose_changes는 interrupt()로 멈춰 사용자 승인을 받는다.
           LLM은 값을 '제안'만 하고, 적용 여부와 값의 범위는 코드·사용자·브라우저가 정한다.

스트리밍 이벤트(custom 스트림):
  {"type": "step", "phase": "start" | "done", "node": 노드, "label": 이름, "detail"?: 설명}
  {"type": "token", "text": 답변 조각}
  {"type": "approval", "thread_id", "reason", "changes", "patch", "rejected", "notes"}   (승인 대기, 호출이 여기서 끝남)
"""
import re
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from src.agent.context import describe, describe_checks
from src.agent.settings import diff_items, format_settings, validate_patch

MAX_LOOPS = 5  # agent가 도구를 부르며 돌 수 있는 최대 횟수

SYSTEM_PROMPT = """너는 '데스크 셋업 시뮬레이터'에 붙어 있는 도우미야. 사용자가 3D로 꾸민 책상 셋업을 보고 한국어로 짧고 친근하게 답하고, 요청하면 설정을 바꿔 줘.

규칙:
- 아래 '현재 셋업' 글과 도구 결과에 적힌 값만 근거로 말해. 없는 수치는 지어내지 말고 모른다고 해.
- 점검 결과의 '문제'·'주의' 항목을 먼저 짚고, 어떻게 바꾸면 좋을지 구체적인 값(mm 등)으로 제안해.
- 설정을 바꾸려면: 먼저 list_settings로 정확한 키와 허용값을 확인하고, propose_changes로 제안해. 제안은 사용자가 승인해야 적용돼.
- propose_changes는 한 번에 한 번만 호출하고, 관련된 변경은 changes에 모아서 한 번에 넣어.
- 도구 결과에 '적용했어요'가 나오기 전에는 바꿨다고 말하지 마. 결과의 새 점검 결과를 보고 좋아졌는지 알려 줘.
- 사용자가 거절하면 다시 졸라대지 말고 다른 대안만 짧게 말해.
- 사용자가 말하는 스피커·노트북·의자 같은 물건은 모두 시뮬레이터 화면 속 가상의 물건이야. "켜줘/놔줘/추가해줘"는 그 설정을 켜는 요청, "꺼줘/빼줘/치워줘"는 끄는 요청으로 보고 list_settings로 맞는 항목을 찾아.
- 추가 책상·서랍장·세트·타공판처럼 list_settings에 없는 항목은 아직 바꿀 수 없다고 안내해.
- 데스크 셋업과 관계없는 요청은 정중히 사양하고 셋업 얘기로 돌아와.
- 답은 5문장 이내의 평범한 글로 써. 화면이 마크다운을 그리지 못하니 **굵게**, # 제목, 표, 목록 기호를 쓰지 마.

현재 셋업:
{context}"""

STEP_LABELS = {"context": "셋업 읽기", "agent": "생각하기", "act": "도구 실행"}


class ListSettings(BaseModel):
    """조절할 수 있는 설정 항목의 키·범위·선택지를 조회한다. 값을 바꾸기 전에 먼저 호출해 정확한 키와 허용값을 확인한다."""

    section: str | None = Field(
        default=None, description="섹션 이름(책상, 방, 메인 모니터, 보조 모니터, 입력 장치, 기타 물건). 비우면 섹션별 키 목록만 돌려준다."
    )


class ProposeChanges(BaseModel):
    """설정 변경을 사용자에게 제안한다. 사용자가 승인해야 적용된다. 한 번에 한 번만 호출하고 관련 변경은 모아서 넣는다."""

    changes: dict[str, Any] = Field(description='{설정 키: 새 값}. 예: {"deskW": 1600, "mon": "32"}')
    reason: str = Field(description="왜 바꾸는지 한 줄 설명(사용자에게 보인다)")


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    desk_state: dict
    checks: list[dict]
    settings_schema: list[dict]
    context: str
    messages: Annotated[list, add_messages]
    loops: int
    answer: str


def _emit(event: dict) -> None:
    get_stream_writer()(event)


def _step_done(node: str, detail: str) -> None:
    _emit({"type": "step", "phase": "done", "node": node, "label": STEP_LABELS[node], "detail": detail})


def _text_of(chunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


_BOUNDARY = re.compile(r"[.!?]\s|\n")


def _stream_reply(llm, messages: list) -> AIMessage:
    """LLM 응답을 문장이 끝날 때마다 token 이벤트로 내보내고, 도구 호출까지 합친 AIMessage를 돌려준다."""
    full, buf = None, ""
    stream = llm.stream(messages) if hasattr(llm, "stream") else [llm.invoke(messages)]
    for chunk in stream:
        full = chunk if full is None else full + chunk
        buf += _text_of(chunk)
        while (m := _BOUNDARY.search(buf)):
            segment, buf = buf[: m.end()], buf[m.end():]
            if segment.strip():
                _emit({"type": "token", "text": segment})
    if buf.strip():
        _emit({"type": "token", "text": buf})
    # 본문은 문자열로만 보관한다. 도구 호출은 tool_calls로 따로 들고 있어 다시 보낼 때 중복되지 않는다
    return AIMessage(content=_text_of(full) if full is not None else "", tool_calls=list(getattr(full, "tool_calls", None) or []))


def _propose(state: AgentState, args: dict) -> str:
    """propose_changes 처리. 검증하고, 바뀌는 게 있으면 interrupt로 멈춰 승인을 받은 뒤 결과를 글로 돌려준다."""
    schema = state.get("settings_schema") or []
    raw = args.get("changes")
    clean, rejected, notes = validate_patch(raw if isinstance(raw, dict) else {}, schema)
    items, same = diff_items(clean, state.get("desk_state") or {}, schema)
    problems = [f"{k}: {why}" for k, why in rejected.items()]
    if not items:
        parts = ["바꿀 항목이 없어서 제안하지 않았어요."]
        if same:
            parts.append("이미 그 값이에요: " + ", ".join(same))
        if problems:
            parts.append("거절된 항목: " + "; ".join(problems) + ". list_settings로 키와 허용값을 다시 확인하세요.")
        return " ".join(parts)

    decision = interrupt({
        "reason": str(args.get("reason") or ""),
        "changes": items,
        "patch": {k: clean[k] for k in clean if k not in same},
        "rejected": rejected,
        "notes": notes,
    })

    if not decision.get("approve"):
        _step_done("act", "변경 제안을 사용자가 취소했어요")
        return "사용자가 변경을 거절했어요. 아무것도 적용하지 않았어요."
    result = decision.get("result") or {}
    applied, failed = result.get("applied") or {}, result.get("rejected") or {}
    _step_done("act", f"항목 {len(applied)}개를 적용했어요")
    text = "적용했어요: " + ", ".join(f"{k}={v}" for k, v in applied.items())
    if failed:
        text += " / 적용하지 못한 항목: " + "; ".join(f"{k}: {why}" for k, why in failed.items())
    if problems:
        text += " / 검증에서 거절된 항목: " + "; ".join(problems)
    if decision.get("checks"):
        text += "\n" + describe_checks(decision["checks"])
    return text


def build_graph(llm, checkpointer=None):
    """llm: ChatBedrockConverse 호환(bind_tools와 stream을 가진 객체). checkpointer는 interrupt에 필요해서 기본으로 메모리를 쓴다."""
    tool_llm = llm.bind_tools([ListSettings, ProposeChanges])

    def context(state: AgentState) -> AgentState:
        _emit({"type": "step", "phase": "start", "node": "context", "label": STEP_LABELS["context"]})
        text = describe(state.get("desk_state") or {}, state.get("checks") or [])
        messages = [SystemMessage(SYSTEM_PROMPT.format(context=text))]
        for m in state.get("history") or []:
            messages.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
        messages.append(HumanMessage(state["question"]))
        _step_done("context", f"점검 항목 {len(state.get('checks') or [])}개를 읽었어요")
        return {"context": text, "messages": messages, "loops": 0, "answer": ""}

    def agent(state: AgentState) -> AgentState:
        _emit({"type": "step", "phase": "start", "node": "agent", "label": STEP_LABELS["agent"]})
        reply = _stream_reply(tool_llm, state["messages"])
        names = ", ".join(c["name"] for c in reply.tool_calls)
        _step_done("agent", f"도구를 골랐어요: {names}" if reply.tool_calls else "답변을 마쳤어요")
        out: dict = {"messages": [reply], "loops": state.get("loops", 0) + 1}
        if not reply.tool_calls:
            out["answer"] = reply.content
        return out

    def act(state: AgentState) -> AgentState:
        schema = state.get("settings_schema") or []
        results, proposed = [], False
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call.get("args") or {}
            if name == "ListSettings":
                content = format_settings(schema, args.get("section"))
                _step_done("act", f"설정 목록을 조회했어요({args.get('section') or '전체'})")
            elif name == "ProposeChanges" and not proposed:
                proposed = True
                content = _propose(state, args)
            elif name == "ProposeChanges":
                content = "propose_changes는 한 번에 하나만 처리해요. 이 제안은 건너뛰었어요."
            else:
                content = f"알 수 없는 도구예요: {name}"
            results.append(ToolMessage(content, tool_call_id=call["id"]))
        return {"messages": results}

    def stop(state: AgentState) -> AgentState:
        text = "생각이 길어져서 여기서 멈출게요. 원하는 걸 조금 더 구체적으로 다시 말씀해 주세요."
        _emit({"type": "token", "text": text})
        return {"answer": text}

    def route(state: AgentState) -> str:
        if not state["messages"][-1].tool_calls:
            return END
        return "act" if state.get("loops", 0) < MAX_LOOPS else "stop"

    g = StateGraph(AgentState)
    g.add_node("context", context)
    g.add_node("agent", agent)
    g.add_node("act", act)
    g.add_node("stop", stop)
    g.add_edge(START, "context")
    g.add_edge("context", "agent")
    g.add_conditional_edges("agent", route, ["act", "stop", END])
    g.add_edge("act", "agent")
    g.add_edge("stop", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def build_default_graph():
    """.env 설정으로 Bedrock LLM을 만들어 그래프를 구성한다."""
    from langchain_aws import ChatBedrockConverse

    from src.core.config import bedrock_client_config, load_settings

    s = load_settings()
    retry = bedrock_client_config(s["max_retries"])
    llm = ChatBedrockConverse(model=s["llm_model_id"], region_name=s["region"], temperature=0.3, config=retry)
    return build_graph(llm)


def _run(graph, payload, thread_id: str):
    final: dict = {}
    for mode, chunk in graph.stream(payload, {"configurable": {"thread_id": thread_id}}, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield chunk
        else:
            final = chunk
    pending = final.get("__interrupt__")
    if pending:  # 사용자의 승인을 기다리는 중. 화면이 resume_stream으로 이어 간다
        yield {"type": "approval", "thread_id": thread_id, **pending[0].value}
        return
    graph.checkpointer.delete_thread(thread_id)
    yield {"type": "result", "answer": final.get("answer", "")}


def ask_stream(graph, question: str, history: list[dict], desk_state: dict, checks: list[dict], settings_schema: list[dict],
               thread_id: str | None = None):
    """진행 이벤트(step·token)를 차례로 내보낸다. 끝은 {"type": "result", "answer"} 또는 승인 대기 {"type": "approval", ...}."""
    payload = {"question": question, "history": history, "desk_state": desk_state, "checks": checks, "settings_schema": settings_schema}
    yield from _run(graph, payload, thread_id or uuid.uuid4().hex)


def resume_stream(graph, thread_id: str, approve: bool, result: dict | None = None, checks: list[dict] | None = None):
    """승인(True)·취소(False)로 멈춘 호출을 이어서 진행한다. result는 브라우저가 적용한 결과, checks는 적용 뒤 점검 결과."""
    decision = {"approve": bool(approve), "result": result or {}, "checks": checks or []}
    yield from _run(graph, Command(resume=decision), thread_id)
