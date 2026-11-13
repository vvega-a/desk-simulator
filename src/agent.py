"""데스크 셋업 에이전트 그래프(LangGraph): ReAct 도구 호출 + 사람 승인 + 입력·출력 가드레일.

  context ─┬─ 프롬프트 인젝션·범위 밖 요청 ─► shield ─► END   (LLM을 부르지 않는다)
           └─► agent ─┬─ 도구 호출 없음 ─► END
                      └─ 도구 호출 있음 ─► act ─► agent ...   (최대 MAX_LOOPS번)

  context: 상태·점검 결과를 글로 정리하고 대화를 메시지로 만든다. 미들웨어 역할도 한다.
           - 질문·이전 대화의 개인정보를 가린다(src/guard.py)
           - 인젝션 질문과 범위 밖 요청(날씨·코드·주가 등)을 규칙으로 잡아 shield로 보낸다
           - 오래된 대화는 요약해 시스템 프롬프트 뒤에 붙인다(src/memory.py, 요약할 때만 LLM 호출)
  shield : 막은 질문에 고정 안내만 한다(인젝션·범위 밖 각각 다른 안내)
  agent  : LLM이 답하거나 도구를 고른다(문장 단위로 스트리밍)
  act    : 도구를 실행한다.
           list_settings   바로 실행(읽기)
           search_guides   가이드 문서 하이브리드 검색(RAG, 읽기) — src/retriever.py
           search_products 지금 물건과 치수가 맞는 상품을 네이버 쇼핑에서 검색(외부 API, 읽기) — src/tools.py
                           결과는 화면에 상품 카드로 보낸다(products 이벤트)
           set_view        화면 보기 변경은 되돌리기 쉬워 승인 없이 브라우저에 알린다(client_action 이벤트)
           propose_changes 단일 설정 변경 + 아이템 추가·수정·삭제 → interrupt()로 멈춰 사용자 승인
           scene_action    초기화·슬롯 저장/불러오기 → interrupt()로 멈춰 사용자 승인
           LLM은 값을 '제안'만 하고, 적용 여부와 값의 범위는 코드·사용자·브라우저가 정한다.

스트리밍 이벤트(custom 스트림):
  {"type": "step", "phase": "start" | "done", "node": 노드, "label": 이름, "detail"?: 설명}
  {"type": "token", "text": 답변 조각}   (개인정보는 내보내기 전에 한 번 더 가린다)
  {"type": "client_action", "action": "set_view", "view"?, "auto_rotate"?}   (브라우저가 바로 실행)
  {"type": "products", "label", "query", "url", "items": [{title, link, price, mall, match, size}], "mode", "error", "open"}   (상품 카드. open이면 검색 페이지를 새 탭으로 연다)
  {"type": "approval", "thread_id", "reason", "changes", "patch", "rejected", "notes"}   (승인 대기, 호출이 여기서 끝남)
"""
import re
import threading
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from src import memory
from src.context import describe, describe_changes, describe_checks, describe_collections
from src.settings import (
    BAD_VALUE_PREFIX, VIEW_NAMES, diff_items, format_settings, validate_ops, validate_patch, validate_scene_action,
)
from src import guard
from src.retriever import format_hits
from src import tools as shop

MAX_LOOPS = 6  # agent가 도구를 부르며 돌 수 있는 최대 횟수
SEARCH_K = 3  # search_guides가 돌려주는 문서 조각 수

SYSTEM_PROMPT = """너는 '데스크 셋업 시뮬레이터'에 붙어 있는 도우미야. 사용자가 3D로 꾸민 책상 셋업을 보고 한국어로 짧고 친근하게 답하고, 요청하면 화면의 기능을 대신 써 줘.

할 수 있는 일:
- 단일 설정 바꾸기(책상·모니터·키보드·방 등): propose_changes의 changes
- 물건 추가·수정·삭제(모니터 2 이상·추가 책상·서랍장·모니터+노트북 세트·타공판): propose_changes의 array_ops. 모니터 2부터는 각각 독립이라 모델·거치·책상·위치·높이·각도를 따로 정해
- 노트북·PC 본체·스피커·의자·방·데스크매트·키보드/마우스 추가·삭제: changes에서 해당 설정 켜기/끄기(list_settings로 키 확인). 켜고 끄는 키: 방=room, 스피커=speakers, 의자=chair, 치수선=dims. '방을 보이게 해줘'는 시점 변경이 아니라 room 설정을 켜는 요청이야
- 초기화, 저장 슬롯(A·B·C·D·E·F) 저장·불러오기: scene_action
- 화면 시점(사선·정면·위에서·옆·앉은 시점)과 자동 회전: set_view
- 지금 셋업의 물건과 치수가 맞는 실제 상품 찾기(예: 지금 책상에 맞는 모델 찾아줘, 이 모니터 어디서 사?): search_products
- 권장 기준·이유·고르는 요령 같은 일반 지식(예: 모니터 높이는 어느 정도가 좋아? 목이 아파, 울트라와이드랑 듀얼 중 뭐가 나아?): search_guides

규칙:
- 아래 '현재 셋업' 글과 도구 결과에 적힌 값만 근거로 말해. 없는 수치는 지어내지 말고 모른다고 해.
- '현재 셋업'은 이번 질문을 받은 때의 화면이야. 사용자는 대화 사이에 화면에서 직접 물건을 옮기거나 지우거나 값을 바꿀 수 있어. 지난 질문들의 대화·도구 결과(예: 예전의 '적용 뒤 현재 번호')에 나온 물건·번호·값이 '현재 셋업'과 다르면 '현재 셋업'을 따라. 단, 이번 질문에서 네가 도구로 적용한 결과('적용했어요', '적용 뒤 현재 번호')는 '현재 셋업'보다 최신이니 그것을 따라. 질문 앞에 '[화면에서 직접 바꾼 것]'이 있으면 그 변화를 알고 있는 것처럼 반영해.
- 위치 값의 방향: x·monX·좌우 이동은 +가 오른쪽(사용자가 앉아서 볼 때), z·앞뒤 이동은 +가 앞쪽(사용자 쪽), back·monBack(책상 뒤 끝 → 화면)은 클수록 사용자 쪽이야. 서랍장의 x·z는 where(기본 위치)에서 옮긴 양이야.
- 권장 기준·이유·요령을 물으면 search_guides로 가이드 문서를 찾아 근거로 답하고, 쓴 문서 제목을 짧게 밝혀(예: '가이드 「모니터 높이와 각도」에 따르면'). 문서와 점검 결과에 없는 기준은 지어내지 마. 현재 셋업의 수치는 점검 결과를 근거로 말해.
- 점검 결과의 '문제'·'주의' 항목을 먼저 짚고, 어떻게 바꾸면 좋을지 구체적인 값(mm 등)으로 제안해.
- 설정을 바꾸기 전에 list_settings로 정확한 키와 허용값을 확인해. 물건을 추가·수정할 때는 해당 물건 이름(예: 서랍장)을 section으로 조회해.
- 물건을 추가할 때 사용자가 말하지 않은 값(위치·높이·각도·책상 등)은 비워 둬. 시뮬레이터가 다른 물건과 겹치지 않게 알맞은 기본값으로 자동 배치하고, 사용자는 승인 카드에서 확인하고 나중에 고칠 수 있어. 그러니 위치를 되묻지 말고 바로 제안해.
- propose_changes·scene_action은 사용자가 승인해야 적용돼. 관련된 변경은 한 번에 모아서 제안하고, 한 턴에 propose_changes와 scene_action은 각각 한 번만 호출해.
- 모든 물건에는 화면과 같은 '이름 번호'가 있어(책상 1, 추가 책상은 책상 2부터, 서랍장 1, 타공판 1 …). 사용자에게도 이 이름으로 말해.
- array_ops에서 수정·삭제할 물건은 그 번호를 number로 그대로 써(서랍장 2면 number=2, 책상 2면 number=2). 0부터 세지 마. 책상 1과 모니터 1은 array_ops가 아니라 changes의 설정('책상', '모니터 1' 섹션)으로 바꿔. 여러 개를 지울 때는 번호가 당겨지니 큰 번호부터 지워.
- 도구 결과에 '적용했어요'가 나오기 전에는 바꿨다고 말하지 마. 결과의 새 점검 결과를 보고 좋아졌는지 알려 줘.
- 이전 대화의 '완료했어요'는 그때 도구를 불러 실제로 적용한 결과였어. 이번 요청의 변경·추가·삭제·저장도 반드시 도구를 새로 불러야 하고, 도구를 부르지 않았다면 했다고 절대 말하지 마.
- 사용자가 거절하면 다시 졸라대지 말고 다른 대안만 짧게 말해. 같은 요청에서 승인 카드를 또 띄우지 마.
- 사용자가 허용 범위를 벗어난 값(예: 책상 가로 5000)을 말해도 되묻지 말고 말한 값 그대로 제안해. 범위 안으로 맞추는 건 네가 계산하지 않아도 코드가 하고, 어떻게 맞췄는지 승인 카드에 보여 줘.
- 개수 한도에 이미 도달한 물건(현재 셋업에 '더 추가할 수 없음'이라고 적힌 것)은 추가를 제안하지 말고 한도를 알려 줘.
- 사용자가 말하는 스피커·노트북·의자·서랍장 같은 물건은 모두 시뮬레이터 화면 속 가상의 물건이야. "켜줘/놔줘/추가해줘"는 추가, "꺼줘/빼줘/치워줘"는 삭제 요청으로 봐.
- 상품을 찾을 때는 search_products의 target에 화면 이름(책상 1, 모니터 2, 서랍장 1, 의자 …)을 넣어. 치수·규격은 코드가 현재 셋업에서 읽으니 네가 숫자를 넣지 마. 사용자가 말한 색·브랜드·가격대만 keywords에 넣어.
- 상품 이름·가격·링크는 화면에 카드로 나와. 답에서는 치수가 맞는 것 한두 개만 짧게 짚고, 검색 결과에 없는 상품·가격은 절대 지어내지 마. 치수는 상품명 기준이라 판매 페이지에서 확인하라고 한 번 말해. 상품을 사거나 주문하는 일은 하지 않아.
- 기능과 관계없는 요청은 정중히 사양하고 셋업 얘기로 돌아와.
- 답은 5문장 이내의 평범한 글로 써. 화면이 마크다운을 그리지 못하니 **굵게**, # 제목, 표, 목록 기호를 쓰지 마.

현재 셋업:
{context}"""

STEP_LABELS = {"context": "셋업 읽기", "agent": "생각하기", "act": "도구 실행", "shield": "안전 점검"}


class ListSettings(BaseModel):
    """조절할 수 있는 설정 항목의 키·범위·선택지를 조회한다. 값을 바꾸거나 물건을 추가·수정하기 전에 먼저 호출해 정확한 키와 허용값을 확인한다."""

    section: str | None = Field(
        default=None,
        description="섹션 이름(책상, 방, 모니터 1, 입력 장치, 기타 물건) 또는 물건 이름(모니터, 추가 책상, 서랍장, 모니터+노트북 세트, 타공판). 비우면 전체 키 목록.",
    )


class SearchGuides(BaseModel):
    """인체공학·배치 가이드 문서를 검색한다(읽기, 승인 불필요). 권장 기준, 그 이유, 고르는 요령을 근거와 함께 답할 때 쓴다."""

    query: str = Field(description="찾을 내용을 짧은 한국어 문장으로(예: '모니터 높이와 눈높이', '스탠딩 데스크 높이')")


class SearchProducts(BaseModel):
    """지금 셋업의 물건과 치수·규격이 맞는 상품을 네이버 쇼핑에서 찾는다(읽기, 승인 불필요). 결과는 화면에 상품 카드로 뜬다."""

    target: str = Field(description="찾을 물건의 화면 이름: 책상 1, 책상 2, 모니터 1, 모니터 2, 서랍장 1, 타공판 1, 데스크매트, 키보드, 의자, 모니터암")
    keywords: str | None = Field(default=None, description="사용자가 말한 색·브랜드·가격대 같은 추가 검색어(예: '화이트', '모션데스크'). 치수는 넣지 않는다")


class ArrayOp(BaseModel):
    """추가 책상·서랍장·모니터+노트북 세트·타공판 하나에 대한 작업."""

    op: Literal["add", "update", "remove"] = Field(description="add=새로 놓기, update=값 바꾸기, remove=치우기")
    collection: Literal["monitors", "extraDesks", "drawers", "sets", "pegs"] = Field(
        description="monitors=모니터 2 이상, extraDesks=추가 책상, drawers=서랍장, sets=모니터+노트북 세트, pegs=타공판"
    )
    number: int | None = Field(
        default=None, description="update·remove에서 대상의 화면 번호(서랍장 2 → 2, 책상 2 → 2). 0부터 세지 않는다. add에서는 쓰지 않는다."
    )
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="add·update에서 정할 값 {키: 값}. add에서는 사용자가 말한 것만 넣고 나머지는 비워 둔다(위치·높이·각도 등은 시뮬레이터가 자동으로 정한다).",
    )


class ProposeChanges(BaseModel):
    """설정 변경과 물건 추가·수정·삭제를 사용자에게 제안한다. 사용자가 승인해야 적용된다. 한 번만 호출하고 관련 변경은 모아서 넣는다."""

    changes: dict[str, Any] = Field(default_factory=dict, description='단일 설정 {키: 새 값}. 예: {"deskW": 1600, "speakers": true}')
    array_ops: list[ArrayOp] = Field(default_factory=list, description="물건 추가·수정·삭제 작업 목록(순서대로 적용)")
    reason: str = Field(description="왜 바꾸는지 한 줄 설명(사용자에게 보인다)")


class SceneAction(BaseModel):
    """장면 전체에 대한 동작을 사용자에게 제안한다. 사용자가 승인해야 실행된다. reset=모든 설정 초기화, save_slot=현재 설정을 슬롯에 저장, load_slot=슬롯 내용으로 교체."""

    action: Literal["reset", "save_slot", "load_slot"]
    slot: Literal["A", "B", "C", "D", "E", "F"] | None = Field(default=None, description="save_slot·load_slot에서 슬롯 이름")
    reason: str = Field(description="왜 하는지 한 줄 설명(사용자에게 보인다)")


class SetView(BaseModel):
    """3D 화면의 보기를 바꾼다. 승인 없이 바로 바뀐다. view=시점(iso 사선, front 정면, top 위에서, side 옆, pov 앉은 시점), auto_rotate=자동 회전."""

    view: Literal["iso", "front", "top", "side", "pov"] | None = None
    auto_rotate: bool | None = None


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    desk_state: dict
    prev_state: dict  # 지난 답변을 마친 때의 상태(브라우저가 보냄). 지금 상태와 비교해 사용자가 직접 바꾼 것을 알린다
    checks: list[dict]
    settings_schema: list[dict]
    collections: dict
    slots: dict
    context: str
    messages: Annotated[list, add_messages]
    loops: int
    answer: str
    blocked: bool  # 프롬프트 인젝션·범위 밖 요청으로 보여 LLM 없이 막았는가
    notice: str  # 막았을 때 shield가 내보낼 고정 안내
    declined: bool  # 이번 요청에서 사용자가 변경 제안을 거절했는가(그러면 같은 요청에서 다시 제안하지 않는다)


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


RETRY_BASE_WAIT, RETRY_MAX_WAIT = 1.0, 8.0  # 응답 도중 쓰로틀링 재시도 대기(초): 1, 2, 4, 8, 8 …
_sleep = time.sleep


def _is_stream_throttle(error: BaseException | None) -> bool:
    """응답을 받는 도중(스트림 이벤트)에 온 쓰로틀링인가.

    요청을 보낼 때의 쓰로틀링(ThrottlingException)은 boto3가 BEDROCK_MAX_RETRIES번 이미 재시도했으므로 여기서 또 하지 않는다.
    스트림이 열린 뒤에 오는 쓰로틀링(EventStreamError)은 boto3가 재시도하지 않아서 직접 다시 시도한다.
    """
    for _ in range(5):  # 감싸진 예외까지 본다
        if error is None:
            return False
        if type(error).__name__ == "EventStreamError" and "throttl" in str(error).lower():
            return True
        error = error.__cause__ or error.__context__
    return False


def _stream_reply(llm, messages: list, max_retries: int = 0) -> AIMessage:
    """LLM 응답을 문장이 끝날 때마다 token 이벤트로 내보내고, 도구 호출까지 합친 AIMessage를 돌려준다.

    응답 도중 쓰로틀링이 나면 max_retries번까지 처음부터 다시 받는다. 이미 화면에 내보낸 글이 있으면 겹치므로 다시 받지 않는다.
    """
    for attempt in range(max_retries + 1):
        full, buf, emitted = None, "", False
        try:
            stream = llm.stream(messages) if hasattr(llm, "stream") else [llm.invoke(messages)]
            for chunk in stream:
                full = chunk if full is None else full + chunk
                buf += _text_of(chunk)
                while (m := _BOUNDARY.search(buf)):
                    segment, buf = buf[: m.end()], buf[m.end():]
                    if segment.strip():
                        emitted = True
                        _emit({"type": "token", "text": guard.redact(segment)})  # 출력 가드: LLM이 개인정보를 말해도 가린다
            break
        except Exception as e:
            if emitted or attempt >= max_retries or not _is_stream_throttle(e):
                raise
            _sleep(min(RETRY_BASE_WAIT * 2 ** attempt, RETRY_MAX_WAIT))
    if buf.strip():
        _emit({"type": "token", "text": guard.redact(buf)})
    # 본문은 문자열로만 보관한다. 도구 호출은 tool_calls로 따로 들고 있어 다시 보낼 때 중복되지 않는다
    content = guard.redact(_text_of(full)) if full is not None else ""
    return AIMessage(content=content, tool_calls=list(getattr(full, "tool_calls", None) or []))


REJECTED_TEXT = "사용자가 거절했어요. 아무것도 적용하지 않았어요."
ALREADY_DECLINED_TEXT = "사용자가 이번 요청에서 이미 거절했어요. 다시 제안하지 말고 말로만 짧게 대안을 안내하세요."
MAX_TRACE_RESULT = 800  # 도구 결과를 기록에 남길 때 자르는 길이
APPLIED_PREFIXES = ("적용했어요", "화면을 바꿨어요")  # 실제로 무언가 적용됐을 때 도구 결과가 시작하는 말
CLAIM = re.compile(r"저장했|추가했|바꿨|변경했|삭제했|제거했|치웠|초기화했|불러왔|적용했|완료했|놓았|켰|껐|돌아갔")
NOT_APPLIED_NOTICE = "\n※ 참고: 이번 요청에서는 실제로 적용된 것이 없어요. 바꾸려면 한 번 더 말씀해 주세요."


def _history_messages(history: list[dict]) -> list:
    """이전 대화를 메시지로 복원한다. 어시스턴트 턴에 도구 호출 내역(tools)이 있으면 호출과 결과를 함께 되살린다.

    텍스트만 남기면 '완료했어요'라는 말만 보고 따라 해서, 도구를 부르지 않고 했다고 주장하게 된다.
    """
    out: list = []
    for i, m in enumerate(history):
        if m["role"] == "user":
            out.append(HumanMessage(guard.redact(m["content"])))
            continue
        for n, t in enumerate(m.get("tools") or []):
            call_id = f"h{i}_{n}"
            out.append(AIMessage(content="", tool_calls=[{"name": t["name"], "args": t.get("args") or {}, "id": call_id}]))
            out.append(ToolMessage(t.get("result") or "", tool_call_id=call_id))
        out.append(AIMessage(guard.redact(m["content"])))
    return out


def _trace_of(messages: list) -> list[dict]:
    """이번 질문에서 실제로 실행된 도구 호출을 [{name, args, result}]로 뽑아 대화 기록에 남길 수 있게 한다."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1) + 1
    results = {m.tool_call_id: m.content for m in messages[start:] if isinstance(m, ToolMessage)}
    trace = []
    for m in messages[start:]:
        for c in getattr(m, "tool_calls", None) or []:
            if c["id"] in results:
                trace.append({"name": c["name"], "args": c.get("args") or {}, "result": str(results[c["id"]])[:MAX_TRACE_RESULT]})
    return trace


def _outcome_text(decision: dict, problems: list[str]) -> str:
    """승인 뒤 브라우저가 적용한 결과를 LLM이 읽을 글로 만든다."""
    result = decision.get("result") or {}
    applied, failed = result.get("applied") or {}, result.get("rejected") or {}
    ops = result.get("ops") or []
    done = [f"{k}={v}" for k, v in applied.items()] + [o["text"] for o in ops if o.get("ok")]
    bad = [f"{k}: {why}" for k, why in failed.items()] + [o["text"] for o in ops if not o.get("ok")]
    text = "적용했어요: " + (", ".join(done) if done else "(적용된 것 없음)")
    if bad:
        text += " / 적용하지 못한 것: " + "; ".join(bad)
    if problems:
        text += " / 검증에서 거절된 것: " + "; ".join(problems)
    after = describe_collections(decision.get("state") or {})
    if after:  # 지우면 뒤 번호가 당겨지므로, 적용 뒤의 번호를 알려 줘 낡은 번호로 말하지 않게 한다
        text += "\n적용 뒤 현재 번호:\n" + "\n".join(after)
    if decision.get("checks"):
        text += "\n" + describe_checks(decision["checks"])
    return text


def _ask_approval(reason: str, items: list[dict], patch: dict, rejected: dict, notes: dict, problems: list[str]) -> str:
    """승인 카드를 띄우고 멈춘다. 취소면 그 사실을, 승인되면 적용 결과 글을 돌려준다."""
    decision = interrupt({"reason": guard.redact(reason), "changes": items, "patch": patch, "rejected": rejected, "notes": notes, "problems": problems})
    if not decision.get("approve"):
        _step_done("act", "제안을 사용자가 취소했어요")
        return REJECTED_TEXT
    ops = (decision.get("result") or {}).get("ops") or []
    applied = (decision.get("result") or {}).get("applied") or {}
    _step_done("act", f"{len(applied) + sum(1 for o in ops if o.get('ok'))}개를 적용했어요")
    return _outcome_text(decision, problems)


def _propose(state: AgentState, args: dict) -> str:
    """propose_changes 처리. 검증하고, 바뀌는 게 있으면 interrupt로 멈춰 승인을 받은 뒤 결과를 글로 돌려준다."""
    schema = state.get("settings_schema") or []
    desk = state.get("desk_state") or {}
    raw = args.get("changes")
    clean, rejected, notes = validate_patch(raw if isinstance(raw, dict) else {}, schema)
    items, same = diff_items(clean, desk, schema)
    ops_raw = args.get("array_ops")
    clean_ops, op_items, op_problems, op_notes = validate_ops(ops_raw if isinstance(ops_raw, list) else [], state.get("collections") or {}, desk)
    problems = [f"{k}: {why}" for k, why in rejected.items()] + op_problems
    notes = {**notes, **op_notes}
    bad_values = [f"{BAD_VALUE_PREFIX}{k}: {why}" for k, why in rejected.items()] + [p for p in op_problems if p.startswith(BAD_VALUE_PREFIX)]
    if bad_values:
        # 모델이 없는 키나 선택지를 추측한 제안은 카드로 띄우지 않는다(버려진 값을 빼고 띄우면 사용자가 모르고 승인할 수 있다).
        # 틀린 이유와 허용 키를 돌려줘 모델이 스스로 고쳐서 다시 제안하게 한다.
        collections = state.get("collections") or {}
        used = {op.get("collection") for op in (ops_raw if isinstance(ops_raw, list) else []) if isinstance(op, dict)}
        help_text = "\n".join(format_settings(schema, collections[n]["label"], collections) for n in sorted(used & set(collections)))
        return ("승인 카드를 띄우지 않았어요. " + "; ".join(bad_values) + ". 아래 키와 허용값을 확인해 고쳐서 다시 제안하세요."
                + ("\n" + help_text if help_text else ""))
    if not items and not clean_ops:
        parts = ["바꿀 항목이 없어서 제안하지 않았어요."]
        if same:
            parts.append("이미 그 값이에요: " + ", ".join(same))
        if problems:
            parts.append("거절된 항목: " + "; ".join(problems) + ". list_settings로 키와 허용값을 다시 확인하세요.")
        return " ".join(parts)
    patch = {"changes": {k: clean[k] for k in clean if k not in same}, "array_ops": clean_ops}
    return _ask_approval(str(args.get("reason") or ""), items + op_items, patch, rejected, notes, problems)


def _scene_action(state: AgentState, args: dict) -> str:
    action, item, problem = validate_scene_action(args.get("action"), args.get("slot"), state.get("slots") or {})
    if action is None:
        return f"실행하지 않았어요: {problem}"
    return _ask_approval(str(args.get("reason") or ""), [item], {"scene_action": action}, {}, {}, [])


def _set_view(args: dict) -> str:
    view, rotate = args.get("view"), args.get("auto_rotate")
    if view is not None and view not in VIEW_NAMES:
        return f"알 수 없는 시점이에요: {view}. 가능: " + ", ".join(f"{k}={v}" for k, v in VIEW_NAMES.items())
    if view is None and not isinstance(rotate, bool):
        return "바꿀 것이 없어요. view나 auto_rotate를 지정하세요."
    event: dict = {"type": "client_action", "action": "set_view"}
    done = []
    if view is not None:
        event["view"] = view
        done.append(f"시점={VIEW_NAMES[view]}")
    if isinstance(rotate, bool):
        event["auto_rotate"] = rotate
        done.append("자동 회전=" + ("켬" if rotate else "끔"))
    _emit(event)
    _step_done("act", "화면 보기를 바꿨어요")
    return "화면을 바꿨어요: " + ", ".join(done)


def _search_guides(retriever, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if retriever is None:
        _step_done("act", "가이드 문서를 쓸 수 없어요")
        return "가이드 문서 검색을 쓸 수 없어요. 문서에 없는 기준은 지어내지 말고 점검 결과만으로 답하세요."
    if not query:
        return "검색어가 비어 있어요. query에 찾을 내용을 넣어 다시 호출하세요."
    hits = retriever.search(query[:200], k=SEARCH_K)
    _step_done("act", f"가이드 문서 {len(hits)}개를 찾았어요({retriever.mode})")
    return format_hits(hits, query)


SUMMARY_HEADER = "\n\n앞선 대화 요약(오래된 대화라 원문은 빠짐. 적용 여부는 이 요약이 아니라 도구 결과와 현재 셋업으로 판단해):\n"


def default_summarizer(llm):
    """LCEL 요약 체인: 프롬프트 | LLM | 글 추출. Runnable이 아닌 LLM(테스트의 가짜 LLM)이면 None — 요약 없이 최근 대화만 쓴다."""
    if not isinstance(llm, Runnable):
        return None
    return PromptTemplate.from_template(memory.SUMMARY_PROMPT) | llm | RunnableLambda(_text_of)


def _summarize(summarizer, older: list[dict]) -> str:
    """오래된 대화 요약. 같은 구간은 캐시를 쓰고, 처음 요약할 때만 LLM을 부른다. 실패하면 요약 없이 진행한다."""
    if not older or summarizer is None:
        return ""
    key = memory.cache_key(older)
    summary = memory.cached(key)
    if summary is None:
        try:
            summary = guard.redact(str(summarizer.invoke({"conversation": memory.format_lines(older)})).strip())
        except Exception:  # 요약이 안 돼도 답변은 해야 한다. 최근 대화만으로 계속한다
            return ""
        memory.store(key, summary)
    return summary


def _search_products(state: AgentState, args: dict, shop_keys: tuple[str, str]) -> str:
    target = shop.target_from_state(str(args.get("target") or ""), state.get("desk_state") or {})
    if isinstance(target, str):
        _step_done("act", "찾을 물건을 알아보지 못했어요")
        return target
    r = shop.search(target, str(args.get("keywords") or ""), *shop_keys)
    # 상품 목록이 없으면(키 없음·실패) 브라우저가 네이버 쇼핑 검색을 새 탭으로 바로 연다(open)
    _emit({"type": "products", "open": not r["items"], **{k: r[k] for k in ("label", "query", "url", "items", "mode", "error")}})
    _step_done("act", f"{target.label}에 맞는 상품을 찾았어요({len(r['items'])}개)" if r["items"] else "네이버 쇼핑 검색 링크를 만들었어요")
    return shop.format_result(r)


def build_graph(llm, checkpointer=None, retriever=None, summarizer="default", shop_keys=("", ""), max_retries=0):
    """llm: ChatBedrockConverse 호환(bind_tools와 stream을 가진 객체). checkpointer는 interrupt에 필요해서 기본으로 메모리를 쓴다.
    retriever: 가이드 문서 검색기(src.retriever.HybridRetriever). None이면 search_guides가 '쓸 수 없음'을 돌려준다.
    summarizer: 오래된 대화 요약 Runnable({conversation} → 글). 기본은 llm으로 만든 LCEL 체인, None이면 요약하지 않는다.
    shop_keys: 네이버 검색 API (Client ID, Client Secret). 비어 있으면 상품 검색은 검색 링크만 돌려준다.
    max_retries: 응답 도중 쓰로틀링이 났을 때 다시 받는 횟수(BEDROCK_MAX_RETRIES). 요청 단계의 쓰로틀링은 llm의 boto3 설정이 같은 횟수로 재시도한다."""
    tool_llm = llm.bind_tools([ListSettings, SearchGuides, SearchProducts, ProposeChanges, SceneAction, SetView])
    if summarizer == "default":
        summarizer = default_summarizer(llm)

    def context(state: AgentState) -> AgentState:
        _emit({"type": "step", "phase": "start", "node": "context", "label": STEP_LABELS["context"]})
        raw_q = state["question"]
        if guard.looks_like_injection(raw_q):
            _step_done("context", "지시를 바꾸려는 요청으로 보여 LLM에 보내지 않았어요")
            return {"blocked": True, "notice": guard.INJECTION_NOTICE, "messages": [], "loops": 0, "answer": "", "declined": False}
        if guard.looks_off_topic(raw_q):  # 범위 밖 요청도 프롬프트가 아니라 코드로 막는다
            _step_done("context", "데스크 셋업과 관계없는 요청으로 보여 LLM에 보내지 않았어요")
            return {"blocked": True, "notice": guard.OFF_TOPIC_NOTICE, "messages": [], "loops": 0, "answer": "", "declined": False}
        text = describe(state.get("desk_state") or {}, state.get("checks") or [], state.get("slots") or {}, state.get("collections") or {})
        older, recent = memory.plan(state.get("history") or [])
        summary = _summarize(summarizer, older)
        system = SYSTEM_PROMPT.format(context=text) + (SUMMARY_HEADER + summary if summary else "")
        question = guard.redact(raw_q)  # 입력 가드: 개인정보는 LLM에 보내기 전에 가린다
        changes = describe_changes(state.get("prev_state"), state.get("desk_state") or {}, state.get("settings_schema"), state.get("collections"))
        if changes:  # 질문 바로 앞에 둬야 앞선 대화보다 무게가 실린다
            question = "[화면에서 직접 바꾼 것(지난 답변 이후, 이미 적용됨)]\n- " + "\n- ".join(changes) + "\n\n[질문]\n" + question
        messages = [SystemMessage(system)] + _history_messages(recent) + [HumanMessage(question)]
        notes = [f"점검 항목 {len(state.get('checks') or [])}개를 읽었어요"]
        if summary:
            notes.append(f"앞선 대화 {len(older)}개를 요약했어요")
        if guard.redact(raw_q) != raw_q:
            notes.append("질문 속 개인정보를 가리고 보냈어요")
        if changes:
            notes.append(f"화면에서 직접 바꾼 것 {len(changes)}건을 반영했어요")
        _step_done("context", " · ".join(notes))
        return {"context": text, "messages": messages, "loops": 0, "answer": "", "declined": False, "blocked": False}

    def shield(state: AgentState) -> AgentState:
        notice = state.get("notice") or guard.INJECTION_NOTICE
        _emit({"type": "token", "text": notice})
        return {"answer": notice}

    def agent(state: AgentState) -> AgentState:
        _emit({"type": "step", "phase": "start", "node": "agent", "label": STEP_LABELS["agent"]})
        reply = _stream_reply(tool_llm, state["messages"], max_retries)
        names = ", ".join(c["name"] for c in reply.tool_calls)
        _step_done("agent", f"도구를 골랐어요: {names}" if reply.tool_calls else "답변을 마쳤어요")
        out: dict = {"messages": [reply], "loops": state.get("loops", 0) + 1}
        if not reply.tool_calls:
            answer = reply.content
            if CLAIM.search(answer) and not any(t["result"].startswith(APPLIED_PREFIXES) for t in _trace_of(state["messages"])):
                # LLM이 도구를 부르지 않고 했다고 말한 경우. 거짓으로 남지 않게 사실을 덧붙인다
                _emit({"type": "token", "text": NOT_APPLIED_NOTICE})
                answer += NOT_APPLIED_NOTICE
            out["answer"] = answer
        return out

    def act(state: AgentState) -> AgentState:
        schema = state.get("settings_schema") or []
        results, asked = [], set()
        declined = bool(state.get("declined"))
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call.get("args") or {}
            if name == "ListSettings":
                content = format_settings(schema, args.get("section"), state.get("collections"))
                _step_done("act", f"설정 목록을 조회했어요({args.get('section') or '전체'})")
            elif name == "SearchGuides":
                content = _search_guides(retriever, args)
            elif name == "SearchProducts":
                content = _search_products(state, args, shop_keys)
            elif name == "SetView":
                content = _set_view(args)
            elif name in ("ProposeChanges", "SceneAction") and declined:
                content = ALREADY_DECLINED_TEXT  # 거절한 사용자에게 같은 요청에서 승인 카드를 또 띄우지 않는다(프롬프트가 아니라 코드로 막는다)
            elif name in ("ProposeChanges", "SceneAction") and name in asked:
                content = f"{name}는 한 번에 하나만 처리해요. 이 제안은 건너뛰었어요."
            elif name == "ProposeChanges":
                asked.add(name)
                content = _propose(state, args)
            elif name == "SceneAction":
                asked.add(name)
                content = _scene_action(state, args)
            else:
                content = f"알 수 없는 도구예요: {name}"
            declined = declined or content == REJECTED_TEXT
            results.append(ToolMessage(content, tool_call_id=call["id"]))
        return {"messages": results, "declined": declined}

    def stop(state: AgentState) -> AgentState:
        text = "생각이 길어져서 여기서 멈출게요. 원하는 걸 조금 더 구체적으로 다시 말씀해 주세요."
        _emit({"type": "token", "text": text})
        return {"answer": text}

    def after_context(state: AgentState) -> str:
        return "shield" if state.get("blocked") else "agent"

    def route(state: AgentState) -> str:
        if not state["messages"][-1].tool_calls:
            return END
        return "act" if state.get("loops", 0) < MAX_LOOPS else "stop"

    g = StateGraph(AgentState)
    g.add_node("context", context)
    g.add_node("agent", agent)
    g.add_node("act", act)
    g.add_node("stop", stop)
    g.add_node("shield", shield)
    g.add_edge(START, "context")
    g.add_conditional_edges("context", after_context, ["agent", "shield"])
    g.add_edge("shield", END)
    g.add_conditional_edges("agent", route, ["act", "stop", END])
    g.add_edge("act", "agent")
    g.add_edge("stop", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def build_default_graph():
    """.env 설정으로 Bedrock LLM을 만들어 그래프를 구성한다."""
    from langchain_aws import ChatBedrockConverse

    from src.config import bedrock_client_config, load_settings

    from src.retriever import build_retriever

    s = load_settings()
    retry = bedrock_client_config(s["max_retries"])
    llm = ChatBedrockConverse(model=s["llm_model_id"], region_name=s["region"], temperature=0.3, config=retry)
    retriever = build_retriever(s["embed_model_id"], s["region"], retry)
    # 문서 조각 임베딩(약 5초)을 미리 해 둬 첫 검색이 느리지 않게 한다. 실패하면 키워드 검색만으로 계속한다
    threading.Thread(target=retriever.warm_up, daemon=True).start()
    return build_graph(llm, retriever=retriever, shop_keys=(s["naver_client_id"], s["naver_client_secret"]), max_retries=s["max_retries"])


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
    yield {"type": "result", "answer": final.get("answer", ""), "tools": _trace_of(final.get("messages") or [])}


def ask_stream(graph, question: str, history: list[dict], desk_state: dict, checks: list[dict], settings_schema: list[dict],
               collections: dict | None = None, slots: dict | None = None, thread_id: str | None = None, prev_state: dict | None = None):
    """진행 이벤트(step·token·client_action)를 차례로 내보낸다. 끝은 {"type": "result", "answer"} 또는 승인 대기 {"type": "approval", ...}."""
    payload = {"question": question, "history": history, "desk_state": desk_state, "checks": checks, "settings_schema": settings_schema,
               "collections": collections or {}, "slots": slots or {}, "prev_state": prev_state or {}}
    yield from _run(graph, payload, thread_id or uuid.uuid4().hex)


def resume_stream(graph, thread_id: str, approve: bool, result: dict | None = None, checks: list[dict] | None = None,
                  state: dict | None = None):
    """승인(True)·취소(False)로 멈춘 호출을 이어서 진행한다. result는 브라우저가 적용한 결과, checks·state는 적용 뒤 점검 결과와 상태."""
    decision = {"approve": bool(approve), "result": result or {}, "checks": checks or [], "state": state or {}}
    yield from _run(graph, Command(resume=decision), thread_id)
