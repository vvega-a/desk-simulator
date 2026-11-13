"""입력·출력 가드레일: 개인정보(PII) 마스킹과 프롬프트 인젝션 차단. 고양이 챗봇(pjt)의 src/core/guard.py를 이 앱에 맞게 옮겼다.

  1. 입력 마스킹: 사용자가 질문·이전 대화에 쓴 전화번호·이메일·주민등록번호·카드번호·주소·"제 이름은 ○○○"를 LLM에 보내기 전에 가린다.
     데스크 셋업에는 개인정보가 필요 없어서, 가린 뒤 그대로 진행하고 진행 단계에 '가렸다'고만 알린다(승인을 묻지 않는다).
  2. 프롬프트 인젝션: 지시 무시·시스템 프롬프트 캐내기·역할 바꾸기 말투는 규칙으로 잡아 LLM 없이 고정 안내만 한다(graph의 shield).
  3. 출력 마스킹: LLM 답변 조각·승인 카드의 이유 글에도 같은 마스킹을 한 번 더 적용한다.

pjt와 다른 점
  - 병원·수의사 이름, 환자·접수 번호 규칙은 이 앱과 관계없어서 뺐다.
  - 인젝션 대상 말에서 '설정'·'제한'을, 동작 말에서 '변경해'·'바꿔'를 뺐다. 이 앱에서 "설정 바꿔줘"는 정상 요청이다.
  - 정규식 기반이라 새로운 말투는 놓칠 수 있다(놓친 인젝션은 시스템 프롬프트의 규칙과 모델 판단이 2차로 막는다).
"""
import re

NAME_MASK = "○○○"
ADDRESS_MASK = "[주소 비공개]"
PHONE_MASK = "[전화번호 비공개]"
NUMBER_MASK = "[번호 비공개]"
EMAIL_MASK = "[이메일 비공개]"

INJECTION_NOTICE = (
    "도우미의 지시나 규칙을 바꾸거나 알아내려는 요청으로 보여서 처리하지 않았어요. "
    "책상·모니터 배치나 점검 결과에 대해 물어봐 주세요."
)

_PHONE = re.compile(r"(?<!\d)0\d{1,2}[-.\s]\d{3,4}[-.\s]\d{4}(?!\d)")
_PHONE_PLAIN = re.compile(r"(?<!\d)01[016789]\d{7,8}(?!\d)")  # 하이픈·공백 없는 휴대전화 번호
_CARD = re.compile(r"(?<!\d)(?:\d{4}[-\s]){3}\d{4}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_RRN = re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)")
_ADDRESS = re.compile(
    r"[가-힣]+(?:특별시|광역시|특별자치시|특별자치도|도)\s*[가-힣]+(?:시|군|구)\s*[가-힣0-9]+(?:로|길)\s*\d+(?:-\d+)?"
    r"(?:\s*,\s*\d+호)?(?:\s*\([가-힣0-9]+\))?"
)
# 본인 이름: "제 이름은 홍길동". "제 이름이 뭐야?"처럼 묻는 말은 건드리지 않는다
_SELF_NAME = re.compile(r"((?:제|내|저의|나의)\s*(?:이름|성함)\s*(?:은|는|이)?\s*[:：]?\s*)(?!뭐|뭔|무엇|무슨|어떻|어떤|누구|알|기억|맞)([가-힣]{2,4})")

# 인젝션: '지시·규칙' 같은 대상 말과 '무시·공개' 같은 동작 말이 함께 있을 때만 본다("규칙대로 배치해줘"는 걸리지 않는다)
_INJ_TARGET = r"(?:지시|명령|규칙|지침|프롬프트|가드레일)(?:사항|문)?"
_INJECTION = re.compile(
    rf"{_INJ_TARGET}\s*(?:을|를|은|는|이|가|도)?\s*(?:모두\s*|전부\s*|다\s*)?(?:무시|잊어|잊고|어기|어겨|해제|풀어|우회|따르지|취소)"
    rf"|(?:무시|잊어|해제|우회)\w*\s*(?:하고|해|해라|해줘|하세요)?\s*.{{0,12}}{_INJ_TARGET}"
    rf"|(?:시스템|초기|숨겨진|내부)\s*(?:프롬프트|지시|지침|명령)|system\s*prompt"
    rf"|{_INJ_TARGET}\s*(?:을|를)?\s*(?:알려|보여|출력|공개|말해|복사|그대로)"
    r"|(?:지금|이제)\s*부터\s*(?:너|넌|당신)"
    r"|탈옥|개발자\s*모드|developer\s*mode|jailbreak|\bDAN\b"
    r"|ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions?|rules?|prompts?|guidelines?)"
    r"|disregard\s+(?:all\s+|the\s+|your\s+)?(?:previous|prior|above)?\s*(?:instructions?|rules?)"
    r"|(?:reveal|show|print|repeat)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+)?(?:prompt|instructions?)"
    r"|you\s+are\s+now\b|act\s+as\s+(?:an?\s+)?(?:unrestricted|different)",
    flags=re.I,
)


def redact(text: str) -> str:
    """텍스트 속 개인정보를 마스킹한다. 개인정보가 없으면 원문 그대로 돌려준다."""
    if not text:
        return text
    text = _CARD.sub(NUMBER_MASK, text)
    text = _PHONE.sub(PHONE_MASK, text)
    text = _PHONE_PLAIN.sub(PHONE_MASK, text)
    text = _RRN.sub(NUMBER_MASK, text)
    text = _EMAIL.sub(EMAIL_MASK, text)
    text = _ADDRESS.sub(ADDRESS_MASK, text)
    return _SELF_NAME.sub(lambda m: f"{m.group(1)}{NAME_MASK}", text)


def looks_like_injection(question: str) -> bool:
    """도우미의 지시를 무시시키거나 시스템 프롬프트를 캐내려는 질문인지 규칙으로 판단한다(LLM 호출 없음)."""
    return bool(_INJECTION.search(question or ""))
