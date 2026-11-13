"""Bedrock 없이 도는 쓰로틀링 재시도 테스트: 응답 도중 쓰로틀링이 나면 BEDROCK_MAX_RETRIES번까지 다시 받는다."""
import pytest

from src import agent
from src.app import _relay
from tests.test_agent import CHECKS, SCHEMA, STATE, FakeLLM, text


class EventStreamError(Exception):
    """botocore.exceptions.EventStreamError 흉내: 스트림이 열린 뒤에 오는 오류."""


class Throttled(FakeLLM):
    """처음 fails번은 조각을 before만큼 내보낸 뒤 오류로 끊기고, 그다음부터는 정상 응답한다."""

    def __init__(self, fails, error=None, before=()):
        super().__init__(text("괜찮아요."))
        self.fails, self.before = fails, before
        self.error = error or EventStreamError("throttlingException: Too many requests")

    def stream(self, messages):
        self.calls.append(list(messages))
        if len(self.calls) <= self.fails:
            yield from self.before
            raise self.error
        yield from self.turns[0]


@pytest.fixture
def waits(monkeypatch):
    slept = []
    monkeypatch.setattr(agent, "_sleep", slept.append)
    return slept


def ask(llm, max_retries):
    g = agent.build_graph(llm, max_retries=max_retries)
    return list(agent.ask_stream(g, "어때?", [], STATE, CHECKS, SCHEMA, thread_id="t1"))


def test_mid_stream_throttling_is_retried_up_to_max_retries(waits):
    llm = Throttled(fails=3)
    events = ask(llm, 3)
    assert events[-1]["answer"] == "괜찮아요." and len(llm.calls) == 4  # 첫 시도 + 재시도 3번
    assert waits == [1.0, 2.0, 4.0]


def test_throttling_gives_up_after_max_retries(waits):
    llm = Throttled(fails=99)
    with pytest.raises(EventStreamError):
        ask(llm, 2)
    assert len(llm.calls) == 3 and len(waits) == 2


def test_zero_retries_means_no_retry(waits):
    llm = Throttled(fails=1)
    with pytest.raises(EventStreamError):
        ask(llm, 0)
    assert len(llm.calls) == 1 and waits == []


@pytest.mark.parametrize("error", [RuntimeError("boom"), RuntimeError("ThrottlingException")])
def test_other_errors_and_request_time_throttling_are_not_retried_here(waits, error):
    llm = Throttled(fails=1, error=error)  # 요청 단계 쓰로틀링은 boto3가 이미 같은 횟수로 재시도했다
    with pytest.raises(RuntimeError):
        ask(llm, 3)
    assert len(llm.calls) == 1 and waits == []


def test_throttling_after_text_was_shown_is_not_retried(waits):
    llm = Throttled(fails=1, before=text("첫 문장이에요. "))  # 이미 화면에 나간 글이 있으면 다시 받지 않는다(글이 겹친다)
    with pytest.raises(EventStreamError):
        ask(llm, 3)
    assert len(llm.calls) == 1 and waits == []


def test_wait_is_capped(waits):
    ask(Throttled(fails=6), 6)
    assert waits == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_boto_config_retries_exactly_max_retries_times():
    from src.config import bedrock_client_config

    cfg = bedrock_client_config(3)
    assert cfg.retries == {"max_attempts": 3, "mode": "standard"}


def test_throttling_error_message_asks_to_try_later():
    def boom():
        raise EventStreamError("throttlingException: secret-detail")
        yield

    out = "".join(_relay(boom()))
    assert "잠시 뒤 다시" in out and "secret-detail" not in out
