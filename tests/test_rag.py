"""Bedrock 없이 도는 RAG 테스트: 조각 나누기, 토큰화·쿼리 확장, BM25·RRF 융합, 임베딩 실패 대비, search_guides 도구, LCEL 판정 체인."""
import json
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from src import agent
from src import evaluate as ev
from src import rag_eval
from src import retriever as rag
from tests.test_agent import FakeLLM, call, run, text, tool_messages

ROOT = Path(__file__).resolve().parents[1]
DOC = """# 모니터 높이
## 화면 위쪽과 눈높이
화면 위쪽이 눈높이 근처면 좋다.
## 모니터암
모니터암은 높이 조절 범위가 넓다.
"""


class FakeEmbeddings:
    """글에 든 낱말 몇 개로 벡터를 만드는 가짜 임베딩. fail=True면 호출할 때 오류를 낸다."""

    WORDS = ["눈높이", "모니터암", "서랍장", "의자"]

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def embed_query(self, t):
        self.calls += 1
        if self.fail:
            raise PermissionError("no access")
        return [float(w in t) for w in self.WORDS] + [0.01]


def chunks(tmp_path):
    (tmp_path / "height.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "README.md").write_text("# 설명\n## 무시\n검색 대상이 아님", encoding="utf-8")
    return rag.load_chunks(tmp_path)


# ---- 조각·토큰 ----------------------------------------------------------------------------

def test_load_chunks_splits_by_section_and_skips_readme(tmp_path):
    cs = chunks(tmp_path)
    assert [c.id for c in cs] == ["height#1", "height#2"]
    assert cs[0].title == "모니터 높이 > 화면 위쪽과 눈높이" and "눈높이 근처" in cs[0].text


def test_real_knowledge_base_loads_every_document():
    cs = rag.load_chunks()
    assert len(cs) >= 20 and len({c.source for c in cs}) >= 6
    assert all(c.text.strip() for c in cs) and len({c.id for c in cs}) == len(cs)


def test_tokenize_adds_bigrams_so_particles_still_match():
    assert "눈높" in rag.tokenize("눈높이를") and "눈높" in rag.tokenize("눈높이")
    assert "27" in rag.tokenize('27" 모니터')


def test_expand_query_maps_symptoms_to_document_words():
    assert "고개" in rag.expand_query("거북목이 걱정돼") and "서서 쓰는 책상" in rag.expand_query("스탠딩 데스크")
    assert rag.expand_query("안녕") == []


# ---- 검색 ---------------------------------------------------------------------------------

def test_rrf_rewards_documents_found_by_both_retrievers():
    fused = rag.rrf({"bm25": [0, 1], "vector": [1, 2]})
    assert fused[0][0] == 1 and fused[0][2] == {"bm25": 2, "vector": 1}


def test_keyword_only_when_no_embeddings(tmp_path):
    r = rag.HybridRetriever(chunks(tmp_path))
    hits = r.search("모니터암 높이 조절")
    assert r.mode == "keyword" and hits[0].chunk.id == "height#2" and "vector" not in hits[0].ranks


def test_hybrid_uses_both_retrievers(tmp_path):
    r = rag.HybridRetriever(chunks(tmp_path), FakeEmbeddings())
    hits = r.search("눈높이")
    assert r.mode == "hybrid" and hits[0].chunk.id == "height#1" and set(hits[0].ranks) == {"bm25", "vector"}


def test_documents_are_embedded_once(tmp_path):
    emb = FakeEmbeddings()
    r = rag.HybridRetriever(chunks(tmp_path), emb)
    r.search("눈높이")
    r.search("모니터암")
    assert emb.calls == 2 + 2  # 조각 2개는 한 번만, 그다음은 질의마다 1번


def test_embedding_failure_falls_back_to_keywords(tmp_path):
    r = rag.HybridRetriever(chunks(tmp_path), FakeEmbeddings(fail=True))
    hits = r.search("모니터암")
    assert r.mode == "keyword" and r.vector_error == "PermissionError" and hits[0].chunk.id == "height#2"


def test_format_hits_cites_sources_and_says_when_nothing_found(tmp_path):
    r = rag.HybridRetriever(chunks(tmp_path))
    out = rag.format_hits(r.search("눈높이"), "눈높이")
    assert "[1] 모니터 높이 > 화면 위쪽과 눈높이 (height.md)" in out
    assert "지어내지" in rag.format_hits([], "날씨")


# ---- 에이전트 도구 ------------------------------------------------------------------------

def test_search_guides_tool_feeds_documents_to_the_llm(tmp_path):
    llm = FakeLLM(call("SearchGuides", {"query": "모니터 높이"}), text("가이드에 따르면 화면 위쪽을 눈높이에 맞추세요."))
    g = agent.build_graph(llm, retriever=rag.HybridRetriever(chunks(tmp_path)))
    events = list(agent.ask_stream(g, "모니터 높이?", [], {}, [], [], thread_id="t1"))
    got = tool_messages(llm, 1)
    assert "화면 위쪽이 눈높이 근처면 좋다" in got[0]
    result = events[-1]
    assert result["type"] == "result" and result["tools"][0]["name"] == "SearchGuides"
    assert not any(e["type"] == "approval" for e in events)  # 읽기 도구라 승인이 없다


def test_search_guides_without_retriever_says_unavailable():
    llm = FakeLLM(call("SearchGuides", {"query": "모니터 높이"}), text("점검 결과로만 말할게요."))
    run(llm)
    assert "쓸 수 없어요" in tool_messages(llm, 1)[0]


# ---- 평가 ---------------------------------------------------------------------------------

def test_rag_eval_set_points_at_real_chunks():
    ids = {c.id for c in rag.load_chunks()}
    queries = json.loads((ROOT / "evaluation" / "rag_queries.json").read_text(encoding="utf-8"))["queries"]
    assert queries and all(set(q["relevant"]) <= ids for q in queries)


def test_rag_eval_scores_hits_and_mrr(tmp_path):
    r = rag.HybridRetriever(chunks(tmp_path))
    out = rag_eval.score_queries(r, [{"id": "a", "query": "모니터암", "relevant": ["height#2"]},
                                     {"id": "b", "query": "없는말", "relevant": ["height#1"]}], rag_eval.CONFIGS["bm25"])
    assert out["hit@1"] == 0.5 and out["hit@3"] == 0.5 and out["mrr"] == 0.5


def test_tools_any_expectation():
    first = {"error": None, "approval": None, "client_actions": [], "tokens": "", "answer": "눈높이",
             "tools": [{"name": "ListSettings", "args": {}, "result": ""}]}
    c = {"id": "x", "category": "in", "question": "q", "expect": {"approval": False, "tools_any": ["SearchGuides"]}}
    assert any("기대한 도구" in f for f in ev.check_run(c, first, None))
    first["tools"].append({"name": "SearchGuides", "args": {}, "result": ""})
    assert ev.check_run(c, first, None) == []


class FakeJudge:
    """with_structured_output이 프롬프트를 받아 Verdict를 돌려주는 Runnable을 주는 가짜 판정 LLM."""

    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        def judge(prompt):
            self.prompts.append(prompt.to_string())
            return schema(ok=True, reason="정중히 사양함")
        return RunnableLambda(judge)


def test_judge_is_an_lcel_chain_returning_a_dict():
    judge = FakeJudge()
    out = ev.judge_out_of_scope(judge, "날씨 알려줘", "셋업 얘기만 도울 수 있어요")
    assert out == {"ok": True, "reason": "정중히 사양함"}
    assert "사용자 요청: 날씨 알려줘" in judge.prompts[0] and "도우미 답변: 셋업 얘기만" in judge.prompts[0]
