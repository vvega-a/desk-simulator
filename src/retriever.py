"""가이드 문서 하이브리드 검색: 쿼리 확장 → BM25(키워드) + 벡터(Bedrock Titan 임베딩) → RRF 융합.

  data/*.md ─► '## ' 제목 단위 조각(Chunk)
  질문 ─► expand_query(도메인 동의어 사전, LLM 없음) ─► BM25 순위 ─┐
       └──────────────────────────────────────────► 벡터 순위 ─┴─► RRF(1/(60+순위) 합) ─► 상위 k개

- 한국어는 띄어쓰기·조사가 제각각이라 BM25 토큰에 어절과 함께 글자 2-gram을 넣는다('눈높이를' ↔ '눈높이').
- 임베딩이 없거나(BEDROCK_EMBED_MODEL_ID 미설정) 호출이 실패하면 키워드 검색만으로 계속 동작한다(mode로 알린다).
- 리랭커는 쓰지 않는다. 조각이 30여 개뿐이라 두 검색기의 순위 융합으로 충분했고, 계정에서 Bedrock 리랭크 모델을 쓸 수 없었다.
"""
import math
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data"
RRF_K = 60  # RRF 상수. 순위 차이가 점수에 미치는 영향을 완만하게 한다(흔히 쓰는 값)
CANDIDATES = 10  # 검색기마다 융합에 넘기는 후보 수

# 사용자가 쓰는 말 → 문서에 적힌 말. 증상·속어로 물어도 해당 절을 찾게 한다(BM25 쪽에만 더한다)
SYNONYMS = {
    "목": ["고개", "눈높이"], "고개": ["눈높이", "시야각"], "거북목": ["고개", "눈높이", "모니터 높이"],
    "눈높이": ["화면 위쪽"], "모니터 높이": ["화면 위쪽", "눈높이"], "화면 높이": ["화면 위쪽", "눈높이"],
    "멀": ["시청 거리"], "떨어": ["시청 거리"], "가까": ["시청 거리"], "거리": ["시청 거리"],
    "눈": ["시청 거리", "눈의 피로"], "피로": ["눈의 피로", "20-20-20"], "침침": ["눈의 피로"], "눈부": ["눈부심"],
    "손목": ["손목 자세", "키보드"], "어깨": ["팔꿈치", "책상 높이", "마우스"], "팔꿈치": ["책상 높이"], "허리": ["의자", "등받이"],
    "스탠딩": ["서서 쓰는 책상"], "서서": ["서서 쓰는 책상", "스탠딩"], "모션데스크": ["높이 조절 책상", "서서 쓰는 책상"],
    "듀얼": ["모니터 두 대", "여러 대"], "트리플": ["여러 대"], "와이드": ["울트라와이드"], "광시야": ["시야각"],
    "암": ["모니터암"], "스탠드": ["기본 스탠드", "받침"], "받침": ["기본 스탠드"], "피벗": ["세로 피벗"], "세로": ["세로 피벗"],
    "통로": ["책상 앞 공간"], "지나": ["책상 앞 공간", "통로"], "서랍": ["서랍장", "다리 공간"], "다리": ["다리 공간"],
    "인치": ["모니터 종류", "선택 기준"], "추천": ["선택 기준"], "고르": ["선택 기준", "고르기"], "몇": ["선택 기준"],
    "텐키리스": ["키보드 배열"], "배열": ["키보드 배열"], "매트": ["데스크매트"],
    "커브": ["커브드"], "곡면": ["커브드"], "본체": ["PC 본체"], "정리": ["타공판"], "의자": ["책상 앞 공간"],
}
_WORD = re.compile(r"[0-9a-z]+|[가-힣]+")


@dataclass
class Chunk:
    id: str  # 파일 이름#순번 (예: viewing-distance#1)
    title: str  # 문서 제목 > 절 제목
    source: str  # 파일 이름
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict = field(default_factory=dict)  # {"bm25": 1, "vector": 3} 어느 검색기에서 몇 위였나


def tokenize(text: str) -> list[str]:
    """어절(영문·숫자·한글 덩어리) + 한글 2-gram. 조사가 붙어도 2-gram이 겹쳐서 찾는다."""
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        out.append(w)
        if len(w) > 2 and "가" <= w[0] <= "힣":
            out += [w[i:i + 2] for i in range(len(w) - 1)]
    return out


def expand_query(query: str) -> list[str]:
    """질문에 사전의 말이 들어 있으면 문서 쪽 표현을 덧붙인다. 덧붙인 말만 돌려준다(중복 제외)."""
    q = query.lower()
    extra: list[str] = []
    for key, words in SYNONYMS.items():
        if key in q:
            extra += [w for w in words if w not in extra and w not in q]
    return extra


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    """'# 제목' 아래 '## 절' 하나를 조각 하나로 만든다. README.md는 문서 설명이라 뺀다."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        doc_title, section, lines, n = path.stem, None, [], 0

        def flush():
            nonlocal n
            body = "\n".join(lines).strip()
            if section and body:
                n += 1
                chunks.append(Chunk(f"{path.stem}#{n}", f"{doc_title} > {section}", path.name, body))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                section, lines = line[3:].strip(), []
            elif line.startswith("# "):
                doc_title = line[2:].strip()
            else:
                lines.append(line)
        flush()
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = [Counter(d) for d in docs], k1, b
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / len(docs) if docs else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.docs, self.lens):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, dict]]:
    """{검색기 이름: [조각 번호, ...](잘한 순)}를 RRF로 합친다. [(조각 번호, 점수, {검색기: 순위})]를 점수 순으로."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict] = {}
    for name, order in rankings.items():
        for r, idx in enumerate(order, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
            ranks.setdefault(idx, {})[name] = r
    return sorted(((i, s, ranks[i]) for i, s in scores.items()), key=lambda x: -x[1])


class HybridRetriever:
    """embeddings: LangChain Embeddings 호환(embed_documents·embed_query). None이면 키워드 검색만 한다."""

    def __init__(self, chunks: list[Chunk], embeddings=None, workers: int = 4):
        self.chunks, self.embeddings, self.workers = chunks, embeddings, workers
        self.bm25 = BM25([tokenize(c.title + " " + c.text) for c in chunks])
        self._vectors: list[list[float]] | None = None
        self._lock = threading.Lock()
        self.vector_error: str | None = None  # 임베딩이 실패하면 그 종류(키워드 검색으로 계속한다)

    @property
    def mode(self) -> str:
        return "hybrid" if self.embeddings is not None and self.vector_error is None else "keyword"

    def warm_up(self) -> None:
        """문서 조각을 한 번만 임베딩해 메모리에 둔다(첫 검색 전에 서버 시작 때 불러 두면 첫 질문이 빨라진다)."""
        if self.embeddings is None or self._vectors is not None or self.vector_error:
            return
        with self._lock:
            if self._vectors is not None or self.vector_error:
                return
            texts = [c.title + "\n" + c.text for c in self.chunks]
            try:
                # Titan은 한 번에 문장 하나만 받아서 조각마다 호출한다. 몇 개씩 동시에 보내 시간을 줄인다
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    self._vectors = list(pool.map(lambda t: self.embeddings.embed_query(t), texts))
            except Exception as e:  # 권한·네트워크 오류. 키워드 검색만으로 계속한다
                self.vector_error = type(e).__name__

    def _vector_order(self, query: str) -> list[int]:
        self.warm_up()
        if self._vectors is None:
            return []
        try:
            q = self.embeddings.embed_query(query)
        except Exception as e:
            self.vector_error = type(e).__name__
            return []
        sims = [_cosine(q, v) for v in self._vectors]
        return sorted(range(len(sims)), key=lambda i: -sims[i])[:CANDIDATES]

    def search(self, query: str, k: int = 3, use_bm25: bool = True, use_vector: bool = True, expand: bool = True) -> list[Hit]:
        """하이브리드 검색. use_*·expand는 평가에서 검색기별 성능을 비교하려고 둔 스위치다."""
        rankings: dict[str, list[int]] = {}
        if use_bm25:
            terms = tokenize(query + (" " + " ".join(expand_query(query)) if expand else ""))
            scores = self.bm25.scores(terms)
            rankings["bm25"] = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i]) if scores[i] > 0][:CANDIDATES]
        if use_vector and self.embeddings is not None:
            order = self._vector_order(query)
            if order:
                rankings["vector"] = order
        return [Hit(self.chunks[i], s, r) for i, s, r in rrf(rankings)[:k]]


def format_hits(hits: list[Hit], query: str) -> str:
    """도구 결과 글. 출처 제목을 붙여 LLM이 근거를 밝히고, 없는 기준을 지어내지 않게 한다."""
    if not hits:
        return f"'{query}'에 맞는 가이드 문서를 찾지 못했어요. 문서에 없는 기준은 지어내지 말고, 현재 셋업의 점검 결과만으로 답하세요."
    parts = [f"[{n}] {h.chunk.title} ({h.chunk.source})\n{h.chunk.text}" for n, h in enumerate(hits, start=1)]
    return "가이드 문서 검색 결과(이 내용과 점검 결과만 근거로 답하고, 쓴 문서 제목을 짧게 밝히세요):\n\n" + "\n\n".join(parts)


def build_retriever(embed_model_id: str = "", region: str = "", client_config=None) -> HybridRetriever:
    """embed_model_id가 있으면 Bedrock 임베딩으로 하이브리드, 없으면 키워드 검색만."""
    embeddings = None
    if embed_model_id:
        from langchain_aws import BedrockEmbeddings

        embeddings = BedrockEmbeddings(model_id=embed_model_id, region_name=region or None, config=client_config)
    return HybridRetriever(load_chunks(), embeddings)
