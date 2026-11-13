"""가이드 문서 검색 평가: evaluation/rag_queries.json의 질의로 검색기 구성별 hit@1·hit@3·MRR을 잰다.

실행:  python -m src.evaluation.rag_eval
결과:  evaluation/reports/rag_latest.json, rag_latest.md

비교하는 구성
  bm25        키워드(BM25)만, 쿼리 확장 없음
  bm25+exp    키워드 + 도메인 동의어 쿼리 확장
  vector      Bedrock 임베딩(코사인)만
  hybrid      키워드+확장 + 벡터를 RRF로 합침 (에이전트가 실제로 쓰는 구성)
임베딩 모델(BEDROCK_EMBED_MODEL_ID)이 없으면 vector·hybrid는 건너뛴다.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.rag.retriever import HybridRetriever, build_retriever

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "evaluation"
CONFIGS = {
    "bm25": {"use_bm25": True, "use_vector": False, "expand": False},
    "bm25+exp": {"use_bm25": True, "use_vector": False, "expand": True},
    "vector": {"use_bm25": False, "use_vector": True, "expand": False},
    "hybrid": {"use_bm25": True, "use_vector": True, "expand": True},
}


def score_queries(retriever: HybridRetriever, queries: list[dict], config: dict, k: int = 3) -> dict:
    """질의마다 정답 조각의 첫 순위를 구해 hit@1·hit@k·MRR(상위 k 안에서)을 낸다."""
    rows = []
    for q in queries:
        ids = [h.chunk.id for h in retriever.search(q["query"], k=k, **config)]
        rank = next((i for i, cid in enumerate(ids, start=1) if cid in q["relevant"]), None)
        rows.append({"id": q["id"], "rank": rank, "top": ids})
    n = len(rows) or 1
    return {
        "hit@1": sum(1 for r in rows if r["rank"] == 1) / n,
        f"hit@{k}": sum(1 for r in rows if r["rank"]) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "rows": rows,
    }


def render(meta: dict, results: dict, k: int) -> str:
    lines = ["# 가이드 문서 검색 평가", "",
             f"- 실행 시각: {meta['when']} · 임베딩: `{meta['embed_model'] or '(없음)'}` · 질의 {meta['queries']}개 · 문서 조각 {meta['chunks']}개", "",
             f"| 구성 | hit@1 | hit@{k} | MRR@{k} |", "|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['hit@1']:.0%} | {r[f'hit@{k}']:.0%} | {r['mrr']:.3f} |")
    lines += ["", "## 질의별 정답 순위 (— = 상위 k 밖)", "", "| id | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    ids = [row["id"] for row in next(iter(results.values()))["rows"]] if results else []
    for i, qid in enumerate(ids):
        lines.append(f"| `{qid}` | " + " | ".join(str(r["rows"][i]["rank"] or "—") for r in results.values()) + " |")
    return "\n".join(lines) + "\n"


def main():
    from src.core.config import bedrock_client_config, load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=str(EVAL_DIR / "reports" / "rag_latest"))
    args = ap.parse_args()

    s = load_settings()
    retriever = build_retriever(s["embed_model_id"], s["region"], bedrock_client_config(s["max_retries"]))
    queries = json.loads((EVAL_DIR / "rag_queries.json").read_text(encoding="utf-8"))["queries"]
    retriever.warm_up()
    configs = {n: c for n, c in CONFIGS.items() if not c["use_vector"] or retriever.mode == "hybrid"}
    results = {name: score_queries(retriever, queries, c, args.k) for name, c in configs.items()}

    meta = {"when": datetime.now().strftime("%Y-%m-%d %H:%M"), "embed_model": s["embed_model_id"] if retriever.mode == "hybrid" else "",
            "queries": len(queries), "chunks": len(retriever.chunks), "vector_error": retriever.vector_error}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(render(meta, results, args.k), encoding="utf-8")
    for name, r in results.items():
        print(f"{name:9s} hit@1 {r['hit@1']:.0%} · hit@{args.k} {r[f'hit@{args.k}']:.0%} · MRR {r['mrr']:.3f}")
    print("→", out.with_suffix(".md"))


if __name__ == "__main__":
    main()
