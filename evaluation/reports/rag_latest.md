# 가이드 문서 검색 평가

- 실행 시각: 2026-11-13 11:49 · 임베딩: `amazon.titan-embed-text-v2:0` · 질의 21개 · 문서 조각 27개

| 구성 | hit@1 | hit@3 | MRR@3 |
|---|---|---|---|
| bm25 | 62% | 90% | 0.754 |
| bm25+exp | 86% | 100% | 0.921 |
| vector | 81% | 90% | 0.857 |
| hybrid | 100% | 100% | 1.000 |

## 질의별 정답 순위 (— = 상위 k 밖)

| id | bm25 | bm25+exp | vector | hybrid |
|---|---|---|---|---|
| `q_dist_formula` | 1 | 1 | 1 | 1 |
| `q_dist_32` | 2 | 1 | — | 1 |
| `q_dist_fix` | 1 | 1 | 2 | 1 |
| `q_fov` | 1 | 1 | 1 | 1 |
| `q_eye_strain` | — | 1 | 1 | 1 |
| `q_mon_height` | 2 | 2 | 1 | 1 |
| `q_neck` | 1 | 1 | 1 | 1 |
| `q_laptop` | 1 | 1 | — | 1 |
| `q_pivot` | — | 1 | 2 | 1 |
| `q_dual` | 1 | 1 | 1 | 1 |
| `q_desk_height` | 2 | 3 | 1 | 1 |
| `q_standing` | 1 | 1 | 1 | 1 |
| `q_chair` | 3 | 1 | 1 | 1 |
| `q_kb_gap` | 2 | 1 | 1 | 1 |
| `q_kb_layout` | 1 | 1 | 1 | 1 |
| `q_wrist` | 1 | 1 | 1 | 1 |
| `q_aisle` | 2 | 2 | 1 | 1 |
| `q_drawer` | 1 | 1 | 1 | 1 |
| `q_arm` | 1 | 1 | 1 | 1 |
| `q_uw_vs_dual` | 1 | 1 | 1 | 1 |
| `q_window` | 1 | 1 | 1 | 1 |
