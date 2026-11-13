#!/usr/bin/env bash
# 서버 실행: ./run.sh [포트]   (.env에 AWS 키 등이 있어야 AI 채팅이 동작한다)
set -euo pipefail
cd "$(dirname "$0")"
exec python -m uvicorn src.app:app --host 0.0.0.0 --port "${1:-8000}"
