# 데스크 셋업 시뮬레이터 + AI 도우미 서버
#   빌드: docker build -t desk-simulator .
#   실행: docker run --rm -p 8000:8000 --env-file .env desk-simulator   → http://localhost:8000
# .env(AWS 키 등)는 이미지에 넣지 않는다(.dockerignore). 실행할 때 --env-file로 넘긴다.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
