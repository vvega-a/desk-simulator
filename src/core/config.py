"""Bedrock 설정. 값은 .env(.env.sample 참고)에서만 읽고 코드에는 키도, 기본값도 두지 않는다.

필수 값이 하나라도 없으면 ConfigError를 낸다. 서버는 시작할 때 이를 확인하므로(app.py lifespan) 설정이 빠지면 뜨지 않는다.
"""
import os

from botocore.config import Config
from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """필수 설정이 없거나 값 형식이 잘못됐을 때."""


def _env(*names: str) -> str:
    """여러 이름 중 처음으로 값이 있는 환경변수를 돌려준다. 없거나 공백뿐이면 빈 문자열."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def load_settings() -> dict:
    load_dotenv()
    values = {
        # AWS_REGION이 있으면 그것을, 없으면 boto3 표준 이름인 AWS_DEFAULT_REGION(.env.sample)을 쓴다
        "region": (_env("AWS_REGION", "AWS_DEFAULT_REGION"), "AWS_REGION 또는 AWS_DEFAULT_REGION"),
        "llm_model_id": (_env("BEDROCK_LLM_MODEL_ID"), "BEDROCK_LLM_MODEL_ID"),
        "max_retries": (_env("BEDROCK_MAX_RETRIES"), "BEDROCK_MAX_RETRIES"),
    }
    missing = [label for value, label in values.values() if not value]
    if missing:
        raise ConfigError(
            "필수 설정이 없어요: " + ", ".join(missing) + ". .env.sample을 참고해 .env에 값을 채워 주세요."
        )
    settings = {key: value for key, (value, _) in values.items()}
    try:
        settings["max_retries"] = int(settings["max_retries"])
        if settings["max_retries"] < 0:
            raise ValueError
    except ValueError:
        raise ConfigError("BEDROCK_MAX_RETRIES는 0 이상의 정수여야 해요.") from None
    # 선택 값: 가이드 문서 검색(RAG)의 임베딩 모델. 비우면 벡터 검색 없이 키워드(BM25) 검색만 한다
    settings["embed_model_id"] = _env("BEDROCK_EMBED_MODEL_ID")
    return settings


def bedrock_client_config(max_retries: int) -> Config:
    """boto3 클라이언트용 재시도 설정. standard 모드는 스로틀링 오류를 백오프 후 다시 시도한다(첫 시도는 제외)."""
    return Config(retries={"max_attempts": max_retries, "mode": "standard"})
