"""
config.py — 경로/모델 설정 + LLM·임베딩 팩토리.
개발(api) ↔ 배포(local) 전환이 LLM_MODE 한 줄로 끝나도록.
"""
import os
from pathlib import Path

try:
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
PAGES_JSON = DATA_DIR / "pages.json"
PERSIST_ROOT = PROJECT_ROOT / "chroma_store"
COL_DOCS = "dongjin_pages"
PERSIST_DOCS = str(PERSIST_ROOT / COL_DOCS)

load_dotenv(dotenv_path=str(ENV_PATH))


def get_secret(key: str, default=None):
    if _HAS_ST:
        try:
            val = st.secrets.get(key, None)
            if val is not None:
                return val
        except StreamlitSecretNotFoundError:
            pass
        except Exception:
            pass
    return os.environ.get(key, default)


LLM_MODE = (get_secret("LLM_MODE", "api") or "api").lower()
API_LLM_MODEL = get_secret("API_LLM_MODEL", "gpt-4o-mini")
LOCAL_LLM_MODEL = get_secret("LOCAL_LLM_MODEL", "gemma3:4b")

API_EMBED_MODEL = get_secret("API_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = get_secret("LOCAL_EMBED_MODEL", "jhgan/ko-sroberta-multitask")

_openai_key = get_secret("OPENAI_API_KEY", None)
if _openai_key:
    os.environ["OPENAI_API_KEY"] = _openai_key


def get_embeddings():
    if LLM_MODE == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=API_EMBED_MODEL)


def get_llm(temperature: float = 0.2):
    # 요약은 약간의 유연성이 필요하므로 temperature 소폭 허용
    if LLM_MODE == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LOCAL_LLM_MODEL, temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=API_LLM_MODEL, temperature=temperature)


def embedding_signature() -> str:
    return f"{LLM_MODE}:{LOCAL_EMBED_MODEL if LLM_MODE=='local' else API_EMBED_MODEL}"
