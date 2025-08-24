import os

# -----------------------------
# 데이터 경로 설정
# -----------------------------
DATA_PATH = "data/"
LAW_PATH = "laws/" 
CHROMA_PERSIST_DIRECTORY = "chroma_db"
INDEX_BASE_DIR = "rag_index"    
INDEX_NS_LAWS = "laws"
INDEX_NS_SUPP = "supplement"

# -----------------------------
# 모델 관련 설정
# -----------------------------
MODEL_NAME = "LGAI-EXAONE/EXAONE-Deep-7.8B"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LOCAL_DIR_EXAONE = "/workspace/models/EXAONE-Deep-7.8B"


CHROMA_LAWS_DIR = os.path.join(INDEX_BASE_DIR, "laws")
CHROMA_SUPP_DIR = os.path.join(INDEX_BASE_DIR, "supplement")
SUPP_PATH = "supplement/"


# -----------------------------
# 모델 캐시 경로 설정 (환경변수 기반)
# -----------------------------
# 환경변수 MODEL_CACHE가 있으면 그것 사용
# 없으면 기본 경로로 /root/.cache/huggingface 사용
CACHE_DIR = os.environ.get(
    "MODEL_CACHE",
    "/root/.cache/huggingface/EXAONE-Deep-7.8B"
)

# 임시 파일/작업 디렉토리

TMP_DIR = os.environ.get(
    "TMP_DIR",
    "/dev/shm/tmp"
)

# -----------------------------
# 실행 환경 플래그 및 캐시/오프라인 설정
# -----------------------------
# 오프라인(대회 환경)에서 네트워크 호출을 막기 위한 플래그
OFFLINE = os.environ.get("HF_HUB_OFFLINE", "1") == "1"

# 캐시/작업 디렉토리 존재 보장
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# 허깅페이스 캐시 경로를 통일
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
os.environ.setdefault("HF_HOME", CACHE_DIR)
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"