import os

DATA_PATH = "data/"
FAISS_INDEX_PATH = "faiss_index"
LAW_PATH = "laws/"  # 실제 폴더 구조에 맞게 수정

# 현재 사용 모델
MODEL_NAME = "LGAI-EXAONE/EXAONE-Deep-7.8B"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
LOCAL_DIR_EXAONE = "/workspace/models/EXAONE-Deep-7.8B"
CACHE_DIR = "/dev/shm/models/EXAONE-Deep-7.8B"
TMP_DIR = "/dev/shm/tmp"
CHROMA_PERSIST_DIRECTORY = "chroma_db"
TOP_K = 30
SCORE_THRESHOLD = 0.85


# 실험/참고용 모델
# SOLAR_MODEL_NAME = "QuantFactory/SOLAR-10.7B-Instruct-v1.0-GGUF"
# SOLAR_MODEL_FILE = "SOLAR-10.7B-Instruct-v1.0.Q8_0.gguf"
# EXAONE_GGUF_MODEL_NAME = "Mungert/EXAONE-Deep-7.8B-GGUF"
# EXAONE_GGUF_MODEL_FILE = "EXAONE-Deep-7.8B-q8_0.gguf"