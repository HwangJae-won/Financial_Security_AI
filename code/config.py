DATA_PATH = "data/"
FAISS_INDEX_PATH = "faiss_index"
LAW_PATH = "laws/"

# MODEL_NAME = "QuantFactory/SOLAR-10.7B-Instruct-v1.0-GGUF"
# MODEL_FILE = "SOLAR-10.7B-Instruct-v1.0.Q8_0.gguf"
MODEL_NAME = "Mungert/EXAONE-Deep-7.8B-GGUF"
MODEL_FILE = "EXAONE-Deep-7.8B-q8_0.gguf"
# EMBEDDING_MODEL_NAME = "upskyy/kf-deberta-multitask" #재원 실험
EMBEDDING_MODEL_NAME ="intfloat/multilingual-e5-small" #유경 실험 

CACHE_DIR = "/dev/shm/huggingface_cache"
TMP_DIR = "/dev/shm/tmp"

TOP_K = 1
SCORE_THRESHOLD = 0.8