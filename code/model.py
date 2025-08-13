import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# --- 진짜 문제 해결을 위한 핵심 코드 ---

# 1. 공간이 넉넉한 /workspace 내에 캐시 및 임시 폴더 경로를 지정합니다.
CACHE_DIR = "/workspace/huggingface_cache"
TMP_DIR = "/workspace/tmp"

# 2. 모든 라이브러리가 이 경로를 사용하도록 '환경 변수'를 설정합니다.
#    이 작업은 다른 어떤 코드보다 먼저 실행되어야 합니다.
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TMPDIR'] = TMP_DIR
os.environ['TEMP'] = TMP_DIR
os.environ['TMP'] = TMP_DIR

# 3. 스크립트 실행 시 해당 폴더들이 존재하도록 생성해줍니다.
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

print(f"✅ Hugging Face 캐시 폴더가 다음으로 설정되었습니다: {os.environ.get('HF_HOME')}")
print(f"✅ 임시 파일 폴더가 다음으로 설정되었습니다: {os.environ.get('TMPDIR')}")


MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"

def load_model():
    """
    지정된 Hugging Face 모델과 토크나이저를 로드합니다.
    캐시 파일과 임시 파일 모두 /workspace 내의 지정된 경로를 사용합니다.
    """
    print(f"Tokenizer를 로딩합니다: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )
    
    print(f"Model을 로딩합니다: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    
    print("Text generation pipeline을 생성합니다.")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    print("✅ 모델 및 파이프라인 로딩 완료!")
    return pipe
