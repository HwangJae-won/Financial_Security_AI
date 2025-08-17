# 필요한 라이브러리들을 import 합니다.
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download

from config import MODEL_NAME, MODEL_FILE, CACHE_DIR

def load_llm(model_name, model_file, cache_dir, **kwargs):
    print(f"'{model_name}'에서 모델 파일을 다운로드합니다...")
    model_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename=MODEL_FILE,
        cache_dir=CACHE_DIR,
    )
    print(f"✅ 모델 파일 다운로드 완료: {model_path}")
    
    print("LlamaCpp를 사용하여 모델을 로딩합니다 (안정화 설정)...")
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=True,
        **kwargs # 재시도 파라미터를 받아서 초기화
    )
    print("✅ GGUF 모델 로딩 완료!")
    return llm


def setup_model(temperature=0.0, top_p=1.0):
    # LlamaCpp 모델 초기화 시점에 파라미터 전달
    return load_llm(MODEL_NAME, MODEL_FILE, CACHE_DIR, temperature=temperature, top_p=top_p)