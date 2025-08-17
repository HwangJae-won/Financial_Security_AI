# 필요한 라이브러리들을 import 합니다.
# from langchain_community.llms import LlamaCpp
# from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import MODEL_NAME, CACHE_DIR

def load_llm(model_name, cache_dir):
    """
    허깅 페이스에서 모델과 토크나이저 로딩
    """
    
    print(f"'{CACHE_DIR}'에서 모델과 토크나이저를 다운로드합니다...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)                                             
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            )
    print("✅ 모델 및 토크나이저 로딩 완료!")

    # pipeline
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # <-- 추가된 부분: 최대 생성 토큰 수 설정
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05
    )
    return llm



def load_llm_light(model_name, model_file, cache_dir, **kwargs):
    """
    gguf 포맷의 모델 로드 for 경량화
    """
    
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

