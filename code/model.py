# 필요한 라이브러리들을 import 합니다.
# from langchain_community.llms import LlamaCpp
# from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, os
from config import LOCAL_DIR_EXAONE, CACHE_DIR, MODEL_NAME
#LOCAL_DIR_EXAONE = "/workspace/models/EXAONE-Deep-7.8B"
# CACHE_DIR = "/workspace/models/EXAONE-Deep-7.8B"
os.environ['HUGGINGFACE_HUB_CACHE'] = '/dev/shm/huggingface_cache'

def load_llm_and_tokenizer(model_name, cache_dir):
    """
    Hugging Face Hub에서 모델을 다운로드하여 지정된 캐시에 저장합니다.
    """
    print(f"'{cache_dir}'에 모델과 토크나이저를 다운로드합니다...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            )
    print("✅ 모델 및 토크나이저 로딩 완료!")

    return model, tokenizer



# def load_llm_light(model_name, model_file, cache_dir, **kwargs):
#     """
#     gguf 포맷의 모델 로드 for 경량화
#     """
    
#     print(f"'{model_name}'에서 모델 파일을 다운로드합니다...")
#     model_path = hf_hub_download(
#         repo_id=MODEL_NAME,
#         filename=MODEL_FILE,
#         cache_dir=CACHE_DIR,
#     )
#     print(f"✅ 모델 파일 다운로드 완료: {model_path}")
    
#     print("LlamaCpp를 사용하여 모델을 로딩합니다 (안정화 설정)...")
#     llm = LlamaCpp(
#         model_path=model_path,
#         n_gpu_layers=-1,
#         n_ctx=4096,
#         verbose=True,
#         **kwargs # 재시도 파라미터를 받아서 초기화
#     )
#     print("✅ GGUF 모델 로딩 완료!")
#     return llm

