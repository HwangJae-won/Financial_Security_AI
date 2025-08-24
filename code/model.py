from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder 
import torch, os
from config import CACHE_DIR, MODEL_NAME, OFFLINE, RERANKER_MODEL_NAME

def load_llm_and_tokenizer(
    model_name: str = MODEL_NAME,
    cache_dir: str = CACHE_DIR,
    offline: bool = OFFLINE,
):
    """
    모델과 토크나이저를 로컬 캐시에서 로드
    오프라인 환경 지원을 위해 HF_HUB_OFFLINE 및 local_files_only를 활용
    """
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        local_files_only = True
    else:
        local_files_only = False

    print(f"[load_llm_and_tokenizer] cache_dir={cache_dir}, offline={offline}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model.eval()
    print("✅ 모델 및 토크나이저 로딩 완료!")
    return model, tokenizer


def build_text_generator(model, tokenizer, max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.9):
    """
    text-generation 파이프라인을 생성합니다.
    """
    device = 0 if torch.cuda.is_available() else -1
    pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

def load_reranker_model(model_name: str = RERANKER_MODEL_NAME):
    """
    리랭커 모델을 로드합니다.
    오프라인 환경에서는 미리 캐시된 가중치를 사용합니다(HF_HUB_OFFLINE=1 설정에 따름).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_name, max_length=512, device=device)