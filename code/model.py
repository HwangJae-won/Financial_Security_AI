from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import TextGenerationPipeline
import torch, os

LOCAL_DIR_EXAONE = "/workspace/models/EXAONE-Deep-7.8B"

def load_model():
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_DIR_EXAONE, trust_remote_code=True, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR_EXAONE,
        device_map="auto",          # accelerate가 자동으로 배치
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    # 🚨 device 인자 제거!
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    return pipe



