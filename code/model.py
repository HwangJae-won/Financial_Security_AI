import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'

MODEL_NAME = "LGAI-EXAONE/EXAONE-Deep-7.8B"
cache_dir_path = "/dev/shm/huggingface_cache"
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=cache_dir_path,
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return pipe