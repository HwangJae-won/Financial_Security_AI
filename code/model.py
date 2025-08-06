import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'

MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
cache_dir_path = "/dev/shm/huggingface_cache"
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=cache_dir_path
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return pipe