
#경로 지정
import os
os.chdir("/content/drive/MyDrive/데이콘/")

import re
import os
import pandas as pd
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

path = "/content/drive/MyDrive/데이콘/"
# 모델 캐싱 경로
cache_dir_path = "/content/drive/MyDrive/huggingface_cache"
model_name = "beomi/gemma-ko-7b"

# Data Load & Define utils
test = pd.read_csv(path +'test.csv')
print(test)

# 객관식 여부 판단 함수
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

# 질문과 선택지 분리 함수
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())

    question = " ".join(q_lines)
    return question, options

# 프롬프트 생성기
def make_prompt_auto(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
                "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 아래 질문에 대해 가장 정확하고 신뢰성 있는 답변을 제공하세요.\\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 아래 질문에 대해 가장 정확하고 신뢰성 있는 답변을 제공하세요.\\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
                )
    return prompt

# Model Load
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("✅ 저장된 모델을 로드하는 중...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        cache_dir=cache_dir_path
    )
    print("✅ 모델 로딩 성공!")

    # Inference pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

except Exception as e:
    print(f"❌ 모델 로딩 중 오류가 발생했습니다: {e}")

model_name = "beomi/gemma-ko-7b"

# Tokenizer 및 모델 로드 (4bit)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inference
# 후처리 함수
def extract_answer_only(generated_text: str, original_question: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
    - 주관식 문제면: 전체 텍스트 그대로 반환
    - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
    """
    # "답변:" 기준으로 텍스트 분리
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()

    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            # 숫자 추출 실패 시 "0" 반환
            return "0"
    else:
        return text

# preds = []
# for q in tqdm(test['Question'], desc="Inference"):
#     prompt = make_prompt_auto(q)
#     output = pipe(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)
#     pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
#     preds.append(pred_answer)

# Submission
# sample_submission = pd.read_csv('/content/drive/MyDrive/데이콘/sample_submission.csv')
# sample_submission['Answer'] = preds
# sample_submission.to_csv('./baseline_submission.csv', index=False, encoding='utf-8-sig')

fewshot_examples = """
질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?
선택지:
1. 데이터 암호화
2. 서버 인증
3. 클라이언트 인증
4. 무결성 확인
답변: 3

질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.
답변: 윈도우 서버를 대상으로 한 해킹 공격을 방어하기 위해서는 보안 업데이트를 주기적으로 적용하고, 불필요한 서비스 포트를 차단해야 합니다. 또한, 강력한 계정 정책을 사용하고, IDS/IPS와 같은 보안 시스템을 도입하여 실시간으로 공격을 탐지하고 차단하는 것이 중요합니다.
"""

# 2. 프롬프트 생성 함수 수정 (Few-shot, 역할 부여 포함)
def make_prompt_fewshot(text):
    base_role_prompt = "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 아래 질문에 대해 가장 정확하고 신뢰성 있는 답변을 제공하세요.\\n"

    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            f"{base_role_prompt}"
            f"아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\\n\\n"
            f"{fewshot_examples}\\n\\n"
            f"질문: {question}\\n"
            f"선택지:\\n"
            f"{chr(10).join(options)}\\n\\n"
            "답변:"
        )
    else:
        # Chain-of-Thought (CoT) Prompting 적용 예시
        prompt = (
            f"{base_role_prompt}"
            "아래 주관식 질문에 대해 먼저 단계적으로 생각하고, 그 과정을 바탕으로 정확하고 간략한 설명을 작성하세요.\\n\\n"
            f"질문: {text}\\n\\n"
            "생각 과정:"
        )
    return prompt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_name = "beomi/gemma-ko-7b"
cache_path = "/content/drive/MyDrive/데이콘/huggingface_cache"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)

from transformers import  BitsAndBytesConfig

# !pip install bitsandbytes

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("모델을 로드하는 중...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    cache_dir=cache_path,
)
print("✅ 모델 로딩 성공!")

# Inference pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# 추론
BATCH_SIZE = 8
preds = []
prompts = [make_prompt_fewshot(q) for q in tqdm(test['Question'], desc="프롬프트 생성")]
batches = [prompts[i:i + BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

with tqdm(batches, desc="추론 진행") as pbar:
    for batch_prompts in pbar:
        outputs = pipe(
            batch_prompts,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.9,
            batch_size=BATCH_SIZE
        )
        for i, batch_output in enumerate(outputs):
            output_text = batch_output[0]["generated_text"]
            current_question_index = (pbar.n - 1) * BATCH_SIZE + i
            original_question = test['Question'].iloc[current_question_index]
            pred_answer = extract_answer_only(output_text, original_question=original_question)
            preds.append(pred_answer)

print("\n✅ 추론 완료!")
print(f"생성된 답변 개수: {len(preds)}")

sample_submission = pd.read_csv('/content/drive/MyDrive/데이콘/sample_submission.csv')
sample_submission['Answer'] = preds
sample_submission.to_csv('/content/drive/MyDrive/데이콘/fewshot_change_prompt.csv', index=False, encoding='utf-8-sig')