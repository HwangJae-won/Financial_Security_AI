import re
import random

def is_multiple_choice(question_text):
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text):
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
def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
    - 주관식 문제면: 전체 텍스트 그대로 반환
    - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
    - 모델의 출력에서 프롬프트가 반복되는 부분을 제거
    """
    # 1. generated_text의 시작 부분에서 prompt를 찾아 제거
    if generated_text.startswith(prompt):
        text = generated_text[len(prompt):].strip()
    else:
        text = generated_text.strip()

    # 2. "답변:" 기준으로 텍스트 분리 (기존 로직 유지)
    if "답변:" in text:
        text = text.split("답변:")[-1].strip()
    
    # 3. 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 4. 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출 (기존 로직 유지)
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            return str(random.randint(1, 5))
    else:
        # 주관식 답변은 그대로 반환 (기존 로직 유지)
        return text
# def extract_answer_only(generated_text: str, original_question: str) -> str:
#     if "답변:" in generated_text:
#         text = generated_text.split("답변:")[-1].strip()
#     else:
#         text = generated_text.strip()
#     if not text:
#         return "미응답"
#     is_mc = is_multiple_choice(original_question)
#     if is_mc:
#         match = re.match(r"\D*([1-9][0-9]?)", text)
#         if match:
#             return match.group(1)
#         else:
#             # 숫자가 없으면 첫 번째 선택지를 반환
#             _, options = extract_question_and_choices(original_question)
#             return options[0] if options else "미응답"
#     else:
#         return text