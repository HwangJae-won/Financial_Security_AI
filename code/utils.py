import re
import random

def is_multiple_choice(question_text):
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2, option_count

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
    
# def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
#     """
#     - "답변:" 이후 텍스트만 추출
#     - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
#     - 주관식 문제면: 전체 텍스트 그대로 반환
#     - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
#     - 모델의 출력에서 프롬프트가 반복되는 부분을 제거
#     """
#     # 1. generated_text의 시작 부분에서 prompt를 찾아 제거
#     if generated_text.startswith(prompt):
#         text = generated_text[len(prompt):].strip()
#     else:
#         text = generated_text.strip()

#     # 2. "답변:" 기준으로 텍스트 분리 (기존 로직 유지)
#     if "답변:" in text:
#         text = text.split("답변:")[-1].strip()
    
#     # 3. 공백 또는 빈 문자열일 경우 기본값 지정
#     if not text:
#         text = "미응답"

#     # 4. 객관식 여부 판단
#     is_mc, option_count = is_multiple_choice(original_question)

#     if is_mc:
#         # 숫자만 추출 (기존 로직 유지)
#         match = re.match(r"\D*([1-9][0-9]?)", text)
#         if match:
#             num = int(match.group(1))
#             if 1 <= num <= option_count:   # 선택지 이상이면 무효 처리
#                 return str(num)
#             else:
#                 return '0'
#         else:
#             return '0'
#     else:
#         # 주관식 답변은 그대로 반환 (기존 로직 유지)
#         return text

import re

STOP_MARKERS = [
    "\n답변:", "\n---", "\n***",
    "\n###", "\n[참고", "\n참고자료",
    "\n질문:", "\n[예시",
    "\n지시", "\nInstruction", "\nReferences", "\nAnswer:"
]


def _first_chunk_after_answer(text: str) -> str:
    """'답변:' 이후 첫 블록 전체 반환"""
    idx = text.find("답변:")
    if idx != -1:
        seg = text[idx + len("답변:"):].lstrip()
    else:
        seg = text.lstrip()

    # '다음 답변:'이 다시 나오면 거기서 끊기
    nxt = seg.find("답변:")
    if nxt != -1:
        seg = seg[:nxt]

    # 다른 STOP_MARKERS는 최소한만 적용
    cut = len(seg)
    for m in STOP_MARKERS:
        k = seg.find(m)
        if k != -1:
            cut = min(cut, k)
    return seg[:cut].strip()

def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
    """
    - '답변:' 이후 첫 청크만 채택
    - 객관식: 첫 청크에서의 첫 유효 숫자만 사용 (범위 밖이면 '0')
    - 주관식: 첫 청크 그대로 반환
    - 비어 있으면 '미응답'
    - 프롬프트 에코 제거
    """
    # 1) 프롬프트 에코 제거
    if generated_text.startswith(prompt):
        text = generated_text[len(prompt):].strip()
    else:
        text = generated_text.strip()

    # 2) 첫 번째 답변 청크만 추출
    first = _first_chunk_after_answer(text)
    if not first:
        first = "미응답"

    # 3) 객관식/주관식 분기
    is_mc, option_count = is_multiple_choice(original_question)

    if is_mc:
        # 첫 청크에서 '첫 유효 숫자'만 채택
        m = re.search(r"(^|\D)([1-9][0-9]?)(\D|$)", first)
        if m:
            num = int(m.group(2))
            return str(num) if 1 <= num <= option_count else "0"
        return "0"
    else:
        # 주관식: 첫 청크만 반환 (뒤 반복/참고/지시 등은 이미 제거됨)
        return first

