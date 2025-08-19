import re

OPTION_PATTERN = re.compile(r"^\s*[1-9][0-9]?\s")

def is_multiple_choice(question_text):
    """
    객관식 여부와 선택지 개수 반환
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(OPTION_PATTERN.match(line)) for line in lines)
    return option_count >= 2, option_count

def extract_question_and_choices(full_text):
    """
    질문과 선택지 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []
    for line in lines:
        if OPTION_PATTERN.match(line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options

def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식: 정답 숫자만 추출 (실패 시 '0' 반환)
    - 주관식: 전체 텍스트 반환
    - 빈 응답은 '미응답' 반환
    """
    # 프롬프트 제거
    text = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
    # "답변:" 이후만 추출
    if "답변:" in text:
        text = text.split("답변:")[-1].strip()
    if not text:
        return "미응답"

    is_mc, option_count = is_multiple_choice(original_question)
    if is_mc:
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            num = int(match.group(1))
            return str(num) if 1 <= num <= option_count else '0'
        return '0'
    return text


def clean_markdown(text: str) -> str:
    """
    텍스트에서 볼드체, 이탤릭체 등 마크다운 특수문자를 제거합니다.
    """
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # **text** 제거
    text = re.sub(r'_(.*?)_', r'\1', text)     # _text_ 제거
    text = re.sub(r'~(.*?)~', r'\1', text)     # ~text~ 제거
    return text