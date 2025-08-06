from utils import is_multiple_choice, extract_question_and_choices

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

def make_prompt_auto(text):
    role_instruction = "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 한국어로 주어진 질문에 대해 다음 지시에 따라 답변하세요." # <-- 한국어 답변 지시 추가

    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            f"{role_instruction}\n\n"
            "**지시:** 가장 적절한 정답 선택지 번호만 출력하세요.\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        question_text = text
        prompt = (
            f"{role_instruction}\n\n"
            "**지시:** 핵심 키워드를 포함하여 정확하고 간략한 설명을 작성하세요.\n\n"
            f"질문: {question_text}\n\n"
            "답변:"
        )
    return prompt

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
        prompt = (
            f"{base_role_prompt}"
            "아래 주관식 질문에 대해 먼저 단계적으로 생각하고, 그 과정을 바탕으로 정확하고 간략한 설명을 작성하세요.\\n\\n"
            f"질문: {text}\\n\\n"
            "생각 과정:"
        )
    return prompt