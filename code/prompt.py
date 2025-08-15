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


def make_prompt_rag_solar( text, contexts, use_fewshot = False):
    """
    SOLAR 모델에 맞춘 RAG 프롬프트.
    - 명확한 역할, 컨텍스트, 지시 구조를 사용.
    - 객관식/주관식에 따른 지시를 명확히 구분.
    - 주관식에 Chain-of-Thought(CoT) 기법을 활용.
    """
    
    # Few-shot 예시 정의
    fewshot_examples = """
[예시]
질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?
선택지:
1. 데이터 암호화
2. 서버 인증
3. 클라이언트 인증
4. 무결성 확인
답변: 3

[예시]
질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.
답변: 윈도우 서버를 대상으로 한 해킹 공격을 방어하기 위해서는 보안 업데이트를 주기적으로 적용하고, 불필요한 서비스 포트를 차단해야 합니다. 또한, 강력한 계정 정책을 사용하고, IDS/IPS와 같은 보안 시스템을 도입하여 실시간으로 공격을 탐지하고 차단하는 것이 중요합니다.
"""
    fewshot_block_mc = ""
    fewshot_block_gen = ""
    if use_fewshot:
        # 객관식 예시
        fewshot_block_mc = fewshot_examples.split("[예시]")[1].strip()
        # 주관식 예시
        fewshot_block_gen = fewshot_examples.split("[예시]")[2].strip()

    # 역할 및 컨텍스트 블록
    role_persona = "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 한국어로 주어진 질문에 대해 다음 지시에 따라 가장 정확하고 신뢰성 있는 답변을 제공하세요."
    
    context_block = ""
    if contexts:
        context_block = "### 참고자료\n" + "\n\n---\n".join(contexts) + "\n\n"

    is_mc, _ = is_multiple_choice(text)

    # 프롬프트 구성
    if is_mc:
        q, opts = extract_question_and_choices(text)
        return (
            f"### 역할\n{role_persona}\n\n"
            + context_block
            + (f"### 예시\n{fewshot_block_mc}\n\n" if use_fewshot else "")
            + "### 지시\n"
            + "- [참고자료]에 근거하여 [질문]의 정답을 선택하세요.\n"
            + "- 오직 **정답 선택지의 번호**만 출력하세요. 다른 근거 설명이나 추가적인 텍스트는 절대 포함하지 마세요.\n\n"
            + "### 질문\n"
            + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
            + "### 답변\n"
        )
    else:
        # 주관식 질문에 Chain-of-Thought(CoT) 기법 적용
        return (
            f"### 역할\n{role_persona}\n\n"
            + context_block
            + (f"### 예시\n{fewshot_block_gen}\n\n" if use_fewshot else "")
            + "### 지시\n"
            + "- [참고자료]를 바탕으로, 먼저 정답의 핵심 키워드를 생각하세요. 그런 다음 그 키워드를 포함하여 3문장 이내로 답변을 작성하세요.\n"
            + "- 답변은 한국어로 작성하고, [참고자료]의 내용을 그대로 복사하지 마세요.\n\n"
            + "### 질문\n"
            + f"{text}\n\n"
            + "### 답변\n"
        )