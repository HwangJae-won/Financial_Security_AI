from typing import List, Optional, Tuple
from utils import is_multiple_choice, extract_question_and_choices, is_negated_question

# ===== 1) Baseline =====

def make_prompt_baseline(text: str) -> str:
    is_mc, _ = is_multiple_choice(text)
    if is_mc:
        q, opts = extract_question_and_choices(text)
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
            f"질문: {q}\n"
            f"선택지:\n{chr(10).join(opts)}\n\n"
            "답변:"
        )
    else:
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )

# ===== 2) RAG EXAONE =====

def make_prompt_rag_exaone(
    text: str,
    contexts: Optional[List[str]] = None,
    use_fewshot: bool = False  # 예시 1개 사용할지 여부
) -> str:
    """
    EXAONE-Deep-7.8B 등 EXAONE 계열에 맞춘 RAG 프롬프트.
    - 컨텍스트는 하단 [참고자료] 블록으로만 추가(파서 안전).
    - use_fewshot=True면, MC/주관식 각각 예시 1개를 상단에 추가.
    """
    # few-shot (간결·형식 고정)
    fewshot_block_mc = ""
    fewshot_block_gen = ""
    if use_fewshot:
        fewshot_block_mc = (
            "[예시]\n"
            "질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?\n"
            "선택지:\n"
            "1 데이터 암호화\n"
            "2 서버 인증\n"
            "3 클라이언트 인증\n"
            "4 무결성 확인\n"
            "답변: 3\n\n"
        )
        fewshot_block_gen = (
            "[예시]\n"
            "질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.\n"
            "답변: 정기적 보안 업데이트, 불필요 포트 차단, 강력한 계정 정책 적용, IDS/IPS 도입을 통해 실시간 탐지·차단이 필요합니다.\n\n"
        )

    # 컨텍스트 블록
    context_block = ""
    if contexts:
        context_block = "[참고자료]\n" + "\n\n---\n".join(contexts) + "\n\n"

    is_mc, _ = is_multiple_choice(text)

    # 역할·지시: EXAONE는 지시 준수/형식 엄수에 강함. 근거 설명 출력 금지로 형식 안정화
    role = (
        "### 역할\n"
        "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다.\n\n"
    )

    if is_mc:
        q, opts = extract_question_and_choices(text)
        if contexts:
            return (
                role
                + "### 지시\n"
                "- [참고자료]에 근거하여 [질문]의 정답을 선택하세요.\n"
                "- 출력은 반드시 **두 줄**.\n"
                "- '근거: [참고자료] 속 근거 문장' 그대로 출력하세요.\n"
                "- `답변: <번호>`**만 허용됩니다.\n"
                "- 추가 문장, 불릿 출력 금지.\n\n"
                + (fewshot_block_mc if use_fewshot else "")
                + context_block
                + "### 질문\n"
                + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
                + "근거:\n"
                + "답변:"
            )
        else:
            return (
                role
                + "### 지시\n"
                "- [질문]의 정답을 선택하세요.\n"
                "- 출력은 반드시 **한 줄**, 형식은 **`답변: <번호>`**만 허용됩니다.\n"
                "- 추가 문장, 근거 설명, 불릿 출력 금지.\n\n"
                + (fewshot_block_mc if use_fewshot else "")
                + "### 질문\n"
                + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
                + "답변:"
            )
    else:
        if contexts:
            return (
                role
                + "### 지시\n"
                "- [질문]을 자세히 읽고 묻는 바에 빠짐없이 답변하세요.\n"
                "- 반드시 [참고자료]에 근거하여 답변을 서술하세요"
                "- 먼저 문제 해결을 위한 사고 과정을 작성하고, 그 후에 최종 답변을 출력하세요.\n"
                "- 사고 과정과 최종 답변은 명확히 구분하여 작성해야 합니다.\n"
                "- 출력은 반드시 한국어로, 형식은 아래 예시를 따르세요.\n\n"
                + (fewshot_block_gen if use_fewshot else "") # 기존 few-shot 유지
                + context_block
                + "### 질문\n"
                + f"{text}\n\n"
                + "### 사고 과정\n"
                + "### 최종 답변\n"
                + "답변:"
            )
        else:
            return (
                role
                + "### 지시\n"
                "- [질문]을 자세히 읽고, 묻는 바에 빠짐없이 답변하세요.\n"
                "- 먼저 문제 해결을 위한 사고 과정을 충분히 거친 뒤, 최종 답변을 출력하세요.\n"
                "- 답변은 핵심 키워드를 중심으로 서술하세요.\n"
                "- 출력은 반드시 한국어로, 형식은 아래 예시를 따르세요.\n\n"
                + (fewshot_block_gen if use_fewshot else "")
                + "### 질문\n"
                + f"{text}\n\n"
                + "### 최종 답변\n"
                + "답변:"
            )

# ===== 2.5) SUBJECTIVE ====

def make_prompt_subjective(
    text: str,
    contexts: Optional[List[str]] = None,
    use_fewshot: bool = False,  # 예시 1개 사용할지 여부
    question_type: Optional[str] = None
) -> str:

    # few-shot (간결·형식 고정)
    fewshot_block_mc = ""
    fewshot_block_gen = ""
    if use_fewshot:
        fewshot_block_gen = (
            "[예시]\n"
            "질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.\n"
            "답변: 정기적 보안 업데이트, 불필요 포트 차단, 강력한 계정 정책 적용, IDS/IPS 도입을 통해 실시간 탐지·차단이 필요합니다.\n\n"
        )

    # 컨텍스트 블록
    context_block = ""
    if contexts:
        context_block = "[참고자료]\n" + "\n\n---\n".join(contexts) + "\n\n"


    # 역할·지시: EXAONE는 지시 준수/형식 엄수에 강함. 근거 설명 출력 금지로 형식 안정화
    role = (
        "### 역할\n"
        "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다.\n\n"
    )

    if question_type == '기술하세요.':
        if contexts:
            return (
                role
                + "### 지시\n"
                "- [질문]을 자세히 읽고 묻는 바에 빠짐없이 답변하세요.\n"
                "- 반드시 [참고자료]에 근거하여 답변을 서술하세요"
                "- 먼저 문제 해결을 위한 사고 과정을 작성하고, 그 후에 최종 답변을 출력하세요.\n"
                "- 사고 과정과 최종 답변은 명확히 구분하여 작성해야 합니다.\n"
                "- 출력은 반드시 한국어로, 형식은 아래 예시를 따르세요.\n\n"
                + (fewshot_block_gen if use_fewshot else "") # 기존 few-shot 유지
                + context_block
                + "### 질문\n"
                + f"{text}\n\n"
                + "### 사고 과정\n"
                + "### 최종 답변\n"
                + "답변:"
            )
        else:
            return (
                role
                + "### 지시\n"
                "- [질문]을 자세히 읽고 묻는 바에 빠짐없이 답변하세요.\n"
                "- 먼저 문제 해결을 위한 사고 과정을 충분히 거친 뒤, 최종 답변을 출력하세요.\n"
                "- 답변은 핵심 키워드를 중심으로 서술하세요.\n"
                "- 출력은 반드시 한국어로, 형식은 아래 예시를 따르세요.\n\n"
                + (fewshot_block_gen if use_fewshot else "")
                + "### 질문\n"
                + f"{text}\n\n"
                + "### 최종 답변\n"
                + "답변:"
            )
    else:
        if contexts:
            return (
                role
                + "### 지시\n"
                "- [질문]을 자세히 읽고 묻는 바에 빠짐없이 답변하세요.\n"
                "- 반드시 [참고자료]에 근거하여 답변을 서술하세요"
                "- 먼저 문제 해결을 위한 사고 과정을 작성하고, 그 후에 최종 답변을 출력하세요.\n"
                "- 사고 과정과 최종 답변은 명확히 구분하여 작성해야 합니다.\n"
                "- 출력은 반드시 한국어로, 형식은 아래 예시를 따르세요.\n\n"
                + (fewshot_block_gen if use_fewshot else "") # 기존 few-shot 유지
                + context_block
                + "### 질문\n"
                + f"{text}\n\n"
                + "### 사고 과정\n"
                + "### 최종 답변\n"
                + "답변:"
            )
        else:
            return (
                role
                + "### 지시\n"
                "- [질문]의 정답을 서술하세요.\n"
                "- 출력은 반드시 한국어로, 형식은 '답변: <내용>'만 허용됩니다.\n"
                "- 생각의 과정 없이, 한 번만 답변하세요\n"
                "- 추가 문장, 근거 설명 출력 금지.\n\n"
                + (fewshot_block_gen if use_fewshot else "")
                + "### 질문\n"
                + f"{text}\n\n"
                + "답변:"
            )


# ===== 3) RECHECK =====

def make_prompt_recheck(
    text: str,
    contexts: Optional[List[str]] = None,
) -> str:
    """
    RECHECK 프롬프트: 문제(text), 초안답(first_answer), 참고자료(contexts)를 주고
    '최종 형식만' 출력하도록 강제.
    """
    context_block = ""
    if contexts:
        context_block = "[참고자료]\n" + "\n\n---\n".join(contexts) + "\n\n"
    
    is_mc, _ = is_multiple_choice(text)
    role = (
        "### 역할\n"
        "당신은 금융보안 전문가입니다. 아래 문제와 참고자료를 보고 올바른 답을 결정하세요.\n\n"
    )
    
    if is_mc:
        q, opts = extract_question_and_choices(text)
        n = len(opts)
        return (
            role
            + "### 지시\n"
            "- 반드시 [참고자료]에 근거하여 [질문]의 정답을 선택하세요.\n"
            "- [질문]의 정답을 선택지의 번호(정수)로만 결정하세요.\n"
            "- [참고자료]에서 선택지를 직접 뒷받침하는 **단 하나의 문장**을 그대로 인용하여 근거로 제시하세요.\n"
            "- 출력은 반드시 **두 줄**.\n"
            "  - `답변: <번호>`  (허용 범위: 0, 1, 2, 3, 4, 5)\n"
            "  - `근거: <[참고자료]의 원문 한 문장>`  (문장부호/한글 괄호/띄어쓰기 보존)\n"
            "- [참고자료]에서 **직접적 근거 문장**을 찾을 수 없으면\n"
            "- 추가 설명, 불릿, 공백 줄, 서술, 체인오브소트 출력 절대 금지.\n\n"
            + context_block
            + "### 질문\n"
            + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
            + "근거:\n"
            + "답변:"
        )
    else:
        return (
            role
            + "### 지시\n"
            "- [질문]의 정답을 서술하세요.\n"
            "- 출력은 반드시 한국어로, 형식은 '답변: <내용>'만 허용됩니다.\n"
            "- 생각의 과정 없이, 한 번만 답변하세요\n"
            "- 추가 문장, 근거 설명 출력 금지.\n\n"
            + (fewshot_block_gen if use_fewshot else "")
            + context_block
            + "### 질문\n"
            + f"{text}\n\n"
            + "답변:"
        )

        
    