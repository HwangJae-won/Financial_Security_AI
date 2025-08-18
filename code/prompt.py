from typing import List, Optional, Tuple
from utils import is_multiple_choice, extract_question_and_choices

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

# ===== 2) V2 (few-shot 1개 고정) =====
def make_prompt_v2(text: str) -> str:
    is_mc, _ = is_multiple_choice(text)

    # 네가 쓰던 예시 1개만 고정 삽입
    fewshot_block_mc = (
        "[예시]\n"
        "질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?\n"
        "선택지:\n"
        "1 데이터 암호화\n"
        "2 서버 인증\n"
        "3 클라이언트 인증\n"
        "4 무결성 확인\n"
        "[근거] SSL/TLS는 데이터 암호화, 서버 인증, 무결성 확인을 제공하지만, 클라이언트 인증은 필수 기능이 아니다.\n"
        "답변: 3\n\n"
    )
    fewshot_block = (
        "[예시]\n"
        "질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.\n"
        "답변: 윈도우 서버를 대상으로 한 해킹 공격을 방어하기 위해서는 보안 업데이트를 주기적으로 적용하고, "
        "불필요한 서비스 포트를 차단해야 합니다. 또한, 강력한 계정 정책을 사용하고, IDS/IPS와 같은 보안 시스템을 "
        "도입하여 실시간으로 공격을 탐지하고 차단하는 것이 중요합니다.\n\n"
    )

    if is_mc:
        q, opts = extract_question_and_choices(text)
        return (
            "당신은 한국인 금융보안 전문가입니다.\n"
            "아래 질문에 대해 [예시]를 참고하여 신중하게 생각한 후 답변을 도출하세요.\n"
            "답변에는 번호만 있어야 하며, 설명은 생략합니다.\n\n"
            f"{fewshot_block_mc}"
            f"질문: {q}\n"
            f"선택지:\n{chr(10).join(opts)}\n\n"
            "답변:"
        )
    else:
        return (
            "당신은 한국인 금융보안 전문가입니다.\n"
            "아래 질문에 대해 [예시]를 참고하여 신중하게 생각한 후 답변을 도출하세요.\n"
            "답변에는 핵심 키워드를 중심으로 2줄 이내로 작성하세요.\n\n"
            f"{fewshot_block}"
            f"질문: {text}\n\n"
            "답변:"
        )

# ===== 3) RAG (컨텍스트는 하단에만) =====
def make_prompt_rag(
    text: str,
    contexts: Optional[List[str]] = None,
    use_fewshot: bool = False  # ← 예시 1개 사용할지 여부
) -> str:
    """
    - 컨텍스트는 하단 [참고자료] 블록으로만 추가(파서 안전).
    - use_fewshot=True면, 네가 쓰던 예시 1개를 상단에 추가.
    """
    # few-shot 1개 (고정)
    fewshot_block_mc = ""
    fewshot_block = ""
    
    if use_fewshot:
        fewshot_block_mc = (
        "[예시]\n"
        "질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?\n"
        "선택지:\n"
        "1 데이터 암호화\n"
        "2 서버 인증\n"
        "3 클라이언트 인증\n"
        "답변: 3\n\n"
        )
        fewshot_block = (
            "[예시]\n"
            "질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.\n"
            "답변: 윈도우 서버를 대상으로 한 해킹 공격을 방어하기 위해서는 보안 업데이트를 주기적으로 적용하고, "
            "불필요한 서비스 포트를 차단해야 합니다. 또한, 강력한 계정 정책을 사용하고, IDS/IPS와 같은 보안 시스템을 "
            "도입하여 실시간으로 공격을 탐지하고 차단하는 것이 중요합니다.\n\n"
        )

    # 컨텍스트 블록
    context_block = ""
    if contexts:
        context_block = "[참고자료]\n" + "\n\n---\n".join(contexts) + "\n\n"

    is_mc, _ = is_multiple_choice(text)

    role = (
        "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 한국어로 아래 지시를 따르세요.\n\n"
    )

    if is_mc:
        q, opts = extract_question_and_choices(text)
        return (
            role
            + "**지시:** [질문]에 대한 답변을 [참고자료]를 바탕으로 도출하세요.  "
              " [예시]와 같이 가장 적절한 **답변: <번호 1개만>** 출력하세요.\n\n"
            + fewshot_block_mc
            + context_block
            + f"[질문]: {q}\n선택지:\n{chr(10).join(opts)}\n\n"
            + "\n답변:"
        )
    else:
        return (
            role
            + "**지시:** [참고자료]의 원문을 그대로 복사하지 말고, 질문에 맞는 핵심 내용을 3문장 이내로 요약·재구성하세요. "
              "가능하면 [참고자료]에서 **정확한 용어 한두 개**를 포함하세요. 답변은 한국어로 작성하세요.\n\n"
            + fewshot_block
            + context_block
            + f"질문: {text}\n\n"
            + "답변:"
        )



# ===== 4) RAG EXAONE =====

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
                "- '근거: [참고자료] 속 근거 문장' 그대로 출력하세요."
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
                "- [참고자료]를 그대로 복사하지 말고 핵심만 3문장 이내로 요약·재구성하세요.\n"
                "- 가능하면 [참고자료]의 정확한 용어 한두 개를 포함하세요.\n"
                "- 출력은 반드시 한국어로, 형식은 '답변: <내용>'만 허용됩니다.\n"
                "- 생각의 과정 없이, 한 번만 답변하세요\n"
                "- 추가 문장, 근거 설명 출력 금지.\n\n"
                + (fewshot_block_gen if use_fewshot else "")
                + context_block
                + "### 질문\n"
                + f"{text}\n\n"
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