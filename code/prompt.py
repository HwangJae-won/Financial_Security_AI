from typing import List, Optional
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
        "4 무결성 확인\n"
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

    if is_mc:
        q, opts = extract_question_and_choices(text)
        return (
            "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 한국어로 아래 지시를 따르세요.\n\n"
            "**지시:** 다음 질문은 객관식입니다. 참고자료를 바탕으로 **가장 적절한 정답 번호만** 출력하세요.\n\n"
            f"{fewshot_block_mc}"
            f"질문: {q}\n"
            "선택지:\n"
            f"{chr(10).join(opts)}\n\n"
            f"{context_block}"
            "답변:"
        )
    else:
        return (
            "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 한국어로 아래 지시를 따르세요.\n\n"
            "**지시:** 참고자료의 원문을 절대 그대로 복사하지 말고, 질문에 맞는 핵심 내용을 3문장 이내로 요약·재구성하세요. "
            "불필요한 세부사항은 생략하고, 핵심 키워드만 포함하세요.\n\n"
            f"{fewshot_block}"
            f"질문: {text}\n\n"
            f"{context_block}"
            "답변:"
        )

