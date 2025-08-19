from utils import is_multiple_choice, extract_question_and_choices
import re
from typing import Optional, List, Tuple
from utils import is_multiple_choice, extract_question_and_choices

# --- 이전에 논의되었던 few-shot 예시 블록 ---
fewshot_examples = """
질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?
선택지:
1. 데이터 암호화
2. 서버 인증
3. 클라이언트 인증
4. 무결성 확인
답변: 3

질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.
답변: 정기적 보안 업데이트, 불필요 포트 차단, 강력한 계정 정책 적용, IDS/IPS 도입을 통해 실시간 탐지·차단이 필요합니다.
"""

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
# def make_prompt_rag_exaone(
#     text: str,
#     contexts: Optional[List[str]] = None,
#     use_fewshot: bool = False
# ) -> str:
#     # few-shot (간결·형식 고정)
#     fewshot_block_mc = ""
#     fewshot_block_gen = ""
#     if use_fewshot:
#         fewshot_block_mc = (
#             "[예시]\n"
#             "질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?\n"
#             "선택지:\n"
#             "1 데이터 암호화\n"
#             "2 서버 인증\n"
#             "3 클라이언트 인증\n"
#             "4 무결성 확인\n"
#             "답변: 3\n\n"
#         )
#         fewshot_block_gen = (
#             "[예시]\n"
#             "질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.\n"
#             "답변: 정기적 보안 업데이트, 불필요 포트 차단, 강력한 계정 정책 적용, IDS/IPS 도입을 통해 실시간 탐지·차단이 필요합니다.\n\n"
#         )

#     # 컨텍스트 블록
#     context_block = ""
#     if contexts:
#         context_block = "[참고자료]\n" + "\n\n---\n".join(contexts) + "\n\n"

#     is_mc, _ = is_multiple_choice(text)

#     # 역할·지시: EXAONE는 지시 준수/형식 엄수에 강함. 근거 설명 출력 금지로 형식 안정화
#     role = (
#         "### 역할\n"
#         "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다.\n\n"
#     )

#     if is_mc:
#         q, opts = extract_question_and_choices(text)
#         if contexts:
#             return (
#                 role
#                 + "### 지시\n"
#                 "- [참고자료]에 근거하여 [질문]의 정답을 선택하세요.\n"
#                 "- 출력은 반드시 **한 줄**, 형식은 **`답변: <번호>`**만 허용됩니다.\n"
#                 "- 추가 문장, 근거 설명, 불릿 출력 금지.\n\n"
#                 + (fewshot_block_mc if use_fewshot else "")
#                 + context_block
#                 + "### 질문\n"
#                 + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
#                 + "답변:"
#             )
#         else:
#             return (
#                 role
#                 + "### 지시\n"
#                 "- [질문]의 정답을 선택하세요.\n"
#                 "- 출력은 반드시 **한 줄**, 형식은 **`답변: <번호>`**만 허용됩니다.\n"
#                 "- 추가 문장, 근거 설명, 불릿 출력 금지.\n\n"
#                 + (fewshot_block_mc if use_fewshot else "")
#                 + "### 질문\n"
#                 + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
#                 + "답변:"
#             )
#     else:
#         if contexts:
#             return (
#                 role
#                 + "### 지시\n"
#                 "- [참고자료]를 그대로 복사하지 말고 핵심만 3문장 이내로 요약·재구성하세요.\n"
#                 "- 가능하면 [참고자료]의 정확한 용어 한두 개를 포함하세요.\n"
#                 "- 출력은 반드시 한국어로, 형식은 '답변: <내용>'만 허용됩니다.\n"
#                 "- 생각의 과정 없이, 한 번만 답변하세요\n"
#                 "- 추가 문장, 근거 설명 출력 금지.\n\n"
#                 + (fewshot_block_gen if use_fewshot else "")
#                 + context_block
#                 + "### 질문\n"
#                 + f"{text}\n\n"
#                 + "답변:"
#             )
#         else:
#             return (
#                 role
#                 + "### 지시\n"
#                 "- [질문]의 정답을 서술하세요.\n"
#                 "- 출력은 반드시 한국어로, 형식은 '답변: <내용>'만 허용됩니다.\n"
#                 "- 생각의 과정 없이, 한 번만 답변하세요\n"
#                 "- 추가 문장, 근거 설명 출력 금지.\n\n"
#                 + (fewshot_block_gen if use_fewshot else "")
#                 + context_block
#                 + "### 질문\n"
#                 + f"{text}\n\n"
#                 + "답변:"
#             )
