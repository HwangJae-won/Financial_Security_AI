from utils import is_multiple_choice, extract_question_and_choices
import re
from typing import Optional, List, Tuple
# prompt.py 파일 전체 내용

from typing import List

# --- EXAONE-DEEP 모델 특성에 맞는 프롬프트 템플릿 ---

# # 객관식 문제용 템플릿: 정답 번호만 출력하도록 엄격하게 지시
# MC_PROMPT_TEMPLATE = """
# ### 지시사항
# 주어진 '문맥'과 '보기'를 참고하여, 질문에 대한 정답 번호만 답변하세요.
# 답변은 오직 숫자 1, 2, 3, ... 중 하나여야 하며, 다른 부가적인 설명은 절대 포함하지 마세요.
# 만약 정답을 찾을 수 없다면 '미응답'이라고만 답변하세요.

# ### 문맥
# {contexts}

# ### 질문
# {text}

# ### 답변
# 정답:"""

# # 주관식 문제용 템플릿: 핵심 내용을 요약하고 명확한 형식으로 출력하도록 지시
# SUB_PROMPT_TEMPLATE = """
# ### 지시사항
# 주어진 '문맥'을 참고하여, 질문에 대한 답변을 작성하세요.
# - 답변은 핵심 용어 2~3개를 포함하여 3문장 이내의 간결한 문장으로 요약하세요.
# - 답변은 반드시 '답변: <내용>' 형식으로만 출력하세요.
# - 만약 답변에 필요한 내용이 문맥에 포함되어 있지 않다면 '미응답'이라고만 답변하세요.
# - 생각의 과정 없이, 한 번만 답변하세요.

# ### 문맥
# {contexts}

# ### 질문
# {text}

# ### 답변:"""


# # Few-shot 예시 (모델의 성능을 향상시키는 중요한 요소)
# FEWSHOT_EXAMPLE = [
#     {
#         "question": "금융보안에 관한 법률 제12조에 따르면, 금융회사가 준수해야 할 사항은 무엇인가요?",
#         "context": "금융보안에 관한 법률 제12조: 금융회사는 안전한 전자금융거래를 위하여 정보통신기술부의 권고를 준수하여야 한다.",
#         "answer": "금융회사는 정보통신기술부의 권고를 준수하여야 합니다."
#     },
#     {
#         "question": "개인정보보호법 제12조에 따르면 개인정보는 어떻게 처리되어야 하는가?",
#         "context": "개인정보보호법 제12조: 개인정보는 법령에서 정한 목적 범위 내에서만 처리하여야 한다.",
#         "answer": "개인정보는 법령에서 정한 목적 범위 내에서만 처리되어야 합니다."
#     }
# ]

# def make_dynamic_rag_prompt(
#     question: str,
#     contexts: List[str],
#     is_multiple_choice: bool,
#     use_fewshot: bool = True
# ) -> str:
#     """
#     질문 유형에 따라 동적으로 RAG 프롬프트를 생성합니다.
#     """
    
#     context_str = "\n".join(contexts)
    
#     # 1. Few-shot 예시 추가
#     fewshot_str = ""
#     if use_fewshot:
#         fewshot_str = "\n".join([
#             f"### 질문\n{ex['question']}\n### 문맥\n{ex['context']}\n### 답변\n{ex['answer']}"
#             for ex in FEWSHOT_EXAMPLE
#         ]) + "\n\n"

#     # 2. 질문 유형에 맞는 템플릿 선택
#     prompt_template = MC_PROMPT_TEMPLATE if is_multiple_choice else SUB_PROMPT_TEMPLATE
    
#     # 3. 최종 프롬프트 생성
#     final_prompt = prompt_template.format(
#         contexts=context_str,
#         text=question
#     )

#     # Few-shot 예시를 지시사항 아래에 삽입
#     final_prompt = final_prompt.replace("### 문맥", fewshot_str + "\n### 문맥")

#     return final_prompt
# # --- 이전에 논의되었던 few-shot 예시 블록 ---
# fewshot_examples = """
# 질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?
# 선택지:
# 1. 데이터 암호화
# 2. 서버 인증
# 3. 클라이언트 인증
# 4. 무결성 확인
# 답변: 3

# 질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.
# 답변: 정기적 보안 업데이트, 불필요 포트 차단, 강력한 계정 정책 적용, IDS/IPS 도입을 통해 실시간 탐지·차단이 필요합니다.
# """


# def make_prompt_rag_exaone(
#     text: str,
#     contexts: Optional[List[str]] = None,
#     use_fewshot: bool = False  # 예시 1개 사용할지 여부
# ) -> str:
#     """
#     EXAONE-Deep-7.8B 등 EXAONE 계열에 맞춘 RAG 프롬프트.
#     - 컨텍스트는 하단 [참고자료] 블록으로만 추가(파서 안전).
#     - use_fewshot=True면, MC/주관식 각각 예시 1개를 상단에 추가.
#     """
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
#                 "- 출력은 반드시 **두 줄**.\n"
#                 "- '근거: [참고자료] 속 근거 문장' 그대로 출력하세요."
#                 "- `답변: <번호>`**만 허용됩니다.\n"
#                 "- 추가 문장, 불릿 출력 금지.\n\n"
#                 + (fewshot_block_mc if use_fewshot else "")
#                 + context_block
#                 + "### 질문\n"
#                 + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
#                 + "근거:\n"
#                 + "답변:"
#             )
#         else:
#             return (
#                 role
#                 + "### 지시\n"
#                 "- [질문]의 정답을 선택하세요.\n"
#                 "- 출력은 반드시 **한 줄**, 형식은 **`답변: <번호>`**만 허용됩니다.\n"
#                 "- 추가 문장, 근거 설명, 불릿 출력 금지.\n"
#                 "- 답변을 할 때 문장이 끊기지 않도록 끝까지 완결된 문장으로 작성하세요.\n\n"
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
                "- 추가 문장, 근거 설명, 불릿 출력 금지.\n"
                "- 답변을 할 때 문장이 끊기지 않도록 끝까지 완결된 문장으로 작성하세요.\n\n"
                + (fewshot_block_mc if use_fewshot else "")
                + "### 질문\n"
                + f"{q}\n\n선택지:\n{chr(10).join(opts)}\n\n"
                + "답변:"
            )
    else:
        # ★ 주관식 프롬프트 템플릿 변경 (여기서부터) ★
        if contexts:
            return (
                role
                + "### 지시\n"
                "- [참고자료]를 그대로 복사하지 말고 핵심만 요약·재구성하여 답변하세요.\n"  # 문장 수 제약 제거
                "- 가능하면 [참고자료]의 정확한 용어 한두 개를 포함하세요.\n"
                "- 출력은 반드시 한국어로, '답변:'으로 시작해야 합니다.\n"  # <내용> 태그 제거
                "- 생각의 과정 없이, 한 번만 답변하세요\n"
                "- 추가 문장, 근거 설명 출력 금지.\n\n"
                + (fewshot_block_gen if use_fewshot else "")
                + context_block
                + "### 질문\n"
                + f"{text}\n\n"
                + "답변:"  # <내용> 태그 제거
            )
        else:
            return (
                role
                + "### 지시\n"
                "- [질문]의 정답을 서술하세요.\n"
                "- 출력은 반드시 한국어로, '답변:'으로 시작해야 합니다.\n"  # <내용> 태그 제거
                "- 생각의 과정 없이, 한 번만 답변하세요\n"
                "- 추가 문장, 근거 설명 출력 금지.\n\n"
                + (fewshot_block_gen if use_fewshot else "")
                + context_block
                + "### 질문\n"
                + f"{text}\n\n"
                + "답변:"  # <내용> 태그 제거
            )
