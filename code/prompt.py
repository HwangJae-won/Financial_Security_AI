from utils import is_multiple_choice, extract_question_and_choices



# def make_prompt_auto(text, tokenizer, version="baseline"):
#     """
#     반환값:
#       rendered_prompt: chat 템플릿이 적용된 최종 문자열
#       gen_kwargs: pipe(...)에 그대로 넘길 추천 generation 인자
#     """
#     def render(messages):
#         return tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,  # assistant 턴을 열어줌
#         )

#     # ===== 객관식 =====
#     if is_multiple_choice(text):
#         question, options = extract_question_and_choices(text)

#         if version == "v2":
#             system = "너는 한국인 금융보안 평가위원이다. 정답의 '번호'만 출력한다. 설명/문장/기호 금지."
#             user = (
#                 "아래 문제에 대해 정답 번호 하나만 출력하세요.\n"
#                 "예시: 질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?\n"
#                 "선택지:\n1. 데이터 암호화\n2. 서버 인증\n3. 클라이언트 인증\n4. 무결성 확인\n"
#                 "답변: 3\n\n"
#                 f"질문: {question}\n선택지:\n" + "\n".join(options) + "\n정답:"
#             )
            
#         else:  # baseline
#             system = "너는 금융보안 전문가다. 정답의 번호만 출력한다."
#             user = f"질문: {question}\n선택지:\n" + "\n".join(options) + "\n정답:"

#         messages = [
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ]
#         rendered_prompt = render(messages)

#         gen_kwargs = {
#             "max_new_tokens": 3,          # 숫자 하나면 충분
#             "do_sample": False,           # 객관식은 결정적으로
#             "eos_token_id": tokenizer.eos_token_id,
#             "no_repeat_ngram_size": 3,    # 반복 억제
#             "repetition_penalty": 1.15,
#             "return_full_text": False,
#         }
#         return rendered_prompt, gen_kwargs

#     # ===== 주관식 =====
#     else:
#         if version == "v2":
#             system = "너는 한국인 금융보안 컨설턴트다. 답변은 2줄 이내로 핵심만 한국어로 작성한다."
#             user = (
#                 "아래 질문에 대해 고객에게 설명하듯 핵심 키워드 중심으로 2줄 이내로 작성하세요.\n\n"
#                 f"질문: {text}\n답변:"
#             )
#         else:  # baseline
#             system = "너는 금융보안 전문가다. 간결하고 정확하게 한국어로 답한다."
#             user = f"질문: {text}\n답변:"

#         messages = [
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ]
#         rendered_prompt = render(messages)

#         gen_kwargs = {
#             "max_new_tokens": 128,
#             "temperature": 0.2,           # 정확성 위주
#             "top_p": 0.9,
#             "eos_token_id": tokenizer.eos_token_id,
#             "no_repeat_ngram_size": 3,
#             "repetition_penalty": 1.05,
#             "return_full_text": False,
#         }
#         return rendered_prompt, gen_kwargs


def make_prompt_auto(text, version="baseline"):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)

        if version == "v2":
            prompt = (
                "당신은 한국인 금융보안 전문가입니다.\n"
                "아래 질문에 대해 [예시]를 참고하여 신중하게 생각한 후 답변을 도출하세요.\n"
                "답변에는 번호만 있어야 하며, 설명은 생략합니다.\n\n"
                "[예시]\n"
                "질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?\n"
                "선택지:\n"
                "1 데이터 암호화\n"
                "2 서버 인증\n"
                "3 클라이언트 인증\n"
                "4 무결성 확인\n"
                "답변: 3\n\n"
                f"질문: {question}\n"
                f"선택지:\n{chr(10).join(options)}\n\n"
                "답변:"
            )
        elif version == "v1":
            prompt = (
                "당신은 한국인 금융보안 전문가입니다.\n"
                "아래 질문에 대해 한국어로 신중하게 생각한 후 답변을 도출하세요.\n\n"
                #"답변에는 정답 번호만 있어야 하며, 설명은 생략합니다.\n\n"
                f"질문: {question}\n"
                f"선택지:\n{chr(10).join(options)}\n\n"
                "답변:"
            )
        else:  # baseline
            prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                f"선택지:\n{chr(10).join(options)}\n\n"
                "답변:"
            )
            
    else:
        if version == "v2":
            prompt = (
                "당신은 한국인 금융보안 전문가입니다.\n"
                "아래 질문에 대해 [예시]를 참고하여 신중하게 생각한 후 답변을 도출하세요.\n"
                "답변에는 핵심 키워드를 중심으로 2줄 이내로 작성하세요.\n\n"
                "[예시]\n"
                "질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.\n"
                "답변: 윈도우 서버를 대상으로 한 해킹 공격을 방어하기 위해서는 보안 업데이트를 주기적으로 적용하고, "
                "불필요한 서비스 포트를 차단해야 합니다. 또한, 강력한 계정 정책을 사용하고, IDS/IPS와 같은 보안 시스템을 "
                "도입하여 실시간으로 공격을 탐지하고 차단하는 것이 중요합니다.\n\n"
                f"질문: {text}\n\n"
                "답변:"
            )
        elif version == "v1":
            prompt = (
                "당신은 한국인 금융보안 전문가입니다.\n"
                "아래 질문에 대해 한국어로 신중하게 생각한 후 답변을 도출하세요.\n"
                "답변은 핵심 키워드를 중심으로 2줄 이내로 **한국어로** 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
            )
        else:  # baseline
            prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
            )

    return prompt

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
