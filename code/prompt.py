from utils import is_multiple_choice, extract_question_and_choices

def make_prompt_auto(text, version="baseline"):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)

        if version == "v2":
            prompt = (
                "금융보안 평가 위원으로서, 아래 질문에 대해 **가장 적절한 선택지 번호 하나만 출력**하세요.\n"
                "답변에는 번호만 있어야 하며, 설명은 생략합니다.\n"
                "[예시]\n"
                "질문: SSL/TLS 프로토콜의 주요 기능이 아닌 것은?\n"
                "선택지:\n"
                "1. 데이터 암호화\n"
                "2. 서버 인증\n"
                "3. 클라이언트 인증\n"
                "4. 무결성 확인\n"
                "답변: 3\n\n"
                f"질문: {question}\n"
                f"선택지:\n{chr(10).join(options)}\n\n"
                "답변:"
            )

        else:  # baseline
            prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                "예시:\n"
                "질문: 윈도우 서버를 대상으로 한 해킹 공격 방어 방법을 설명하세요.\n"
                "답변: 윈도우 서버를 대상으로 한 해킹 공격을 방어하기 위해서는 보안 업데이트를 주기적으로 적용하고, "
                "불필요한 서비스 포트를 차단해야 합니다. 또한, 강력한 계정 정책을 사용하고, IDS/IPS와 같은 보안 시스템을 "
                "도입하여 실시간으로 공격을 탐지하고 차단하는 것이 중요합니다.\n\n"
                f"질문: {question}\n"
                f"선택지:\n{chr(10).join(options)}\n\n"
                "답변:"
            )
            
    else:
        if version == "v2":
            prompt = (
                "금융보안 컨설턴트로서 고객에게 설명하듯, 아래 질문에 대해 핵심 키워드를 중심으로 2줄 이내로 작성하세요.\n\n"
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