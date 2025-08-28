import re
import random
import unicodedata


_NEG_PATTS = [
    r"옳지\s*않", r"맞지\s*않", r"아닌\s*것", r"아니(?:다|며?)",
    r"불가", r"금지", r"제외", r"해당하지\s*않",
    r"옳은\s*것이\s*아닌", r"타당하지\s*않",
    r"\bNOT\b", r"\bFALSE\b", r"\bincorrect\b", r"\bexcept\b"
]

def _nfkc_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()

def is_negated_question(q: str) -> bool:
    t = _nfkc_lower(q)
    return any(re.search(p, t) for p in _NEG_PATTS)



def is_multiple_choice(question_text):
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2, option_count

def extract_question_and_choices(full_text):
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options
    


def classify_question_type_tail(full_text: str) -> str:
    """
    extract_question_and_choices()로 뽑은 question의 '끝'만 보고 분류:
    '설명하세요.' / '기술하세요.' / '무엇인가요?' / '기타'
    """
    question, _ = extract_question_and_choices(full_text)
    q = (question or "").strip()

    # 공백-문장부호 정리, 닫는 따옴표/괄호 제거(문장부호는 유지)
    q = re.sub(r"\s+([?.!])$", r"\1", q)
    q = re.sub(r'["“”\'’\)\]\}〉》」』\s]+$', "", q)

    if re.search(r"설명하세요\.$", q):
        return "설명하세요."
    if re.search(r"기술하세요\.$", q):
        return "기술하세요."
    if re.search(r"무엇인가요\?$", q):
        return "무엇인가요?"
    return "기타"


STOP_MARKERS = [
    "\n---", "\n***",
    "\n###", "\n### 최종답변", "\n[참고", "\n참고자료",
    "\n질문:", "\n[예시",
    "\n지시", "\nInstruction", "\nReferences", "\nAnswer:"
]

_HANGUL = re.compile(r"[가-힣]")
# 줄 시작(^) 또는 개행 뒤(\n) 등장하는 '답변:'도 경계로 사용
_BOUNDARY_RE = re.compile(r"(^|\n)답변:|" + "|".join(map(re.escape, STOP_MARKERS)), re.MULTILINE)

def _has_korean(s: str) -> bool:
    return _HANGUL.search(s) is not None

def _is_numeric_short(s: str) -> bool:
    s = s.strip()
    return s.isdigit() and 1 <= len(s) <= 2  # 1~99 같은 객관식 단답 허용

def _split_by_boundaries(text: str):
    """STOP_MARKERS 또는 줄 시작의 '답변:'을 경계로 잘라, 마커 '사이'의 토막들을 순서대로 반환"""
    parts = []
    p = 0
    for m in _BOUNDARY_RE.finditer(text):
        if m.start() > p:
            parts.append(text[p:m.start()])
        p = m.end()  # 마커는 버리고 그 뒤부터 다음 토막 시작
    if p < len(text):
        parts.append(text[p:])
    # 공백 제거 + 빈 토막 제거
    return [seg.strip() for seg in parts if seg.strip()]

def _first_chunk_after_answer(text: str, prompt: str = "") -> str:
    """
    1) 프롬프트 에코 제거
    2) STOP_MARKERS 또는 줄 시작의 '답변:' 기준으로 토막
    3) 앞에서부터 유효성(한국어 포함 or 숫자 단답) 검사, 첫 유효 토막 즉시 반환
    4) 모두 무효면 첫 토막(있으면) 반환, 없으면 빈 문자열
    """
    t = text.lstrip()
    if prompt and t.startswith(prompt):
        t = t[len(prompt):].lstrip()

    chunks = _split_by_boundaries(t)
    for seg in chunks:
        if _has_korean(seg) or _is_numeric_short(seg):
            return seg
    return chunks[0] if chunks else ""


def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
    """
    - '답변:' 이후 첫 청크만 채택
    - 객관식: 첫 청크에서의 첫 유효 숫자만 사용 (범위 밖이면 '0')
    - 주관식: 첫 청크 그대로 반환
    - 비어 있으면 '미응답'
    - 프롬프트 에코 제거
    """
    # 1) 프롬프트 에코 제거
    if generated_text.startswith(prompt):
        text = generated_text[len(prompt):].strip()
    else:
        text = generated_text.strip()

    # 2) 첫 번째 답변 청크만 추출
    first = _first_chunk_after_answer(text)
    if not first:
        first = "미응답"

    # 3) 객관식/주관식 분기
    is_mc, option_count = is_multiple_choice(original_question)

    if is_mc:
        # 첫 청크에서 '첫 유효 숫자'만 채택
        m = re.search(r"(^|\D)([1-9][0-9]?)(\D|$)", first)
        if m:
            num = int(m.group(2))
            return str(num) if 1 <= num <= option_count else "0"
        return "0"
    else:
        # 주관식: 첫 청크만 반환 (뒤 반복/참고/지시 등은 이미 제거됨)
        return first

