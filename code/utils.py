import re
import random
import unicodedata
from typing import List, Tuple, Optional, Dict
def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
    """
    안전 추출 규칙(보강판)
    1) 프롬프트 에코 제거
    2) '### 최종 답변' 블록 우선 추출 → 같은 블록의 '답변:' 라인 뒤부터 사용
    3) 폴백: 전역에서 첫 '답변:' 블록
    4) 섹션/구분선/두번째 최종답변/질문/참고/근거에서 컷
    5) 객관식: 첫 유효 숫자(1~option_count), 주관식: 블록(단락) 그대로 반환
    """
    import re

    # --- 공용 STOP 토큰(look-behind 없음) ---
    _STOP_TOKENS = (
        r"\n\s*###\s*질문", r"\n\s*###\s*사고\s*과정", r"\n\s*###\s*역할", r"\n\s*###\s*지시",
        r"\n\s*\[참고자료\]", r"\n\s*근거\s*:", r"\n\s*---\s*(\n|$)", r"\n\s*###\s*최종\s*답변"
    )
    _STOP_RE = re.compile("|".join(_STOP_TOKENS), flags=re.IGNORECASE)

    def cut_at_stops(s: str) -> str:
        m = _STOP_RE.search(s)
        return s[:m.start()] if m else s

    def looks_like_question_block(s: str) -> bool:
        t = s.lstrip()
        return t.startswith("### 질문") or t.startswith("질문:")

    # --------------------------
    # 1) 프롬프트 에코 제거
    # --------------------------
    t = generated_text or ""
    if prompt and t.startswith(prompt):
        t = t[len(prompt):]
    t = t.lstrip()

    # --------------------------
    # 2) 최우선: '### 최종 답변' 섹션에서 추출
    # --------------------------
    body = ""
    m = re.search(r"(^|\n)\s*###\s*최종\s*답변[^\n]*\n", t, flags=re.IGNORECASE)
    if m:
        seg = t[m.end():]
        # 같은 블록 선두의 '답변:' 마커 제거(있으면)
        seg = re.sub(r"^\s*답변\s*:\s*", "", seg, count=1, flags=re.IGNORECASE)
        body = cut_at_stops(seg).strip()

    # --------------------------
    # 3) 폴백: 전역 '답변:' 첫 블록
    # --------------------------
    if not body:
        m2 = re.search(r"(^|\n)\s*답변\s*:\s*", t, flags=re.IGNORECASE)
        if m2:
            seg = t[m2.end():]
            body = cut_at_stops(seg).strip()

    # --------------------------
    # 4) 정리: 질문/헤더 라인 제거, '답변:' 에코 제거
    # --------------------------
    if looks_like_question_block(body):
        # 질문 블록이 잘못 들어온 경우 비움 → 다음 폴백으로
        body = ""

    if not body:
        # 마지막 폴백: 전체에서 STOP 전까지 블록 하나 가져오되 질문 라인은 버림
        tmp_lines = []
        for ln in t.splitlines():
            if re.match(r"^\s*(###\s*질문|질문\s*:)", ln, flags=re.IGNORECASE):
                continue
            tmp_lines.append(ln)
        body = cut_at_stops("\n".join(tmp_lines)).strip()

    # 남아있을 수 있는 선두 '답변:' 정리
    body = re.sub(r"^\s*답변\s*:\s*", "", body, flags=re.IGNORECASE).strip()
    # '(형식에 맞게)' 같은 라벨 제거
    body = re.sub(r"^\s*\(?\s*형식에\s*맞게\)?\s*", "", body, flags=re.IGNORECASE).strip()

    # --------------------------
    # 5) 객관식/주관식 분기
    # --------------------------
    is_mc, option_count = is_multiple_choice(original_question)
    if is_mc:
        m = re.search(r"(^|\D)([1-9][0-9]?)(\D|$)", body)
        if m:
            num = int(m.group(2))
            return str(num) if 1 <= num <= option_count else "0"
        return "0"

    # 주관식: "첫 문장만"이 아니라 블록(단락) 그대로 반환
    # 단, 본문 중간에 또 '답변:'이 있으면 그 이전만 사용
    body = re.split(r"(^|\n)\s*답변\s*:\s*", body, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # 본문 맨 앞에 질문 라인이 섞이면 제거
    body = re.sub(r"(^|\n)\s*(###\s*질문|질문\s*:).*$", "", body, flags=re.IGNORECASE).strip()

    return body or "미응답"
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


# STOP_MARKERS = [
#     "\n---", "\n***",
#     "\n###", "\n### 최종답변", "\n[참고", "\n참고자료",
#     "\n질문:", "\n[예시",
#     "\n지시", "\nInstruction", "\nReferences", "\nAnswer:"
# ]

# _HANGUL = re.compile(r"[가-힣]")
# # 줄 시작(^) 또는 개행 뒤(\n) 등장하는 '답변:'도 경계로 사용
# _BOUNDARY_RE = re.compile(r"(^|\n)답변:|" + "|".join(map(re.escape, STOP_MARKERS)), re.MULTILINE)

# def _has_korean(s: str) -> bool:
#     return _HANGUL.search(s) is not None

# def _is_numeric_short(s: str) -> bool:
#     s = s.strip()
#     return s.isdigit() and 1 <= len(s) <= 2  # 1~99 같은 객관식 단답 허용

# def _split_by_boundaries(text: str):
#     """STOP_MARKERS 또는 줄 시작의 '답변:'을 경계로 잘라, 마커 '사이'의 토막들을 순서대로 반환"""
#     parts = []
#     p = 0
#     for m in _BOUNDARY_RE.finditer(text):
#         if m.start() > p:
#             parts.append(text[p:m.start()])
#         p = m.end()  # 마커는 버리고 그 뒤부터 다음 토막 시작
#     if p < len(text):
#         parts.append(text[p:])
#     # 공백 제거 + 빈 토막 제거
#     return [seg.strip() for seg in parts if seg.strip()]

# def _first_chunk_after_answer(text: str, prompt: str = "") -> str:
#     """
#     1) 프롬프트 에코 제거
#     2) STOP_MARKERS 또는 줄 시작의 '답변:' 기준으로 토막
#     3) 앞에서부터 유효성(한국어 포함 or 숫자 단답) 검사, 첫 유효 토막 즉시 반환
#     4) 모두 무효면 첫 토막(있으면) 반환, 없으면 빈 문자열
#     """
#     t = text.lstrip()
#     if prompt and t.startswith(prompt):
#         t = t[len(prompt):].lstrip()

#     chunks = _split_by_boundaries(t)
#     for seg in chunks:
#         if _has_korean(seg) or _is_numeric_short(seg):
#             return seg
#     return chunks[0] if chunks else ""


# def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
#     """
#     - '답변:' 이후 첫 청크만 채택
#     - 객관식: 첫 청크에서의 첫 유효 숫자만 사용 (범위 밖이면 '0')
#     - 주관식: 첫 청크 그대로 반환
#     - 비어 있으면 '미응답'
#     - 프롬프트 에코 제거
#     """
#     # 1) 프롬프트 에코 제거
#     if generated_text.startswith(prompt):
#         text = generated_text[len(prompt):].strip()
#     else:
#         text = generated_text.strip()

#     # 2) 첫 번째 답변 청크만 추출
#     first = _first_chunk_after_answer(text)
#     if not first:
#         first = "미응답"

#     # 3) 객관식/주관식 분기
#     is_mc, option_count = is_multiple_choice(original_question)

#     if is_mc:
#         # 첫 청크에서 '첫 유효 숫자'만 채택
#         m = re.search(r"(^|\D)([1-9][0-9]?)(\D|$)", first)
#         if m:
#             num = int(m.group(2))
#             return str(num) if 1 <= num <= option_count else "0"
#         return "0"
#     else:
#         # 주관식: 첫 청크만 반환 (뒤 반복/참고/지시 등은 이미 제거됨)
#         return first
    
    
    
    
    #======
# _SENT_END_RE = re.compile(r"([.!?。]|다\.|요\.|니다\.)\s+")

# def _split_kor_sentences(text: str) -> list[str]:
#     # 구두점 뒤 공백을 마커로 치환 → 마커 기준 split
#     tmp = _SENT_END_RE.sub(r"\1§", text)
#     return [s.strip() for s in tmp.split("§") if s.strip()]

# _NEG_PATTS = [
#     r"옳지\s*않", r"맞지\s*않", r"아닌\s*것", r"아니(?:다|며?)",
#     r"불가", r"금지", r"제외", r"해당하지\s*않",
#     r"옳은\s*것이\s*아닌", r"타당하지\s*않",
#     r"\bNOT\b", r"\bFALSE\b", r"\bincorrect\b", r"\bexcept\b"
# ]

# def _nfkc_lower(s: str) -> str:
#     return unicodedata.normalize("NFKC", s).lower()

# def is_negated_question(q: str) -> bool:
#     t = _nfkc_lower(q)
#     return any(re.search(p, t) for p in _NEG_PATTS)



# def is_multiple_choice(question_text):
#     lines = question_text.strip().split("\n")
#     option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
#     return option_count >= 2, option_count

# def extract_question_and_choices(full_text):
#     lines = full_text.strip().split("\n")
#     q_lines = []
#     options = []
#     for line in lines:
#         if re.match(r"^\s*[1-9][0-9]?\s", line):
#             options.append(line.strip())
#         else:
#             q_lines.append(line.strip())
#     question = " ".join(q_lines)
#     return question, options
    


# STOP_MARKERS = [
#     "\n---", "\n***",
#     "\n###", "\n[참고", "\n참고자료",
#     "\n질문:", "\n[예시",
#     "\n지시", "\nInstruction", "\nReferences", "\nAnswer:"
# ]

# _HANGUL = re.compile(r"[가-힣]")
# # 줄 시작(^) 또는 개행 뒤(\n) 등장하는 '답변:'도 경계로 사용
# _BOUNDARY_RE = re.compile(r"(^|\n)답변:|" + "|".join(map(re.escape, STOP_MARKERS)), re.MULTILINE)

# def _has_korean(s: str) -> bool:
#     return _HANGUL.search(s) is not None

# def _is_numeric_short(s: str) -> bool:
#     s = s.strip()
#     return s.isdigit() and 1 <= len(s) <= 2  # 1~99 같은 객관식 단답 허용

# def _split_by_boundaries(text: str):
#     """STOP_MARKERS 또는 줄 시작의 '답변:'을 경계로 잘라, 마커 '사이'의 토막들을 순서대로 반환"""
#     parts = []
#     p = 0
#     for m in _BOUNDARY_RE.finditer(text):
#         if m.start() > p:
#             parts.append(text[p:m.start()])
#         p = m.end()  # 마커는 버리고 그 뒤부터 다음 토막 시작
#     if p < len(text):
#         parts.append(text[p:])
#     # 공백 제거 + 빈 토막 제거
#     return [seg.strip() for seg in parts if seg.strip()]

# def _first_chunk_after_answer(text: str, prompt: str = "") -> str:
#     """
#     1) 프롬프트 에코 제거
#     2) STOP_MARKERS 또는 줄 시작의 '답변:' 기준으로 토막
#     3) 앞에서부터 유효성(한국어 포함 or 숫자 단답) 검사, 첫 유효 토막 즉시 반환
#     4) 모두 무효면 첫 토막(있으면) 반환, 없으면 빈 문자열
#     """
#     t = text.lstrip()
#     if prompt and t.startswith(prompt):
#         t = t[len(prompt):].lstrip()

#     chunks = _split_by_boundaries(t)
#     for seg in chunks:
#         if _has_korean(seg) or _is_numeric_short(seg):
#             return seg
#     return chunks[0] if chunks else ""




# def _simple_tokens(s: str) -> set:
#     s = re.sub(r"[^0-9A-Za-z가-힣]+", " ", s)
#     toks = [t for t in s.split() if len(t) >= 2]
#     return set(toks)

# def _overlap(a: str, b: str) -> float:
#     ta, tb = _simple_tokens(a), _simple_tokens(b)
#     if not ta or not tb:
#         return 0.0
#     return len(ta & tb) / max(1, len(ta))

# def _cut_at_stops(s: str) -> str:
#     m = _STOP_RE.search(s)
#     return s[:m.start()] if m else s

# # def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
# #     """
# #     - '답변:' 이후 첫 청크만 채택
# #     - 객관식: 첫 청크에서의 첫 유효 숫자만 사용 (범위 밖이면 '0')
# #     - 주관식: 첫 청크 그대로 반환
# #     - 비어 있으면 '미응답'
# #     - 프롬프트 에코 제거
# #     """
# #     # 1) 프롬프트 에코 제거
# #     if generated_text.startswith(prompt):
# #         text = generated_text[len(prompt):].strip()
# #     else:
# #         text = generated_text.strip()

# #     # 2) 첫 번째 답변 청크만 추출
# #     first = _first_chunk_after_answer(text)
# #     if not first:
# #         first = "미응답"

# #     # 3) 객관식/주관식 분기
# #     is_mc, option_count = is_multiple_choice(original_question)

# #     if is_mc:
# #         # 첫 청크에서 '첫 유효 숫자'만 채택
# #         m = re.search(r"(^|\D)([1-9][0-9]?)(\D|$)", first)
# #         if m:
# #             num = int(m.group(2))
# #             return str(num) if 1 <= num <= option_count else "0"
# #         return "0"
# #     else:
# #         # 주관식: 첫 청크만 반환 (뒤 반복/참고/지시 등은 이미 제거됨)
# #         return first
# def extract_answer_only(generated_text: str, original_question: str, prompt: str) -> str:
#     """
#     안전 추출 규칙
#     1) 프롬프트 에코 제거
#     2) '### 최종 답변' 앵커 ▷ (가능하면) 같은 블록의 '답변:' 라인 이후만 슬라이스
#     3) 위가 비면 전역에서 '답변:' 마커들을 순회하며 첫 유효 청크 선택
#     4) STOP_MARKERS/섹션 헤더(### 질문/사고 과정/참고자료/근거/---) 이전까지만 채택
#     5) 객관식: 첫 유효 숫자(1~option_count)만. 주관식: 앞에서부터 최대 5문장(또는 불릿 최대 5개)
#     6) 질문/헤더가 섞여 들어오면 제거
#     """
#     import re
#     import unicodedata

#     # ---- 내부 유틸(프로젝트 전역에 있으면 그걸 사용) ----
#     try:
#         _stop_re = _STOP_RE
#     except NameError:
#         _STOP_TOKENS = (
#                 r"\n\s*###\s*질문", r"\n\s*질문\s*:", r"\n\s*###\s*사고\s*과정",
#                 r"\n\s*\[참고자료\]", r"\n\s*근거\s*:", r"\n\s*---\s*\n"
#             )
#         _stop_re = re.compile("|".join(_STOP_TOKENS), flags=re.IGNORECASE)

#     try:
#         sent_split = _split_kor_sentences
#     except NameError:
#         _SENT_END_RE = re.compile(r"([.!?。]|다\.|요\.|니다\.)\s+")
#         def sent_split(text: str) -> list[str]:
#             tmp = _SENT_END_RE.sub(r"\1§", text)
#             return [s.strip() for s in tmp.split("§") if s.strip()]

#     def has_korean(s: str) -> bool:
#         return re.search(r"[가-힣]", s) is not None

#     def looks_like_question(s: str) -> bool:
#         t = s.lstrip()
#         return t.startswith("### 질문") or t.startswith("질문:")

#     def cut_at_stops(s: str) -> str:
#         m = _stop_re.search(s)
#         return s[:m.start()] if m else s

#     def nfkc(s: str) -> str:
#         return unicodedata.normalize("NFKC", s)

#     def slice_after_anchors(text: str) -> str:
#         # '### 최종 답변' 섹션 우선
#         m = re.search(r"###\s*최종\s*답변", text, flags=re.IGNORECASE)
#         start = None
#         if m:
#             start = m.end()
#         if start is None:
#             m2 = re.search(r"(^|\n)\s*최종\s*답변\s*[:：]?\s*$", text, flags=re.IGNORECASE)
#             if m2:
#                 start = m2.end()
#         if start is None:
#             return ""
#         seg = text[start:].lstrip()
#         m3 = re.match(r"^\s*답변\s*:\s*", seg, flags=re.IGNORECASE)
#         if m3:
#             seg = seg[m3.end():]
#         return cut_at_stops(seg).strip()

#     def best_chunk_after_any_answer(text: str) -> str:
#         chunks = []
#         for m in re.finditer(r"(^|\n)\s*답변\s*:\s*", text, flags=re.IGNORECASE):
#             seg = cut_at_stops(text[m.end():]).strip()
#             if seg:
#                 chunks.append(seg)
#         is_mc, _ = is_multiple_choice(original_question)
#         for seg in chunks:
#             if is_mc:
#                 if re.search(r"(^|\D)([1-9][0-9]?)(\D|$)", seg):
#                     return seg
#             else:
#                 if has_korean(seg):
#                     return seg
#         return chunks[0] if chunks else ""

#     # 1) 프롬프트 에코 제거
#     t = generated_text or ""
#     if prompt and t.startswith(prompt):
#         t = t[len(prompt):]
#     t = t.lstrip()

#     # 2) '### 최종 답변' 섹션 우선
#     body = slice_after_anchors(t)

#     # 3) 폴백: 전역 '답변:' 마커
#     if not body or looks_like_question(body):
#         body = best_chunk_after_any_answer(t)

#     # 4) 헤더/질문 제거
#     if looks_like_question(body):
#         body = ""

#     # 5) 완전 폴백: 전체에서 최초 한국어 문장
#     if not body:
#         tmp = cut_at_stops(t)
#         sents = sent_split(tmp)
#         body = sents[0] if sents else (tmp.strip() or "미응답")

#     # --------------------------
#     # 6) 객관식/주관식 분기
#     # --------------------------
#     is_mc, option_count = is_multiple_choice(original_question)
#     if is_mc:
#         m = re.search(r"(^|\D)([1-9][0-9]?)(\D|$)", body)
#         if m:
#             num = int(m.group(2))
#             return str(num) if 1 <= num <= option_count else "0"
#         return "0"
#     else:
#         # 주관식: 앞에서부터 최대 5문장(또는 불릿 최대 5개) 채택
#         body = cut_at_stops(body).strip()
#         # '답변:' 에코 제거
#         body = re.sub(r"^\s*답변\s*:\s*", "", body, flags=re.IGNORECASE).strip()

#         # 6-1) 문장 기반 수집
#         MAX_SENTS = 5
#         sents = sent_split(body)
#         picked = []
#         for s in sents:
#             if looks_like_question(s):
#                 break
#             picked.append(s)
#             if len(picked) >= MAX_SENTS:
#                 break

#         # 6-2) 문장이 거의 없으면 불릿 라인 보조(최대 5개)
#         ans = " ".join(picked).strip()
#         if len(ans) < 8:
#             lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
#             bullets = [ln for ln in lines if re.match(r"^(\d+[\.\)]|[-•*])\s+", ln)]
#             if bullets:
#                 ans = " ".join(bullets[:5]).strip()

#         # 너무 짧으면 원문 폴백
#         if len(ans) < 2:
#             ans = body

#         return nfkc(ans).strip() or "미응답"