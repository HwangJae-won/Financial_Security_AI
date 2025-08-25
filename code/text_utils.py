import os
import re
from typing import List
import pdfplumber

from config import CHUNK_SIZE, CHUNK_OVERLAP


def _ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_pdf_text(pdf_path: str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            texts.append(t)
    return "\n".join(texts)





def _clean_text(t: str) -> str:
    # 기본 정리
    t = t.replace("\u3000", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)

    # --- "삭제<날짜>"가 포함된 '모든 줄' 제거 ---
    # 예: "1. 삭제<2020. 2. 4.>", "제28조의6 삭제 <2023. 3. 14.>", "…삭제＜2021.1.1.＞…"
    # - (?m): 줄 단위 매칭
    # - .*삭제\s*[<＜][^>＞]+[>＞].*$ : 해당 줄에 '삭제<...>' 또는 '삭제＜...＞' 패턴이 있으면 그 줄 전체 삭제
    t = re.sub(r'(?m)^.*삭제\s*[<＜][^>＞]+[>＞].*$', '', t)

    # 연속 빈 줄 정리 (앞에서 줄을 지웠으니 마지막에 수행)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def _chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = _clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        # 문장 경계 맞추기(가볍게): 마지막 마침표/줄바꿈 기준으로 자르기
        if end < len(text):
            cut = max(chunk.rfind("\n"), chunk.rfind("."), chunk.rfind("다."), chunk.rfind("다\n"))
            if cut > int(chunk_size * 0.6):
                chunk = chunk[:cut+1]
                end = start + len(chunk)
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    # 중복/빈 제거
    uniq = []
    seen = set()
    for c in chunks:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


# --- 조 헤더 정규식: '제n조(…)' 또는 '제n조의m(…)' + 줄 시작 + 괄호 존재 보장 + '조제' 참조 제외 ---
# 전각 괄호(（ ）)까지 허용
ARTICLE_RE_STRICT = re.compile(
    r'(?m)^'                                  # 줄 시작
    r'(?P<header>' 
       r'제\s*\d+\s*조'                        # 제n조
       r'(?!\s*제)'                            # '조제…항' 참조는 제외
       r'(?:\s*의\s*\d+)?'                     # '의m' (제n조의m) 허용
    r')'
    r'(?=\s*[（(])'                            # 바로 괄호가 존재해야 함(lookahead)
    r'\s*[（(]'                                # 괄호 여는 기호 소모
    r'(?P<title>[^）)]*)'                      # 제목(비워둘 수도 있음)
    r'[）)]',                                  # 괄호 닫기
    re.UNICODE
)

# 폴백: 혹시 일부 문서에서 괄호가 누락된 헤더가 존재하는 경우 대비
ARTICLE_RE_FALLBACK = re.compile(
    r'(?m)^'
    r'(?P<header>제\s*\d+\s*조(?!\s*제)(?:\s*의\s*\d+)?)'
    r'(?:\s*[（(](?P<title>[^）)]*)[）)])?',    # 괄호가 없어도 허용
    re.UNICODE
)


def parse_korean_law_articles(raw_text: str):
    text = _clean_text(raw_text)

    # 헤더가 줄 맨 앞에 떨어지도록 약간 정규화 (PDF 추출 잡음 완화)
    # '제176조제3항' 같은 붙은 참조는 띄어쓰기 보정
    text = re.sub(r"(제\s*\d+\s*조)(\s*제\s*\d+\s*항)", r"\1 \2", text)

    # 1) 엄격 규칙으로 시도(제목 괄호 필수)
    matches = list(ARTICLE_RE_STRICT.finditer(text))
    if not matches:
        # 2) 괄호 없는 헤더가 섞인 문서 대응
        matches = list(ARTICLE_RE_FALLBACK.finditer(text))
        if not matches:
            return [{"article": "전체", "title": "", "text": text}]

    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        header = m.group("header")
        title = (m.groupdict().get("title") or "").strip()
        body = text[start:end].strip()

        # 헤더 행을 깔끔하게 앞줄로 정렬
        head_full = header + (f"({title})" if title else "")
        # 전각 괄호를 일반 괄호로 통일(보기도 좋고 후처리 쉬움)
        head_full_alt = header + (f"（{title}）" if title else "")
        body_norm = body
        # 헤더 라벨 넣기
        if head_full in body_norm:
            body_norm = body_norm.replace(head_full, head_full + "\n", 1)
        elif head_full_alt in body_norm:
            body_norm = body_norm.replace(head_full_alt, head_full + "\n", 1)

        articles.append({
            "article": header.replace(" ", ""),   # 예: "제11조", "제11조의2"
            "title": title,
            "text": body_norm.strip(),
        })
    return articles


# 항/호 마커 (다양한 표기 대응: ①②… / '1.' 등)
PARA_SPLIT_RE = re.compile(
    r'(?m)^(?=(?:[①-⑳]))'
)


def split_article_if_long(article_text: str, max_len: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    
    """
    조(條)가 너무 길면 항/호 표기(PARA_SPLIT_RE)를 기준으로 우선 분할,
    그래도 길면 안전하게 길이 기반 분할까지 적용.
    """
    text = article_text.strip()
    if len(text) <= max_len:
        return [text]

    # 1) 항/호 기준 1차 분할
    parts = []
    last = 0
    for m in PARA_SPLIT_RE.finditer(text):
        idx = m.start()
        if idx != last:
            parts.append(text[last:idx].strip())
        last = idx
    parts.append(text[last:].strip())
    parts = [p for p in parts if p]

    # 2) 각 파트가 너무 길면 길이 기반 2차 분할
    chunks = []
    for p in parts:
        if len(p) <= max_len:
            chunks.append(p)
        else:
            s = 0
            while s < len(p):
                e = min(len(p), s + max_len)
                chunk = p[s:e]
                # 문장 경계 보정(선택): 마침표/줄바꿈 근처
                cut = max(chunk.rfind("\n"), chunk.rfind("."), chunk.rfind("다."), chunk.rfind("다\n"))
                if cut > int(len(chunk) * 0.6) and e < len(p):
                    e = s + cut + 1
                    chunk = p[s:e]
                chunks.append(chunk.strip())
                s = max(e - overlap, e)
    return [c for c in chunks if c]
    

# 기존 _chunk_text를 사용하되, 법령 파일엔 '조 단위' 우선 적용
def chunk_law_text(raw_text: str, by_article: bool = True,
                   chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not by_article:
        return _chunk_text(raw_text, chunk_size, overlap)

    articles = parse_korean_law_articles(raw_text)
    chunks = []
    labels = []  # "제n조(제목) - 청크i" 같은 메타 라벨
    for a in articles:
        title = f"{a['article']}" + (f"({a['title']})" if a['title'] else "")
        subchunks = split_article_if_long(a["text"], max_len=chunk_size, overlap=overlap)
        for i, sc in enumerate(subchunks):
            chunks.append(sc)
            labels.append(f"{title}#{i+1}")
    return chunks, labels