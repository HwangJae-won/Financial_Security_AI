import os
import re
import pdfplumber
from typing import List, Tuple

# --- 정규 표현식: 법령의 조(條)와 항/호(項/號)를 파싱하는 핵심 패턴 ---

# 조 헤더 정규식: '제n조(…)' 또는 '제n조의m(…)' + 줄 시작 + 괄호 존재 보장 + '조제' 참조 제외
# 전각 괄호(（ ）)까지 허용
ARTICLE_RE_STRICT = re.compile(
    r'(?m)' +  # 줄 시작 앵커 제거 (`^`를 제거했습니다)
    r'(?P<header>' +
        r'제\s*\d+\s*조' +
        r'(?!\s*제)' +
        r'(?:\s*의\s*\d+)?' +
    r')' +
    r'(?=\s*[（(])' +
    r'\s*[（(]' +
    r'(?P<title>[^）)]*)' +
    r'[）)]',
    re.UNICODE
)

# 폴백: 혹시 일부 문서에서 괄호가 누락된 헤더가 존재하는 경우 대비
ARTICLE_RE_FALLBACK = re.compile(
    r'(?m)' +  # 줄 시작 앵커 제거 (`^`를 제거했습니다)
    r'(?P<header>제\s*\d+\s*조(?!\s*제)(?:\s*의\s*\d+)?)' +
    r'(?:\s*[（(](?P<title>[^）)]*)[）)])?',
    re.UNICODE
)

# 항/호 마커 (다양한 표기 대응: ①②… / '1항' / '1.' / '가.' / 괄호 숫자 등)
PARA_SPLIT_RE = re.compile(
    r'(?m)^(?=(?:[①-⑳]|[0-9]+\.?\s*항|[0-9]+\)|[가-하]\.|[ㄱ-ㅎ]\)|\([0-9]+\)|\([가-하]\)))'
)

# --- 텍스트 로딩 및 정제 ---

def load_pdf_text(pdf_path: str) -> str:
    """
    PDF 파일에서 모든 페이지의 텍스트를 추출
    """
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            texts.append(t)
    return "\n".join(texts)

def _clean_text(t: str) -> str:
    """
    텍스트에서 불필요한 공백, 특수문자 등을 정제하고, 법령 특유의 패턴을 제거
    """
    t = t.replace('\xa0', ' ')
    t = t.replace("\u3000", " ").strip()
    
    # --- "삭제<날짜>"가 포함된 '모든 줄' 제거 (가장 중요한 추가 로직) ---
    t = re.sub(r'(?m)^.*삭제\s*[<＜][^>＞]+[>＞].*$', '', t)
    
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    return t

# --- 법령 구조 기반 분할 ---

def parse_korean_law_articles(raw_text: str) -> List[dict]:
    """
    한국어 법령 텍스트를 조별로 파싱하고 정규화
    """
    text = _clean_text(raw_text)
    
    # '제176조제3항' 같은 붙은 참조는 띄어쓰기 보정
    text = re.sub(r"(제\s*\d+\s*조)(\s*제\s*\d+\s*항)", r"\1 \2", text)
    
    # 1) 엄격 규칙으로 시도(제목 괄호 필수)
    matches = list(ARTICLE_RE_STRICT.finditer(text))
    if not matches:
        # 2) 괄호 없는 헤더가 섞인 문서 대응
        matches = list(ARTICLE_RE_FALLBACK.finditer(text))
        if not matches:
            print("⚠️ 경고: '제n조' 헤더 패턴을 찾을 수 없습니다.")
            return [{"article": "전체", "title": "", "text": text}]
    
    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        header = m.group("header")
        title = (m.groupdict().get("title") or "").strip()
        body = text[start:end].strip()

        # 헤더 행을 깔끔하게 앞줄로 정렬 및 괄호 통일
        head_full = header.replace(" ", "") + (f"({title})" if title else "")
        head_full_alt = header + (f"（{title}）" if title else "")
        
        body_norm = body
        if head_full in body_norm:
            body_norm = body_norm.replace(head_full, head_full + "\n", 1)
        elif head_full_alt in body_norm:
            body_norm = body_norm.replace(head_full_alt, head_full + "\n", 1)

        articles.append({
            "article": header.replace(" ", ""),
            "title": title,
            "text": body_norm.strip(),
        })
    return articles

def split_article_if_long(
    article_text: str,
    max_len: int,
    overlap: int
) -> List[str]:
    """
    조(條)가 너무 길면 항/호 표기(PARA_SPLIT_RE)를 기준으로 우선 분할하고,
    그래도 길면 안전하게 길이 기반 분할까지 적용합니다.
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
            start = 0
            while start < len(p):
                end = min(len(p), start + max_len)
                chunk = p[start:end]
                # 문장 경계 보정: 마침표/줄바꿈 근처
                cut = max(
                    chunk.rfind("\n"),
                    chunk.rfind("."),
                    chunk.rfind("다."),
                    chunk.rfind("다\n")
                )
                if cut > int(len(chunk) * 0.6) and end < len(p):
                    end = start + cut + 1
                    chunk = p[start:end]
                
                chunks.append(chunk.strip())
                start = max(end - overlap, end)
    
    # 3) 중복/빈 청크 제거
    seen = set()
    uniq_chunks = []
    for c in chunks:
        if c and c not in seen:
            uniq_chunks.append(c)
            seen.add(c)

    return uniq_chunks

def chunk_law_text(
    raw_text: str,
    by_article: bool = True,
    chunk_size: int = 700,
    overlap: int = 50
) -> Tuple[List[str], List[str]]:
    """
    법령 텍스트를 구조 기반으로 분할하고, 메타데이터 레이블을 반환합니다.
    """
    articles = parse_korean_law_articles(raw_text)
    chunks = []
    labels = []  # '법률명/제n조(제목) - 청크i' 같은 메타 라벨
    for a in articles:
        subchunks = split_article_if_long(
            a["text"], 
            max_len=chunk_size, 
            overlap=overlap
        )
        for i, sc in enumerate(subchunks):
            chunks.append(sc)
            title = f"({a['title']})" if a['title'] else ""
            labels.append(f"{a['article']}{title}#Part{i+1}")
            
    return chunks, labels