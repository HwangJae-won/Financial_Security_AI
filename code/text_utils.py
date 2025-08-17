import re
import pdfplumber
from typing import List

def _clean_text(t: str) -> str:
    """
    텍스트에서 불필요한 공백, 특수문자 등을 정제
    """
    t = t.replace('\xa0', ' ')
    t = re.sub(r'\s+', ' ', t)
    t = t.strip()
    return t

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    긴 텍스트를 chunk_size 단위로 겹치게 분할
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

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

ARTICLE_RE_STRICT = re.compile(
    r'(?m)^' + r'(?P<header>' + r'제\s*\d+\s*조' + r'(?!\s*제)' + r'(?:\s*의\s*\d+)?' + r')' + r'(?=\s*[（(])' + r'\s*[（(]' + r'(?P<title>[^）)]*)' + r'[）)]',
    re.UNICODE
)

ARTICLE_RE_FALLBACK = re.compile(
    r'(?m)^' + r'(?P<header>제\s*\d+\s*조(?!\s*제)(?:\s*의\s*\d+)?)' + r'(?:\s*[（(](?P<title>[^）)]*)[）)])?',
    re.UNICODE
)

PARA_SPLIT_RE = re.compile(
    r'(?m)^(?=(?:[①-⑳]|[0-9]+\.?\s*항|[0-9]+\)|[가-하]\.|[ㄱ-ㅎ]\)|\([0-9]+\)|\([가-하]\)))'
)

def parse_korean_law_articles(raw_text: str) -> List[dict]:
    """
    한국어 법령 텍스트를 조별로 파싱
    """
    text = _clean_text(raw_text)
    text = re.sub(r"(제\s*\d+\s*조)(\s*제\s*\d+\s*항)", r"\1 \2", text)
    
    matches = list(ARTICLE_RE_STRICT.finditer(text))
    
    if not matches:
        matches = list(ARTICLE_RE_FALLBACK.finditer(text))
        if not matches:
            # --- 파싱에 모두 실패했을 경우 ---
            print("⚠️ 경고: '제n조' 헤더 패턴을 찾을 수 없습니다.")
            return [{"article": "전체", "title": "", "text": text}]
    
    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        articles.append({
            "header": m.group("header"),
            "title": m.groupdict().get("title", ""),
            "text": text[start:end].strip()
        })
    return articles

def split_article_if_long(article_text: str, max_len: int, overlap: int) -> List[str]:
    """
    조별 텍스트가 너무 길면 항/목 단위로 추가 분할
    """
    text = article_text.strip()
    if len(text) <= max_len:
        return [text]
    parts = []
    last = 0
    for m in PARA_SPLIT_RE.finditer(text):
        idx = m.start()
        if idx != last:
            parts.append(text[last:idx].strip())
        last = idx
    parts.append(text[last:].strip())
    parts = [p for p in parts if p]
    chunks = []
    for p in parts:
        if len(p) <= max_len:
            chunks.append(p)
        else:
            chunks.extend(_chunk_text(p, max_len, overlap))
    return [c for c in chunks if c]

def chunk_law_text(raw_text: str, by_article: bool, chunk_size: int, overlap: int) -> List[str]:
    """
    법령 텍스트를 조별 또는 전체로 분할
    """
    if not by_article:
        return _chunk_text(raw_text, chunk_size, overlap)
    articles = parse_korean_law_articles(raw_text)
    chunks = []
    for a in articles:
        subchunks = split_article_if_long(a["text"], max_len=max(800, chunk_size*2//1), overlap=overlap)
        chunks.extend(subchunks)
    return chunks