import os
import re
from typing import List
import pdfplumber
import pikepdf 

import contextlib

from config import CHUNK_SIZE, CHUNK_OVERLAP


def _ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# def load_pdf_text(pdf_path: str) -> str:
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             return "\n".join(p.extract_text() or "" for p in pdf.pages)
#     except Exception:
#         # 1) 임시 수리본 저장
#         tmp = pdf_path + ".fixed"
#         with pikepdf.open(pdf_path, allow_overwriting_input=True) as doc:
#             doc.save(tmp, linearize=True)
#         # 2) 원본을 수리본으로 원자적 교체 → 이후 모든 단계가 같은 경로(원래 경로)를 사용
#         os.replace(tmp, pdf_path)

#         # 3) 다시 열기
#         with pdfplumber.open(pdf_path) as pdf:
#             return "\n".join(p.extract_text() or "" for p in pdf.pages)


def load_pdf_text(pdf_path: str) -> str:
    # 0) 1차: pdfplumber (pdfminer 기반)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e1:
        first_err = e1  # 마지막에 메시지로 남김

    # 1) 2차: pikepdf로 임시 수리본 만들기(원본은 그대로, 성공 시에만 교체)
    tmp = pdf_path + ".fixed.pdf"
    try:
        with pikepdf.open(pdf_path) as doc:
            doc.save(tmp, linearize=True)
        # 수리본 열어보기
        with pdfplumber.open(tmp) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        # 문제 없으면 원자적 교체
        os.replace(tmp, pdf_path)
        return text
    except Exception:
        with contextlib.suppress(Exception):
            if os.path.exists(tmp):
                os.remove(tmp)

    # 2) 3차: PyMuPDF (깨진 PDF에 강함)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") or "" for page in doc)
        if text.strip():
            return text
    except Exception:
        pass

    # 3) 4차: pypdf (strict=False) 시도
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path, strict=False)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass

    # 4) 5차: pypdfium2 (PDFium 바인딩)
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        out = []
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            out.append(tp.get_text_range())
            tp.close()
        text = "\n".join(out)
        if text.strip():
            return text
    except Exception:
        pass

    # 5) (옵션) OCR 폴백: 환경변수 ENABLE_OCR=1일 때만
    try:
        if os.environ.get("ENABLE_OCR", "0") == "1":
            from pdf2image import convert_from_path
            import pytesseract
            pages = convert_from_path(pdf_path, dpi=300)
            text = "\n".join(pytesseract.image_to_string(img, lang="kor+eng") for img in pages)
            if text.strip():
                return text
    except Exception:
        pass

    # 모두 실패
    raise RuntimeError(
        f"Failed to extract text from {pdf_path}. First error was {type(first_err).__name__}: {first_err}"
    )


BULLET_MAP = {
    "": "•",     # 윙딩스 점
    "▶": "•",
    "▷": "•",
    "※": "•",
    "": "•" 
}
def _normalize_bullets(t: str) -> str:
    for k, v in BULLET_MAP.items():
        t = t.replace(k, v)
    # 줄 앞 글머리 정규화: "•" 뒤 공백 1개
    t = re.sub(r'(?m)^\s*[•\-]\s*', "• ", t)
    return t

def _newline_before_bullets(t: str) -> str:
    # 한 줄 안에서 처음 글자가 아닌 '•' 앞에는 개행을 넣어 각각 한 줄로 분리
    lines = []
    for s in t.splitlines():
        s = re.sub(r'(?<!^)•\s*', r'\n• ', s)  # 줄 맨 앞이 아닌 불릿 앞에 \n 추가
        lines.append(s)
    return "\n".join(lines)
    
def _join_vertical_korean_blocks(t: str) -> str:
    lines = t.splitlines()
    out, i, n = [], 0, len(lines)
    is_kchar = re.compile(r'^[가-힣]{1,2}$')  # 1~2글자짜리 한글 라인
    while i < n:
        j = i
        while j < n and is_kchar.match(lines[j].strip() or ""):
            j += 1
        if j - i >= 3:  # 3줄 이상 연속이면 제목/머리말로 판단 → 결합
            token = "".join(s.strip() for s in lines[i:j])
            out.append(token)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)

def _despace_korean_runs(t: str) -> str:
    # 4글자 이상 연속해서 '한글 + 공백' 패턴이면 공백 제거
    return re.sub(r'((?:[가-힣]\s+){3,}[가-힣])',
                  lambda m: re.sub(r'\s+', '', m.group(0)), t)

def _drop_lonely_pagenums(t: str) -> str:
    # 숫자만 있는 줄, 1~4자리 → 삭제
    return re.sub(r'(?m)^\s*\d{1,4}\s*$', '', t)

def _glue_bullet_paragraphs(t: str) -> str:
    out, buf = [], []
    for line in t.splitlines():
        if line.strip().startswith("• "):
            if buf: out.append(" ".join(buf)); buf=[]
            out.append(line.strip())
        elif not line.strip():
            if buf: out.append(" ".join(buf)); buf=[]
            out.append("")
        else:
            buf.append(line.strip())
    if buf: out.append(" ".join(buf))
    return "\n".join(out)

def _clean_text(t: str) -> str:
    t = t.replace("\u3000", " ")  # 전각 공백
    t = t.replace("\r\n","\n").replace("\r","\n")
    t = re.sub(r"[ \t]+", " ", t)

    # PDF 노이즈 정리
    t = _join_vertical_korean_blocks(t)
    t = _despace_korean_runs(t)
    t = _normalize_bullets(t)
    t = _drop_lonely_pagenums(t)

    # 불릿 단락 접기(여러 줄 → 한 줄)
    t = _glue_bullet_paragraphs(t)

    # 너가 쓰던 '삭제<날짜>' 라인 제거(법령에도 그대로 유효)
    t = re.sub(r'(?m)^.*삭제\s*[<＜][^>＞]+[>＞].*$', '', t)

    # 남은 공백/빈줄 정리
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
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


def _chunk_generic(raw_text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = _chunk_text(raw_text, chunk_size, overlap)
    labels = [f"chunk#{i+1}" for i in range(len(chunks))]
    return chunks, labels