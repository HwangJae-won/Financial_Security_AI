# rag.py
# -*- coding: utf-8 -*-
"""
    # 1) 인덱스 빌드
    !python code/rag.py build --pdf "data/전자금융거래법(법률)(제19734호)(20240915).pdf"
    !python code/rag.py build --dir "laws/"

    # 2) 단일 질문
    !python code/rag.py ask --question "전자자금이체의 지급 효력 발생 시점을 전자금융거래법 기준에 따라 설명하세요."
    !python code/rag.py ask --question $'전자금융거래법 제44조에 따르면, 청문 절차가 필요한 경우는 무엇인가?\n1 전자금융거래의 중단\n2 전자금융거래의 보안 점검\n3 전자금융업자의 등록 취소\n4 전자금융거래의 수수료 변경'
    !python code/rag.py ask --question $'국내대리인이 법을 위반한 경우, 그 책임은 누구에게 있는가?\n1 국내대리인\n2 정부기관\n3 법원\n4 정보통신서비스 제공자\n5 개인정보 처리 위탁업체'
    !python code/rag.py ask --question $'개인정보보호법 제63조에 따르면, 보호위원회가 자료제출 요구 및 검사를 통해 수집한 서류나 자료를 제3자에게 제공하거나 일반에 공개할 수 있는 경우는?\n1 자료가 비밀이 아닌 경우\n2 개인정보처리자의 동의가 있는 경우\n3 정보주체가 개인정보 열람을 요청한 경우\n4 법에 따른 경우\n5 보호위원회의 내부 규정에 따른 경우'

    # 3) test.csv 한 번에 추론(컬럼명: Question)
    !python code/rag.py run --csv "data/test.csv"

"""

import os
import re
import json
import argparse
import pickle
from typing import List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import math

# 외부 패키지
import pdfplumber
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# 기존 프로젝트 모듈
from model import load_model
from utils import is_multiple_choice, extract_question_and_choices, extract_answer_only
from prompt import make_prompt_rag_exaone 
  



# -----------------------------
# 설정
# -----------------------------
INDEX_DIR = "./rag_index"
MODEL_NAME = "/workspace/models/multilingual-e5-small"   # 한글 안정: e5-base 다국어
CHUNK_SIZE = 700        # 청크 길이(문자 수 기준)
CHUNK_OVERLAP = 50     # 청크 겹침
TOP_K = 1             # 검색 상위 k개

SCORE_THRESHOLD = 0.89
OUTPUT_PATH = "results/"



os.environ["ANONYMIZED_TELEMETRY"] = "false"      # 크로마 텔레메트리 끄기
# 혹시 환경에 따라 아래 키도 지원됩니다(둘 다 넣어도 무해).
os.environ["CHROMADB_TELEMETRY_ENABLED"] = "false"

# --- CHROMA: imports & constants ---
from pathlib import Path
import chromadb
from chromadb import PersistentClient

# Chroma 영구 저장 경로와 컬렉션 이름
CHROMA_DIR = Path(INDEX_DIR) / "chroma"     # 기존 INDEX_DIR 활용
CHROMA_COLLECTION = "rag_index"  



# ---- E5 임베딩 클래스 (sentence-transformers 버전) ----
from sentence_transformers import SentenceTransformer
import numpy as np

class E5Embedder:
    """
    intfloat/multilingual-e5-small 를 SentenceTransformer로 로드.
    - query에는 'query: ' 프리픽스
    - passage에는 'passage: ' 프리픽스
    - SBERT 내부 mean-pooling + L2 normalize 사용
    """
    def __init__(self, model_name=MODEL_NAME, device=None):
        # device: 'cpu' 또는 'cuda'
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        # E5는 512 토큰권장. 필요 시 명시적으로 고정 가능
        self.model.max_seq_length = 512

    def _encode(self, texts, batch_size=16, show_progress=None):
        # SBERT가 pooling/정규화까지 처리
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # FAISS IP(코사인)와 일치
            show_progress_bar=(show_progress if show_progress is not None else (len(texts) > batch_size))
        )
        # float32로 고정(FAISS/메모리 일관성)
        return embs.astype(np.float32, copy=False)

    def encode_passages(self, passages):
        return self._encode([f"passage: {p}" for p in passages])

    def encode_queries(self, queries):
        return self._encode([f"query: {q}" for q in queries])




# -----------------------------
# 유틸
# -----------------------------
def _ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

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

def load_pdf_text(pdf_path: str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            texts.append(t)
    return "\n".join(texts)




# --- 법령 전용 분할기: '제n조(제n조의m)' 단위로 자르기 + 길면 항/호로 재분할 ---

# --- 조 헤더 정규식: '제n조(…)' 또는 '제n조의m(…)' + 줄 시작 + 괄호 존재 보장 + '조제' 참조 제외 ---

# 제목 괄호를 반드시 요구(ASCII '(' ')' 또는 전각 '（' '）')
ARTICLE_RE = re.compile(
    r'(?m)^'
    r'(?P<header>'
        r'제\s*\d+\s*조'          # 제n조
        r'(?!\s*제)'              # '조제…항' 참조 제외
        r'(?:\s*의\s*\d+)?'       # '의m' 허용(제n조의m)
    r')'
    r'\s*[（(]'                   # 여는 괄호(필수)
    r'(?P<title>[^）)]+)'         # 제목(최소 1자)
    r'[）)]'                      # 닫는 괄호
    , re.UNICODE
)

# 항/호 마커 (다양한 표기 대응: ①②… / '1항' / '1.' 등)
PARA_SPLIT_RE = re.compile(
    r'(?m)^(?=(?:[①-⑳]|[0-9]+\.?\s*항|[0-9]+\)))'
)

def parse_korean_law_articles(raw_text: str):
    text = _clean_text(raw_text)
    # '제176조제3항' 같은 붙은 참조 띄어쓰기 보정(기존 유지)
    text = re.sub(r"(제\s*\d+\s*조)(\s*제\s*\d+\s*항)", r"\1 \2", text)

    matches = list(ARTICLE_RE.finditer(text))  # 🔒 괄호 필수 정규식만 사용
    if not matches:
        return [{"article": "전체", "title": "", "text": text}]

    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        header = m.group("header")
        title  = m.group("title").strip()      # 항상 존재
        body   = text[start:end].strip()

        head_full = f"{header}({title})"
        head_full_alt = f"{header}（{title}）"
        body_norm = body
        if head_full in body_norm:
            body_norm = body_norm.replace(head_full, head_full + "\n", 1)
        elif head_full_alt in body_norm:
            body_norm = body_norm.replace(head_full_alt, head_full + "\n", 1)

        articles.append({
            "article": header.replace(" ", ""),   # 예: "제9조의2"
            "title": title,                       # 예: "전자자금이체의 지급 효력 발생시기의 지연"
            "text": body_norm,
        })
    return articles



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


# -----------------------------
# Reranker
# -----------------------------

class STReranker:
    """
    Sentence-Transformers CrossEncoder 기반 재정렬기.
    - 기본 모델: "Alibaba-NLP/gte-multilingual-reranker-base" (멀티링궐)
    - 입력: (query, passages[list[str]])
    - 출력: scores[list[float]] (클수록 관련성 높음)
    """
    def __init__(self, model_name_or_path: str = "/workspace/models/gte-multilingual-reranker-base", 
                 device: str | None = None, max_length: int = 512, trust_remote_code=True):

         # model_name_or_path에 로컬 디렉토리 or HF 모델명 모두 허용
        path = model_name_or_path
        if os.path.isdir(model_name_or_path):
            # 로컬 디렉토리 우선 사용
            path = model_name_or_path
            
        self.model = CrossEncoder(model_name_or_path, device=device, max_length=max_length, trust_remote_code=True)

    @torch.no_grad()
    def score(self, query: str, passages: list[str], batch_size: int = 32) -> list[float]:
        if not passages:
            return []
        # 너무 긴 본문이면 대략 자르기(토큰 기준 아님, 안전장치)
        trimmed = [p[:4000] for p in passages]
        pairs = [(query, p) for p in trimmed]
        scores = self.model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
        return scores.tolist()



# -----------------------------
# 인덱서/검색기
# -----------------------------


_ART_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", re.UNICODE)  # 제22조의2 → (22, 2)
_CLAUSE_RE = re.compile(r"(?:제)?\s*(\d+)\s*항")
LABEL_RE = re.compile(r'^(제\s*\d+\s*조(?:\s*의\s*\d+)?)(?:\(([^)]*)\))?#(\d+)$')


def _norm_law_name_from_filename(base: str) -> str:
    name = os.path.splitext(base)[0]
    name = re.sub(r"\(.*?\)", "", name)        # 괄호군 제거
    name = name.replace(" ", "")
    name = name.replace("개인정보 보호법", "개인정보보호법")  # 흔한 표기 정규화 예시
    return name

def _parse_label_to_meta(label: str):
    s = label.replace(" ", "")
    a = _ART_RE.search(s)
    a_num = int(a.group(1)) if a else None
    a_bis = int(a.group(2)) if (a and a.group(2)) else None
    clause = None
    c = _CLAUSE_RE.search(label)
    if c: clause = int(c.group(1))
    article_key = str(a_num) + (f"-{a_bis}" if (a_num and a_bis) else "") if a_num else None
    return {"article_num": a_num, "article_bis": a_bis, "article_key": article_key, "clause": clause}

def _sanitize_meta(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if v is None:
            continue  # None은 키째로 제거
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)  # 혹시 모를 비허용 타입 방지
    return out


class RAGIndexer:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.model_name = model_name
        self.embedder = E5Embedder(MODEL_NAME, device='cpu')

    def build_many(self, pdf_paths: List[str], index_dir=INDEX_DIR):
        all_chunks, all_sources = [], []
        metas = []   # ★ 추가: 청크별 메타 담을 리스트
        total = 0
        for pdf_path in pdf_paths:
            print(f"📄 PDF 읽는 중: {pdf_path}")
            raw = load_pdf_text(pdf_path)
            if not raw.strip():
                raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. OCR이 필요할 수 있습니다.")

            print("🔪 '제n조' 단위 청크 분할 중...")
            chunks, labels = chunk_law_text(raw, by_article=True, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            base = os.path.basename(pdf_path)
            all_chunks.extend(chunks)
            all_sources.extend([f"{base}::{label}" for label in labels])

            # ★ 추가: 라벨→메타 파싱 + 법령명 주입
            law_name = _norm_law_name_from_filename(base)
            
            for lbl in labels:
                # lbl 예: "제11조(다른 법률의 개정)#2" 또는 "제11조#1"
                m_lbl = LABEL_RE.match(lbl)
                if m_lbl:
                    article_label = m_lbl.group(1).replace(" ", "")             # "제11조" / "제11조의2"
                    article_title = (m_lbl.group(2) or "").strip()              # "다른 법률의 개정" / ""
                    chunk_index   = int(m_lbl.group(3))
                else:
                    article_label, article_title, chunk_index = None, "", None
            
                m_art = _ART_RE.search(article_label or "")
                article_num = int(m_art.group(1)) if m_art else None
                article_bis = int(m_art.group(2)) if (m_art and m_art.group(2)) else None
                article_key = str(article_num) + (f"-{article_bis}" if article_bis else "") if article_num else None
            
                metas.append(_sanitize_meta({
                    "law": law_name,                       # 필터용 법령명
                    "source": f"{base}::{lbl}",            # 사람이 보기 쉬운 원본 라벨
                    "article_label": article_label,        # "제11조" / "제11조의2"
                    "article_title": article_title,        # 제목(없으면 빈 문자열)
                    "article_num": article_num,            # 11
                    "article_bis": article_bis,            # 2 (없으면 None)
                    "article_key": article_key,            # "11" / "11-2"
                    "chunk_index": chunk_index,            # 1부터 시작
                    # "clause": None,  # 항/호까지 필요하면 분할 시점에 넣는 게 정확
                }))

            total += len(chunks)
            print(f"  → {base}: 조/항 기준 {len(chunks)}개")

        if total == 0:
            raise ValueError("인덱싱할 청크가 없습니다.")

        print(f"🧠 임베딩 계산({self.model_name})... 총 청크 {total}개")
        vectors = self.embedder.encode_passages(all_chunks)  # (N, dim) float32
        if vectors.shape[0] != len(all_chunks):
            raise RuntimeError(f"벡터 수({vectors.shape[0]})와 청크 수({len(all_chunks)}) 불일치")
        dim = vectors.shape[1]
        print(f"  → dim={dim}")

        # --- CHROMA: 영구 클라이언트/컬렉션 생성 ---
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = PersistentClient(path=str(CHROMA_DIR))

        # 중복 빌드를 피하려면 기존 컬렉션 삭제 후 재생성(선택)
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"🗑️ 기존 컬렉션 '{CHROMA_COLLECTION}' 삭제")
        except Exception:
            pass

        collection = client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={
                "hnsw:space": "cosine",   # E5는 L2 정규화 → cosine 추천
                "model_name": self.model_name,
            },
        )

        # --- CHROMA: upsert ---
        print("📦 ChromaDB 업서트(add) 중...")
        # ids는 고유 문자열 필요
        ids = [f"doc-{i}" for i in range(len(all_chunks))]
        # ★ 교체: 위에서 만든 metas 사용 (길이 검증)
        assert len(metas) == len(all_chunks), f"metas({len(metas)}) != chunks({len(all_chunks)})"
        # chroma는 list-of-list/pythonic 타입 권장
        collection.add(
            ids=ids,
            documents=list(all_chunks),
            embeddings=vectors.tolist(),
            metadatas=metas,
        )

        print(f"✅ 저장 완료: Chroma @ {CHROMA_DIR}, collection='{CHROMA_COLLECTION}' | 총 청크 {total}개")


# === 상단 공용 ===
_LAW_NAME_RE = re.compile(r"([가-힣A-Za-z0-9·\s]+?(?:법|령|칙|규정|고시|지침))")

def _extract_explicit_law_and_article(q: str):
    # 법령명
    law = None
    cand = _LAW_NAME_RE.findall(q)
    if cand:
        law = max(cand, key=len).replace(" ", "")
        law = law.replace("개인정보 보호법", "개인정보보호법")
    # 조/조의
    a = _ART_RE.search(q.replace(" ", ""))
    a_num, a_bis = None, None
    if a:
        a_num = int(a.group(1))
        a_bis = int(a.group(2)) if a.group(2) else None
    # 항
    clause = None
    c = _CLAUSE_RE.search(q)
    if c:
        clause = int(c.group(1))
    # article_key(문자열)는 보조용으로 필요 시 구성
    article_key = str(a_num) + (f"-{a_bis}" if (a_num and a_bis) else "") if a_num else None
    return law, a_num, a_bis, clause, article_key

class RAGRetriever:
    def __init__(self, index_dir=INDEX_DIR, device=None):
        # --- CHROMA: 로드 ---
        if not (CHROMA_DIR.exists()):
            raise FileNotFoundError("Chroma 인덱스가 없습니다. 먼저 `python rag.py build --pdf <path>`를 실행하세요.")
        client = PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = client.get_collection(CHROMA_COLLECTION)
        except Exception as e:
            raise FileNotFoundError(f"Chroma 컬렉션 '{CHROMA_COLLECTION}'을 찾을 수 없습니다.") from e

        # 메모리 캐시(FAISS 호환 인터페이스 유지를 위해)
        got = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids: List[str] = got["ids"]
        self.chunks: List[str] = got["documents"]
        # 메타에 source 없으면 'unknown'
        self.sources: List[str] = [
            (m.get("source") if m else "unknown") for m in got.get("metadatas", [])
        ]

        # id -> local index 맵
        self._id2idx = {id_: i for i, id_ in enumerate(self.doc_ids)}
        self._n_index = len(self.doc_ids)
        self._n_meta = len(self.chunks)

        # 검색시 사용할 동일 임베더 로드
        self.embedder = E5Embedder(MODEL_NAME, device='cpu')

        # 모델명 (옵션)
        try:
            self.model_name = self.collection.metadata.get("model_name", MODEL_NAME)
        except Exception:
            self.model_name = MODEL_NAME

        # 불일치 보정
        if self._n_index != self._n_meta:
            n = min(self._n_index, self._n_meta)
            print(f"⚠️ meta/Index 불일치. index={self._n_index}, meta={self._n_meta}. {n}개로 보정합니다.")
            self.doc_ids = self.doc_ids[:n]
            self.chunks = self.chunks[:n]
            self.sources = self.sources[:n]
            self._id2idx = {id_: i for i, id_ in enumerate(self.doc_ids)}
            self._n_index = n
            self._n_meta = n


    def _where_all(self, **kv):
        terms = [{k: v} for k, v in kv.items() if v is not None]
        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]              # ✅ 단일 조건은 그대로 반환 (예: {"law": "..."} )
        return {"$and": terms}           # ✅ 2개 이상일 때만 $and

    
    def _to_hits(self, res):
        """
        Chroma query/get 응답을 (idx, similarity) 리스트로 변환
        - res["ids"]      : List[List[str]]
        - res["distances"]: List[List[float]]  (include=["distances"]로 요청해야 옴)
        """
        # 빈 결과 방어
        if not res or "ids" not in res or not res["ids"] or not res["ids"][0]:
            return []
    
        ids = res["ids"][0]
        # distances가 없을 수도 있으니 0.0으로 폴백
        dists = res.get("distances", [[0.0] * len(ids)])[0]
    
        out = []
        for id_, dist in zip(ids, dists):
            idx = self._id2idx.get(id_)
            if idx is None:
                continue
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            out.append((idx, sim))
        return out

    # RAGRetriever 내부 메서드로 추가
    def _format_for_rerank(self, i: int) -> str:
        """리랭커에 넣을 passage 앞에 [법][조][제목] 헤더를 붙인다."""
        # 메타 안전 접근
        meta = {}
        try:
            if hasattr(self, "metadatas") and self.metadatas:
                meta = self.metadatas[i] or {}
        except Exception:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
    
        law   = (meta.get("law") or "").strip()
        albl  = (meta.get("article_label") or meta.get("article") or "").strip()
        atitle= (meta.get("article_title") or meta.get("title") or "").strip()
    
        head = " ".join(p for p in [
            f"[법:{law}]"     if law   else "",
            f"[조:{albl}]"    if albl  else "",
            f"[제목:{atitle}]"if atitle else "",
        ] if p)
    
        body = self.chunks[i]
        return (head + "\n" if head else "") + body

    def search_mix_and_rerank(
        self,
        query: str,
        top_k: int = TOP_K,
        M_generic: int = 1,     # 전역 검색 후보 수
        M_filtered: int = 1,    # 필터 검색 후보 수 (질의에 법+조문 있을 때만)
        reranker: Optional["STReranker"] = None,
        use_clause: bool = True, # '항'까지 있으면 더 좁히기
    ) -> list[tuple[int, float]]:
        """
        1) 전역 검색 M_generic
        2) (질의에 '법+조문'이 명시된 경우) where 필터로 M_filtered
        3) 두 후보를 합쳐서 CrossEncoder로 rerank → Top-K 반환
           반환 score는 reranker 점수
        """
        if self._n_index <= 0:
            return []
    
        k = min(int(top_k), self._n_index)
    
        # --- 1) 전역 검색
        q_emb = self.embedder.encode_queries([query]).astype("float32").tolist()
        res_global = self.collection.query(
            query_embeddings=q_emb,
            n_results=min(M_generic, self._n_index),
            include=["distances"],
        )
        hits_global = self._to_hits(res_global)
    
        # --- 2) 법+조문 파싱 후(있을 때만) 메타 필터 검색
        law, a_num, a_bis, clause, _article_key = _extract_explicit_law_and_article(query)
        hits_filtered = []
        if law and (a_num is not None):
            # (기존 그대로) 조/조의 [+ 항] 필터
            where = self._where_all(law=law, article_num=a_num, article_bis=a_bis)
            if use_clause and (clause is not None):
                where_clause = self._where_all(law=law, article_num=a_num, article_bis=a_bis, clause=clause)
                res_f1 = self.collection.query(
                    query_embeddings=q_emb, n_results=min(M_filtered, self._n_index),
                    where=where_clause, include=["distances"]
                )
                hits_filtered = self._to_hits(res_f1)
                if len(hits_filtered) < min(M_filtered, self._n_index):
                    res_f2 = self.collection.query(
                        query_embeddings=q_emb, n_results=min(M_filtered, self._n_index),
                        where=where, include=["distances"]
                    )
                    hits_filtered += [h for h in self._to_hits(res_f2) if h not in hits_filtered]
            else:
                res_f = self.collection.query(
                    query_embeddings=q_emb, n_results=min(M_filtered, self._n_index),
                    where=where, include=["distances"]
                )
                hits_filtered = self._to_hits(res_f)
        
        elif law:
            # ✅ 추가: 법령명만 명시된 경우에도 해당 법령으로 1차 좁히기
            where_law = self._where_all(law=law)
            res_law = self.collection.query(
                query_embeddings=q_emb,
                n_results=min(M_filtered, self._n_index),
                where=where_law,
                include=["distances"],
            )
            hits_filtered = self._to_hits(res_law)

    
        # --- 3) 후보 합치기(중복 제거)
        # idx 기준 dedup, 우선순위는 filtered > global (동일 idx면 한 번만)
        seen = set()
        combined = []
        for h in hits_filtered + hits_global:
            if h[0] in seen:
                continue
            seen.add(h[0])
            combined.append(h)
        if not combined:
            return []
    
        # --- 4) Rerank (없으면 combined 유사도 점수 그대로 정렬)
        if reranker is None:
            combined.sort(key=lambda x: x[1], reverse=True)
            return combined[:k]
        
        # ✅ 헤더 주입한 passage로 리랭크
        aug_passages = [self._format_for_rerank(i) for i, _ in combined]
        scores = reranker.score(query, aug_passages, batch_size=32)
        
        reranked = sorted(
            zip([i for i, _ in combined], scores),
            key=lambda x: x[1],
            reverse=True
        )
        return reranked[:k]
    
    # === RAGRetriever 내부 ===
    def search(self, query: str, top_k=TOP_K) -> List[Tuple[int, float]]:
        if self._n_index <= 0:
            return []
        k = min(int(top_k), self._n_index)
        q_emb = self.embedder.encode_queries([query]).astype("float32").tolist()
            
        law, a_num, a_bis, clause, article_key = _extract_explicit_law_and_article(query)
        
        # 1) 명시적 법 + '조'(및 '조의')가 있으면 '숫자'로 필터
        if law and a_num is not None:
            where = self._where_all(law=law, article_num=a_num, article_bis=a_bis)
            # (선택) 항까지 있으면 더 좁히기 시도
            if clause is not None:
                where_clause = self._where_all(law=law, article_num=a_num, article_bis=a_bis, clause=clause)
                res = self.collection.query(query_embeddings=q_emb, n_results=k, where=where_clause, include=["distances"])
                hits = self._to_hits(res)
                if hits:   # 최소 1건 나오면 여기서 반환
                    # 부족하면 같은 조문(where)로 보충
                    if len(hits) < k:
                        res2 = self.collection.query(query_embeddings=q_emb, n_results=k, where=where, include=["distances"])
                        more = self._to_hits(res2)
                        seen = {i for i, _ in hits}
                        hits.extend([(i, s) for i, s in more if i not in seen])
                        hits = sorted(hits, key=lambda x: x[1], reverse=True)[:k]
                    return hits
            # 항이 없거나 0건이면 조문 수준으로 재시도
            res = self.collection.query(query_embeddings=q_emb, n_results=k, where=where, include=["distances"])
            hits = self._to_hits(res)
            if hits:
                return hits

        # ✅ 추가: 법령명만 있을 때는 법령 필터로 한 번 좁혀본다
        if law:
            where_law = self._where_all(law=law)
            res_law = self.collection.query(query_embeddings=q_emb, n_results=k, where=where_law, include=["distances"])
            hits = self._to_hits(res_law)
            if hits:
                return hits
                
        # 2) 필터 결과가 0이면 전역 검색 폴백 (원래대로)
        res = self.collection.query(query_embeddings=q_emb, n_results=k, include=["distances"])
        return self._to_hits(res)



    def get_passages(self, hits: List[Tuple[int, float]]) -> List[str]:
        return [self.chunks[i] for i, _ in hits]

    def get_sources(self, hits: List[Tuple[int, float]]) -> List[str]:
        return [self.sources[i] for i, _ in hits]



# -----------------------------
# RAG 추론 함수
# -----------------------------


def answer_with_rag(
    question: str,
    retriever: RAGRetriever,
    pipe,
    top_k=TOP_K,
    score_threshold: float = SCORE_THRESHOLD,   # 임베딩 검색용 임계값(폴백)
    reranker: Optional["STReranker"] = None,       # ★ 추가: CrossEncoder reranker
    rerank_threshold: float = 0.0,              # ★ 추가: reranker 점수 임계값
    M_generic: int = 1,                        # ★ 추가: 전역 후보 수
    M_filtered: int = 1,                       # ★ 추가: 법/조문 필터 후보 수
):
    
    is_mc, _ = is_multiple_choice(question)
    # --- 1) 검색 ---
    if reranker is not None and is_mc:
        # 혼합 검색(전역 + 명시적 법/조문 필터) → rerank
        hits = retriever.search_mix_and_rerank(
            question,
            top_k=top_k,
            M_generic=M_generic,
            M_filtered=M_filtered,
            reranker=reranker,
        )
        passages = retriever.get_passages(hits)
        top_score = hits[0][1] if hits else float("-inf")  # reranker 점수
        use_context = (len(hits) > 0) and (top_score >= rerank_threshold)
    else:
        # 기존 전역 검색
        hits = retriever.search(question, top_k=2)
        passages = retriever.get_passages(hits)
        top_score = hits[0][1] if hits else 0.0            # 임베딩 유사도
        use_context = (top_score >= score_threshold)

    contexts = passages if use_context else []

    # --- 2) 프롬프트 ---
    prompt = make_prompt_rag_exaone(question, contexts, use_fewshot=True)

    # --- 3) 1차 생성 (greedy) ---
    
    try:
        if is_mc:
            out = pipe(prompt, max_new_tokens=2, do_sample=False)
        else:
            out = pipe(prompt, max_new_tokens=256, do_sample=False)
        gen = out[0]["generated_text"]
    except Exception:
        out = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95)
        gen = out[0]["generated_text"]

    # --- 4) 추출 실패 시 샘플링 백업 ---
    ans = extract_answer_only(gen, original_question=question, prompt=prompt)
    if ans in ("0", "미응답"):
        if is_mc:
            outs = pipe(prompt, max_new_tokens=2, do_sample=True, temperature=0.6, top_p=0.95,
                        num_return_sequences=3, repetition_penalty=1.05)
        else:
            outs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95,
                        num_return_sequences=3, repetition_penalty=1.05)
        for o in outs:
            cand = extract_answer_only(o["generated_text"], original_question=question, prompt=prompt)
            if cand not in ("0", "미응답"):
                gen = o["generated_text"]
                break

    # --- 5) 객관식 후처리: 숫자 강제 ---
    if is_mc:
        m = re.search(r"\b([1-9][0-9]?)\b", gen)
        if not m:
            out = pipe(prompt, max_new_tokens=2, do_sample=True, temperature=0.6, top_p=0.95)
            gen = out[0]["generated_text"]

    return prompt, gen, passages, hits, use_context, contexts




# -----------------------------
# CLI
# -----------------------------
def cmd_build(args):
    _ensure_dir(INDEX_DIR)
    indexer = RAGIndexer(MODEL_NAME, device="cpu")
    pdfs = []
    if args.pdf:
        pdfs.extend(args.pdf)
    if args.dir:
        for name in os.listdir(args.dir):
            if name.lower().endswith(".pdf"):
                pdfs.append(os.path.join(args.dir, name))
    if not pdfs:
        raise ValueError("PDF가 없습니다. --pdf 다중 또는 --dir를 지정하세요.")
    indexer.build_many(pdfs, INDEX_DIR)



def cmd_ask(args):
    import os
    import torch
    pipe = load_model()
    retr = RAGRetriever(INDEX_DIR, device="cpu")

    # --- Reranker 준비 (없으면 CPU로도 동작) ---
    rr_model = getattr(args, "rerank_model", "/workspace/models/gte-multilingual-reranker-base")
    rr_device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = STReranker(model_name_or_path=rr_model, device=rr_device, max_length=512)

    # 하이퍼파라미터
    top_k = getattr(args, "top_k", TOP_K)
    score_threshold = getattr(args, "threshold", SCORE_THRESHOLD)      # 임베딩 폴백용
    rerank_threshold = getattr(args, "rerank_threshold", 0.0)          # reranker 컨텍스트 게이트
    M_generic = getattr(args, "M_generic", 1)
    M_filtered = getattr(args, "M_filtered", 1)

    q = args.question

    # --- 혼합 검색 + rerank 사용 ---
    prompt, gen, passages, hits, use_context, contexts = answer_with_rag(
        q,
        retr,
        pipe,
        top_k=top_k,
        score_threshold=score_threshold,
        reranker=reranker,                 # ★ 중요: reranker 주입
        rerank_threshold=rerank_threshold, # ★ 중요: rerank 임계값
        M_generic=M_generic,
        M_filtered=M_filtered,
    )

    print("\n===== 생성된 답변 =====")
    print(gen)
    ans = extract_answer_only(gen, original_question=q, prompt=prompt)
    print("\n===== 채택된 답변 =====")
    print(ans)

    print("\n===== 검색 결과 요약 =====")
    if hits:
        # reranker 점수(상위1) 기준으로 표시
        print(f"Top-1 rerank_score={hits[0][1]:.3f} | rerank_threshold={rerank_threshold:.2f} | context_used={use_context}")
    else:
        print("검색 결과 없음 | context_used=False")

    if use_context:
        srcs = retr.get_sources(hits)
        print("\n===== 참고된 청크 (점수순) =====")
        for (idx, score), p, s in zip(hits, passages, srcs):
            print(f"\n[rerank_score={score:.3f}] chunk#{idx} | source={s}\n{p[:400]}...")



def cmd_run(args):
    import os
    import re
    import torch
    import pandas as pd

    # 1) 모델/인덱스 로드
    pipe = load_model()
    retr = RAGRetriever(INDEX_DIR, device="cpu")

    # 2) Reranker 준비 (없으면 CPU로도 동작)
    rr_model = getattr(args, "rerank_model", "/workspace/models/gte-multilingual-reranker-base")
    rr_device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = STReranker(model_name_or_path=rr_model, device=rr_device, max_length=800)

    # 3) CSV 로드
    df = pd.read_csv(args.csv)

    preds = []
    context_flags = []         # context 사용 여부 (True/False)
    full_context = []          # context 원문
    generated_texts = []       # 출력값
    top_sources = []           # (디버깅) 선택된 패시지 소스
    top_scores  = []           # (디버깅) reranker 또는 embed 스코어

    # 하이퍼파라미터(없으면 기본값 사용)
    top_k = getattr(args, "top_k", TOP_K)
    score_threshold = getattr(args, "threshold", SCORE_THRESHOLD)          # embed 검색 폴백용
    rerank_threshold = getattr(args, "rerank_threshold", 0.0)              # reranker 점수 임계값
    M_generic = getattr(args, "M_generic", 1)                              # 전역 후보 수
    M_filtered = getattr(args, "M_filtered", 1)                            # 필터 후보 수

    from tqdm import tqdm
    for idx, q in enumerate(tqdm(df['Question'], desc="Inference")):
        # === 핵심 변경: 혼합 검색 + rerank 사용 ===
        prompt, gen, passages, hits, use_context, contexts = answer_with_rag(
            q,
            retr,
            pipe,
            top_k=top_k,
            score_threshold=score_threshold,
            reranker=reranker,                 # ★ 추가
            rerank_threshold=rerank_threshold, # ★ 추가
            M_generic=M_generic,               # ★ 추가
            M_filtered=M_filtered,             # ★ 추가
        )

        ans = extract_answer_only(gen, original_question=q, prompt=prompt)

        preds.append(ans)
        context_flags.append(bool(use_context))
        full_context.append(contexts)
        generated_texts.append(gen)

        # (옵션) 디버깅용 메타 저장
        try:
            srcs = retr.get_sources(hits)
            top_sources.append(srcs[:top_k])
            top_scores.append([float(s) for _, s in hits[:top_k]])
        except Exception:
            top_sources.append([])
            top_scores.append([])

    # 4) 제출 파일 저장
    experiment_name = "result.csv"
    print("📄 제출 파일 생성 중...")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    sample_submission['Answer'] = preds

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    sample_submission.to_csv(OUTPUT_PATH + experiment_name, index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {OUTPUT_PATH + experiment_name}")

    # 5) 부가 정보 저장
    result_with_info = sample_submission.copy()
    result_with_info["ContextUsed"] = context_flags
    result_with_info["Contexts"] = full_context
    result_with_info["Generated"] = generated_texts
    result_with_info["TopSources"] = top_sources
    result_with_info["TopScores"] = top_scores

    result_with_info_path = os.path.join(OUTPUT_PATH, "result_with_info.csv")
    result_with_info.to_csv(result_with_info_path, index=False, encoding='utf-8-sig')
    print(f"✅ 부가 정보 파일 저장 완료: {result_with_info_path}")




def main():
    parser = argparse.ArgumentParser(description="RAG for 전자금융거래법")
    sub = parser.add_subparsers()

    p_build = sub.add_parser("build", help="PDF에서 인덱스 생성(여러 개 가능)")
    p_build.add_argument("--pdf", nargs="+", help="PDF 파일 경로(공백으로 여러 개)")
    p_build.add_argument("--dir", help="PDF 폴더 경로(내부 *.pdf 일괄)")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="단일 질문에 RAG 적용")
    p_ask.add_argument("--question", required=True, help="질문 텍스트")
    p_ask.add_argument("--top_k", type=int, default=TOP_K)
    p_ask.add_argument("--threshold", type=float, default=SCORE_THRESHOLD, help="컨텍스트 사용 점수 임계값")
    p_ask.set_defaults(func=cmd_ask)

    p_run = sub.add_parser("run", help="CSV(Question 컬럼) 일괄 추론")
    p_run.add_argument("--csv", required=True, help="CSV 경로 (Question 컬럼 필요)")
    p_run.add_argument("--top_k", type=int, default=TOP_K)
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument("--threshold", type=float, default=SCORE_THRESHOLD, help="컨텍스트 사용 점수 임계값")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
