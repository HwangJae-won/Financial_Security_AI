# rag.py
# -*- coding: utf-8 -*-
"""
- PDF → 텍스트 청크 → 임베딩 → FAISS 인덱스 생성/저장
- 질문 시 상위 k개 청크 검색 후, 컨텍스트를 포함한 프롬프트로 LLM 호출

사용 예:
    # 1) 인덱스 빌드
    !python code/rag.py build --pdf "data/전자금융거래법(법률)(제19734호)(20240915).pdf"
    !python code/rag.py build --dir "laws/"

    # 2) 단일 질문
    !python code/rag.py ask --question "정보보호의 3대 요소에 해당하는 보안 목표를 3가지 기술하세요."
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

# 외부 패키지
import pdfplumber
from transformers import AutoTokenizer, AutoModel
import torch
import faiss

# 기존 프로젝트 모듈
from model import load_model
from utils import is_multiple_choice, extract_question_and_choices, extract_answer_only
from prompt import make_prompt_rag_exaone 


# -----------------------------
# 설정
# -----------------------------
INDEX_DIR = "./rag_index"
INDEX_BIN = os.path.join(INDEX_DIR, "faiss.index")
META_PKL = os.path.join(INDEX_DIR, "meta.pkl")
MODEL_NAME = "/workspace/models/multilingual-e5-small"   # 한글 안정: e5-base 다국어
CHUNK_SIZE = 200        # 청크 길이(문자 수 기준)
CHUNK_OVERLAP = 50     # 청크 겹침
TOP_K = 1             # 검색 상위 k개

SCORE_THRESHOLD = 0.89
OUTPUT_PATH = "results/"

RETRIEVE_K = 3      # 벡터검색 1차 후보 개수
RERANK_KEEP = 1      # 리랭크 후 LLM에 줄 개수
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"  # 멀티링구얼 추천(리소스 여유 없으면 MiniLM)

# ---- E5 임베딩 클래스 (sentence-transformers 대체) ----
class E5Embedder:
    """
    intfloat/multilingual-e5-small 를 transformers로 직접 로드.
    - query에는 'query: ' 프리픽스
    - passage에는 'passage: ' 프리픽스
    - mean-pooling + L2 normalize
    """
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    @torch.no_grad()
    def _encode(self, texts, batch_size=16):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            out = self.model(**tokens)
            last_hidden = out.last_hidden_state  # [B, T, H]
            mask = tokens["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            # mean pooling
            summed = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            emb = summed / lengths
            # L2 normalize
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0).numpy().astype("float32")

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

# 항/호 마커 (다양한 표기 대응: ①②… / '1항' / '1.' / '가.' / 괄호 숫자 등)
PARA_SPLIT_RE = re.compile(
    r'(?m)^(?=(?:[①-⑳]|[0-9]+\.?\s*항|[0-9]+\)|[가-하]\.|[ㄱ-ㅎ]\)|\([0-9]+\)|\([가-하]\)))'
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
        subchunks = split_article_if_long(a["text"], max_len=max(800, chunk_size*2//1), overlap=overlap)
        for i, sc in enumerate(subchunks):
            chunks.append(sc)
            labels.append(f"{title}#{i+1}")
    return chunks, labels


# -----------------------------
# 인덱서/검색기
# -----------------------------

class RAGIndexer:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.model_name = model_name
        self.embedder = E5Embedder(MODEL_NAME, device='cpu')

    def build_many(self, pdf_paths: List[str], index_dir=INDEX_DIR):
        all_chunks, all_sources = [], []
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
            # 소스에 파일명 + 조 라벨을 함께 저장 (검색 결과 설명에 바로 활용)
            all_sources.extend([f"{base}::{label}" for label in labels])
            total += len(chunks)
            print(f"  → {base}: 조/항 기준 {len(chunks)}개")

        if total == 0:
            raise ValueError("인덱싱할 청크가 없습니다.")

        print(f"🧠 임베딩 계산({self.model_name})... 총 청크 {total}개")
        vectors = self.embedder.encode_passages(all_chunks)
        if vectors.shape[0] != len(all_chunks):
            raise RuntimeError(f"벡터 수({vectors.shape[0]})와 청크 수({len(all_chunks)}) 불일치")
        dim = vectors.shape[1]

        print("📦 FAISS 인덱스 생성/저장...")
        _ensure_dir(index_dir)
        index = faiss.IndexFlatIP(dim)  # L2 정규화된 코사인 유사도
        index.add(vectors)

        tmp_idx = INDEX_BIN + ".tmp"
        tmp_meta = META_PKL + ".tmp"
        faiss.write_index(index, tmp_idx)
        meta = {
            "chunks": all_chunks,
            "sources": all_sources,           # 예: "전자서명법.pdf::제22조(분쟁의 조정)#1"
            "model_name": self.model_name,
            "n_vectors": int(vectors.shape[0]),
        }
        with open(tmp_meta, "wb") as f:
            pickle.dump(meta, f)
        os.replace(tmp_idx, INDEX_BIN)
        os.replace(tmp_meta, META_PKL)

        print(f"✅ 저장 완료: {INDEX_BIN}, {META_PKL} | 총 청크 {total}개")



class RAGRetriever:
    def __init__(self, index_dir=INDEX_DIR, device=None):
        if not (os.path.exists(INDEX_BIN) and os.path.exists(META_PKL)):
            raise FileNotFoundError("인덱스가 없습니다. 먼저 `python rag.py build --pdf <path>`를 실행하세요.")
        self.index = faiss.read_index(INDEX_BIN)
        with open(META_PKL, "rb") as f:
            meta = pickle.load(f)
        self.chunks = list(meta["chunks"])
        self.sources = list(meta.get("sources", ["unknown"] * len(self.chunks)))
        self.model_name = meta.get("model_name", MODEL_NAME)
        self._n_meta = len(self.chunks)
        self._n_index = int(self.index.ntotal)
        # 불일치 보정(더 작은 쪽으로 자르기)
        expect = int(meta.get("n_vectors", self._n_meta))
        if expect != self._n_index or self._n_meta != self._n_index:
            print(f"⚠️ meta/Index 불일치. meta.chunks={self._n_meta}, meta.n_vectors={expect}, index.ntotal={self._n_index}. "
                  f"{min(self._n_meta, self._n_index)}개로 보정합니다.")
            n = min(self._n_meta, self._n_index)
            self.chunks = self.chunks[:n]
            self.sources = self.sources[:n]
            self._n_meta = n
            self._n_index = n

        # 검색시 사용할 동일 임베더 로드
        self.embedder = E5Embedder(MODEL_NAME, device='cpu')

    def search(self, query: str, top_k=TOP_K) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode_queries([query]).astype("float32")
        k = min(int(top_k), self._n_index) if self._n_index > 0 else 0
        if k <= 0:
            return []
        D, I = self.index.search(q_emb, k)
        raw = list(zip(I[0].tolist(), D[0].tolist()))
        # 유효 인덱스만 필터(-1/범위 초과 제거)
        results = [(idx, float(score)) for idx, score in raw if (0 <= idx < self._n_meta)]
        return results

    def get_passages(self, hits: List[Tuple[int, float]]) -> List[str]:
        return [self.chunks[i] for i, _ in hits]

    def get_sources(self, hits: List[Tuple[int, float]]) -> List[str]:
        return [self.sources[i] for i, _ in hits]



# -----------------------------
# reranker
# -----------------------------


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class CrossEncoderReranker:
    """
    (질문, 문서청크) 쌍을 입력받아 관련성 점수를 산출하고 재정렬.
    - 기본 모델: 가볍고 빠른 영어/멀티도 가능한 ms-marco 미니LM
    - 대안: 'BAAI/bge-reranker-v2-m3' (멀티링구얼, 성능↑, 약간 무거움)
            'jinaai/jina-reranker-v2-base-multilingual'
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        scores = []
        for i in range(0, len(passages), batch_size):
            batch_psg = passages[i:i+batch_size]
            encoded = self.tokenizer(
                [query] * len(batch_psg), batch_psg,
                padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**encoded).logits.squeeze(-1)  # [B]
            # 점수 스케일을 안정화하려면 시그모이드(0~1)로 보정해도 됨
            probs = torch.sigmoid(logits).tolist()
            scores.extend(probs)
        return scores

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """
        return: [(원래인덱스, rerank_score)] 높은 점수 순
        """
        if not passages:
            return []
        scores = self.score(query, passages)
        order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order]



# -----------------------------
# RAG 추론 함수
# -----------------------------

def answer_with_rag(
    question: str, retriever: RAGRetriever, pipe, top_k=TOP_K, score_threshold: float = SCORE_THRESHOLD):
    hits = retriever.search(question, top_k=top_k)
    passages = retriever.get_passages(hits)
    top_score = hits[0][1] if hits else 0.0

    # --- 핵심: 점수 낮으면 컨텍스트 제거 ---
    use_context = (top_score >= score_threshold)
    contexts = passages if use_context else []

    prompt = make_prompt_rag_exaone(question, contexts, use_fewshot=True)

    # 너 환경의 decode 정책 맞춤: 1차 greedy → 실패 시 샘플링
    is_mc, mc_num = is_multiple_choice(question)
    if is_mc:
        out = pipe(prompt, max_new_tokens=2, do_sample=False)
    else:
        out = pipe(prompt, max_new_tokens=256, do_sample=False)
    gen = out[0]["generated_text"]
    ans = extract_answer_only(gen, original_question=question, prompt=prompt)
    if ans in ("0", "미응답"):
        if is_mc:
            outs = pipe(prompt, max_new_tokens=2, do_sample=True, temperature=0.6, top_p=0.95, 
                        num_return_sequences=3, repetition_penalty=1.05)
        else:
            outs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95, 
                        num_return_sequences=3, repetition_penalty=1.05)
        picked = None
        for o in outs:
            cand = extract_answer_only(o["generated_text"], original_question=question, prompt=prompt)
            if cand not in ("0", "미응답"):
                picked = cand
                gen = o["generated_text"]
                break

    # 간단 후처리(필요시 강화)
    # 객관식이면 숫자만, 주관식이면 문장 정리
    is_mc, _ = is_multiple_choice(question)
    if is_mc:
        # 가능한 숫자만 추출(1~99), 없으면 샘플링 재시도
        m = re.search(r"\b([1-9][0-9]?)\b", gen)
        if not m:
            out = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95)
            gen = out[0]["generated_text"]
    return prompt, gen, passages, hits, use_context

# # 초기화 시점 어딘가에(전역 혹은 객체 내부)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# reranker = CrossEncoderReranker(model_name=RERANK_MODEL, device=device)


# def answer_with_rag(
#     question: str, retriever: RAGRetriever, pipe,
#     top_k=TOP_K, score_threshold: float = SCORE_THRESHOLD,
#     debug: bool = True,  # ← 디버그 출력 스위치
#     return_debug: bool = False  # ← 디버그 정보를 함께 반환할지
# ):
#     # 1) 1차: 벡터 검색 (넉넉히)
#     k_retrieve = RETRIEVE_K if RETRIEVE_K > top_k else top_k
#     hits = retriever.search(question, top_k=k_retrieve)              # [(faiss_idx, vec_score)]
#     passages = retriever.get_passages(hits)                          # [str]
#     sources_all = retriever.get_sources(hits)                        # [str]
#     faiss_indices = [i for i, _ in hits]                             # [int]
#     vec_scores = [s for _, s in hits]                                # [float]
#     top_score = vec_scores[0] if vec_scores else 0.0

#     # 초기 순위표 (FAISS)
#     initial_table = [
#         {
#             "rank": r+1,
#             "faiss_idx": faiss_indices[r],
#             "vec_score": float(vec_scores[r]),
#             "source": sources_all[r],
#             "passage": passages[r]
#         }
#         for r in range(len(passages))
#     ]

#     # 2) 임계치로 컨텍스트 사용 여부 판단 (벡터스코어 기준 유지)
#     use_context = (top_score >= score_threshold)

#     # 3) use_context면 리랭크 → 상위 n개만 선택
#     rerank_table = []
#     kept_idx_local = []
#     kept_sources = []
#     if use_context and passages:
#         reranked = reranker.rerank(question, passages)               # [(local_idx, rerank_score)]
#         # rerank 전체 테이블 (변화 추적용)
#         for new_rank, (loc_idx, rr_score) in enumerate(reranked, start=1):
#             rerank_table.append({
#                 "new_rank": new_rank,
#                 "from_initial_rank": loc_idx + 1,
#                 "faiss_idx": faiss_indices[loc_idx],
#                 "rerank_score": float(rr_score),
#                 "vec_score": float(vec_scores[loc_idx]),
#                 "source": sources_all[loc_idx],
#                 "passage": passages[loc_idx]
#             })
#         # 상위만 유지
#         kept_idx_local = [i for i, _ in reranked[:RERANK_KEEP]]
#         passages = [passages[i] for i in kept_idx_local]
#         kept_sources = [sources_all[i] for i in kept_idx_local]
#     else:
#         passages = []
#         kept_sources = []

#     # 4) 프롬프트 생성 및 생성
#     prompt = make_prompt_rag_exaone(question, passages, use_fewshot=True)

#     is_mc, mc_num = is_multiple_choice(question)
#     if is_mc:
#         out = pipe(prompt, max_new_tokens=2, do_sample=False)
#     else:
#         out = pipe(prompt, max_new_tokens=256, do_sample=False)

#     gen = out[0]["generated_text"]
#     ans = extract_answer_only(gen, original_question=question, prompt=prompt)

#     if ans in ("0", "미응답"):
#         if is_mc:
#             outs = pipe(prompt, max_new_tokens=2, do_sample=True, temperature=0.6, top_p=0.95, 
#                         num_return_sequences=3, repetition_penalty=1.05)
#         else:
#             outs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95, 
#                         num_return_sequences=3, repetition_penalty=1.05)
#         for o in outs:
#             cand = extract_answer_only(o["generated_text"], original_question=question, prompt=prompt)
#             if cand not in ("0", "미응답"):
#                 gen = o["generated_text"]
#                 break

#     # 5) 객관식 숫자 후처리
#     is_mc, _ = is_multiple_choice(question)
#     if is_mc:
#         m = re.search(r"\b([1-9][0-9]?)\b", gen)
#         if not m:
#             out = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95)
#             gen = out[0]["generated_text"]

#     # ---- 디버그 출력/반환 ----
#     if debug:
#         print("\n[FAISS 초기 순위]")
#         for row in initial_table[:10]:  # 너무 길면 상위 10개만 출력
#             print(f"{row['rank']:>2}. faiss_idx={row['faiss_idx']}, vec={row['vec_score']:.4f}, src={row['source']}")
#         if use_context and rerank_table:
#             print("\n[Re-Rank 결과]")
#             for row in rerank_table[:10]:
#                 print(f"{row['new_rank']:>2}. (from {row['from_initial_rank']:>2}) "
#                       f"faiss_idx={row['faiss_idx']}, rr={row['rerank_score']:.4f}, vec={row['vec_score']:.4f}, src={row['source']}")
#             if kept_idx_local:
#                 kept_str = ", ".join([f"#{i+1}" for i in kept_idx_local])
#                 print(f"\n→ LLM에 전달한 문맥(local rank): {kept_str} (총 {len(kept_idx_local)}개)")

#     debug_payload = None
#     if return_debug:
#         debug_payload = {
#             "initial": initial_table,      # FAISS 결과 전체
#             "reranked": rerank_table,      # rerank 결과 전체
#             "kept_local_indices": kept_idx_local,
#             "use_context": use_context
#         }

#     # 기존 반환형을 유지하며, 필요 시 debug_payload 추가
#     if return_debug:
#         return (prompt, gen, passages, hits, use_context, debug_payload)
#     else:
#         return (prompt, gen, passages, hits, use_context)




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
    pipe = load_model()
    retr = RAGRetriever(INDEX_DIR, device="cpu")
    q = args.question
    prompt, gen, passages, hits, use_context = answer_with_rag(
        q, retr, pipe, top_k=args.top_k, score_threshold=args.threshold)
    print("\n===== 생성된 답변 =====")
    print(gen)
    ans = extract_answer_only(gen, original_question=q, prompt=prompt)
    print("\n===== 채택된 답변 =====")
    print(ans)
    print("\n===== 검색 결과 요약 =====")
    if hits:
        print(f"Top-1 score={hits[0][1]:.3f} | threshold={args.threshold:.2f} | context_used={use_context}")
    else:
        print("검색 결과 없음 | context_used=False")
    if use_context:
        srcs = retr.get_sources(hits)
        print("\n===== 참고된 청크 (점수순) =====")
        for (idx, score), p, s in zip(hits, passages, srcs):
            print(f"\n[score={score:.3f}] chunk#{idx} | source={s}\n{p[:400]}...")

def cmd_run(args):
    import pandas as pd
    pipe = load_model()
    retr = RAGRetriever(INDEX_DIR, device="cpu")
    df = pd.read_csv(args.csv)
    preds = []
    for idx, q in enumerate(tqdm(df['Question'], desc="Inference")):
        prompt, gen, passages, hits, use_context = answer_with_rag(
            q, retr, pipe, top_k=args.top_k, score_threshold=args.threshold)
        print(f"\n===== [문항{idx}] 생성된 답변 =====")
        print(gen)
        if hits:
            print(f"Top-1 score={hits[0][1]:.3f} | threshold={args.threshold:.2f} | context_used={use_context}")
        else:
            print("검색 결과 없음 | context_used=False")
        ans = extract_answer_only(gen, original_question=q, prompt=prompt)
        
        preds.append(ans)
        if args.verbose and i < 5:
            print(f"\n[#{i}] Q={q[:80]}...")
            print(f"[use_context={use_context}] top1={hits[0][1] if hits else None}")
            print(f"[gen] {gen[:200]}...")

    
    experiment_name = "result.csv"
    print("📄 제출 파일 생성 중...")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    sample_submission['Answer'] = preds
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    sample_submission.to_csv(OUTPUT_PATH + experiment_name, index=False, encoding='utf-8-sig')
    print(f"✅ 제출 파일 저장 완료: {OUTPUT_PATH + experiment_name}")
    

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
