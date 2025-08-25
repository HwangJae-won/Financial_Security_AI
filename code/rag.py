import os
import re
import json
import pickle
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

import numpy as np
import math

# 외부 패키지
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# 기존 프로젝트 모듈
from utils import is_multiple_choice, extract_question_and_choices, extract_answer_only
from prompt import make_prompt_rag_exaone 
from text_utils import _ensure_dir, load_pdf_text, chunk_law_text, _chunk_generic
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SCORE_THRESHOLD, M_GENERIC, M_FILTERED, OUTPUT_PATH, CHROMA_COLLECTION,
    MODEL_NAME, INDEX_DIR, RERANK_THRESHOLD
)
  

from pathlib import Path
import chromadb
from chromadb import PersistentClient



import hashlib 

# -----------------------------
# 설정
# -----------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "false"      # 크로마 텔레메트리 끄기
# 혹시 환경에 따라 아래 키도 지원됩니다(둘 다 넣어도 무해).
os.environ["CHROMADB_TELEMETRY_ENABLED"] = "false"



# ---- E5 임베딩 클래스 (sentence-transformers 버전) ----

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
            normalize_embeddings=True, 
            show_progress_bar=(show_progress if show_progress is not None else (len(texts) > batch_size))
        )
        return embs.astype(np.float32, copy=False)

    def encode_passages(self, passages):
        return self._encode([f"passage: {p}" for p in passages])

    def encode_queries(self, queries):
        return self._encode([f"query: {q}" for q in queries])


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

def _guess_kind_from_text(t: str) -> str:
    # 간단 휴리스틱: '제 n 조' 패턴 많으면 법령으로 간주
    return "law" if re.search(r"제\s*\d+\s*조", t) else "generic"

class RAGIndexer:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.model_name = model_name
        self.embedder = E5Embedder(MODEL_NAME, device='cpu')

    def build_many(self, pdf_paths: List[str], index_dir=INDEX_DIR, kind: str = "law"):
        all_chunks, all_sources = [], []
        metas = []   # ★ 추가: 청크별 메타 담을 리스트
        total = 0
        for pdf_path in pdf_paths:
            print(f"📄 PDF 읽는 중: {pdf_path}")
            raw = load_pdf_text(pdf_path)
            if not raw.strip():
                raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. OCR이 필요할 수 있습니다.")

            
            # kind 자동 추정(명시 인자가 'auto'면 추정, 아니면 그대로 사용)
            use_kind = _guess_kind_from_text(raw) if kind == "auto" else kind

            print(f"🔪 청크 분할 ({use_kind}) 중...")
            if use_kind == "law":
                chunks, labels = chunk_law_text(raw, by_article=True, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            else:
                chunks, labels = _chunk_generic(raw, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

            base = os.path.basename(pdf_path)
            all_chunks.extend(chunks)
            all_sources.extend([f"{base}::{label}" for label in labels])
            
            # --- 메타 작성 ---
            if use_kind == "law":
                law_name = _norm_law_name_from_filename(base)
                for lbl in labels:
                    m = _parse_label_to_meta(lbl)  # (조/항 숫자 추출; 없으면 None들)
                    m.update({
                        "law": law_name,
                        "source": f"{base}::{lbl}",
                        "doc_type": "law",
                        "filename": base,
                    })
                    metas.append(_sanitize_meta(m))
            else:
                for lbl in labels:
                    metas.append(_sanitize_meta({
                        "source": f"{base}::{lbl}",
                        "doc_type": "attachment",
                        "filename": base,
                    }))
                    
            total += len(chunks)
            print(f"  → {base}: {use_kind} 기준 {len(chunks)}개")

        if total == 0:
            raise ValueError("인덱싱할 청크가 없습니다.")

        print(f"🧠 임베딩 계산({self.model_name})... 총 청크 {total}개")
        vectors = self.embedder.encode_passages(all_chunks)  # (N, dim) float32
        if vectors.shape[0] != len(all_chunks):
            raise RuntimeError(f"벡터 수({vectors.shape[0]})와 청크 수({len(all_chunks)}) 불일치")
        dim = vectors.shape[1]
        print(f"  → dim={dim}")

        # --- CHROMA: 영구 클라이언트/컬렉션 생성 ---
        chroma_dir = Path(index_dir)                      # ★ 변경 포인트
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = PersistentClient(path=str(chroma_dir))

        # 중복 빌드를 피하려면 기존 컬렉션 삭제 후 재생성(선택)
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"🗑️ 기존 컬렉션 '{CHROMA_COLLECTION}' 삭제")
        except Exception:
            pass

        collection = client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine", "model_name": self.model_name,},
        )

        print("📦 ChromaDB 업서트(add) 중...")
        ids = [f"doc-{i}" for i in range(len(all_chunks))]
        assert len(metas) == len(all_chunks), f"metas({len(metas)}) != chunks({len(all_chunks)})"
        collection.add(ids=ids, documents=list(all_chunks), embeddings=vectors.tolist(), metadatas=metas)

        print(f"✅ 저장 완료 | 총 청크 {total}개") 


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

def _where_all(**kv):
    terms = [{k: v} for k, v in kv.items() if v is not None]
    if not terms:
        return None
    return {"$and": terms}
    

class RAGRetriever:
    def __init__(self, index_dir=INDEX_DIR, device=None, collection_name=CHROMA_COLLECTION):
        # --- CHROMA: 로드 (빌드 시 사용한 경로 그대로) ---
        chroma_dir = Path(index_dir)

        # (옵션) 사용자가 "laws" 같은 네임스페이스만 준 경우를 위한 폴백
        if not chroma_dir.exists() and not chroma_dir.is_absolute():
            maybe = Path(INDEX_DIR) / index_dir
            if maybe.exists():
                chroma_dir = maybe

        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"Chroma 인덱스가 없습니다: {chroma_dir}\n"
                "먼저 `python rag.py build --dir <폴더>` 또는 `--pdf <파일...>`로 빌드하세요."
            )

        client = PersistentClient(path=str(chroma_dir))
        try:
            self.collection = client.get_collection(collection_name)
        except Exception as e:
            raise FileNotFoundError(
                f"Chroma 컬렉션 '{collection_name}'을(를) {chroma_dir}에서 찾을 수 없습니다."
            ) from e

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

    def _hits_to_dicts(self, hits: List[tuple[int, float]], retriever_tag: str) -> List[Dict]:
        out = []
        for idx, sim in hits:
            out.append({
                "idx": idx,
                "text": self.chunks[idx],
                "sim": float(sim),
                "source": self.sources[idx] if idx < len(self.sources) else "unknown",
                "retriever": retriever_tag,   # "A" or "B" 등 외부 식별 태그
            })
        return out

    def global_search(self, query: str, n: int, retriever_tag: str) -> List[Dict]:
        if self._n_index <= 0:
            return []
        q_emb = self.embedder.encode_queries([query]).astype("float32").tolist()
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=min(int(n), self._n_index),
            include=["distances"],
        )
        hits = self._to_hits(res)
        return self._hits_to_dicts(hits, retriever_tag)

    def filtered_search(self, query: str, n: int, retriever_tag: str, use_clause: bool = True) -> List[Dict]:
        if self._n_index <= 0:
            return []
        law, a_num, a_bis, clause, _article_key = _extract_explicit_law_and_article(query)
        if not (law and (a_num is not None)):
            return []  # 명시적 법/조문 없으면 필터 검색 생략

        q_emb = self.embedder.encode_queries([query]).astype("float32").tolist()

        where = _where_all(law=law, article_num=a_num, article_bis=a_bis)
        hits_filtered = []

        if use_clause and (clause is not None):
            where_clause = _where_all(law=law, article_num=a_num, article_bis=a_bis, clause=clause)
            res_f1 = self.collection.query(
                query_embeddings=q_emb, n_results=min(int(n), self._n_index),
                where=where_clause, include=["distances"]
            )
            hits_filtered = self._to_hits(res_f1)
            if len(hits_filtered) < min(int(n), self._n_index):
                res_f2 = self.collection.query(
                    query_embeddings=q_emb, n_results=min(int(n), self._n_index),
                    where=where, include=["distances"]
                )
                add = self._to_hits(res_f2)
                # idx 기준 dedup
                seen = {i for i, _ in hits_filtered}
                hits_filtered += [h for h in add if h[0] not in seen]
        else:
            res_f = self.collection.query(
                query_embeddings=q_emb, n_results=min(int(n), self._n_index),
                where=where, include=["distances"]
            )
            hits_filtered = self._to_hits(res_f)

        return self._hits_to_dicts(hits_filtered, retriever_tag)
        
    def get_passages(self, hits: List[Tuple[int, float]]) -> List[str]:
        return [self.chunks[i] for i, _ in hits]

    def get_sources(self, hits: List[Tuple[int, float]]) -> List[str]:
        return [self.sources[i] for i, _ in hits]


# -----------------------------
# RAG 추론 함수
# -----------------------------


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def _key_pair(c: Dict) -> str:
    # (retriever_tag, idx) 페어 기준으로 중복 제거 (가장 안전)
    return f'{c["retriever"]}:{c["idx"]}'

def fuse_rrf(candidate_lists: List[List[Dict]], keep_for_ce: int = 50) -> List[Dict]:
    merged = {}
    for cand in candidate_lists:
        for rank, c in enumerate(cand, start=1):
            k = _key_pair(c)
            if k not in merged:
                merged[k] = {**c, "rrf": _rrf(rank)}
            else:
                merged[k]["rrf"] += _rrf(rank)
    return sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)[:keep_for_ce]

def fuse_rrf_global_only(candidate_lists: List[List[Dict]], keep_for_ce: int = 50) -> List[Dict]:
    merged = {}
    for cand in candidate_lists:
        for rank, c in enumerate(cand, start=1):
            k = _key_pair(c)
            if k not in merged:
                merged[k] = {**c, "rrf": _rrf(rank)}
            else:
                merged[k]["rrf"] += _rrf(rank)
    return sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)[:keep_for_ce]
    
def rerank_and_pick(query: str, candidates: List[Dict], reranker, topn: int = 1, batch_size: int = 32) -> List[Dict]:
    if not candidates:
        return []
    passages = [c["text"] for c in candidates]
    ce_scores = reranker.score(query, passages, batch_size=batch_size)
    for c, s in zip(candidates, ce_scores):
        c["ce_score"] = float(s)
    return sorted(candidates, key=lambda x: x["ce_score"], reverse=True)[:topn]

def search_two_indexes_pipeline(
    query: str,
    retrA, retrB,          # RAGRetriever
    M_generic_A: int, M_filtered_A: int, M_generic_B: int,
    reranker, keep_for_ce: int, topn: int, use_clause: bool = True
) -> List[Dict]:
    # A: 전역 + (있으면) 필터
    cand_gA = retrA.global_search(query, n=M_generic_A, retriever_tag="A")
    cand_fA = retrA.filtered_search(query, n=M_filtered_A, retriever_tag="A", use_clause=use_clause)
    # B: 전역
    cand_gB = retrB.global_search(query, n=M_generic_B, retriever_tag="B")

    # 융합 → rerank
    candidates = fuse_rrf([cand_fA, cand_gA, cand_gB], keep_for_ce=keep_for_ce)
    final = rerank_and_pick(query, candidates, reranker=reranker, topn=topn, batch_size=32)
    return final            # Dict 리스트 (각 원소에 text/source/retriever/idx/ce_score 포함)

def search_two_indexes_global_only(
    query: str,
    retrA, retrB,                 # RAGRetriever
    M_generic_A: int = 20,
    M_generic_B: int = 20,
    reranker=None,
    keep_for_ce: int = 50,
    topn: int = 3,
) -> List[Dict]:
    # 1) 전역 검색만 실행 (A/B)
    cand_gA = retrA.global_search(query, n=M_generic_A, retriever_tag="A")
    cand_gB = retrB.global_search(query, n=M_generic_B, retriever_tag="B")

    # 2) 융합(RRF) → 3) CE rerank
    candidates = fuse_rrf_global_only([cand_gA, cand_gB], keep_for_ce=keep_for_ce)
    final = rerank_and_pick(query, candidates, reranker=reranker, topn=topn, batch_size=32)
    return final  # [{"idx","text","source","retriever","ce_score",...}, ...]
    
def answer_with_rag(
    question: str,
    retrieverA: RAGRetriever,                # ★ 바뀜: A 인덱스
    retrieverB: RAGRetriever,                # ★ 바뀜: B 인덱스
    pipe,
    top_k=TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    reranker: Optional["STReranker"] = None,
    rerank_threshold: float = RERANK_THRESHOLD,
    M_generic: int = M_GENERIC,              # A의 전역 후보 (기존 의미 유지)
    M_filtered: int = M_FILTERED,            # A의 필터 후보
    M_generic_B: int = None,                 # ★ 추가: B 전역 후보 (기본 없으면 M_generic과 동일)
    keep_for_ce: int = 50,                   # ★ 추가: CE에 태울 최대 후보 수
    use_clause: bool = True,                 # ★ 추가: 항 단위까지 필터
):
    if M_generic_B is None:
        M_generic_B = M_generic

    is_mc, _ = is_multiple_choice(question)

    # --- 1) 검색 ---
    if reranker is not None and is_mc:
        # A(전역+필터) + B(전역) → RRF → CE rerank
        final = search_two_indexes_pipeline(
            query=question,
            retrA=retrieverA, retrB=retrieverB,
            M_generic_A=M_generic, M_filtered_A=M_filtered, M_generic_B=M_generic_B,
            reranker=reranker, keep_for_ce=keep_for_ce, topn=max(1, top_k),
            use_clause=use_clause,
        )
        passages = [c["text"] for c in final[:top_k]]
        top_score = final[0]["ce_score"] if final else float("-inf")
        use_context = (len(final) > 0) and (top_score >= rerank_threshold)

        # 디버깅/호환용: hits는 (pseudo) 튜플로 만들어 두되, 인덱스 구분을 위해 문자열 키 사용
        hits = [(f'{c["retriever"]}:{c["idx"]}', c["ce_score"]) for c in final[:top_k]]

        # 멀티 인덱스이므로 contexts는 passages 자체를 씀
        contexts = passages if use_context else []
        top_meta_for_debug = [{"retriever": c["retriever"], "source": c.get("source","unknown"), "score": c["ce_score"]} for c in final[:top_k]]

    else:
        # --- 주관식 처리: 두 인덱스 전역 검색만 수행 후 rerank ---
        final = search_two_indexes_global_only(
            query=question,
            retrA=retrieverA,
            retrB=retrieverB,
            M_generic_A=M_generic,     # laws 인덱스 전역 후보 수
            M_generic_B=M_generic_B,   # supplement 인덱스 전역 후보 수
            reranker=reranker,
            keep_for_ce=keep_for_ce,
            topn=max(1, top_k),
        )

        passages = [c["text"] for c in final[:top_k]]
        top_score = final[0]["ce_score"] if final else float("-inf")
        use_context = (len(final) > 0) and (top_score >= score_threshold)  # 게이트는 score_threshold 재사용

        # hits와 meta_dbg 준비
        hits = [(f'{c["retriever"]}:{c["idx"]}', c["ce_score"]) for c in final[:top_k]]
        contexts = passages if use_context else []
        top_meta_for_debug = [
            {"retriever": c["retriever"], "source": c.get("source", "unknown"), "score": c["ce_score"]}
            for c in final[:top_k]
        ]

    # --- 2) 프롬프트 ---
    prompt = make_prompt_rag_exaone(question, contexts, use_fewshot=True)

    # --- 3) 1차 생성 ---
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

    # --- 5) 객관식 후처리 ---
    if is_mc:
        m = re.search(r"\b([1-9][0-9]?)\b", gen)
        if not m:
            out = pipe(prompt, max_new_tokens=2, do_sample=True, temperature=0.6, top_p=0.95)
            gen = out[0]["generated_text"]

    # 멀티 인덱스 디버깅 정보를 되돌려 주면 cmd_run에서 소스 로깅이 쉬워짐
    return prompt, gen, passages, hits, use_context, contexts, top_meta_for_debug
