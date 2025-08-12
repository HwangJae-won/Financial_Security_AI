# rag.py
# -*- coding: utf-8 -*-
"""
전자금융거래법 PDF를 RAG로 활용하는 모듈/스크립트.
- PDF → 텍스트 청크 → 임베딩 → FAISS 인덱스 생성/저장
- 질문 시 상위 k개 청크 검색 후, 컨텍스트를 포함한 프롬프트로 LLM 호출
- 객관식/주관식 모두 지원(기존 utils.py의 is_multiple_choice, extract_question_and_choices 사용)

필요 패키지:
    pip install pdfplumber sentence-transformers faiss-cpu

사용 예:
    # 1) 인덱스 빌드
    python rag.py build --pdf ./data/전자금융거래법_제19734호.pdf

    # 2) 단일 질문
    python rag.py ask --question "전자금융거래법 제6조의 핵심은 무엇인가?"

    # 3) test.csv 한 번에 추론(컬럼명: Question)
    python rag.py run --csv ./data/test.csv

인덱스/메타는 ./rag_index/ 아래 저장됩니다.
"""

import os
import re
import json
import argparse
import pickle
from typing import List, Tuple, Optional

import numpy as np

# 외부 패키지
import pdfplumber
from transformers import AutoTokenizer, AutoModel
import torch
import faiss

# 기존 프로젝트 모듈
from model import load_model
from utils import is_multiple_choice, extract_question_and_choices
from prompt import make_prompt_auto  # 필요시만 사용(우리는 전용 프롬프트도 제공)


# ---- E5 임베딩 클래스 (sentence-transformers 대체) ----
class E5Embedder:
    """
    intfloat/multilingual-e5-base 를 transformers로 직접 로드.
    - query에는 'query: ' 프리픽스
    - passage에는 'passage: ' 프리픽스
    - mean-pooling + L2 normalize
    """
    def __init__(self, model_name="intfloat/multilingual-e5-base", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)  # torch==2.1.0 안전
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
# 설정
# -----------------------------
INDEX_DIR = "./rag_index"
INDEX_BIN = os.path.join(INDEX_DIR, "faiss.index")
META_PKL = os.path.join(INDEX_DIR, "meta.pkl")
MODEL_NAME = "intfloat/multilingual-e5-base"   # 한글 안정: e5-base 다국어
CHUNK_SIZE = 256        # 청크 길이(문자 수 기준)
CHUNK_OVERLAP = 120     # 청크 겹침
TOP_K = 3               # 검색 상위 k개

# -----------------------------
# 유틸
# -----------------------------
def _ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _clean_text(t: str) -> str:
    t = t.replace("\u3000", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
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

# -----------------------------
# 인덱서/검색기
# -----------------------------
class RAGIndexer:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.model_name = model_name
        self.embedder = E5Embedder(model_name, device='cpu')

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        return emb.astype("float32")

    def build(self, pdf_path: str, index_dir=INDEX_DIR):
        print(f"📄 PDF 읽는 중: {pdf_path}")
        raw = load_pdf_text(pdf_path)
        if not raw.strip():
            raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. OCR이 필요할 수 있습니다.")

        print("🔪 청크 분할 중...")
        chunks = _chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"✅ 청크 개수: {len(chunks)}")

        print(f"🧠 임베딩 계산({self.model_name})...")
        vectors = self.embedder.encode_passages(chunks)
        dim = vectors.shape[1]

        print("📦 FAISS 인덱스 생성/저장...")
        _ensure_dir(index_dir)
        index = faiss.IndexFlatIP(dim)  # 내적(코사인 유사도는 정규화 완료 기준)
        index.add(vectors)
        faiss.write_index(index, INDEX_BIN)
        meta = {"chunks": chunks, "model_name": self.model_name}
        with open(META_PKL, "wb") as f:
            pickle.dump(meta, f)
        print(f"✅ 저장 완료: {INDEX_BIN}, {META_PKL}")

class RAGRetriever:
    def __init__(self, index_dir=INDEX_DIR, device=None):
        if not (os.path.exists(INDEX_BIN) and os.path.exists(META_PKL)):
            raise FileNotFoundError("인덱스가 없습니다. 먼저 `python rag.py build --pdf <path>`를 실행하세요.")
        self.index = faiss.read_index(INDEX_BIN)
        with open(META_PKL, "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        self.model_name = meta.get("model_name", MODEL_NAME)

        # 검색시 사용할 동일 임베더 로드
        self.embedder = E5Embedder(self.model_name, device='cpu')

    def search(self, query: str, top_k=TOP_K) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode_queries([query]).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            results.append((idx, float(score)))
        return results

    def get_passages(self, hits: List[Tuple[int, float]]) -> List[str]:
        return [self.chunks[i] for i, _ in hits]

# -----------------------------
# 프롬프트 생성 (RAG 전용)
# -----------------------------
def make_prompt_with_context(original_text: str, contexts: List[str]) -> str:
    """
    - 객관식: 기존 질문/선택지는 그대로 유지하고, 맨 아래에 [참고자료] 블록으로 컨텍스트 추가
    - 주관식: 질문 아래에 [참고자료] 블록 추가
    (주의) 질문 문자열 자체에 컨텍스트를 섞어 넣지 말아야 utils의 선택지 파서가 깨지지 않음.
    """
    context_block = "[참고자료]\n" + "\n\n---\n".join(contexts) if contexts else ""
    is_mc, _ = is_multiple_choice(original_text)
    if is_mc:
        question, options = extract_question_and_choices(original_text)
        role_instruction = "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 한국어로 아래 지시를 따르세요."
        prompt = (
            f"{role_instruction}\n\n"
            f"**지시:** 다음 질문은 객관식입니다. 참고자료를 바탕으로 **가장 적절한 정답 번호만** 출력하세요.\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            f"{context_block}\n\n"
            "답변:"
        )
        return prompt
    else:
        role_instruction = "당신은 금융보안 전문가이자, 금융보안원 소속의 베테랑 연구원입니다. 한국어로 아래 지시를 따르세요."
        prompt = (
            f"{role_instruction}\n\n"
            f"**지시:** 참고자료의 원문을 절대 그대로 복사하지 말고, 질문에 맞는 핵심 내용을 3문장 이내로 요약·재구성하세요. 불필요한 세부사항은 생략하고, 핵심 키워드만 포함하세요.\n\n"
            f"질문: {original_text}\n\n"
            f"{context_block}\n\n"
            "답변:"
        )
        return prompt

# -----------------------------
# RAG 추론 함수
# -----------------------------
def answer_with_rag(question: str, retriever: RAGRetriever, pipe, top_k=TOP_K):
    hits = retriever.search(question, top_k=top_k)
    passages = retriever.get_passages(hits)
    prompt = make_prompt_with_context(question, passages)

    # 너 환경의 decode 정책 맞춤: 1차 greedy → 실패 시 샘플링
    out = pipe(prompt, max_new_tokens=128, temperature=0.0)
    gen = out[0]["generated_text"]

    # 간단 후처리(필요시 강화)
    # 객관식이면 숫자만, 주관식이면 문장 정리
    is_mc, _ = is_multiple_choice(question)
    if is_mc:
        # 가능한 숫자만 추출(1~99), 없으면 샘플링 재시도
        m = re.search(r"\b([1-9][0-9]?)\b", gen)
        if not m:
            out = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.6, top_p=0.95)
            gen = out[0]["generated_text"]
    return prompt, gen, passages, hits

# -----------------------------
# CLI
# -----------------------------
def cmd_build(args):
    _ensure_dir(INDEX_DIR)
    indexer = RAGIndexer(MODEL_NAME, device="cpu")
    indexer.build(args.pdf, INDEX_DIR)

def cmd_ask(args):
    pipe = load_model()
    retr = RAGRetriever(INDEX_DIR, device="cpu")
    q = args.question
    prompt, gen, passages, hits = answer_with_rag(q, retr, pipe, top_k=args.top_k)
    print("\n===== 프롬프트 (앞부분) =====")
    print(prompt[:500] + "...")
    print("\n===== 생성된 답변 =====")
    print(gen)
    print("\n===== 참고된 청크 (점수순) =====")
    for (idx, score), p in zip(hits, passages):
        print(f"\n[score={score:.3f}] chunk#{idx}\n{p[:400]}...")

def cmd_run(args):
    import pandas as pd
    pipe = load_model()
    retr = RAGRetriever(INDEX_DIR, device="cpu")
    df = pd.read_csv(args.csv)
    preds = []
    for i, q in enumerate(df["Question"]):
        prompt, gen, passages, hits = answer_with_rag(q, retr, pipe, top_k=args.top_k)
        # 기존 파이프라인의 extract_answer_only를 재사용하고 싶다면:
        from utils import extract_answer_only  # <- 프로젝트 경로에 맞게 수정 or 직접 구현
        # 여기서는 간단히 원문 gen을 그대로 저장(후처리는 기존 코드에 맞춰 적용 권장)
        preds.append(gen)
        if args.verbose and i < 5:
            print(f"\n[#{i}] Q={q[:80]}...")
            print(f"[gen] {gen[:200]}...")

    out_csv = os.path.splitext(args.csv)[0] + "_rag_output.csv"
    df["RAG_Generation"] = preds
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장 완료: {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="RAG for 전자금융거래법")
    sub = parser.add_subparsers()

    p_build = sub.add_parser("build", help="PDF에서 인덱스 생성")
    p_build.add_argument("--pdf", required=True, help="PDF 파일 경로")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="단일 질문에 RAG 적용")
    p_ask.add_argument("--question", required=True, help="질문 텍스트")
    p_ask.add_argument("--top_k", type=int, default=TOP_K)
    p_ask.set_defaults(func=cmd_ask)

    p_run = sub.add_parser("run", help="CSV(Question 컬럼) 일괄 추론")
    p_run.add_argument("--csv", required=True, help="CSV 경로 (Question 컬럼 필요)")
    p_run.add_argument("--top_k", type=int, default=TOP_K)
    p_run.add_argument("--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
