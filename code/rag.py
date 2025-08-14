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
    !python code/rag.py ask --question "전자금융거래법 제6조의 핵심은 무엇인가?"
    !python code/rag.py ask --question "전자금융업자가 전자금융거래법 제35조에 따라 겸업제한을 위반할 경우, 어떤 조치를 받을 수 있는가? 1 과태료 부과 2 형사처벌 3 영업정지 4 경고"
    !python code/rag.py ask --question $'전자서명법 제22조에 따른 분쟁 조정의 주체로 올바른 것은?\n1 한국인터넷진흥원\n2 과학기술정보통신부\n3 금융감독원\n4 개인정보보호위원회'

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


# ---- E5 임베딩 클래스 (sentence-transformers 대체) ----
class E5Embedder:
    """
    intfloat/multilingual-e5-small 를 transformers로 직접 로드.
    - query에는 'query: ' 프리픽스
    - passage에는 'passage: ' 프리픽스
    - mean-pooling + L2 normalize
    """
    def __init__(self, model_name="intfloat/multilingual-e5-small", device=None):
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
MODEL_NAME = "intfloat/multilingual-e5-small"   # 한글 안정: e5-base 다국어
CHUNK_SIZE = 480        # 청크 길이(문자 수 기준)
CHUNK_OVERLAP = 80     # 청크 겹침
TOP_K = 1               # 검색 상위 k개

SCORE_THRESHOLD = 0.9
OUTPUT_PATH = "results/"
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

    def build_many(self, pdf_paths: List[str], index_dir=INDEX_DIR):
        all_chunks, all_sources = [], []
        total = 0
        for pdf_path in pdf_paths:
            print(f"📄 PDF 읽는 중: {pdf_path}")
            raw = load_pdf_text(pdf_path)
            if not raw.strip():
                raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. OCR이 필요할 수 있습니다.")
            print("🔪 청크 분할 중...")
            chunks = _chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
            all_chunks.extend(chunks)
            all_sources.extend([os.path.basename(pdf_path)] * len(chunks))
            total += len(chunks)
            print(f"  → {os.path.basename(pdf_path)}: {len(chunks)}개")
        
        if total == 0:
            raise ValueError("인덱싱할 청크가 없습니다.")

        print(f"🧠 임베딩 계산({self.model_name})... 총 청크 {total}개")
        vectors = self.embedder.encode_passages(all_chunks)
        if vectors.shape[0] != len(all_chunks):
            raise RuntimeError(f"벡터 수({vectors.shape[0]})와 청크 수({len(all_chunks)}) 불일치")
        dim = vectors.shape[1]

        print("📦 FAISS 인덱스 생성/저장...")
        _ensure_dir(index_dir)
        index = faiss.IndexFlatIP(dim)  # 내적(코사인; 임베딩 L2 정규화 가정)
        index.add(vectors)

        # 원자적 저장(권장)
        tmp_idx = INDEX_BIN + ".tmp"
        tmp_meta = META_PKL + ".tmp"
        faiss.write_index(index, tmp_idx)
        meta = {
            "chunks": all_chunks,
            "sources": all_sources,
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
        self.embedder = E5Embedder(self.model_name, device='cpu')

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
