import os
import re
import pickle
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pdfplumber
import faiss

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from prompt import make_prompt_rag_solar
from utils import is_multiple_choice, extract_answer_only
from config import MODEL_NAME, EMBEDDING_MODEL_NAME, TOP_K


# --- 텍스트 전처리 및 청크 분할 함수 (친구 코드) ---
def _clean_text(t: str) -> str:
    t = t.replace("\u3000", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def _chunk_text(text: str, chunk_size, overlap) -> List[str]:
    text = _clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            cut = max(chunk.rfind("\n"), chunk.rfind("."), chunk.rfind("다."), chunk.rfind("다\n"))
            if cut > int(chunk_size * 0.6):
                chunk = chunk[:cut+1]
                end = start + len(chunk)
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
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

# 법령 전용 정규식
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

def parse_korean_law_articles(raw_text: str):
    text = _clean_text(raw_text)
    text = re.sub(r"(제\s*\d+\s*조)(\s*제\s*\d+\s*항)", r"\1 \2", text)
    matches = list(ARTICLE_RE_STRICT.finditer(text))
    if not matches:
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
        head_full = header + (f"({title})" if title else "")
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

def split_article_if_long(article_text: str, max_len: int, overlap: int):
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
            s = 0
            while s < len(p):
                e = min(len(p), s + max_len)
                chunk = p[s:e]
                cut = max(chunk.rfind("\n"), chunk.rfind("."), chunk.rfind("다."), chunk.rfind("다\n"))
                if cut > int(len(chunk) * 0.6) and e < len(p):
                    e = s + cut + 1
                    chunk = p[s:e]
                chunks.append(chunk.strip())
                s = max(e - overlap, e)
    return [c for c in chunks if c]

def chunk_law_text(raw_text: str, by_article: bool, chunk_size: int, overlap: int):
    if not by_article:
        return _chunk_text(raw_text, chunk_size, overlap)
    articles = parse_korean_law_articles(raw_text)
    chunks = []
    for a in articles:
        subchunks = split_article_if_long(a["text"], max_len=max(800, chunk_size*2//1), overlap=overlap)
        chunks.extend(subchunks)
    return chunks

# --- RAG 검색 및 답변 생성 함수 ---
def answer_with_rag(
    question: str,
    vectorstore,
    pipe,
    top_k: int = 20,
    score_threshold: float = 0.89
):
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)
    
    documents = [doc for doc, score in docs_with_scores]
    scores = [score for doc, score in docs_with_scores]

    contexts = [doc.page_content for doc in documents]
    
    # 점수 기반 컨텍스트 제거 로직
    if not scores or scores[0] < score_threshold:
        contexts = []
    
    
    prompt = make_prompt_rag_solar(
        text=question,
        contexts=contexts,
        use_fewshot=True
    )
    
    # LlamaCpp는 __call__에 파라미터를 받지 않습니다.
    answer_text = pipe(prompt)
    
    final_answer = extract_answer_only(
        generated_text=f"답변:{answer_text}",
        original_question=question,
        prompt=prompt
    )

    return final_answer


def setup_retriever(folder_path="data/laws/"):
    print(f"📄 '{folder_path}' 폴더에서 문서를 로딩하고 벡터화합니다.")
    all_documents = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == ".pdf":
                print(f"📄 로딩 중: {file_path}")
                raw_text = load_pdf_text(file_path)
                chunks = chunk_law_text(raw_text, by_article=True, chunk_size=450, overlap=100)
                all_documents.extend([Document(page_content=chunk, metadata={'source': file_path}) for chunk in chunks])
            elif file_extension == ".txt":
                print(f"📄 로딩 중: {file_path}")
                loader = UnstructuredFileLoader(file_path)
                all_documents.extend(loader.load())
            else:
                print(f"⚠️ 경고: 지원되지 않는 파일 형식 스킵 - {file_path}")
    else:
        raise FileNotFoundError(f"'{folder_path}' 폴더를 찾을 수 없습니다.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)
    
    print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("🔍 벡터 스토어(FAISS) 생성 중...")
    vectorstore = LangchainFAISS.from_documents(docs, embeddings)
    print("✅ 벡터 스토어 생성 완료.")
    
    return vectorstore

#### past code
# def preprocess_text(text: str) -> str:
#     """
#     텍스트에서 깨진 문자, 특수 기호, 반복되는 패턴 등을 정제합니다.
#     """
#     # 불필요한 공백과 줄바꿈 정리
#     cleaned_text = re.sub(r'\s+', ' ', text)
#     # 특수 문자 제거 (예: θ, ㎝ 등) - 일반적인 문자가 아닌 것을 제거
#     cleaned_text = re.sub(r'[^\w\s가-힣a-zA-Z0-9.,?!-]', '', cleaned_text)
#     # 과도하게 반복되는 하이픈이나 밑줄 제거
#     cleaned_text = re.sub(r'-{3,}', ' ', cleaned_text)
#     # 한 글자짜리 문자를 제거 (주로 OCR 오류나 잔여물)
#     cleaned_text = ' '.join([word for word in cleaned_text.split() if len(word) > 1 or re.match(r'[a-zA-Z0-9]', word)])
    
#     return cleaned_text.strip()

# def setup_retriever(folder_path="data/laws/"):
#     """
#     Load documents from a folder and set up the retriever.
#     """
#     print(f"📄 '{folder_path}' 폴더에서 문서를 로딩하고 벡터화합니다.")
    
#     all_documents = []
    
#     # 폴더 존재 여부 확인
#     if os.path.exists(folder_path):
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
            
#             file_extension = os.path.splitext(file_path)[1].lower()
            
#             if file_extension == ".pdf":
#                 print(f"📄 로딩 중: {file_path}")
#                 # PyPDFLoader 대신 PyMuPDFLoader 사용
#                 loader = PyMuPDFLoader(file_path)
#                 documents = loader.load()
                
#                 # 사전 텍스트 정제 적용
#                 for doc in documents:
#                     doc.page_content = preprocess_text(doc.page_content)
                
#                 all_documents.extend(documents)
                
#             elif file_extension == ".txt":
#                 print(f"📄 로딩 중: {file_path}")
#                 loader = UnstructuredFileLoader(file_path)
#                 all_documents.extend(loader.load())
#             else:
#                 print(f"⚠️ 경고: 지원되지 않는 파일 형식 스킵 - {file_path}")

#     else:
#         raise FileNotFoundError(f"'{folder_path}' 폴더를 찾을 수 없습니다.")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#     docs = text_splitter.split_documents(all_documents)
    
#     print(f"✨ 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
#     print("🔍 벡터 스토어(FAISS) 생성 중...")
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     print("✅ 벡터 스토어 생성 완료.")
    
#     return vectorstore.as_retriever()



# def rerank_documents(question, documents, k: int = 5):
#     """
#     RAG 시스템에서 검색된 문서들을 질문과의 관련성을 기준으로 재순위화합니다.
#     - question: 사용자 질문
#     - documents: 검색 모듈에서 반환된 문서 목록
#     - k: 최종적으로 반환할 상위 문서의 개수
#     """
#     if not documents:
#         print("경고: 재순위화할 문서가 없습니다.")
#         return []

#     print(f"🔍 {len(documents)}개의 문서를 재순위화합니다...")

#     # 재순위화 모델 로딩 (CrossEncoder 모델 사용)
#     # 이 모델은 질문-문서 쌍의 관련성 점수를 매기는 데 특화되어 있습니다.
#     reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')
    
#     # 질문과 각 문서의 내용을 쌍으로 만들어 리스트를 생성
#     pairs = [[question, doc.page_content] for doc in documents]
    
#     # 모델을 사용하여 각 쌍에 대한 점수를 예측
#     scores = reranker.predict(pairs)
    
#     # 점수와 문서를 튜플로 묶은 후, 점수 기준으로 내림차순 정렬
#     scored_documents = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
#     print("✅ 재순위화 완료. 상위 K개 문서 반환.")
    
#     # 상위 k개 문서만 반환
#     reranked_docs = [doc for score, doc in scored_documents[:k]]
    
#     return reranked_docs